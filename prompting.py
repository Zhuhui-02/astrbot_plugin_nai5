import json
import re
from functools import lru_cache
from pathlib import Path

SKILLS = Path(__file__).parent / "vendor" / "nai5-prompting" / "skills"


class PromptError(ValueError):
    pass


def direct_draw_request(text, presets):
    """Match an explicit drawing request at the start of a chat message."""
    text = text.strip()
    draw = r"(?:帮我画|给我画|为我画|画一张|画一幅)"
    prefix = re.match(r"(?:请\s*)?" + draw + r"(?:一张|一幅)?\s*[：:]?\s*", text)
    scene = text[prefix.end():] if prefix else text
    for name in sorted(presets, key=len, reverse=True):
        style = r"(?:请)?(?:用|按|按照)(?:我的)?[「“]?" + re.escape(name) + r"[」”]?(?:的)?(?:预设|画风|风格)?"
        if prefix:
            match = re.match(style + r"(?:来画|来一张|画一张|画一幅|画|的)?[，,：:\s]*", scene)
        else:
            match = re.match(style + r"(?:帮我画|给我画|来画|画|来一张)(?:一张|一幅)?[，,：:\s]*", scene)
        if match and scene[match.end():].strip():
            return scene[match.end():].strip(), name
    return (scene, "") if prefix and scene else None


def style_presets(config):
    presets = {}
    for entry in config.get("style_presets", []):
        parts = re.split(r"[|｜]", entry, maxsplit=1) if isinstance(entry, str) else []
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise PromptError("画风预设格式有误，请在配置中按「名称｜画风说明」填写每一项。")
        name, brief = (part.strip() for part in parts)
        if name == "无" or name in presets or "】" in name:
            raise PromptError(f"请检查画风名称「{name}」：名称须唯一，且不能用「无」或含有 】。")
        presets[name] = brief
    return presets


def select_style(config, description, preset_name=""):
    name = preset_name.strip() or config.get("default_style", "").strip()
    override = re.match(r"^【画风[:：]([^】]+)】\s*", description)
    if override:
        name = override[1].strip()
        description = description[override.end():]
    if not description.strip():
        raise PromptError("请在画风名称后填写要画的内容。")
    if not name or name == "无":
        return description, "", ""
    presets = style_presets(config)
    if name not in presets:
        raise PromptError(f"找不到画风「{name}」，请用 /nai画风 查看可用名称。")
    return description, name, presets[name]


@lru_cache(maxsize=32)
def section(skill, filename, heading):
    source = (SKILLS / skill / "references" / filename).read_text(encoding="utf-8")
    lines = source.splitlines()
    start = lines.index(heading)
    level = len(heading.split(" ", 1)[0])
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^#{1," + str(level) + r"} ", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def instructions(description, *, use_style=False):
    excerpts = [
        section("nai5-illustration-composition", "composition.md", "## 镜头、空间与主次"),
        section("nai5-writing", "conventions.md", "## 用户事实、原串与本次可见范围"),
        section("nai5-writing", "conventions.md", "## 字段职责：先选画面类型"),
        section("nai5-writing", "conventions.md", "## 完整导出的质量词与 UC"),
        section("nai5-writing", "illustration.md", "### 主提示词里的顺序"),
        section("nai5-writing", "illustration.md", "### 空间关系与归属"),
        section("nai5-writing", "illustration.md", "### 角色栏放多少：按原创或版权角色区分"),
    ]
    if re.search(r"漫画|四格|分镜|comic|manga", description, re.I):
        excerpts.append(section("nai5-comic-writing", "comics.md", "### 字段模板与先后顺序"))
    if re.search(r"服装|穿搭|衣装|服饰|outfit|costume", description, re.I):
        excerpts.append(section("nai5-costume-writing", "costume.md", "### 将服装放入完整画面"))
    if use_style:
        excerpts.append(section("nai5-writing", "illustration.md", "### 样式块：画师与技法串的组织"))
        excerpts.append(
            "本次输入是 JSON，scene 是用户画面，style_brief 是选定的画风资料。"
            "按画风选择与场景呼应的线条、上色、光影或媒材表达，整合进最终英文提示词。"
            "保留用户指定的主体、动作、色彩与文字，发生冲突时以用户的明确要求为准。"
            "资料中明确指定的固定标签串应保留原文、权重及顺序。画风名仅用于标识。"
        )
    return "\n\n".join(excerpts) + """
根据以上方法把用户的画面描述编译成一张 NovelAI V5 插画。只返回一个 JSON 对象。
输入为 JSON 时，scene 是当前用户的要求。仅当用户引用前文（如“沿用刚才的角色”）时，用 context_hint 补足所指对象；发生冲突以 scene 为准。
提交前逐项对照用户指定的人数、外观、左右位置、谁对谁做什么、道具归属和画中文字。让主串、每位角色的描述和坐标彼此一致。
格式 {"base":"主提示词","negative":"","characters":[{"prompt":"角色栏","x":0.5,"y":0.5}]}。
提示词用英文，画面内指定原文逐字保留。无角色栏时返回空数组。x、y 是角色中心的
归一化坐标，从左上角 (0,0) 到右下角 (1,1)。不要输出示例占位符、字段标题或解释。
"""


def parse_prompt(raw):
    try:
        obj = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I))
        if not isinstance(obj, dict) or not isinstance(obj["base"], str) or not obj["base"].strip():
            raise ValueError
        if not isinstance(obj["negative"], str) or not isinstance(obj["characters"], list):
            raise ValueError
        if len(obj["characters"]) > 32:
            raise ValueError
        for char in obj["characters"]:
            if not isinstance(char["prompt"], str) or not char["prompt"].strip():
                raise ValueError
            for axis in ("x", "y"):
                if type(char[axis]) not in (int, float) or not 0 <= char[axis] <= 1:
                    raise ValueError
        return obj
    except (ValueError, AttributeError, TypeError, KeyError):
        raise PromptError("聊天模型返回的提示词格式有误，请重新预览。") from None
