import json
import re
from functools import lru_cache
from pathlib import Path

SKILLS = Path(__file__).parent / "vendor" / "nai5-prompting" / "skills"


class PromptError(ValueError):
    pass


@lru_cache(maxsize=32)
def section(skill, filename, heading):
    source = (SKILLS / skill / "references" / filename).read_text(encoding="utf-8")
    lines = source.splitlines()
    start = lines.index(heading)
    level = len(heading.split(" ", 1)[0])
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^#{1," + str(level) + r"} ", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def instructions(description):
    excerpts = [
        section("nai5-illustration-composition", "composition.md", "## 镜头、空间与主次"),
        section("nai5-writing", "conventions.md", "## 字段职责：先选画面类型"),
        section("nai5-writing", "conventions.md", "## 完整导出的质量词与 UC"),
        section("nai5-writing", "illustration.md", "### 主提示词里的顺序"),
        section("nai5-writing", "illustration.md", "### 角色栏放多少：按原创或版权角色区分"),
    ]
    if re.search(r"漫画|四格|分镜|comic|manga", description, re.I):
        excerpts.append(section("nai5-comic-writing", "comics.md", "### 字段模板与先后顺序"))
    if re.search(r"服装|穿搭|衣装|服饰|outfit|costume", description, re.I):
        excerpts.append(section("nai5-costume-writing", "costume.md", "### 将服装放入完整画面"))
    return "\n\n".join(excerpts) + """
根据以上方法把用户的画面描述编译成一张 NovelAI V5 插画。只返回一个 JSON 对象。
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
