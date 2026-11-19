import asyncio
import json
import os
import secrets
from copy import deepcopy
from collections import OrderedDict
from time import monotonic

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain
from astrbot.api.star import Context, Star, register

from .novelai import GenerationError, MODELS, SIZES, generate, payload
from .source_image import read_source
from .prompting import PromptError, instructions, parse_prompt, select_style, style_presets


@register("astrbot_plugin_nai5", "Zhuhui-02", "使用 nai5-prompting 和 NovelAI V5 生图", "0.4.0")
class NovelAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.previews = OrderedDict()
        self.last_text_draw = OrderedDict()
        self.lock = asyncio.Lock()

    async def compile_prompt(self, event, description):
        scene, name, brief = select_style(self.config, description)
        system = instructions(scene, use_style=bool(name))
        prompt = scene
        if name:
            prompt = json.dumps(
                {"scene": scene, "style_name": name, "style_brief": brief}, ensure_ascii=False
            )
        try:
            provider = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            answer = await asyncio.wait_for(self.context.llm_generate(
                chat_provider_id=provider, system_prompt=system, prompt=prompt
            ), timeout=120)
        except Exception as exc:
            logger.warning("NAI5 聊天模型调用失败 (%s)", type(exc).__name__)
            raise PromptError("无法取得提示词，请检查当前会话的聊天模型配置及连接。") from exc
        plan = parse_prompt(answer.completion_text)
        plan["style_name"] = name
        return plan

    @staticmethod
    def remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > 128:
            cache.popitem(last=False)

    @staticmethod
    def preview_text(plan):
        fields = ["主提示词\n" + plan["base"]]
        if plan.get("style_name"):
            fields.insert(0, "画风：" + plan["style_name"])
        for i, char in enumerate(plan["characters"], 1):
            fields.append(f"角色 {i} ({char['x']}, {char['y']})\n{char['prompt']}")
        if plan["negative"]:
            fields.append("负面提示词\n" + plan["negative"])
        fields.append("10 分钟内发送 /nai确认，使用这份提示词绘图。")
        return "\n\n".join(fields)

    def redraw_body(self, key, seed_text):
        if key not in self.last_text_draw:
            raise GenerationError("请先用 /nai、/nai原词 或 /nai确认 完成一次文生图。")
        if seed_text:
            if not seed_text.isascii() or not seed_text.isdecimal() or len(seed_text) > 10:
                raise GenerationError("种子应为 0 至 4294967295 的整数。")
            seed = int(seed_text)
            if seed > 4294967295:
                raise GenerationError("种子应为 0 至 4294967295 的整数。")
        else:
            seed = secrets.randbits(32)
        body = deepcopy(self.last_text_draw[key])
        body["parameters"]["seed"] = seed
        return body

    async def make_body(self, event, mode, key, description):
        if mode == "redraw":
            return self.redraw_body(key, description)
        model = self.config.get("model", MODELS[0])
        size = self.config.get("size", "方形")
        source = None
        if mode == "img2img":
            if size not in SIZES:
                raise GenerationError("请检查插件的画幅设置。")
            source = await read_source(event, SIZES[size])
        if mode == "confirm":
            saved = self.previews.get(key)
            if not saved or monotonic() > saved[0]:
                self.previews.pop(key, None)
                raise PromptError("当前没有可用的预览，请先发送 /nai预览 画面描述。")
            plan = saved[1]
        elif mode == "raw":
            plan = {"base": description, "negative": "", "characters": []}
        else:
            plan = await self.compile_prompt(event, description)
        return payload(plan, model, size, config=self.config, source=source)

    def image_result(self, event, png, seed):
        content = [Image.fromBytes(png), Plain(f"种子 {seed}")]
        if self.config.get("qq_forward_card", True) and event.get_platform_name() == "aiocqhttp":
            node = Node(uin=event.get_self_id(), name="NovelAI 绘图", content=content)
            return event.chain_result([Nodes([node])])
        return event.chain_result(content)

    async def handle(self, event, mode, description=None):
        if description is None:
            parts = event.message_str.strip().split(maxsplit=1)
            description = parts[1] if len(parts) == 2 else ""
        if mode not in ("confirm", "redraw") and not description:
            return event.plain_result("请在指令后填写提示词或画面描述。输入 /nai帮助 查看示例。")
        token = (os.getenv("NOVELAI_API_TOKEN") or self.config.get("api_token", "")).strip()
        if mode != "preview" and not token:
            return event.plain_result("请管理员先配置 NovelAI Persistent API Token。")
        if self.lock.locked():
            return event.plain_result("正在处理另一条绘图指令，请稍后再试。")
        key = (event.unified_msg_origin, event.get_sender_id())
        async with self.lock:
            try:
                if mode == "preview":
                    plan = await self.compile_prompt(event, description)
                    self.remember(self.previews, key, (monotonic() + 600, plan))
                    return event.plain_result(self.preview_text(plan))
                body = await self.make_body(event, mode, key, description)
                png = await asyncio.to_thread(generate, token, body)
                if mode == "confirm":
                    self.previews.pop(key, None)
                if body["action"] == "generate":
                    self.remember(self.last_text_draw, key, body)
                seed = body["parameters"]["seed"]
                return self.image_result(event, png, seed)
            except (GenerationError, PromptError) as exc:
                return event.plain_result(str(exc))

    @filter.llm_tool(name="nai5_draw_image")
    async def natural_draw(self, event: AstrMessageEvent, description: str):
        """当前用户明确请你画图时，使用 NovelAI 生成并发送一张插画。每条消息只调用一次。

        Args:
            description(string): 完整画面描述。若用户点名某个预设，在开头写【画风：预设名】，其余情况直接写画面。保留角色、动作、构图和文字。
        """
        if event.get_extra("nai5_draw_handled"):
            return "这条消息的绘图请求已经处理。"
        event.set_extra("nai5_draw_handled", True)
        result = await self.handle(event, "draw", description=description.strip())
        await event.send(result)
        return "处理结果已发送到当前聊天。"

    @filter.command("nai")
    async def draw(self, event: AstrMessageEvent):
        """用中文或英文描述生成一张 NovelAI 插画。"""
        yield await self.handle(event, "draw")

    @filter.command("nai预览")
    async def preview(self, event: AstrMessageEvent):
        """查看技能编写的主提示词与角色栏。"""
        yield await self.handle(event, "preview")

    @filter.command("nai确认")
    async def confirm(self, event: AstrMessageEvent):
        """用本人在当前会话的最近一次预览生成图片。"""
        yield await self.handle(event, "confirm")

    @filter.command("nai图生图")
    async def img2img(self, event: AstrMessageEvent):
        """附一张原图并描述目标画面。"""
        yield await self.handle(event, "img2img")

    @filter.command("nai原词")
    async def raw(self, event: AstrMessageEvent):
        """原样使用主提示词，直接请求文生图。"""
        yield await self.handle(event, "raw")

    @filter.command("nai重画")
    async def redraw(self, event: AstrMessageEvent):
        """复用上次文生图参数。可选填整数种子。"""
        yield await self.handle(event, "redraw")

    @filter.command("nai帮助")
    async def help_command(self, event: AstrMessageEvent):
        """查看 NovelAI 绘图指令。"""
        yield event.plain_result(
            "也可以在对话中说「帮我画一张雨后的街道」，需要启用绘图工具。\n"
            "/nai 雨后的街道 — 用描述绘图\n"
            "/nai预览 描述 → /nai确认 — 看词后出图\n"
            "/nai画风 — 列出可用预设；/nai预览 【画风：透明水彩】雨后街道\n"
            "/nai图生图 水彩风格的街道 — 随指令附一张图，或回复原图\n"
            "/nai原词 watercolor, rainy street — 直接提交已有提示词\n"
            "/nai重画 — 为上次文生图换种子；/nai重画 42 — 指定种子"
        )

    @filter.command("nai画风")
    async def list_styles(self, event: AstrMessageEvent):
        """查看可用的画风预设和默认画风。"""
        try:
            presets = style_presets(self.config)
        except PromptError as exc:
            yield event.plain_result(str(exc))
            return
        default = self.config.get("default_style", "").strip() or "无"
        lines = [f"默认画风：{default}", "可用预设：" + ("、".join(presets) or "暂无"),
                 "本次指定：/nai预览 【画风：名称】画面描述；跳过预设用【画风：无】。"]
        yield event.plain_result("\n".join(lines))
