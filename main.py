import asyncio
import secrets
from copy import deepcopy
from collections import OrderedDict
from time import monotonic

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .novelai import GenerationError, MODELS, SIZES, generation_service, generate, payload
from .source_image import read_source
from .prompt_agent import PromptAgent
from .prompting import PromptError, direct_draw_request, style_presets


@register("astrbot_plugin_nai5", "Zhuhui-02", "使用 nai5-prompting 和 NovelAI V5 生图", "0.6.0")
class NovelAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.prompt_agent = PromptAgent(context, config)
        self.previews = OrderedDict()
        self.last_text_draw = OrderedDict()
        self.lock = asyncio.Lock()

    @staticmethod
    def remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > 128:
            cache.popitem(last=False)

    @staticmethod
    def prompt_text(plan):
        fields = ["主提示词\n" + plan["base"]]
        if plan.get("style_name"):
            fields.insert(0, "画风：" + plan["style_name"])
        for i, char in enumerate(plan["characters"], 1):
            fields.append(f"角色 {i} ({char['x']}, {char['y']})\n{char['prompt']}")
        if plan["negative"]:
            fields.append("负面提示词\n" + plan["negative"])
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
        previous_body, _ = self.last_text_draw[key]
        body = deepcopy(previous_body)
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
            plan = await self.prompt_agent.compile(event.unified_msg_origin, description)
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
        if self.lock.locked():
            return event.plain_result("正在处理另一条绘图指令，请稍后再试。")
        key = (event.unified_msg_origin, event.get_sender_id())
        async with self.lock:
            try:
                if mode == "preview":
                    plan = await self.prompt_agent.compile(event.unified_msg_origin, description)
                    self.remember(self.previews, key, (monotonic() + 600, plan))
                    return event.plain_result(self.prompt_text(plan) + "\n\n10 分钟内发送 /nai确认，使用这份提示词绘图。")
                endpoint, token = generation_service(self.config)
                body = await self.make_body(event, mode, key, description)
                png = await asyncio.to_thread(generate, token, body, endpoint)
                if mode == "confirm":
                    self.previews.pop(key, None)
                if body["action"] == "generate":
                    if mode == "redraw":
                        _, source = self.last_text_draw[key]
                    elif mode == "raw":
                        source = "/nai原词 原文直传"
                    elif mode == "confirm":
                        source = "nai5-prompting 预览确认"
                    else:
                        source = "nai5-prompting 编词"
                    self.remember(self.last_text_draw, key, (body, source))
                seed = body["parameters"]["seed"]
                return self.image_result(event, png, seed)
            except (GenerationError, PromptError) as exc:
                return event.plain_result(str(exc))

    @filter.on_llm_request()
    async def dispatch_direct_draw(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("direct_draw", True) or getattr(req, "image_urls", None):
            return
        try:
            direct = direct_draw_request(event.message_str, style_presets(self.config))
        except PromptError:
            direct = direct_draw_request(event.message_str, ())
        if direct:
            event.stop_event()
            if not event.get_extra("nai5_draw_handled"):
                event.set_extra("nai5_draw_handled", True)
                await event.send(await self.handle(event, "draw", description=event.message_str.strip()))

    @filter.llm_tool(name="nai5_draw_image")
    async def natural_draw(self, event: AstrMessageEvent, description: str, preset_name: str = ""):
        """当前用户明确请你画图时，使用 NovelAI 生成并发送一张插画。每条消息只调用一次。

        Args:
            description(string): 兼容字段，插件从当前用户消息直接读取画面描述。
            preset_name(string): 兼容字段，请留空，插件从原文识别预设。
        """
        if event.get_extra("nai5_draw_handled"):
            return "这条消息的绘图请求已经处理。"
        event.set_extra("nai5_draw_handled", True)
        result = await self.handle(event, "draw", description=(event.message_str or "").strip())
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

    @filter.command("nai上次提示词")
    async def last_prompt(self, event: AstrMessageEvent):
        """查看本人在当前会话最近一次成功文生图实际使用的提示词。"""
        record = self.last_text_draw.get((event.unified_msg_origin, event.get_sender_id()))
        if record is None:
            yield event.plain_result("当前会话没有你通过本插件成功生成的文生图记录。")
            return
        body, source = record
        params = body["parameters"]
        characters = [
            {"prompt": char["char_caption"], **char["centers"][0]}
            for char in params["v4_prompt"]["caption"]["char_captions"]
        ]
        plan = {"base": body["input"], "negative": params["negative_prompt"], "characters": characters}
        yield event.plain_result(f"最近一次文生图 · {source} · 种子 {params['seed']}\n\n" + self.prompt_text(plan))

    @filter.command("nai帮助")
    async def help_command(self, event: AstrMessageEvent):
        """查看 NovelAI 绘图指令。"""
        yield event.plain_result(
            "也可以在对话中说「帮我画一张雨后的街道」，可开启「明确绘图请求优先处理」。\n"
            "/nai 雨后的街道 — 用描述绘图\n"
            "/nai预览 描述 → /nai确认 — 看词后出图\n"
            "/nai画风 — 列出可用预设；/nai预览 【画风：预设名称】雨后街道\n"
            "/nai图生图 水彩风格的街道 — 随指令附一张图，或回复原图\n"
            "/nai原词 watercolor, rainy street — 直接提交已有提示词\n"
            "/nai上次提示词 — 查看最近一次文生图实际提交的词和种子\n"
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
