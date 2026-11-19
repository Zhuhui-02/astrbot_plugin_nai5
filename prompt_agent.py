import asyncio
import json

from astrbot.api import logger

from .prompting import OUTPUT_FORMAT, PromptError, PromptFormatError, instructions, parse_prompt, select_style


class PromptAgent:
    """Compile one drawing request with the bundled skill and a fresh model context."""

    def __init__(self, context, config):
        self.context = context
        self.config = config

    async def request_prompt_text(self, provider, system, prompt):
        answer = await asyncio.wait_for(self.context.llm_generate(
            chat_provider_id=provider, system_prompt=system, prompt=prompt, contexts=[], tools=None
        ), timeout=120)
        return answer.completion_text

    async def compile(self, origin, description):
        scene, name, brief = select_style(self.config, description)
        system = instructions(scene, use_style=bool(name))
        fields = {"scene": scene}
        if name:
            fields.update(style_name=name, style_brief=brief)
        prompt = json.dumps(fields, ensure_ascii=False) if len(fields) > 1 else scene
        try:
            provider = self.config.get("prompt_provider_id", "").strip()
            if not provider:
                provider = await self.context.get_current_chat_provider_id(umo=origin)
            raw = await self.request_prompt_text(provider, system, prompt)
            try:
                plan = parse_prompt(raw)
            except PromptFormatError as exc:
                logger.info("NAI5 编词结果需要整理格式：%s", exc)
                repair = (
                    OUTPUT_FORMAT + " 将输入中的 draft 整理为上述结构，保留已有主串、全部角色、"
                    "左右关系、文字和负面词。依据 request 核对主题、角色和选定画风，只整理对应的草稿；如果草稿没有可整理的绘图提示词，返回 null。"
                )
                draft = json.dumps({"request": fields, "draft": raw, "error": str(exc)}, ensure_ascii=False)
                fixed = await self.request_prompt_text(provider, repair, draft)
                if isinstance(fixed, str) and fixed.strip() == "null":
                    raise PromptError("编词模型未给出可用的绘图提示词。请用 /nai预览 检查编词模型。")
                try:
                    plan = parse_prompt(fixed)
                except PromptError as final:
                    raise PromptError(f"编词模型已尝试整理一次，仍无法读取：{final}") from None
        except PromptError:
            raise
        except Exception as exc:
            logger.warning("NAI5 聊天模型调用失败 (%s)", type(exc).__name__)
            raise PromptError("无法取得提示词，请检查插件的编词模型配置及连接。") from exc
        plan["style_name"] = name
        return plan
