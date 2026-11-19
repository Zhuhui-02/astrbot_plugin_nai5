import asyncio
import base64
import io

from PIL import Image as PILImage, ImageOps
from astrbot.api.message_components import Image, Reply

from .novelai import GenerationError

MAX_SOURCE_BYTES = 12 * 1024 * 1024


def prepare_source(encoded, size):
    if len(encoded) > 4 * ((MAX_SOURCE_BYTES + 2) // 3):
        raise GenerationError("原图超过 12 MiB，请缩小后再试。")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with PILImage.open(io.BytesIO(raw)) as original:
            if original.width * original.height > 32_000_000:
                raise GenerationError("原图超过 3200 万像素，请缩小后再试。")
            oriented = ImageOps.exif_transpose(original).convert("RGBA")
            canvas = PILImage.new("RGB", oriented.size, "white")
            canvas.paste(oriented, mask=oriented.getchannel("A"))
            image = ImageOps.fit(canvas, size, method=PILImage.Resampling.LANCZOS)
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                return base64.b64encode(output.getvalue()).decode("ascii")
    except (ValueError, OSError, PILImage.DecompressionBombError) as exc:
        raise GenerationError("无法解码原图，请发送静态 PNG 或 JPEG 图片。") from exc


async def read_source(event, size):
    components = event.get_messages()
    images = [part for part in components if isinstance(part, Image)]
    if not images:
        images = [item for reply in components if isinstance(reply, Reply)
                  for item in (reply.chain or []) if isinstance(item, Image)]
    if len(images) != 1:
        raise GenerationError("图生图需要一张原图。请随指令附图，或引用含一张图片的消息。")
    try:
        encoded = await asyncio.wait_for(images[0].convert_to_base64(), timeout=30)
    except Exception as exc:
        raise GenerationError("读取消息中的图片失败，请重新附图。") from exc
    return await asyncio.to_thread(prepare_source, encoded, size)
