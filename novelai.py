import http.client
import io
import json
import os
import secrets
import zipfile
import zlib
from urllib.parse import urlsplit

from PIL import Image

MODELS = ("nai-diffusion-5-full", "nai-diffusion-5-curated")
SIZES = {"方形": (1024, 1024), "竖图": (832, 1216), "横图": (1216, 832)}
LIMIT = 24 * 1024 * 1024
SERVICES = {
    "NovelAI 官方": ("https://image.novelai.net/ai/generate-image", "api_token", "NOVELAI_API_TOKEN"),
    "蛋糕": ("https://dangao.iisbo.com/ai/generate-image", "dangao_api_key", "DANGAO_API_KEY"),
    "Langbai": ("https://nai.langbai.cc/ai/generate-image", "langbai_api_key", "LANGBAI_API_KEY"),
    "自定义镜像": ("", "custom_api_key", "NAI_MIRROR_API_KEY"),
}


class GenerationError(Exception):
    pass


def generation_service(config):
    provider = config.get("provider", "NovelAI 官方")
    if provider not in SERVICES:
        raise GenerationError("请选择有效的生图服务商。")
    url, field, env_name = SERVICES[provider]
    url = url or config.get("custom_endpoint", "").strip()
    try:
        endpoint = urlsplit(url)
        if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username is not None
                or endpoint.query or endpoint.fragment or endpoint.path in ("", "/")
                or not url.isascii() or any(ord(ch) <= 32 or ord(ch) == 127 for ch in url)
                or endpoint.port == 0):
            raise ValueError
    except ValueError:
        raise GenerationError("请填写完整的 HTTPS 生图地址，包含接口路径，去掉账号、查询参数和页面锚点。") from None
    token = (os.getenv(env_name) or config.get(field, "")).strip()
    if not token:
        raise GenerationError(f"请配置「{provider}」的密钥，或设置 {env_name}。")
    if not token.isascii() or any(ord(ch) <= 32 or ord(ch) == 127 for ch in token):
        raise GenerationError(f"「{provider}」密钥格式有误，请只填写密钥本身。")
    return endpoint, token


def number_setting(config, key, default, lower, upper):
    value = config.get(key, default)
    expected = (int,) if type(default) is int else (float, int)
    if type(value) not in expected or not lower <= value <= upper:
        raise GenerationError(f"请将 {key} 设为 {lower} 至 {upper} 之间的有效数值。")
    return value


def payload(plan, model, size, *, config=None, source=None):
    config = config or {}
    if model not in MODELS or size not in SIZES:
        raise GenerationError("请检查插件配置中的模型和画幅。")
    width, height = SIZES[size]
    positive, negative = [], []
    for char in plan["characters"]:
        centers = [{"x": round(char["x"], 3), "y": round(char["y"], 3)}]
        positive.append({"char_caption": char["prompt"], "centers": centers})
        negative.append({"char_caption": "", "centers": centers})
    base, uc = plan["base"], plan["negative"]
    body = {"model": model, "action": "generate", "input": base, "parameters": {
        "params_version": 4, "width": width, "height": height, "n_samples": 1,
        "scale": number_setting(config, "scale", 5.0, 1, 10),
        "steps": number_setting(config, "steps", 28, 1, 50), "seed": secrets.randbits(32),
        "sampler": "k_euler_ancestral", "noise_schedule": "karras", "uc": uc,
        "qualityToggle": False, "negative_prompt": uc,
        "dynamic_thresholding": False, "cfg_rescale": 0,
        "deliberate_euler_ancestral_bug": False, "prefer_brownian": True,
        "v4_prompt": {"caption": {"base_caption": base, "char_captions": positive},
                      "use_coords": bool(positive), "use_order": True},
        "v4_negative_prompt": {"caption": {"base_caption": uc, "char_captions": negative}},
    }}
    if source is not None:
        params = body["parameters"]
        body["action"] = "img2img"
        params.update(image=source, extra_noise_seed=params["seed"],
                      strength=number_setting(config, "strength", 0.5, 0.1, 0.9),
                      noise=number_setting(config, "noise", 0.2, 0, 1))
    return body


def extract_png(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                if entry.filename.lower().endswith(".png") and 0 < entry.file_size <= LIMIT:
                    image = archive.read(entry)
                    with Image.open(io.BytesIO(image), formats=["PNG"]) as result:
                        result.verify()
                        return image
    except (zipfile.BadZipFile, OSError, RuntimeError, SyntaxError, ValueError,
            Image.DecompressionBombError, zlib.error):
        pass
    raise GenerationError("NovelAI 返回的压缩包中没有可读的 PNG 图片。")


def decode_image(data):
    if data.startswith(b"PK"):
        return extract_png(data)
    try:
        with Image.open(io.BytesIO(data), formats=["PNG", "JPEG", "WEBP"]) as image:
            if image.width * image.height > 32_000_000:
                raise GenerationError("生图服务返回的图像像素过多。")
            if image.format == "PNG":
                image.verify()
                return data
            with io.BytesIO() as out:
                mode = "RGBA" if "A" in image.getbands() else "RGB"
                image.convert(mode).save(out, format="PNG")
                if out.tell() > LIMIT:
                    raise GenerationError("转换后的 PNG 文件过大。")
                return out.getvalue()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, zlib.error):
        raise GenerationError("生图服务未返回可读取的 ZIP、PNG、JPEG 或 WebP 图片。") from None


def generate(token, body, endpoint):
    conn = http.client.HTTPSConnection(endpoint.netloc, timeout=120)
    try:
        conn.request("POST", endpoint.path, body=json.dumps(body).encode(),
                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                              "Accept": "application/zip"})
        response = conn.getresponse()
        if not 200 <= response.status < 300:
            hints = {401: "请检查当前站点的 API Key。", 402: "请检查当前站点的账户余额。",
                     403: "请检查账户的模型权限。", 429: "请求受到限流，请稍后再试。"}
            raise GenerationError(f"{endpoint.hostname} 返回 HTTP {response.status}。" + hints.get(response.status, ""))
        data = response.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise GenerationError("生图服务返回的文件过大。")
        return decode_image(data)
    except (http.client.HTTPException, OSError) as exc:
        raise GenerationError(f"连接 {endpoint.hostname} 中断或超时。再次绘图前可先核对账户用量。") from exc
    finally:
        conn.close()
