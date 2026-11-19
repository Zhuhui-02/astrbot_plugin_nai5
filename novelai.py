import http.client
import io
import json
import secrets
import zipfile
import zlib

from PIL import Image

MODELS = ("nai-diffusion-5-full", "nai-diffusion-5-curated")
SIZES = {"方形": (1024, 1024), "竖图": (832, 1216), "横图": (1216, 832)}
LIMIT = 24 * 1024 * 1024


class GenerationError(Exception):
    pass


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


def generate(token, body):
    conn = http.client.HTTPSConnection("image.novelai.net", timeout=120)
    try:
        conn.request("POST", "/ai/generate-image", body=json.dumps(body).encode(),
                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                              "Accept": "application/zip"})
        response = conn.getresponse()
        if not 200 <= response.status < 300:
            hints = {401: "请检查 API Token。", 402: "请检查 Anlas 余额与订阅。",
                     403: "请检查账户的模型权限。", 429: "请求受到限流，请稍后再试。"}
            raise GenerationError(f"NovelAI 返回 HTTP {response.status}。" + hints.get(response.status, ""))
        data = response.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise GenerationError("NovelAI 返回的文件过大。")
        return extract_png(data)
    except (http.client.HTTPException, OSError) as exc:
        raise GenerationError("连接 NovelAI 中断或超时。再次绘图前可先核对账户用量。") from exc
    finally:
        conn.close()
