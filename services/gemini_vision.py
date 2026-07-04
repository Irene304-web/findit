"""Gemini 3.5 Flash 视觉识别与语音转写（google-genai 官方 SDK）。"""

from __future__ import annotations

import base64
import io
import json
import re

from google import genai
from google.genai import types
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MAX_EDGE = 960
GEMINI_JPEG_QUALITY = 78

MULTI_ITEM_PROMPT = """你是家庭收纳助手。请仔细观察这张图片，识别其中所有可以独立收纳、单独存放的物品。

要求：
1. 列出每个物品的简洁中文名称（2-8 个汉字）
2. 不要重复，不要包含背景、桌面、容器本身（除非容器是主要收纳对象）
3. 只返回 JSON 数组，格式严格为：["物品1", "物品2", "物品3"]
4. 不要 markdown，不要解释
5. 若完全看不清，返回 ["未知物品"]"""

ITEMS_JSON_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}

SPEECH_PROMPT = (
    "请将这段中文语音转写为文字。"
    "只返回转写结果本身，不要标点解释，不要 markdown。"
    "若听不清，返回空字符串。"
)


class GeminiVisionError(Exception):
    """Gemini API 调用失败。"""


def _client(api_key: str) -> genai.Client:
    if not api_key:
        raise GeminiVisionError("GEMINI_API_KEY 未配置，请在 .streamlit/secrets.toml 中设置。")
    return genai.Client(api_key=api_key)


def compress_to_jpeg(image_bytes: bytes, max_edge: int = GEMINI_MAX_EDGE) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=GEMINI_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _clean_name(raw: str) -> str:
    name = re.sub(r'["""\'「」【】\[\]]', "", (raw or "").strip())
    return name or "未知物品"


def parse_items_json(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiVisionError("Gemini 返回格式无法解析。") from exc

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", [])
    else:
        raise GeminiVisionError("Gemini 返回的不是 JSON 数组。")

    if not isinstance(items, list):
        raise GeminiVisionError("Gemini 返回的 items 不是列表。")

    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = _clean_name(str(item))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names or ["未知物品"]


def transcribe_audio(audio_b64: str, mime_type: str, api_key: str) -> str:
    """将录音转写为中文文字（点击语音识别 → Gemini 转写）。"""
    if not audio_b64:
        raise GeminiVisionError("未录到有效语音，请点击「语音识别」后说话。")

    audio_bytes = base64.b64decode(audio_b64)
    client = _client(api_key)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(text=SPEECH_PROMPT),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=128,
            ),
        )
    except Exception as exc:
        raise GeminiVisionError(f"Gemini 语音转写失败: {exc}") from exc

    text = re.sub(r'["""\'「」【】]', "", (response.text or "").strip())
    if not text:
        raise GeminiVisionError("没有识别到语音内容，请再试一次。")
    return text
