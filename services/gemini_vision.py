"""Gemini 2.5 Flash 视觉识别与语音转写（REST API）。"""

from __future__ import annotations

import base64
import io
import json
import re
import time

import requests
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com"
GEMINI_MAX_EDGE = 960
GEMINI_JPEG_QUALITY = 78
GEMINI_TIMEOUT = (20, 120)
GEMINI_RETRIES = 3

MULTI_ITEM_PROMPT = """你是家庭收纳助手。请仔细观察这张图片，识别其中所有可以独立收纳、单独存放的物品。

要求：
1. 列出每个物品的简洁中文名称（2-8 个汉字）
2. 不要重复，不要包含背景、桌面、容器本身（除非容器是主要收纳对象）
3. 只返回 JSON，格式严格为：{"items": ["物品1", "物品2", "物品3"]}
4. 不要 markdown，不要解释
5. 若完全看不清，返回 {"items": ["未知物品"]}"""

SPEECH_PROMPT = (
    "请将这段中文语音转写为文字。"
    "只返回转写结果本身，不要标点解释，不要 markdown。"
    "若听不清，返回空字符串。"
)


class GeminiVisionError(Exception):
    """Gemini API 调用失败。"""


def _compress_to_jpeg(image_bytes: bytes, max_edge: int = GEMINI_MAX_EDGE) -> bytes:
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


def _api_url(model: str, api_base: str | None = None) -> str:
    base = (api_base or DEFAULT_GEMINI_BASE).rstrip("/")
    return f"{base}/v1beta/models/{model}:generateContent"


def _call_gemini(payload: dict, api_key: str, api_base: str | None = None) -> dict:
    url = _api_url(GEMINI_MODEL, api_base)
    last_error: Exception | None = None

    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=GEMINI_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < GEMINI_RETRIES:
                time.sleep(1.5 * attempt)

    raise GeminiVisionError(
        "Gemini API 调用失败（已重试）。云端服务器会代为访问 Google，手机无需 VPN。"
        f" 详情: {last_error}"
    ) from last_error


def _extract_text(body: dict) -> str:
    try:
        return str(body["candidates"][0]["content"]["parts"][0]["text"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiVisionError("Gemini 响应结构异常。") from exc


def _clean_name(raw: str) -> str:
    name = re.sub(r'["""\'「」【】\[\]]', "", (raw or "").strip())
    return name or "未知物品"


def _parse_items_json(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise GeminiVisionError("Gemini 返回格式无法解析。") from None
        data = json.loads(match.group())

    items = data.get("items", [])
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


def recognize_multiple_items(
    image_bytes: bytes, api_key: str, api_base: str | None = None
) -> list[str]:
    """调用 Gemini 2.5 Flash 识别一张图中的多个物品名称。"""
    if not api_key:
        raise GeminiVisionError("GEMINI_API_KEY 未配置，请在 .streamlit/secrets.toml 中设置。")
    if not image_bytes:
        raise GeminiVisionError("图片数据为空。")

    jpeg_bytes = _compress_to_jpeg(image_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": MULTI_ITEM_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }

    body = _call_gemini(payload, api_key, api_base)
    text = _extract_text(body)
    if not text:
        raise GeminiVisionError("Gemini 未返回识别结果。")
    return _parse_items_json(text)


def transcribe_audio(
    audio_b64: str, mime_type: str, api_key: str, api_base: str | None = None
) -> str:
    """将录音转写为中文文字（点击语音识别 → Gemini 转写）。"""
    if not api_key:
        raise GeminiVisionError("GEMINI_API_KEY 未配置，无法使用语音输入。")
    if not audio_b64:
        raise GeminiVisionError("未录到有效语音，请点击「语音识别」后说话。")

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    {"text": SPEECH_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 128,
        },
    }

    body = _call_gemini(payload, api_key, api_base)
    text = _extract_text(body)
    text = re.sub(r'["""\'「」【】]', "", text).strip()
    if not text:
        raise GeminiVisionError("没有识别到语音内容，请再试一次。")
    return text
