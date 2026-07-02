"""ImgBB 永久图床上传服务。"""

from __future__ import annotations

import base64
import io
import time

import requests
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"
IMGBB_URL_PREFIX = "https://i.ibb.co/"
MAX_EDGE = 1600
JPEG_QUALITY = 85
UPLOAD_TIMEOUT = (15, 120)  # (连接秒, 读写秒)
MAX_RETRIES = 3


class ImgBBUploadError(Exception):
    """ImgBB 上传失败。"""


def _compress_image(file_bytes: bytes, max_edge: int = MAX_EDGE) -> bytes:
    """压缩图片，减小上传体积，降低超时概率。"""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _parse_imgbb_url(payload: dict) -> str:
    data = payload.get("data") or {}
    image_block = data.get("image") or {}
    url = image_block.get("url") or data.get("display_url") or data.get("url")
    if not url:
        raise ImgBBUploadError("ImgBB 响应中未找到图片 URL。")
    if not url.startswith(IMGBB_URL_PREFIX):
        raise ImgBBUploadError(
            f"返回的 URL 不符合永久直链格式（期望 {IMGBB_URL_PREFIX} 开头）: {url}"
        )
    return url


def _post_upload(api_key: str, image_bytes: bytes) -> dict:
    """优先 multipart 上传；失败时回退 base64。"""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                IMGBB_UPLOAD_URL,
                data={"key": api_key},
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
                timeout=UPLOAD_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    # multipart 全部失败，尝试 base64（部分网络环境更稳定）
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                IMGBB_UPLOAD_URL,
                data={"key": api_key, "image": encoded},
                timeout=UPLOAD_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    raise ImgBBUploadError(
        "网络请求失败（已重试多次）。请检查网络能否访问 api.imgbb.com，或换更小图片再试。"
        f" 详情: {last_error}"
    ) from last_error


def upload_to_imgbb(file_bytes: bytes, api_key: str) -> str:
    """
    将图片字节流上传至 ImgBB，返回永久直链。

    成功时 URL 以 https://i.ibb.co/ 开头。
    """
    if not api_key:
        raise ImgBBUploadError("IMGBB_API_KEY 未配置，请在 .streamlit/secrets.toml 中设置。")
    if not file_bytes:
        raise ImgBBUploadError("图片数据为空，无法上传。")

    original_kb = len(file_bytes) / 1024
    try:
        compressed = _compress_image(file_bytes)
    except Exception as exc:
        raise ImgBBUploadError(
            f"图片预处理失败: {exc}。"
            "若为 .HEIC 格式，请确认已安装 pillow-heif。"
        ) from exc

    compressed_kb = len(compressed) / 1024

    try:
        payload = _post_upload(api_key, compressed)
    except ImgBBUploadError:
        raise
    except ValueError as exc:
        raise ImgBBUploadError("ImgBB 返回了无效的 JSON 响应。") from exc

    if not payload.get("success"):
        error_msg = payload.get("error", {}).get("message", "未知错误")
        raise ImgBBUploadError(f"ImgBB 拒绝上传: {error_msg}")

    url = _parse_imgbb_url(payload)
    return url
