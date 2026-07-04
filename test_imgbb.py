"""Milestone 1：ImgBB 上传引擎命令行自测。"""
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from services.imgbb import ImgBBUploadError, upload_to_imgbb

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


def _read_secret(name: str) -> str:
    if not SECRETS_PATH.exists():
        return ""
    for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name} ="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _make_test_jpeg() -> bytes:
    img = Image.new("RGB", (320, 240), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 280, 200], fill=(102, 126, 234))
    draw.text((90, 100), "FindIt M1", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main() -> int:
    api_key = _read_secret("IMGBB_API_KEY")
    if not api_key:
        print("❌ 请在 .streamlit/secrets.toml 中配置 IMGBB_API_KEY")
        return 1

    print("正在上传测试图片到 ImgBB …")
    try:
        url = upload_to_imgbb(_make_test_jpeg(), api_key)
    except ImgBBUploadError as exc:
        print(f"❌ 上传失败: {exc}")
        return 1

    print(f"✅ 上传成功")
    print(f"永久直链: {url}")
    if url.startswith("https://i.ibb.co/"):
        print("✅ URL 格式验证通过 (https://i.ibb.co/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
