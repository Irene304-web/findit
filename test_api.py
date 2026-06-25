"""独立测试智谱 GLM-4.6V-Flash API，不依赖 Streamlit。"""
import base64
import io
import os
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"
DEFAULT_KEY = "53254a8bae734950860ec204ce158d40.Hq3xHKoA0ZdTrJBa"
DEFAULT_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.6v-flash"


def _read_secret(name: str) -> str:
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def load_config() -> tuple[str, str, str]:
    api_key = os.environ.get("GLM_API_KEY") or _read_secret("GLM_API_KEY") or DEFAULT_KEY
    base_url = os.environ.get("GLM_BASE_URL") or _read_secret("GLM_BASE_URL") or DEFAULT_URL
    model = os.environ.get("GLM_MODEL") or _read_secret("GLM_MODEL") or DEFAULT_MODEL
    return api_key, base_url, model


def extract_text(message) -> str:
    content = getattr(message, "content", None) or ""
    if content:
        return content.strip()
    return (getattr(message, "reasoning_content", None) or "").strip()


def make_test_jpeg() -> bytes:
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([64, 64, 192, 192], fill=(220, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> int:
    api_key, base_url, model = load_config()

    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"API Key  : {api_key[:8]}...{api_key[-4:]}")
    print("-" * 40)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("[1/2] 文本 API 测试...")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：成功"}],
            max_tokens=10,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = extract_text(resp.choices[0].message)
        print(f"  ✅ 文本 API 成功，回复: {text}")
    except Exception as exc:
        print(f"  ❌ 文本 API 失败: {exc}")
        return 1

    print("[2/2] 视觉 API 测试...")
    b64 = base64.b64encode(make_test_jpeg()).decode()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "识别图片中的物体，只返回中文名称，2-8个字",
                        },
                    ],
                }
            ],
            max_tokens=64,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = extract_text(resp.choices[0].message)
        print(f"  ✅ 视觉 API 成功，回复: {text}")
    except Exception as exc:
        print(f"  ❌ 视觉 API 失败: {exc}")
        return 1

    print("-" * 40)
    print("全部测试通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
