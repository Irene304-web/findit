"""
找到了么 — 家庭寻物 Web 应用
使用 Streamlit + SQLite + 智谱 GLM-4.6V-Flash 多模态视觉识别
"""

import base64
import hashlib
import io
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
from PIL import Image

from components.mobile_capture import mobile_capture_input

# ── 路径与配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "find_goods.db"
IMAGES_DIR = BASE_DIR / "saved_images"

# 密钥优先级：环境变量 > Streamlit secrets > 内置默认值
_DEFAULT_API_KEY = "53254a8bae734950860ec204ce158d40.Hq3xHKoA0ZdTrJBa"


def _load_config() -> tuple[str, str, str]:
    api_key = os.environ.get("GLM_API_KEY", "")
    base_url = os.environ.get("GLM_BASE_URL", "")
    model = os.environ.get("GLM_MODEL", "")

    if not api_key:
        try:
            api_key = st.secrets.get("GLM_API_KEY", "")
            base_url = base_url or st.secrets.get("GLM_BASE_URL", "")
            model = model or st.secrets.get("GLM_MODEL", "")
        except Exception:
            pass

    return (
        api_key or _DEFAULT_API_KEY,
        base_url or "https://open.bigmodel.cn/api/paas/v4",
        model or "glm-4.6v-flash",
    )


GLM_API_KEY, GLM_BASE_URL, GLM_MODEL = _load_config()

VISION_MODELS = ["glm-4.6v-flash"]

VISION_PROMPT = (
    "请观察这张图片，识别其中最核心、最突出的物体。"
    "只返回该物体的中文名称（2-8个字），不要任何解释、标点或其他文字。"
    "如果看不清，返回「未知物品」。"
)

TEXT_FALLBACK_PROMPT = (
    "你是一位家庭收纳助手。根据以下照片的视觉特征，推测照片中最可能的核心物品是什么。"
    "只返回中文物品名称（2-8个字），不要解释。"
    "如果无法判断，返回「未知物品」。\n\n照片特征：\n{features}"
)


# ── 数据库 ──────────────────────────────────────────────────
def init_db() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name  TEXT    NOT NULL,
                location   TEXT    NOT NULL,
                image_path TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )


def save_item(item_name: str, location: str, image_path: str) -> int:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO items (item_name, location, image_path, created_at) "
            "VALUES (?, ?, ?, ?)",
            (item_name, location, image_path, created_at),
        )
        return cur.lastrowid


def search_items(keyword: str) -> list[dict]:
    pattern = f"%{keyword}%"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, item_name, location, image_path, created_at
            FROM items
            WHERE item_name LIKE ? OR location LIKE ?
            ORDER BY created_at DESC
            """,
            (pattern, pattern),
        ).fetchall()
    return [dict(r) for r in rows]


# ── 视觉识别 ────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def _extract_message_text(message) -> str:
    content = getattr(message, "content", None) or ""
    if content:
        return content.strip()
    reasoning = getattr(message, "reasoning_content", None) or ""
    return reasoning.strip()


def _clean_item_name(raw: str) -> str:
    name = re.sub(r'["""\'「」【】]', "", (raw or "").strip())
    return name or "未知物品"


def _image_features(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    thumb = img.resize((64, 64))
    pixels = list(thumb.getdata())
    n = len(pixels) or 1
    avg = tuple(sum(p[i] for p in pixels) // n for i in range(3))
    aspect = "横向" if w > h * 1.2 else "纵向" if h > w * 1.2 else "方形"
    return f"尺寸 {w}x{h}（{aspect}），平均颜色 RGB{avg}"


def _call_text_model(prompt: str, max_tokens: int = 32) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _clean_item_name(_extract_message_text(response.choices[0].message))


def _try_vision_models(image_bytes: bytes) -> str | None:
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    for model in VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": VISION_PROMPT},
                        ],
                    }
                ],
                max_tokens=64,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return _clean_item_name(_extract_message_text(response.choices[0].message))
        except Exception:
            continue
    return None


def recognize_item(image_bytes: bytes) -> tuple[str, str]:
    """返回 (物品名称, 识别方式说明)。"""
    vision_name = _try_vision_models(image_bytes)
    if vision_name and vision_name != "未知物品":
        return vision_name, "vision"

    try:
        features = _image_features(image_bytes)
        text_name = _call_text_model(TEXT_FALLBACK_PROMPT.format(features=features))
        return text_name, "text_fallback"
    except Exception as exc:
        st.warning(f"AI 识别失败：{exc}")
        return "未知物品", "failed"


# ── 图片保存 ────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip())
    return cleaned[:40] or "item"


def save_image(image_bytes: bytes, item_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{sanitize_filename(item_name)}.jpg"
    filepath = IMAGES_DIR / filename
    filepath.write_bytes(image_bytes)
    return str(filepath)


# ── 移动端样式 ──────────────────────────────────────────────
def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        /* 隐藏默认页脚，腾出移动端空间 */
        footer { visibility: hidden; }

        /* 标题区 */
        .app-header {
            text-align: center;
            padding: 0.5rem 0 1rem;
        }
        .app-header h1 {
            font-size: clamp(1.4rem, 5vw, 2rem);
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .app-header p {
            color: #6b7280;
            font-size: 0.9rem;
            margin: 0.3rem 0 0;
        }

        /* 建议名称卡片 */
        .suggest-card {
            background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
            border-left: 4px solid #667eea;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
        }
        .suggest-card .label {
            font-size: 0.8rem;
            color: #6366f1;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .suggest-card .name {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1e1b4b;
        }

        /* 搜索结果卡片 */
        .result-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .result-card .item-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #111827;
            margin: 0.5rem 0 0.2rem;
        }
        .result-card .meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .badge-location {
            background: #ecfdf5;
            color: #065f46;
        }
        .badge-time {
            background: #eff6ff;
            color: #1e40af;
        }

        /* 手机拍照按钮区域 */
        .mobile-capture-wrap {
            margin: 0.5rem 0 1rem;
        }

        /* 移动端适配 */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }
            [data-testid="stCameraInput"] video,
            [data-testid="stCameraInput"] img {
                width: 100% !important;
                border-radius: 12px;
            }
            .stButton > button {
                width: 100%;
                padding: 0.75rem;
                font-size: 1rem;
                border-radius: 12px;
            }
            .result-card img {
                width: 100%;
                border-radius: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── 页面组件 ────────────────────────────────────────────────
def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>📦 找到了么</h1>
            <p>拍照收纳 · 关键词寻找 · 再也不怕东西放哪儿了</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_capture_state() -> None:
    for key in (
        "last_image_hash",
        "image_bytes",
        "suggested_name",
        "recognize_mode",
        "mobile_capture",
        "mobile_upload",
        "store_camera",
    ):
        st.session_state.pop(key, None)


def _process_captured_image(image_bytes: bytes) -> None:
    image_hash = hashlib.md5(image_bytes).hexdigest()
    if st.session_state.get("last_image_hash") == image_hash:
        return
    st.session_state.last_image_hash = image_hash
    st.session_state.image_bytes = image_bytes
    with st.spinner("🔍 AI 正在识别物品…"):
        name, mode = recognize_item(image_bytes)
        st.session_state.suggested_name = name
        st.session_state.recognize_mode = mode


def _render_item_form() -> None:
    suggested = st.session_state.get("suggested_name", "未知物品")
    mode = st.session_state.get("recognize_mode", "")

    if mode == "vision":
        st.success("GLM-4.6V-Flash 已直接识别图片中的物品，请确认名称。")
    elif mode == "text_fallback":
        st.info("视觉识别未返回结果，已根据照片特征推测名称，请确认或修改。")

    st.markdown(
        f"""
        <div class="suggest-card">
            <div class="label">✨ AI 建议名称</div>
            <div class="name">{suggested}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    item_name = st.text_input(
        "物品名称",
        value=suggested,
        placeholder="可修改 AI 建议的名称",
        key="item_name_input",
    )
    location = st.text_input(
        "放置位置",
        placeholder="例如：客厅电视柜第二层抽屉",
        key="location_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        save_clicked = st.button("✅ 确认保存", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 重新拍照", use_container_width=True):
            _clear_capture_state()
            st.rerun()

    if save_clicked:
        if not item_name.strip():
            st.error("请填写物品名称。")
        elif not location.strip():
            st.error("请填写放置位置。")
        else:
            image_path = save_image(st.session_state.image_bytes, item_name)
            item_id = save_item(item_name.strip(), location.strip(), image_path)
            st.success(f"已保存！「{item_name}」→ {location}（编号 #{item_id}）")
            _clear_capture_state()
            st.balloons()


def render_store_tab() -> None:
    st.subheader("📸 我要收纳")
    st.caption("拍下物品照片，AI 自动识别名称，填写存放位置后保存。")

    if GLM_API_KEY:
        st.caption(f"✅ 已连接 GLM 视觉模型 `{GLM_MODEL}`")

    image_bytes: bytes | None = None

    tab_mobile, tab_desktop = st.tabs(["📱 手机拍照（推荐）", "💻 电脑摄像头"])

    with tab_mobile:
        st.markdown(
            """
            <div class="mobile-capture-wrap">
            **手机用户请用此方式：**
            点击下方紫色按钮 → 浏览器会询问是否允许使用摄像头 → 选择「允许」即可拍照。
            </div>
            """,
            unsafe_allow_html=True,
        )
        mobile_bytes = mobile_capture_input(key="mobile_capture")
        if mobile_bytes:
            image_bytes = mobile_bytes
        else:
            uploaded = st.file_uploader(
                "或从相册选择图片",
                type=["jpg", "jpeg", "png", "webp"],
                key="mobile_upload",
                help="若上方按钮无效，可从此处选图或拍照",
            )
            if uploaded is not None:
                image_bytes = uploaded.getvalue()

    with tab_desktop:
        st.caption("电脑端可用实时摄像头；手机请使用「手机拍照」标签页。")
        camera_photo = st.camera_input("实时摄像头拍摄", key="store_camera")
        if camera_photo is not None:
            image_bytes = camera_photo.getvalue()

    if image_bytes:
        _process_captured_image(image_bytes)
    elif st.session_state.get("image_bytes"):
        image_bytes = st.session_state.image_bytes

    if image_bytes:
        st.image(image_bytes, caption="当前照片", use_container_width=True)
        _render_item_form()
    else:
        st.markdown(
            """
            > 💡 **使用提示**
            > 1. **手机**：点「手机拍照」→ 允许摄像头权限 → 拍照
            > 2. **电脑**：切到「电脑摄像头」标签页使用实时预览
            > 3. AI 自动识别物品名称，可手动修改后保存
            """
        )


def render_search_tab() -> None:
    st.subheader("🔍 我要寻找")
    st.caption("输入物品关键词，模糊搜索已收纳的记录。")

    keyword = st.text_input(
        "搜索关键词",
        placeholder="例如：充电器、钥匙、护照…",
        key="search_keyword",
    )

    if keyword.strip():
        results = search_items(keyword.strip())

        if not results:
            st.warning(f"没有找到与「{keyword}」相关的物品，换个关键词试试？")
            return

        st.success(f"找到 {len(results)} 条记录")

        for item in results:
            img_path = Path(item["image_path"])
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.info("图片文件不存在")

                st.markdown(
                    f'<div class="item-title">📌 {item["item_name"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="meta">
                        <span class="badge badge-location">📍 {item["location"]}</span>
                        <span class="badge badge-time">🕐 {item["created_at"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.divider()
    else:
        total = search_items("")  # 空关键词匹配全部
        if total:
            st.info(f"数据库中共有 {len(total)} 件已收纳物品，输入关键词开始搜索。")
        else:
            st.info("还没有收纳记录，去「我要收纳」拍第一张吧！")


# ── 入口 ────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="找到了么",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_db()
    inject_mobile_css()
    render_header()

    tab_store, tab_search = st.tabs(["📸 我要收纳", "🔍 我要寻找"])
    with tab_store:
        render_store_tab()
    with tab_search:
        render_search_tab()


if __name__ == "__main__":
    main()