"""
FindIt 视觉寻物小助手 — 云端完全体
ImgBB 永久图床 + GitHub 云账本 + Gemini 3.5 Flash 多物品识别
"""

import hashlib
import json
import os
import re

import streamlit as st
from google import genai
from google.genai import types

from components.mobile_capture import mobile_capture_input
from components.voice_input import voice_recognition_button
from services.gemini_vision import (
    GEMINI_MODEL,
    ITEMS_JSON_SCHEMA,
    MULTI_ITEM_PROMPT,
    GeminiVisionError,
    compress_to_jpeg,
    transcribe_audio,
)
from services.github_db import (
    GitHubDBError,
    load_global_database,
    make_item_record,
    save_global_database,
)
from services.imgbb import ImgBBUploadError, upload_to_imgbb

UPLOAD_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "heic"]


# ── 密钥配置 ────────────────────────────────────────────────
def _get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    return key or ""


def _get_imgbb_api_key() -> str:
    key = os.environ.get("IMGBB_API_KEY", "")
    if not key:
        try:
            key = st.secrets["IMGBB_API_KEY"]
        except Exception:
            pass
    return key or ""


def _get_github_config() -> tuple[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "")
    try:
        token = token or st.secrets["GITHUB_TOKEN"]
        repo = repo or st.secrets["GITHUB_REPO"]
    except Exception:
        pass
    return token or "", repo or ""


# ── Gemini 多物品识别（官方 SDK + 强锁 JSON 数组）────────────
def _gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _items_generate_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=ITEMS_JSON_SCHEMA,
    )


def items_from_gemini_response(response) -> list[str]:
    """从 Gemini 响应中鲁棒解析物品名称，保证返回干净的字符串列表。"""
    text_content = (response.text or "").strip()
    if not text_content:
        raise GeminiVisionError("Gemini 未返回识别结果。")

    if text_content.startswith("```"):
        parts = text_content.split("```")
        if len(parts) >= 3:
            text_content = parts[1].strip()
            if text_content.startswith("json"):
                text_content = text_content[4:].strip()

    try:
        data = json.loads(text_content)
    except json.JSONDecodeError as exc:
        raise GeminiVisionError("Gemini 返回格式无法解析。") from exc

    if isinstance(data, list):
        final_list = data
    elif isinstance(data, dict):
        final_list = []
        for val in data.values():
            if isinstance(val, list):
                final_list = val
                break
        if not final_list:
            final_list = list(data.keys())
    else:
        final_list = [str(data)]

    names: list[str] = []
    seen: set[str] = set()
    for item in final_list:
        name = re.sub(r'["""\'「」【】\[\]]', "", str(item).strip())
        name = name or "未知物品"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names or ["未知物品"]


def recognize_items_from_image(image_bytes: bytes, api_key: str) -> list[str]:
    """ImgBB 上传完成后，用 Gemini 3.5 Flash 识别图中所有物品，返回干净 JSON 数组。"""
    if not api_key:
        raise GeminiVisionError("GEMINI_API_KEY 未配置，请在 .streamlit/secrets.toml 中设置。")
    if not image_bytes:
        raise GeminiVisionError("图片数据为空。")

    jpeg_bytes = compress_to_jpeg(image_bytes)
    client = _gemini_client(api_key)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=MULTI_ITEM_PROMPT),
            ],
            config=_items_generate_config(),
        )
    except Exception as exc:
        raise GeminiVisionError(
            "Gemini API 调用失败。云端服务器会代为访问 Google，手机无需 VPN。"
            f" 详情: {exc}"
        ) from exc

    return items_from_gemini_response(response)


def commit_items_to_github(
    item_names: list[str],
    location: str,
    img_url: str,
    token: str,
    repo: str,
) -> list[dict]:
    """解析物品数组，逐条写入 GitHub findit_db.json（共用同一张合照 img_url）。"""
    records: list[dict] = []
    for name in item_names:
        records.append(make_item_record(name, location, img_url))

    new_pool = list(st.session_state.get("items_pool", [])) + records
    save_global_database(
        new_pool,
        token,
        repo,
        message=f"Add {len(records)} items",
    )
    st.session_state.items_pool = new_pool
    st.session_state.github_sync_ok = True
    return records


# ── 云端账本 ────────────────────────────────────────────────
def ensure_items_pool_loaded(force: bool = False) -> None:
    if not force and "items_pool" in st.session_state:
        return

    token, repo = _get_github_config()
    if not token or not repo:
        st.session_state.items_pool = []
        st.session_state.github_sync_ok = False
        st.session_state.github_sync_error = "未配置 GITHUB_TOKEN 或 GITHUB_REPO"
        return

    try:
        st.session_state.items_pool = load_global_database(token, repo)
        st.session_state.github_sync_ok = True
        st.session_state.github_sync_error = ""
    except GitHubDBError as exc:
        st.session_state.items_pool = []
        st.session_state.github_sync_ok = False
        st.session_state.github_sync_error = str(exc)


def search_cloud_items(keyword: str, pool: list[dict]) -> list[dict]:
    kw = keyword.strip().lower()
    if not kw:
        return sorted(pool, key=lambda x: x.get("created_at", ""), reverse=True)

    results = []
    for item in pool:
        name = str(item.get("name", "")).lower()
        location = str(item.get("location", "")).lower()
        if kw in name or kw in location:
            results.append(item)
    return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)


# ── 样式 ────────────────────────────────────────────────────
def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        footer { visibility: hidden; }
        .app-header { text-align: center; padding: 0.5rem 0 1rem; }
        .app-header h1 {
            font-size: clamp(1.4rem, 5vw, 2rem); margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .app-header p { color: #6b7280; font-size: 0.9rem; margin: 0.3rem 0 0; }
        .cloud-bar {
            text-align: center; padding: 0.5rem 1rem; margin-bottom: 0.5rem;
            background: #f0fdf4; border-radius: 10px; color: #166534; font-size: 0.9rem;
        }
        .suggest-card {
            background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
            border-left: 4px solid #667eea; border-radius: 12px;
            padding: 1rem 1.2rem; margin: 0.8rem 0;
        }
        .result-card {
            background: #fff; border: 1px solid #e5e7eb; border-radius: 16px;
            padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .result-card .item-title { font-size: 1.2rem; font-weight: 700; color: #111827; margin: 0.5rem 0; }
        .badge {
            display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
            font-size: 0.82rem; margin-right: 0.4rem;
        }
        .badge-location { background: #ecfdf5; color: #065f46; }
        .badge-time { background: #eff6ff; color: #1e40af; }
        @media (max-width: 768px) {
            .block-container { padding: 1rem; max-width: 100%; }
            .stButton > button { width: 100%; padding: 0.75rem; border-radius: 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── UI 辅助 ─────────────────────────────────────────────────
def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>📦 FindIt 找到了么</h1>
            <p>全家云端收纳 · 手机随时访问 · 无需开电脑 · Gemini 智能识物</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pool = st.session_state.get("items_pool", [])
    if st.session_state.get("github_sync_ok"):
        st.markdown(
            f'<div class="cloud-bar">☁️ 云端已同步 <b>{len(pool)}</b> 条记录 · 全家设备共享</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.get("github_sync_error"):
        st.warning(f"云端同步异常：{st.session_state.github_sync_error}")


def _text_input_with_voice(label: str, field_key: str, placeholder: str = "") -> str:
    """文本框 + 下方「按住说话」按钮（Gemini 语音转写）。"""
    if field_key not in st.session_state:
        st.session_state[field_key] = ""

    text = st.text_input(
        label,
        value=st.session_state[field_key],
        placeholder=placeholder,
        key=f"input_{field_key}",
    )

    raw_voice = voice_recognition_button(key=f"voice_{field_key}")
    if raw_voice:
        if raw_voice.startswith("__AUDIO__|"):
            gemini_key = _get_gemini_api_key()
            if not gemini_key:
                st.error("请配置 GEMINI_API_KEY 以使用语音输入。")
            else:
                try:
                    _, mime, b64 = raw_voice.split("|", 2)
                    with st.spinner("🎤 正在识别语音…"):
                        voice_text = transcribe_audio(b64, mime, gemini_key)
                    st.session_state[field_key] = voice_text
                    st.rerun()
                except GeminiVisionError as exc:
                    st.error(str(exc))
        else:
            st.session_state[field_key] = raw_voice
            st.rerun()

    st.session_state[field_key] = text
    return text


def _clear_capture_state() -> None:
    keys_to_clear = [
        "last_image_hash",
        "image_bytes",
        "img_url",
        "detected_items",
        "item_name_fields",
        "mobile_capture",
        "mobile_upload",
        "submit_locked",
        "last_saved_records",
        "location_field",
    ]
    for key in list(st.session_state.keys()):
        if key in keys_to_clear or key.startswith("item_field_") or key.startswith("input_item_field_"):
            st.session_state.pop(key, None)


def _process_captured_image(image_bytes: bytes) -> None:
    """流程：ImgBB 永久图床 → Gemini JSON 数组识物。"""
    if st.session_state.get("submit_locked"):
        return
    image_hash = hashlib.md5(image_bytes).hexdigest()
    if st.session_state.get("last_image_hash") == image_hash:
        return

    st.session_state.last_image_hash = image_hash
    st.session_state.image_bytes = image_bytes

    imgbb_key = _get_imgbb_api_key()
    gemini_key = _get_gemini_api_key()
    if not imgbb_key:
        st.warning("未配置 IMGBB_API_KEY，无法上传图片。")
        return

    with st.spinner("📤 正在上传图片至 ImgBB 永久图床…"):
        try:
            img_url = upload_to_imgbb(image_bytes, imgbb_key)
            st.session_state.img_url = img_url
        except ImgBBUploadError as exc:
            st.error(str(exc))
            return

    if not gemini_key:
        st.session_state.detected_items = ["未知物品"]
        st.session_state.item_field_0 = "未知物品"
        st.warning("未配置 GEMINI_API_KEY，请手动填写物品名称。")
        return

    with st.spinner("🔍 Gemini 正在识别图中所有物品…"):
        try:
            items = recognize_items_from_image(image_bytes, gemini_key)
            st.session_state.detected_items = items
            for i, name in enumerate(items):
                st.session_state[f"item_field_{i}"] = name
        except GeminiVisionError as exc:
            st.session_state.detected_items = ["未知物品"]
            st.session_state.item_field_0 = "未知物品"
            st.warning(str(exc))


def _render_cloud_save_form() -> None:
    if st.session_state.get("submit_locked"):
        records = st.session_state.get("last_saved_records", [])
        st.success(f"🎉 已经送往云端保险箱（共 {len(records)} 件）")
        if records and records[0].get("img_url"):
            st.image(records[0]["img_url"], use_container_width=True)
        for rec in records:
            st.markdown(f"- **{rec.get('name')}** → 📍 {rec.get('location')} · 🕐 {rec.get('created_at')}")
        if st.button("➕ 继续录入下一批", type="primary", use_container_width=True):
            _clear_capture_state()
            st.rerun()
        return

    detected = st.session_state.get("detected_items", ["未知物品"])
    img_url = st.session_state.get("img_url", "")
    st.success(f"✨ Gemini 识别到 **{len(detected)}** 个物品，请确认名称并填写放置位置")
    if img_url:
        st.caption(f"📎 合照已上传：{img_url}")

    st.markdown(
        f'<div class="suggest-card">AI 建议：{json.dumps(detected, ensure_ascii=False)}</div>',
        unsafe_allow_html=True,
    )

    item_names: list[str] = []
    for i in range(len(detected)):
        field_key = f"item_field_{i}"
        if field_key not in st.session_state:
            st.session_state[field_key] = detected[i]
        name = _text_input_with_voice(f"物品 {i + 1} 名称", field_key)
        item_names.append(name)

    if st.button("➕ 手动添加一个物品"):
        new_idx = len(detected)
        st.session_state.detected_items = detected + [""]
        st.session_state[f"item_field_{new_idx}"] = ""
        st.rerun()

    location = _text_input_with_voice(
        "放置位置（本批物品共用）",
        "location_field",
        placeholder="例如：客厅电视柜第二层抽屉",
    )

    col1, col2 = st.columns(2)
    with col1:
        save_clicked = st.button("☁️ 确认保存至云端", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 重新拍照", use_container_width=True):
            _clear_capture_state()
            st.rerun()

    if not save_clicked:
        return

    valid_names = [n.strip() for n in item_names if n.strip()]
    if not valid_names:
        st.error("请至少填写一个物品名称。")
        return
    if not location.strip():
        st.error("请填写放置位置。")
        return

    img_url = st.session_state.get("img_url", "")
    token, repo = _get_github_config()
    if not img_url:
        st.error("图片尚未上传至 ImgBB，请重新拍照。")
        return
    if not token or not repo:
        st.error("未配置 GITHUB_TOKEN 或 GITHUB_REPO。")
        return

    with st.spinner("正在批量写入 GitHub 云账本…"):
        try:
            records = commit_items_to_github(
                valid_names,
                location.strip(),
                img_url,
                token,
                repo,
            )
            st.session_state.submit_locked = True
            st.session_state.last_saved_records = records
            st.balloons()
            st.rerun()
        except GitHubDBError as exc:
            st.error(str(exc))


def render_add_tab() -> None:
    st.subheader("📥 物品录入")
    st.caption("拍照 → ImgBB 永久图床 → Gemini 识别多个物品 → 语音/文字填写 → 一键存入云端")

    if st.session_state.get("submit_locked"):
        _render_cloud_save_form()
        return

    image_bytes: bytes | None = None
    st.caption("点击下方紫色按钮拍照，或从相册选择（支持 HEIC）")
    mobile_bytes = mobile_capture_input(key="mobile_capture")
    if mobile_bytes:
        image_bytes = mobile_bytes
    else:
        uploaded = st.file_uploader(
            "或从相册选择图片",
            type=UPLOAD_IMAGE_TYPES,
            key="mobile_upload",
            help="若拍照按钮无效，可从此处选图",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()

    if image_bytes:
        _process_captured_image(image_bytes)
    elif st.session_state.get("image_bytes"):
        image_bytes = st.session_state.image_bytes

    if image_bytes:
        st.image(image_bytes, caption="当前照片", use_container_width=True)
        _render_cloud_save_form()
    else:
        st.info("请先拍照或选择一张图片，开始录入。")


def render_search_tab() -> None:
    st.subheader("🔍 智能寻物")
    st.caption("输入关键词，从云端账本中模糊搜索物品")

    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input(
            "搜索关键词",
            placeholder="例如：钥匙、充电器、护照…",
            key="search_keyword",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 刷新云端", use_container_width=True):
            ensure_items_pool_loaded(force=True)
            st.rerun()

    pool = st.session_state.get("items_pool", [])

    if keyword.strip():
        results = search_cloud_items(keyword.strip(), pool)
        if not results:
            st.warning(f"没有找到与「{keyword}」相关的物品。")
            return
        st.success(f"找到 {len(results)} 条记录")
    else:
        results = search_cloud_items("", pool)
        if results:
            st.info(f"云端共有 {len(results)} 件物品，输入关键词开始搜索。")
        else:
            st.info("云端还没有记录，去「物品录入」添加第一件吧！")
            return

    for item in results:
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            img_url = item.get("img_url", "")
            if img_url:
                st.image(img_url, use_container_width=True)
            st.markdown(
                f'<div class="item-title">📌 {item.get("name", "未知物品")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="badge badge-location">📍 {item.get("location", "未填写")}</span>'
                f'<span class="badge badge-time">🕐 {item.get("created_at", "")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="FindIt 找到了么",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_mobile_css()
    ensure_items_pool_loaded()
    render_header()

    tab_add, tab_search = st.tabs(["📥 物品录入", "🔍 智能寻物"])
    with tab_add:
        render_add_tab()
    with tab_search:
        render_search_tab()


if __name__ == "__main__":
    main()
