"""手机端拍照组件：通过 capture 属性直接调起系统摄像头。"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).resolve().parent / "index.html"
_mobile_capture = components.declare_component("mobile_capture", path=str(_FRONTEND.parent))


def mobile_capture_input(key: str = "mobile_capture") -> bytes | None:
    data_url = _mobile_capture(key=key)
    if not data_url or not isinstance(data_url, str):
        return None
    if data_url.startswith("data:"):
        b64 = data_url.split(",", 1)[1]
        return base64.b64decode(b64)
    return None
