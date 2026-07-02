"""点击「语音识别」→ 录音 → Gemini 转文字。"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_voice_input = components.declare_component(
    "voice_input",
    path=str(Path(__file__).resolve().parent),
)


def voice_recognition_button(key: str) -> str | None:
    """返回纯文本，或 __AUDIO__|mime|base64。"""
    result = _voice_input(key=key, default=None)
    if result and isinstance(result, str) and result.strip():
        return result.strip()
    return None

# 兼容旧名
voice_hold_input = voice_recognition_button
