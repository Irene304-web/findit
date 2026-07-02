"""GitHub 云端中心账本 findit_db.json 读写服务。"""

from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime

import requests

DB_FILENAME = "findit_db.json"
GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = (15, 60)
MAX_SAVE_RETRIES = 3


class GitHubDBError(Exception):
    """GitHub 账本读写失败。"""


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo_contents_url(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"{GITHUB_API}/repos/{owner}/{name}/contents/{DB_FILENAME}"


def _decode_file_content(content_b64: str) -> list[dict]:
    raw = base64.b64decode(content_b64).decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        return []
    return data


def _fetch_remote(token: str, repo: str) -> tuple[str | None, list[dict]]:
    """返回 (sha, items)。文件不存在时 sha 为 None、items 为 []。"""
    try:
        response = requests.get(
            _repo_contents_url(repo),
            headers=_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise GitHubDBError(f"读取 GitHub 账本失败: {exc}") from exc

    if response.status_code == 404:
        return None, []

    if response.status_code == 401:
        raise GitHubDBError("GitHub Token 无效或已过期，请检查 GITHUB_TOKEN。")

    if response.status_code != 200:
        raise GitHubDBError(
            f"读取 GitHub 账本失败 (HTTP {response.status_code}): {response.text[:200]}"
        )

    body = response.json()
    sha = body.get("sha")
    content = body.get("content", "")
    if not content:
        return sha, []

    try:
        items = _decode_file_content(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise GitHubDBError("云端 findit_db.json 格式损坏，无法解析。") from exc

    return sha, items


def load_global_database(token: str, repo: str) -> list[dict]:
    """从 GitHub 拉取 findit_db.json，写入内存前调用。"""
    if not token:
        raise GitHubDBError("GITHUB_TOKEN 未配置。")
    if not repo or "/" not in repo:
        raise GitHubDBError("GITHUB_REPO 格式应为 owner/repo。")

    _, items = _fetch_remote(token, repo)
    return items


def save_global_database(
    new_data: list[dict],
    token: str,
    repo: str,
    message: str = "Update findit_db.json",
) -> None:
    """将完整账本覆盖写回 GitHub。"""
    if not token:
        raise GitHubDBError("GITHUB_TOKEN 未配置。")
    if not repo or "/" not in repo:
        raise GitHubDBError("GITHUB_REPO 格式应为 owner/repo。")
    if not isinstance(new_data, list):
        raise GitHubDBError("账本数据必须是 list。")

    url = _repo_contents_url(repo)
    encoded = base64.b64encode(
        json.dumps(new_data, ensure_ascii=False, indent=2).encode("utf-8")
    ).decode("utf-8")

    last_error: GitHubDBError | None = None
    for attempt in range(1, MAX_SAVE_RETRIES + 1):
        sha, _ = _fetch_remote(token, repo)
        payload: dict[str, str] = {
            "message": message,
            "content": encoded,
        }
        if sha:
            payload["sha"] = sha

        try:
            response = requests.put(
                url,
                headers=_headers(token),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise GitHubDBError(f"写入 GitHub 账本失败: {exc}") from exc

        if response.status_code == 409 and attempt < MAX_SAVE_RETRIES:
            last_error = GitHubDBError("与其他设备同时写入冲突，正在重试…")
            continue

        if response.status_code == 401:
            raise GitHubDBError("GitHub Token 无效或权限不足，无法写入仓库。")

        if response.status_code not in (200, 201):
            raise GitHubDBError(
                f"写入 GitHub 账本失败 (HTTP {response.status_code}): {response.text[:200]}"
            )
        return

    raise last_error or GitHubDBError("写入 GitHub 账本失败，请稍后重试。")


def make_item_record(name: str, location: str, img_url: str) -> dict:
    """按数据契约生成一条物品记录。"""
    now = datetime.now()
    suffix = secrets.token_hex(2)
    return {
        "id": f"{now.strftime('%Y%m%d_%H%M%S')}_{suffix}",
        "name": name.strip(),
        "location": location.strip(),
        "img_url": img_url.strip(),
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
