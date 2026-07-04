"""Milestone 2：GitHub 云端账本命令行自测。"""
import json
import sys
from pathlib import Path

from services.github_db import (
    GitHubDBError,
    load_global_database,
    make_item_record,
    save_global_database,
)

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


def _read_secret(name: str) -> str:
    if not SECRETS_PATH.exists():
        return ""
    for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name} ="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> int:
    token = _read_secret("GITHUB_TOKEN")
    repo = _read_secret("GITHUB_REPO")

    if not token:
        print("❌ 请在 .streamlit/secrets.toml 中配置 GITHUB_TOKEN")
        return 1
    if not repo:
        print("❌ 请在 .streamlit/secrets.toml 中配置 GITHUB_REPO（格式 owner/repo）")
        return 1

    print(f"仓库: {repo}")
    print("-" * 40)

    print("[1/2] 读取云端账本 …")
    try:
        items = load_global_database(token, repo)
    except GitHubDBError as exc:
        print(f"❌ 读取失败: {exc}")
        return 1
    print(f"  ✅ 当前共 {len(items)} 条记录")

    print("[2/2] 写入测试记录 …")
    test_item = make_item_record(
        name="M2测试物品",
        location="测试抽屉",
        img_url="https://i.ibb.co/S4sRfw6r/photo.jpg",
    )
    new_items = items + [test_item]
    try:
        save_global_database(
            new_items,
            token,
            repo,
            message=f"Add test item: {test_item['name']}",
        )
    except GitHubDBError as exc:
        print(f"❌ 写入失败: {exc}")
        return 1

    print("  ✅ 写入成功")
    print("  新增记录:")
    print(json.dumps(test_item, ensure_ascii=False, indent=2))
    print("-" * 40)
    print("请到 GitHub 仓库根目录查看 findit_db.json 是否已更新。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
