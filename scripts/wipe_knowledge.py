#!/usr/bin/env python3
"""清空主知识库与 LightRAG 图谱（保留路径脑 .chroma_db_memory）。

用法:
  python scripts/wipe_knowledge.py          # 预览
  python scripts/wipe_knowledge.py --yes      # 执行删除
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description="清空主知识库与 LightRAG 存储")
    parser.add_argument("--yes", action="store_true", help="确认执行删除")
    args = parser.parse_args()

    cfg = get_config()
    targets = [
        ("主知识库 ChromaDB", Path(cfg.chroma_dir)),
        ("LightRAG 图谱", Path(cfg.lightrag_working_dir)),
    ]
    if not Path(cfg.lightrag_working_dir).is_absolute():
        targets[1] = (targets[1][0], Path(__file__).resolve().parent.parent / cfg.lightrag_working_dir)

    print("将删除以下目录（路径脑 memory 库保留）:\n")
    for label, path in targets:
        exists = path.exists()
        print(f"  [{label}] {path} {'(存在)' if exists else '(不存在)'}")

    if not args.yes:
        print("\n加 --yes 确认执行")
        return

    for label, path in targets:
        if path.exists():
            shutil.rmtree(path)
            print(f"已删除: {path}")
        else:
            print(f"跳过（不存在）: {path}")
    print("\n✅ 完成。请 ./run.sh restart 后重新录入知识。")


if __name__ == "__main__":
    main()
