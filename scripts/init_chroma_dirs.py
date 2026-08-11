#!/usr/bin/env python3
"""初始化 ChromaDB 目录（清空后或服务首次部署时运行）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from config import get_config, _find_project_root


def main() -> None:
    cfg = get_config()
    root = _find_project_root()
    dirs = [
        Path(cfg.chroma_dir).resolve(),
        Path(cfg.memory_chroma_dir).resolve(),
        (root / cfg.lightrag_working_dir).resolve(),
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {path}")
    print("\n目录已就绪，请 ./run.sh restart")


if __name__ == "__main__":
    main()
