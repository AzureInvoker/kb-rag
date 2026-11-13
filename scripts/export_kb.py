#!/usr/bin/env python3
"""导出 parent 级知识为 JSONL。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_kb")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出知识库 JSONL")
    parser.add_argument("-o", "--output", default="backup/kb_export.jsonl")
    args = parser.parse_args()

    from server.engine import get_engine

    engine = get_engine()
    items = engine._parent_documents(include_retired=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            if it.get("doc_type") == "brain_memory":
                continue
            row = {
                "id": it.get("id"),
                "doc_type": it.get("doc_type"),
                "title": it.get("title"),
                "content": it.get("content"),
                "tags": it.get("tags") or [],
                "metadata": it.get("metadata") or {},
                "category": it.get("category") or "",
                "game": it.get("game") or "",
                "app": it.get("app") or "",
                "governance_status": it.get("governance_status") or "published",
                "version": it.get("version") or 1,
                "created_at": it.get("created_at") or "",
                "updated_at": it.get("updated_at") or "",
                "next_verify_at": it.get("next_verify_at") or "",
                "source_quality": it.get("source_quality") or "curated",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    logger.info("exported %d items → %s", n, out)


if __name__ == "__main__":
    main()
