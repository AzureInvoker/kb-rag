#!/usr/bin/env python3
"""将旧库 documents 字段回填到 metadata.content（无需 reembed）。"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from server.engine import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_content")


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 metadata.content")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    args = parser.parse_args()

    engine = get_engine()
    all_docs = engine.collection.get()
    ids = all_docs.get("ids") or []
    metas = all_docs.get("metadatas") or []
    documents = all_docs.get("documents") or []

    updated = 0
    for i, hit_id in enumerate(ids):
        meta = metas[i] or {}
        if isinstance(meta.get("content"), str) and meta["content"].strip():
            continue
        doc_text = documents[i] if i < len(documents) else ""
        if not doc_text.strip():
            continue
        updated += 1
        if args.dry_run:
            continue
        new_meta = dict(meta)
        new_meta["content"] = doc_text
        if not new_meta.get("parent_id"):
            new_meta["parent_id"] = hit_id
        engine.collection.update(ids=[hit_id], metadatas=[new_meta])

    logger.info("需回填 %d 条%s", updated, "（dry-run）" if args.dry_run else "，已完成")


if __name__ == "__main__":
    main()
