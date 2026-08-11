#!/usr/bin/env python3
"""对已有长文档重新分块索引（会删除旧 parent/chunk 后重建）。"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from server.engine import VectorEngine, get_engine
from server.models import KnowledgeItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="重新分块索引长文档")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--doc-type", default="", help="仅处理指定 doc_type")
    args = parser.parse_args()

    engine: VectorEngine = get_engine()
    all_docs = engine.collection.get()
    ids = all_docs.get("ids") or []
    metas = all_docs.get("metadatas") or []
    documents = all_docs.get("documents") or []

    seen_parents: set[str] = set()
    candidates: list[KnowledgeItem] = []

    for i, hit_id in enumerate(ids):
        meta = metas[i] or {}
        if meta.get("doc_type") == "brain_memory":
            continue
        if args.doc_type and meta.get("doc_type") != args.doc_type:
            continue
        parent_id = (meta.get("parent_id") or hit_id).strip()
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        content = meta.get("content") or (documents[i] if i < len(documents) else "")
        if len(str(content)) <= engine._chunk_min_chars:
            continue

        item = KnowledgeItem(
            id=parent_id,
            doc_type=meta.get("doc_type", "doc"),
            title=meta.get("title", ""),
            content=str(content),
            tags=engine._parse_tags(meta.get("tags", "[]")),
            metadata=engine._extract_brain_meta(meta),
            created_at=meta.get("created_at", ""),
        )
        candidates.append(item)

    logger.info("待重分块 %d 条", len(candidates))
    if args.dry_run:
        return

    for item in candidates:
        engine.add(item)
        logger.info("  ✅ %s (%d chars)", item.title[:40], len(item.content))

    logger.info("完成")


if __name__ == "__main__":
    main()
