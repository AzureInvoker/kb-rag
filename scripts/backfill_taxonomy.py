#!/usr/bin/env python3
"""回填 doc_type / governance / game / app 等顶层 meta（不 reembed）。"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.taxonomy import (  # noqa: E402
    compute_next_verify_at,
    infer_source_quality,
    normalize_doc_type,
    now_str,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_taxonomy")

GAME_HINTS = {
    "blitz": "blitz",
    "starburst": "starburst",
    "jewels": "jewels",
    "yum": "yum",
    "frenzy": "frenzy",
}


def _guess_game(title: str, tags: list, content: str) -> str:
    blob = f"{title} {' '.join(tags)} {content[:200]}".lower()
    for key, val in GAME_HINTS.items():
        if key in blob:
            return val
    return ""


def _guess_app(title: str, tags: list, content: str, meta: dict) -> str:
    if meta.get("app"):
        return str(meta["app"])
    blob = f"{title} {' '.join(tags)} {content[:300]}"
    m = re.search(r"(com\.[a-zA-Z0-9_.]+)", blob)
    if m:
        return m.group(1)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 taxonomy meta")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = parser.parse_args()

    from config import get_config
    from server.engine import get_engine

    cfg = get_config()
    engine = get_engine()
    parents = engine._parent_documents(include_retired=True)

    type_counter = Counter()
    mapped = 0
    updated = 0
    for item in parents:
        if item.get("doc_type") == "brain_memory":
            continue
        old_type = item.get("doc_type") or "doc"
        new_type = normalize_doc_type(old_type)
        type_counter[new_type] += 1
        if old_type != new_type:
            mapped += 1

        tags = item.get("tags") or []
        meta = dict(item.get("metadata") or {})
        game = item.get("game") or _guess_game(item.get("title", ""), tags, item.get("content") or "")
        app = item.get("app") or _guess_app(item.get("title", ""), tags, item.get("content") or "", meta)
        status = item.get("governance_status") or "published"
        sq = item.get("source_quality") or infer_source_quality(tags)
        nva = item.get("next_verify_at") or compute_next_verify_at(new_type, cfg.freshness_sla_days)

        if args.dry_run:
            continue

        fields = {
            "doc_type": new_type,
            "game": game,
            "app": app,
            "governance_status": status,
            "source_quality": sq,
            "next_verify_at": nva,
            "category": item.get("category") or meta.get("category") or "",
        }
        # 强制走 meta-only：不改正文
        result = engine.update(item["id"], **fields)
        if result:
            updated += 1

    logger.info(
        "parents=%d mapped_type=%d updated=%d dry_run=%s by_type=%s",
        len(parents), mapped, updated, args.dry_run, dict(type_counter),
    )


if __name__ == "__main__":
    main()
