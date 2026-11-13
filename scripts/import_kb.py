#!/usr/bin/env python3
"""从 JSONL 导入知识库；--wipe 清空后重建。ui_control 不进 LightRAG。"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_kb")


def main() -> None:
    parser = argparse.ArgumentParser(description="导入知识库 JSONL")
    parser.add_argument("input", help="JSONL 路径")
    parser.add_argument("--wipe", action="store_true", help="清空 chroma 后导入")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    from config import get_config
    from server.engine import VectorEngine, get_engine
    from server.models import KnowledgeItem
    from server.taxonomy import should_index_graph

    cfg = get_config()
    if args.wipe and not args.dry_run:
        chroma = Path(cfg.chroma_dir)
        if chroma.exists():
            bak = chroma.parent / f"{chroma.name}_wipe_bak"
            if bak.exists():
                shutil.rmtree(bak)
            shutil.move(str(chroma), str(bak))
            logger.warning("已备份旧库到 %s", bak)
        # 重置单例
        import server.engine as eng_mod
        eng_mod._engine_instance = None

    engine = get_engine()
    added = 0
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                item = KnowledgeItem(
                    id=row.get("id") or "",
                    doc_type=row.get("doc_type") or "doc",
                    title=row.get("title") or "",
                    content=row.get("content") or "",
                    metadata=row.get("metadata") or {},
                    tags=row.get("tags") or [],
                    created_at=row.get("created_at") or "",
                    category=row.get("category") or "",
                    game=row.get("game") or "",
                    app=row.get("app") or "",
                    governance_status=row.get("governance_status") or "published",
                    version=int(row.get("version") or 1),
                    updated_at=row.get("updated_at") or "",
                    next_verify_at=row.get("next_verify_at") or "",
                    source_quality=row.get("source_quality") or "curated",
                )
                if args.dry_run:
                    added += 1
                    continue
                if not item.id:
                    item.id = item.gen_id()
                engine.add(item)
                added += 1
            except Exception as exc:
                errors += 1
                logger.warning("line %d failed: %s", i, exc)

    logger.info("import done: added=%d errors=%d dry_run=%s", added, errors, args.dry_run)


if __name__ == "__main__":
    main()
