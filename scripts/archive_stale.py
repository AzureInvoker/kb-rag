#!/usr/bin/env python3
"""将长期 stale 的条目归档为 archived。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("archive_stale")


def main() -> None:
    parser = argparse.ArgumentParser(description="归档长期 stale 知识")
    parser.add_argument("--dry-run", action="store_true", default=True, help="只预览（默认）")
    parser.add_argument("--apply", action="store_true", help="实际写入")
    args = parser.parse_args()
    dry_run = not args.apply

    from server.engine import get_engine

    result = get_engine().archive_stale_batch(dry_run=dry_run)
    logger.info("archive_stale: %s", result)


if __name__ == "__main__":
    main()
