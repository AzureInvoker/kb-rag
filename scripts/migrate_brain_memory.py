#!/usr/bin/env python3
"""
脑记忆迁移脚本 — bulk import + dedupe

使用方式：
  python scripts/migrate_brain_memory.py import --file /path/to/brain_memory.json
  python scripts/migrate_brain_memory.py dedupe              # 合并已有重复
  python scripts/migrate_brain_memory.py prune --dry-run      # 预览清理（默认）

JSON 文件格式：
  [
    {
      "target": "查天气",
      "method": "web_search",
      "source": "baidu",
      "params": "",
      "pleasure": 5,
      "note": "顺利返回实时天气",
      "created_at": "2026-07-10 12:00:00"
    },
    ...
  ]
"""

import json
import sys
import os
from datetime import datetime
from collections import defaultdict

# 添加项目根到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import KnowledgeItem
from server.engine import get_engine


def import_brain_memory(filepath: str, dry_run: bool = True):
    """批量导入脑记忆，自动去重"""
    engine = get_engine()

    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("❌ JSON 必须是数组格式")
        return 1

    print(f"📥 从 {filepath} 读取到 {len(records)} 条记录\n")

    # 拉取已有记录做去重
    existing = engine.collection.get(where={"doc_type": "brain_memory"})
    existing_titles = set()
    if existing and existing["ids"]:
        for m in (existing["metadatas"] or []):
            t = m.get("title", "")
            if t:
                existing_titles.add(t)

    imported = 0
    skipped = 0
    for i, rec in enumerate(records):
        target = rec.get("target", "").strip()
        if not target:
            print(f"  ⏭️ 第 {i+1} 条: target 为空，跳过")
            skipped += 1
            continue

        method = rec.get("method", "").strip()
        source = rec.get("source", "").strip()
        params = rec.get("params", "").strip()
        pleasure = max(-10, min(10, int(rec.get("pleasure", 0))))
        note = rec.get("note", "").strip()

        sig_parts = [target]
        if method: sig_parts.append(method)
        if source: sig_parts.append(source)
        if params: sig_parts.append(params)
        pathsig = " | ".join(sig_parts)

        if pathsig in existing_titles:
            print(f"  ⏭️ 已存在: 「{pathsig}」，跳过")
            skipped += 1
            continue

        now = rec.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item = KnowledgeItem(
            title=pathsig,
            doc_type="brain_memory",
            content=note,
            metadata={
                "target": target,
                "method": method,
                "source": source,
                "params": params,
                "pleasure": pleasure,
                "last_pleasure": pleasure,
                "tries": 1,
                "avg_pleasure": float(pleasure),
                "min_pleasure": pleasure,
                "max_pleasure": pleasure,
                "success_count": 1 if pleasure >= 0 else 0,
                "fail_count": 1 if pleasure < 0 else 0,
                "reliability": 1.0 if pleasure >= 0 else 0.0,
                "updated_at": now,
            },
            tags=[target, method] if method else [target],
            created_at=now,
        )
        item.id = item.gen_id()

        if not dry_run:
            engine.add(item)
            existing_titles.add(pathsig)

        imported += 1
        print(f"  ✅ 导入: 「{pathsig}」({pleasure:+d})")

    print(f"\n📊 汇总: 导入 {imported}, 跳过 {skipped} / 共 {len(records)}")
    return 0


def deduplicate(dry_run: bool = True):
    """合并 brain_memory 中的重复 pathsig"""
    engine = get_engine()

    all_docs = engine.collection.get(where={"doc_type": "brain_memory"})
    if not all_docs or not all_docs["ids"]:
        print("无 brain_memory 记录")
        return 0

    total_before = len(all_docs["ids"])
    groups = defaultdict(list)
    for i, eid in enumerate(all_docs["ids"]):
        m = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
        groups[m.get("title", "unknown")].append({
            "id": eid,
            "meta": m,
        })

    to_delete = []
    merged = 0
    for title, records in groups.items():
        if len(records) <= 1:
            continue
        best = max(records, key=lambda r: r["meta"].get("tries", 1))
        for r in records:
            if r["id"] != best["id"]:
                to_delete.append(r["id"])
        merged += 1
        print(f"  🔗 合并「{title}」: {len(records)} 条 → 保留 `{best['id']}` (tries={best['meta'].get('tries', 1)})")

    if not to_delete:
        print("✅ 无重复记录")
        return 0

    print(f"\n共 {merged} 组重复，将删除 {len(to_delete)} 条")
    if not dry_run:
        engine.collection.delete(ids=to_delete)
        print(f"✅ 已删除 {len(to_delete)} 条重复记录")
    else:
        print("🔍 dry_run=true，未实际删除")
    return 0


def show_help():
    print(__doc__.strip())


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        show_help()
        sys.exit(0)

    cmd = sys.argv[1]
    dry_run = "--no-dry-run" not in sys.argv

    if cmd == "import":
        filepath = None
        for i, arg in enumerate(sys.argv):
            if arg == "--file" and i + 1 < len(sys.argv):
                filepath = sys.argv[i + 1]
        if not filepath:
            print("❌ 请提供 --file 参数指定 JSON 文件路径")
            sys.exit(1)
        rc = import_brain_memory(filepath, dry_run=dry_run)
        sys.exit(rc)

    elif cmd == "dedupe":
        rc = deduplicate(dry_run=dry_run)
        sys.exit(rc)

    else:
        print(f"未知命令: {cmd}")
        show_help()
        sys.exit(1)
