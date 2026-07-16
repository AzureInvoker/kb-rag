#!/usr/bin/env python3
"""
从 testcase-rag 迁移数据到 kb-rag

将旧项目中的 TestCase 转换为 KnowledgeItem（doc_type="test_case"）。

用法:
  cd /home/admin/kb-rag
  uv run python3 scripts/migrate_from_testcase_rag.py

说明:
  1. 读取 testcase-rag 的 ChromaDB
  2. 全量转成 KnowledgeItem 写入 kb-rag 的 ChromaDB
  3. 如果 LightRAG 启用，同步插入图谱
  4. 打印迁移统计
"""

import sys
import json
import logging
from pathlib import Path

# 添加 kb-rag 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("migrate")


def migrate():
    # 1. 连接旧项目 ChromaDB
    OLD_CHROMA_DIR = Path("/home/admin/testcase-rag/.chroma_db")
    if not OLD_CHROMA_DIR.exists():
        logger.error(f"❌ 旧项目 ChromaDB 不存在: {OLD_CHROMA_DIR}")
        logger.error("   请确认 /home/admin/testcase-rag/.chroma_db/ 存在")
        sys.exit(1)

    import chromadb
    old_client = chromadb.PersistentClient(path=str(OLD_CHROMA_DIR))
    old_collections = old_client.list_collections()
    # 找 testcases 集合（旧项目可能叫 testcases 或其他名字）
    old_col = None
    for c in old_collections:
        logger.info(f"  找到旧集合: {c.name}")
        if c.name == "testcases":
            old_col = c
            break
    if old_col is None:
        # 取第一个集合
        if old_collections:
            old_col = old_collections[0]
            logger.info(f"  使用旧集合: {old_col.name}")
        else:
            logger.error("❌ 旧项目中没有集合")
            sys.exit(1)

    old_data = old_col.get()
    old_count = len(old_data["ids"])
    logger.info(f"  旧项目共 {old_count} 条数据")

    if old_count == 0:
        logger.info("  旧项目无数据，无需迁移")
        return

    # 2. 读取 kb-rag 配置和引擎
    from server.config import get_config
    from server.engine import get_engine
    from server.models import KnowledgeItem
    from server.lightrag_engine import LightRAGEngine

    cfg = get_config()
    engine = get_engine()
    lightrag = LightRAGEngine(cfg)

    # 3. 转换数据
    items = []
    for i, id_ in enumerate(old_data["ids"]):
        meta = old_data["metadatas"][i] if old_data["metadatas"] else {}
        doc = old_data["documents"][i] if old_data["documents"] else ""

        # 解析 tags
        raw_tags = meta.get("tags", "")
        tags = raw_tags.split(",") if raw_tags else []
        tags = [t.strip() for t in tags if t.strip()]

        # 构建 metadata
        item_meta = {}
        for key in ("module", "sub_module", "priority", "category",
                     "preconditions", "project", "creator", "expected"):
            val = meta.get(key, "")
            if val:
                item_meta[key] = val

        # steps 从 document text 中提取（如果有）
        # 旧版本的 embedding_text 包含步骤信息

        # 构建 KnowledgeItem
        item = KnowledgeItem(
            id=id_,
            doc_type="test_case",
            title=meta.get("title", ""),
            content=doc,
            metadata=item_meta,
            tags=tags,
            created_at=meta.get("created_at", ""),
        )
        # 确保 ID 一致（使用旧 ID）
        if not item.id:
            item.id = item.gen_id()
        items.append(item)

    # 4. 写入 kb-rag
    logger.info(f"  正在写入 kb-rag（{len(items)} 条）...")
    engine.add_many(items)
    logger.info(f"  ✅ 向量库写入完成")

    # 5. 同步到 LightRAG
    if lightrag.is_available():
        logger.info("  正在同步到 LightRAG 图谱...")
        texts = [it.get_embedding_text() for it in items]
        ids = [it.id for it in items]
        result = lightrag.insert(texts, ids=ids)
        if result.get("ok"):
            logger.info(f"  ✅ LightRAG 同步完成")
        else:
            logger.warning(f"  ⚠️ LightRAG 同步失败: {result.get('message')}")
    else:
        logger.info("  跳过 LightRAG 同步（未启用或不可用）")

    # 6. 统计
    type_counts = {}
    for it in items:
        dt = it.doc_type
        type_counts[dt] = type_counts.get(dt, 0) + 1

    logger.info(f"\n📊 迁移完成")
    logger.info(f"  - 来源: /home/admin/testcase-rag ({old_count} 条)")
    logger.info(f"  - 目标: /home/admin/kb-rag ({len(items)} 条)")
    for dt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    - {dt}: {count} 条")
    logger.info(f"\n💡 kb-rag 端口: {cfg.api_port}")
    logger.info(f"   启动: cd /home/admin/kb-rag && bash run.sh start")


if __name__ == "__main__":
    migrate()
