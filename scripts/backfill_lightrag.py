"""回填已有数据到 LightRAG 图谱"""
import asyncio
import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")

from server.config import get_config
from server.engine import get_engine
from server.lightrag_engine import LightRAGEngine


async def main():
    cfg = get_config()
    engine = get_engine()
    lightrag = LightRAGEngine(cfg)

    if not lightrag.is_available():
        logger.error("LightRAG 未启用或初始化失败")
        return

    all_data = engine.get_all_texts()
    if not all_data:
        logger.info("没有需要回填的数据")
        return

    logger.info("共 %d 条父文档，开始回填图谱...", len(all_data))

    texts = []
    ids = []
    for item in all_data:
        body = (item.get("text") or "").strip() or item.get("title", "")
        texts.append(body)
        ids.append(item["id"])

    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        logger.info("回填第 %d-%d 条...", i + 1, i + len(batch_texts))
        result = await lightrag.async_insert(batch_texts, ids=batch_ids)
        logger.info("  结果: %s", result.get("message", "ok"))
        await asyncio.sleep(1)

    logger.info("✅ 回填完成")
    status = lightrag.get_status()
    logger.info("图谱状态: node_count=%s", status.get("node_count", "?"))


if __name__ == "__main__":
    asyncio.run(main())
