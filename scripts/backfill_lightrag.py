"""回填已有数据到 LightRAG 图谱"""
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")

from server.config import get_config
from server.engine import get_engine
from server.lightrag_engine import LightRAGEngine
import asyncio


async def main():
    cfg = get_config()
    engine = get_engine()
    lightrag = LightRAGEngine(cfg)

    if not lightrag.is_available():
        logger.error("LightRAG 未启用或初始化失败")
        return

    # 获取全部数据
    all_data = engine.get_all(doc_type=None)
    if not all_data:
        logger.info("没有需要回填的数据")
        return

    logger.info(f"共 {len(all_data)} 条数据，开始回填图谱...")

    texts = []
    ids = []
    for item in all_data:
        texts.append(item.get("content", item.get("title", "")))
        ids.append(item["id"])

    # 分批插入，每批 10 条
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        logger.info(f"回填第 {i+1}-{i+len(batch_texts)} 条...")
        result = await lightrag.async_insert(batch_texts, ids=batch_ids)
        logger.info(f"  结果: {result.get('message', 'ok')}")
        # LightRAG insert 是 LLM 驱动的，建图慢，等一下
        await asyncio.sleep(1)

    logger.info("✅ 回填完成")
    # 验证
    status = lightrag.get_status()
    logger.info(f"图谱状态: node_count={status.get('node_count', '?')}")


if __name__ == "__main__":
    asyncio.run(main())
