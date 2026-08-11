# kb-rag

通用知识库 —— 向量 + BM25 + RRF + Rerank 混合检索，可选 LightRAG 知识图谱增强。

## v1.1.0 检索增强

- **BM25 全文**：标题 + 标签 + 正文关键词匹配
- **RRF 融合**：向量与 BM25 用 Reciprocal Rank Fusion 合并（替代固定 0.6/0.4）
- **Reranker**：可选 `BAAI/bge-reranker-v2-m3` 二次排序
- **长文档分块**：超过 `chunking.min_chars` 自动切块（路径脑 `mb_*` 不受影响）
- **内置评测**：`python scripts/eval_recall.py`

## 架构

```
kb-rag/
├── config.py / config.example.yaml
├── run.sh
├── server/
│   ├── engine.py          核心引擎（ChromaDB + BM25 + RRF + Rerank）
│   ├── retrieval.py       RRF / 分块 / Reranker
│   ├── lightrag_engine.py LightRAG 知识图谱
│   ├── search.py          SearchRouter
│   ├── api.py             FastAPI + MCP
│   └── mcp_handler.py     MCP 工具
└── scripts/
    ├── backfill_content.py   旧库回填 metadata.content
    ├── reindex_chunks.py     长文档重新分块
    ├── backfill_lightrag.py  图谱回填
    └── eval_recall.py        召回评测
```

## 快速开始

```bash
uv sync
cp config.example.yaml config.yaml
# 编辑 config.yaml（端口 8766、DeepSeek API Key、LightRAG 等）

./run.sh start
./run.sh test
```

## 升级已有库（无需重建路径脑）

```bash
# 1. 回填全文到 metadata（BM25 可搜正文）
python scripts/backfill_content.py

# 2. 可选：长文档重新分块
python scripts/reindex_chunks.py

# 3. 若换了嵌入模型
uv run python -c "from server.engine import get_engine; get_engine().reembed()"

# 4. LightRAG 图谱需单独重建
rm -rf .lightrag_storage && python scripts/backfill_lightrag.py
```

## 配置要点

三处嵌入模型建议保持一致：

```yaml
engine.embed_model: "BAAI/bge-small-zh-v1.5"
memory.embed_model: "BAAI/bge-small-zh-v1.5"
lightrag.embed_model: "BAAI/bge-small-zh-v1.5"
```

检索增强见 `config.example.yaml` 的 `retrieval` 段。

## MCP 工具

| 工具 | 功能 |
|------|------|
| kb_search / kb_agentic_search | 混合检索 |
| kb_add / kb_delete | 写入 / 删除 |
| mb_check / mb_remember / mb_avoid / mb_prune | 路径脑（独立库，不受分块影响） |
