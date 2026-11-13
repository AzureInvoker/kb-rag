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
│   ├── taxonomy.py        doc_type / 治理常量
│   ├── engine.py          核心引擎（ChromaDB + BM25 + RRF）
│   ├── retrieval.py       RRF / 分块 / Reranker
│   ├── lightrag_engine.py LightRAG 知识图谱
│   ├── api.py             FastAPI + MCP
│   ├── mcp_handler.py     MCP 工具
│   └── static/            管理前端（含控件专区）
└── scripts/
    ├── backfill_content.py / backfill_taxonomy.py
    ├── mark_stale.py / archive_stale.py
    ├── export_kb.py / import_kb.py
    └── eval_recall.py
```

## 快速开始

```bash
uv sync
cp config.example.yaml config.yaml
# 编辑 config.yaml（端口 8766、DeepSeek API Key、LightRAG 等）

./run.sh start
./run.sh test
```

## 文档类型

| doc_type | 用途 |
|----------|------|
| `doc` | 文档 / 手册 / 问答（含原 wiki、faq） |
| `test_case` | 测试用例 |
| `rule` | 团队规范 |
| `ui_control` | UI 控件（必填 `app`，**不进** LightRAG） |

路径探索经验只用 `mb_*`，不要写成 kb。

## 升级已有库（无需重建路径脑）

```bash
# A 日常：回填全文 + 分类治理 meta（不 reembed）
python scripts/backfill_content.py
python scripts/backfill_taxonomy.py --dry-run
python scripts/backfill_taxonomy.py
python scripts/mark_stale.py --dry-run
# 确认后：python scripts/mark_stale.py --apply

# B 换嵌入模型 / 灾难恢复
python scripts/export_kb.py -o backup/kb_export.jsonl
python scripts/import_kb.py backup/kb_export.jsonl --wipe

# 可选：长文档重新分块 / 图谱重建
python scripts/reindex_chunks.py
rm -rf .lightrag_storage && python scripts/backfill_lightrag.py
```

## 配置要点

三处嵌入模型建议保持一致：

```yaml
engine.embed_model: "BAAI/bge-small-zh-v1.5"
memory.embed_model: "BAAI/bge-small-zh-v1.5"
lightrag.embed_model: "BAAI/bge-small-zh-v1.5"
```

检索见 `retrieval`；新鲜度/降权见 `governance`（`config.example.yaml`）。

## MCP 工具

| 工具 | 功能 |
|------|------|
| kb_search / kb_agentic_search | 混合检索（归一化 score + app/game 过滤） |
| kb_add / kb_add_batch / kb_update | 写入 / 更新 |
| kb_deprecate / kb_reverify | 淘汰 / 重新验证 |
| kb_delete / kb_list / kb_stats | 删除 / 浏览 / 统计 |
| mb_check / mb_remember / mb_avoid / mb_prune | 路径脑（独立库） |
