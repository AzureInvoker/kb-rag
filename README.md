# kb-rag

通用知识库 —— 支持多种文档类型的向量 + BM25 混合搜索系统，集成 LightRAG 知识图谱增强。

## 架构

```
kb-rag/
├── config.py              配置加载
├── config.yaml            配置文件（gitignore 中，不提交）
├── run.sh                 启动/停止
├── server/
│   ├── models.py          数据模型
│   ├── engine.py          核心引擎（ChromaDB + BM25）
│   ├── config.py          → 项目根 config.py
│   ├── lightrag_engine.py LightRAG 知识图谱封装
│   ├── search.py          SearchRouter 统一检索入口
│   ├── api.py             FastAPI + REST + MCP SSE
│   ├── mcp_handler.py     MCP 工具定义（api 和 stdio 共享）
│   └── static/index.html  前端页面
├── mcp/
│   └── server.py          stdio MCP 服务器
└── scripts/
    └── migrate_from_tc.py 从 testcase-rag 迁移数据
```

## 快速开始

```bash
# 安装依赖
uv pip install -r requirements.txt

# 配置（复制并编辑）
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入 DeepSeek API Key

# 启动
./run.sh start
```

## 文档类型

通过 `doc_type` 字段区分不同知识类型，例如：
- `test_case` — 测试用例（metadata 含 module/priority/preconditions 等）
- `doc` — 普通文档
- `faq` — 常见问题
- `wiki` — 维基条目

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/v1/health | 健康检查 |
| GET | /api/v1/search | 语义搜索 |
| GET | /api/v1/items | 列表/筛选 |
| GET | /api/v1/items/{id} | 详情 |
| POST | /api/v1/items | 添加 |
| POST | /api/v1/items/batch | 批量添加 |
| DELETE | /api/v1/items/{id} | 删除 |
| GET | /api/v1/stats | 统计 |
| GET | /api/v1/graph/status | 图谱状态 |
| GET | /api/v1/graph/search | 图谱搜索 |

## MCP 工具

| 工具名 | 功能 |
|--------|------|
| kb_search | 基础语义搜索 |
| kb_list | 列表浏览 |
| kb_get | 获取详情 |
| kb_add | 添加条目 |
| kb_add_batch | 批量添加 |
| kb_delete | 删除条目 |
| kb_stats | 知识库统计 |
| kb_graph_search | 知识图谱搜索 |
| kb_agentic_search | 自适应检索(向量+图谱) |
| kb_graph_status | 图谱状态诊断 |
