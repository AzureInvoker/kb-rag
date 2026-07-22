"""
通用知识库 — REST API + MCP SSE
"""

import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import get_engine
try:
    from .config import get_config
except ImportError:
    from config import get_config
from .lightrag_engine import LightRAGEngine
from .search import SearchRouter
from .mcp_handler import TOOLS, handle_tool, async_handle_tool, _clean_text, _clean_list, _make_item

logger = logging.getLogger("api")


# ── 依赖注入（用 FastAPI lifespan 管理全局状态） ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    cfg = get_config()
    lightrag = LightRAGEngine(cfg)
    search_router = SearchRouter(engine, lightrag)

    app.state.engine = engine
    app.state.cfg = cfg
    app.state.lightrag = lightrag
    app.state.search_router = search_router

    logger.info(f"kb-rag 启动完成 (port={cfg.api_port}, lightrag={cfg.lightrag_enabled})")
    yield


# ── FastAPI 应用 ──

app = FastAPI(
    title="通用知识库 API",
    description="KB-RAG — 多类型知识条目存储 + 语义检索 + 知识图谱 + Agentic MCP",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic 模型 ──


class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    n_results: int = Field(default=10, le=20, description="返回结果数量")
    doc_type: str = Field(default=None, description="按文档类型筛选")


class ItemCreate(BaseModel):
    title: str = Field(..., description="标题")
    doc_type: str = Field(default="doc", description="文档类型")
    content: str = Field(default="", description="正文内容")
    metadata: dict = Field(default={}, description="类型专属字段")
    tags: list[str] = Field(default=[], description="标签列表")


class ItemBatchCreate(BaseModel):
    items: list[ItemCreate]


# ── REST 端点 ──


@app.get("/api/v1/health")
def health():
    """健康检查"""
    import time
    return {
        "status": "ok",
        "name": "kb-rag",
        "version": "1.0.0",
        "timestamp": int(time.time()),
    }


@app.get("/api/v1/search")
def search(
    query: str = Query(..., description="搜索关键词"),
    n_results: int = Query(default=10, le=20),
    doc_type: str = Query(default=None, description="文档类型筛选"),
):
    """语义搜索"""
    results = app.state.engine.search(
        query=query,
        n_results=n_results,
        doc_type=doc_type,
    )
    # 默认不返回 brain_memory（只有主动筛选才看记忆）
    if not doc_type:
        results = [r for r in results if r.get("doc_type") != "brain_memory"]
    return {"query": query, "total": len(results), "results": results}


@app.get("/api/v1/items")
def list_items(
    doc_type: str = Query(default=None, description="按文档类型筛选"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
):
    """列表/筛选"""
    items = app.state.engine.get_all(doc_type=doc_type, offset=offset, limit=limit)
    total = app.state.engine.count_by_type(doc_type)
    return {"total": total, "returned": len(items), "items": items}


@app.get("/api/v1/items/{item_id}")
def get_item(item_id: str):
    """获取详情"""
    item = app.state.engine.get_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return item


@app.post("/api/v1/items", status_code=201)
def add_item(data: ItemCreate):
    """添加单条"""
    item = KnowledgeItem_from_pydantic(data)
    if not item.id:
        item.id = item.gen_id()
    app.state.engine.add(item)
    if app.state.lightrag.is_available():
        app.state.lightrag.insert([item.get_embedding_text()], ids=[item.id])
    return {"id": item.id, "title": item.title, "doc_type": item.doc_type}


@app.post("/api/v1/items/batch", status_code=201)
def add_items_batch(data: ItemBatchCreate):
    """批量添加"""
    added = []
    errors = []
    for i, d in enumerate(data.items):
        try:
            item = KnowledgeItem_from_pydantic(d)
            if not item.id:
                item.id = item.gen_id()
            app.state.engine.add(item)
            added.append(item)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    if added and app.state.lightrag.is_available():
        texts = [it.get_embedding_text() for it in added]
        ids = [it.id for it in added]
        app.state.lightrag.insert(texts, ids=ids)
    return {
        "added": len(added),
        "errors": len(errors),
        "error_details": errors if errors else None,
        "ids": [it.id for it in added],
    }


@app.delete("/api/v1/items/{item_id}")
def delete_item(item_id: str):
    """删除单条"""
    ok = app.state.engine.delete(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return {"deleted": item_id}


@app.delete("/api/v1/items")
def delete_items_bulk(doc_type: str = Query(..., description="要删除的文档类型")):
    """按 doc_type 批量删除"""
    count = app.state.engine.delete_many(doc_type=doc_type)
    return {"deleted_count": count, "doc_type": doc_type}


@app.get("/api/v1/stats")
def stats():
    """统计信息"""
    return app.state.engine.get_stats()


@app.get("/api/v1/graph/status")
def graph_status():
    """图谱状态"""
    if not app.state.cfg.lightrag_enabled:
        return {"enabled": False, "ready": False, "message": "LightRAG 未启用"}
    return app.state.lightrag.get_status()


@app.get("/api/v1/graph/search")
async def graph_search(
    query: str = Query(..., description="搜索关键词"),
    n_results: int = Query(default=5, le=20),
):
    """知识图谱搜索"""
    if not app.state.lightrag.is_available():
        raise HTTPException(status_code=503, detail="LightRAG 图谱未启用或初始化失败")
    result = await app.state.lightrag.async_search(query, n_results=n_results)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("message", "图谱检索失败"))
    return result


@app.get("/api/v1/graph/data")
async def graph_data():
    """获取图谱完整数据（节点+关系），供前端可视化"""
    if not app.state.lightrag.is_available():
        raise HTTPException(status_code=503, detail="LightRAG 图谱未启用或初始化失败")
    result = await app.state.lightrag.async_get_graph_data()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("message", "获取图谱数据失败"))
    return result


# ── MCP 端点 ──

_mcp_sessions: dict[str, asyncio.Queue] = {}
_next_session_id = 0


@app.get("/mcp/sse")
async def mcp_sse(request: Request):
    """MCP Server-Sent Events 端点"""
    global _next_session_id
    session_id = f"session-{_next_session_id}"
    _next_session_id += 1
    queue: asyncio.Queue = asyncio.Queue()
    _mcp_sessions[session_id] = queue

    async def event_generator():
        try:
            # 发送 endpoint 信息
            base = str(request.base_url).rstrip("/")
            yield f"event: endpoint\ndata: {base}/mcp/message?session_id={session_id}\n\n"
            # 发 initialized 通知
            init_msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })
            yield f"event: message\ndata: {init_msg}\n\n"
            while True:
                try:
                    response = await asyncio.wait_for(queue.get(), timeout=300)
                    yield f"event: message\ndata: {response}\n\n"
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _mcp_sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/mcp/info")
def mcp_info():
    """MCP 服务信息"""
    cfg = app.state.cfg
    return {
        "name": "kb-rag-mcp",
        "version": "1.0.0",
        "transport": "HTTP + SSE",
        "endpoints": {"sse": "/mcp/sse", "message": "/mcp/message", "direct": "POST /mcp"},
        "tools": [t["name"] for t in TOOLS],
    }


class MCPMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: int | None = None
    method: str | None = None
    params: dict = {}


@app.post("/mcp/message")
async def mcp_message(msg: MCPMessage, request: Request, session_id: str = Query("")):
    """MCP 消息端点（SSE 客户端发消息）"""
    msg_id = msg.id
    method = msg.method

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kb-rag-mcp", "version": "1.0.0"},
            },
        }
    if method in ("notifications/initialized",):
        return {"jsonrpc": "2.0"}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/call":
        tool_name = msg.params.get("name", "")
        tool_args = msg.params.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name in ("kb_graph_search", "kb_agentic_search", "kb_graph_status",
                         "kb_add", "kb_add_batch"):
            result = await async_handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag)
        else:
            result = handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag)

        resp = {"jsonrpc": "2.0", "id": msg_id, "result": result}

        # 如果有 SSE session，通过 SSE 推回
        if session_id and session_id in _mcp_sessions:
            await _mcp_sessions[session_id].put(json.dumps(resp))
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"_delivered": "sse"}}

        return resp

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"不支持的方法: {method}"}}


@app.post("/mcp")
async def mcp_direct(msg: MCPMessage):
    """MCP 直连端点（直接 POST JSON-RPC，无需 SSE）"""
    msg_id = msg.id
    method = msg.method

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kb-rag-mcp", "version": "1.0.0"},
            },
        }
    if method in ("notifications/initialized",):
        return {"jsonrpc": "2.0"}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/call":
        tool_name = msg.params.get("name", "")
        tool_args = msg.params.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name in ("kb_graph_search", "kb_agentic_search", "kb_graph_status",
                         "kb_add", "kb_add_batch"):
            result = await async_handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag)
        else:
            result = handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag)

        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"不支持的方法: {method}"}}


# ── 前端 ──

# 挂载静态文件（前端 HTML 在 server/static/ 下）
import os as _os
_static_path = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_path):
    app.mount("/", StaticFiles(directory=_static_path, html=True), name="static")


# ── 辅助函数 ──


def KnowledgeItem_from_pydantic(data) -> "KnowledgeItem":
    """从 Pydantic 模型创建 KnowledgeItem"""
    from .models import KnowledgeItem
    return KnowledgeItem(
        title=_clean_text(data.title),
        doc_type=data.doc_type or "doc",
        content=_clean_text(data.content or ""),
        metadata=data.metadata or {},
        tags=_clean_list(data.tags or []),
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
