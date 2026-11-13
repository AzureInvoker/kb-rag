"""
通用知识库 — REST API + MCP SSE
"""

import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import get_engine, VectorEngine
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
    mem_engine = VectorEngine(
        chroma_dir=cfg.memory_chroma_dir,
        embed_model=cfg.memory_embed_model,
        collection_name=cfg.memory_collection_name,
        enable_chunking=False,
        enable_rerank=False,
    )
    lightrag = LightRAGEngine(cfg)
    search_router = SearchRouter(engine, lightrag)

    app.state.engine = engine
    app.state.mem_engine = mem_engine
    app.state.cfg = cfg
    app.state.lightrag = lightrag
    app.state.search_router = search_router

    # 后台预热
    async def _warmup():
        try:
            logger.info("后台预热引擎中...")
            import asyncio, time
            start = time.time()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: engine.collection)
            await loop.run_in_executor(None, lambda: engine.embedder)
            await loop.run_in_executor(None, lambda: mem_engine.collection)
            await loop.run_in_executor(None, lambda: mem_engine.embedder)
            elapsed = time.time() - start
            logger.info(f"引擎预热完成 (model={cfg.embed_model}, mem={cfg.memory_embed_model}, 耗时 {elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"引擎预热失败（将延迟到首次请求加载）: {e}")
    asyncio.create_task(_warmup())

    logger.info(f"kb-rag 启动完成 (port={cfg.api_port}, lightrag={cfg.lightrag_enabled})")
    yield


# ── FastAPI 应用 ──

app = FastAPI(
    title="通用知识库 API",
    description="KB-RAG — 多类型知识条目存储 + 语义检索 + 知识图谱 + Agentic MCP",
    version="1.1.0",
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
    game: Optional[str] = Field(default=None)
    app: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    include_retired: bool = Field(default=False)


class ItemCreate(BaseModel):
    title: str = Field(..., description="标题")
    doc_type: str = Field(default="doc", description="文档类型: doc|test_case|rule|ui_control")
    content: str = Field(default="", description="正文内容")
    metadata: dict = Field(default_factory=dict, description="类型专属字段")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    category: str = Field(default="", description="业务域")
    game: str = Field(default="", description="游戏")
    app: str = Field(default="", description="应用（ui_control 必填）")
    governance_status: str = Field(default="published")


class ItemUpdate(BaseModel):
    title: Optional[str] = None
    doc_type: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[list[str]] = None
    category: Optional[str] = None
    game: Optional[str] = None
    app: Optional[str] = None
    governance_status: Optional[str] = None


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
        "version": "1.1.0",
        "timestamp": int(time.time()),
    }


@app.get("/api/v1/search")
async def search(
    query: str = Query(..., description="搜索关键词"),
    n_results: int = Query(default=10, le=20),
    doc_type: str = Query(default=None, description="文档类型筛选"),
    game: Optional[str] = Query(default=None),
    # 参数名不能叫 app，否则会遮蔽 FastAPI 的 app 对象
    app_name: Optional[str] = Query(default=None, alias="app", description="应用过滤"),
    category: Optional[str] = Query(default=None),
    include_retired: bool = Query(default=False),
):
    """语义搜索"""
    results = await asyncio.to_thread(
        app.state.engine.search,
        query,
        n_results,
        doc_type,
        game=game,
        app=app_name,
        category=category,
        include_retired=include_retired,
    )
    # 默认不返回 brain_memory（只有主动筛选才看记忆）
    if not doc_type:
        results = [r for r in results if r.get("doc_type") != "brain_memory"]
    return {"query": query, "total": len(results), "results": results}


@app.get("/api/v1/items")
async def list_items(
    doc_type: str = Query(default=None, description="按文档类型筛选"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
    game: Optional[str] = Query(default=None),
    # 参数名不能叫 app，否则会遮蔽 FastAPI 的 app 对象
    app_name: Optional[str] = Query(default=None, alias="app", description="应用过滤"),
    category: Optional[str] = Query(default=None),
    governance_status: Optional[str] = Query(default=None),
    include_retired: bool = Query(default=True),
):
    """列表/筛选"""
    try:
        items, total = await asyncio.to_thread(
            app.state.engine.list_parents,
            doc_type,
            offset,
            limit,
            game=game,
            app=app_name,
            category=category,
            governance_status=governance_status,
            include_retired=include_retired,
        )
        return {"total": total, "returned": len(items), "offset": offset, "limit": limit, "items": items}
    except Exception as exc:
        logger.exception("list_items failed")
        raise HTTPException(status_code=503, detail=f"知识库未就绪: {exc}") from exc


@app.get("/api/v1/items/{item_id}")
async def get_item(item_id: str):
    """获取详情"""
    item = await asyncio.to_thread(app.state.engine.get_by_id, item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return item


@app.post("/api/v1/items", status_code=201)
def add_item(data: ItemCreate):
    """添加单条"""
    from .taxonomy import should_index_graph
    try:
        item = KnowledgeItem_from_pydantic(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not item.id:
        item.id = item.gen_id()
    try:
        app.state.engine.add(item)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if should_index_graph(item.doc_type) and app.state.lightrag.is_available():
        app.state.lightrag.insert([item.get_lightrag_text()], ids=[item.id])
    return {"id": item.id, "title": item.title, "doc_type": item.doc_type, "app": item.app}


@app.patch("/api/v1/items/{item_id}")
def patch_item(item_id: str, data: ItemUpdate):
    """更新条目"""
    payload = {k: v for k, v in data.model_dump().items() if v is not None}
    try:
        updated = app.state.engine.update(item_id, **payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return updated


@app.post("/api/v1/items/{item_id}/deprecate")
def deprecate_item(item_id: str):
    updated = app.state.engine.set_status(item_id, "deprecated")
    if not updated:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return updated


@app.post("/api/v1/items/{item_id}/reverify")
def reverify_item(item_id: str):
    updated = app.state.engine.set_status(item_id, "published", refresh_verify=True)
    if not updated:
        raise HTTPException(status_code=404, detail=f"条目 {item_id} 不存在")
    return updated


@app.post("/api/v1/items/batch", status_code=201)
def add_items_batch(data: ItemBatchCreate):
    """批量添加"""
    from .taxonomy import should_index_graph
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
    graph_items = [it for it in added if should_index_graph(it.doc_type)]
    if graph_items and app.state.lightrag.is_available():
        texts = [it.get_lightrag_text() for it in graph_items]
        ids = [it.id for it in graph_items]
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
    try:
        return app.state.engine.get_stats()
    except Exception as exc:
        logger.exception("stats failed")
        raise HTTPException(status_code=503, detail=f"知识库未就绪: {exc}") from exc


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
async def graph_data(
    query: Optional[str] = Query(default=None, description="实体/关键词；留空或 * 为有限节点概览，建议按关键词拉子图"),
    max_nodes: Optional[int] = Query(default=150, ge=1, le=2000, description="最大节点数（默认150，避免全量）"),
):
    """按关键词获取图谱子图（节点+关系），供前端可视化。默认不全量加载。"""
    if not app.state.lightrag.is_available():
        raise HTTPException(status_code=503, detail="LightRAG 图谱未启用或初始化失败")
    result = await app.state.lightrag.async_get_graph_data(max_nodes=max_nodes, query=query)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("message", "获取图谱数据失败"))
    return result


# ── 脑记忆 REST 端点（前端 💡 记忆 标签页用） ──


@app.get("/api/v1/memory/items")
def memory_items(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
    sort_by: str = Query(default="time", description="排序：time=添加/更新时间倒序（默认）；tries=尝试次数倒序"),
):
    """脑记忆列表/分页（默认按时间倒序）"""
    mem = app.state.mem_engine
    all_mem = mem.collection.get(where={"doc_type": "brain_memory"})
    items = _fmt_memory_items(all_mem)
    sort_key = (sort_by or "time").strip().lower()
    if sort_key == "tries":
        items.sort(key=lambda x: int((x.get("metadata") or {}).get("tries") or 0), reverse=True)
    else:
        # 时间倒序：优先 updated_at，其次 created_at
        def _time_key(x: dict) -> str:
            meta = x.get("metadata") or {}
            return meta.get("updated_at") or x.get("created_at") or ""

        items.sort(key=_time_key, reverse=True)
    total = len(items)
    page = items[offset : offset + limit]
    return {
        "total": total,
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "sort_by": "tries" if sort_key == "tries" else "time",
        "items": page,
    }


@app.get("/api/v1/memory/search")
def memory_search(
    query: str = Query(..., description="搜索关键词"),
    n_results: int = Query(default=10, le=30),
):
    """脑记忆搜索"""
    mem = app.state.mem_engine
    raw = mem.search(query=query, n_results=n_results, doc_type="brain_memory")
    # 用完整 ChromaDB 元数据重新格式化
    items = []
    for r in raw:
        full_meta = mem.collection.get(ids=[r["id"]], include=["metadatas", "documents"])
        if full_meta and full_meta["ids"]:
            m = full_meta["metadatas"][0] if full_meta["metadatas"] else {}
            doc = full_meta["documents"][0] if full_meta["documents"] else ""
            items.append(_fmt_memory_item(m, doc, r["id"]))
    return {"query": query, "total": len(items), "results": items}


def _fmt_memory_items(results: dict) -> list[dict]:
    """将 ChromaDB get 结果格式化为前端脑记忆格式"""
    items = []
    if not results.get("ids"):
        return items
    for i, eid in enumerate(results["ids"]):
        m = results["metadatas"][i] if results["metadatas"] else {}
        doc = results["documents"][i] if results["documents"] else ""
        items.append(_fmt_memory_item(m, doc, eid))
    return items


def _fmt_memory_item(meta: dict, doc: str = "", id_: str = "") -> dict:
    """格式化单条脑记忆"""
    # 解析 brain 元数据，兼容嵌套(engine.add)和扁平(collection.update)格式
    brain = VectorEngine._extract_brain_meta(meta)
    return {
        "id": id_ or meta.get("id", ""),
        "title": meta.get("title", ""),
        "content": doc,
        "metadata": {
            "target": brain.get("target", ""),
            "method": brain.get("method", ""),
            "source": brain.get("source", ""),
            "params": brain.get("params", ""),
            "pleasure": brain.get("pleasure", 0),
            "last_pleasure": brain.get("last_pleasure", brain.get("pleasure", 0)),
            "avg_pleasure": brain.get("avg_pleasure", 0),
            "min_pleasure": brain.get("min_pleasure", 0),
            "max_pleasure": brain.get("max_pleasure", 0),
            "tries": brain.get("tries", 1),
            "success_count": brain.get("success_count", 0),
            "fail_count": brain.get("fail_count", 0),
            "reliability": brain.get("reliability", 0),
            "updated_at": brain.get("updated_at", ""),
        },
        "doc_type": "brain_memory",
        "created_at": meta.get("created_at") or brain.get("updated_at") or meta.get("updated_at", ""),
    }


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
        "version": "1.1.0",
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
                "serverInfo": {"name": "kb-rag-mcp", "version": "1.1.0"},
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
            result = await async_handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag, app.state.mem_engine)
        else:
            result = await asyncio.to_thread(handle_tool, tool_name, tool_args, app.state.engine, app.state.lightrag, app.state.mem_engine)

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
                "serverInfo": {"name": "kb-rag-mcp", "version": "1.1.0"},
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
            result = await async_handle_tool(tool_name, tool_args, app.state.engine, app.state.lightrag, app.state.mem_engine)
        else:
            result = await asyncio.to_thread(handle_tool, tool_name, tool_args, app.state.engine, app.state.lightrag, app.state.mem_engine)

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
    from .taxonomy import compute_next_verify_at, require_ui_control_app

    meta = dict(data.metadata or {})
    app = getattr(data, "app", "") or meta.get("app", "")
    game = getattr(data, "game", "") or meta.get("game", "")
    category = getattr(data, "category", "") or meta.get("category", "")
    doc_type = data.doc_type or "doc"
    app = require_ui_control_app(doc_type, app, meta)
    item = KnowledgeItem(
        title=_clean_text(data.title),
        doc_type=doc_type,
        content=_clean_text(data.content or ""),
        metadata=meta,
        tags=_clean_list(data.tags or []),
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        category=_clean_text(category or ""),
        game=_clean_text(game or ""),
        app=_clean_text(app or ""),
        governance_status=getattr(data, "governance_status", None) or "published",
    )
    item.next_verify_at = compute_next_verify_at(item.doc_type, app.state.cfg.freshness_sla_days)
    return item
