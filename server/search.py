"""
SearchRouter — 统一检索入口

三种模式：
  chroma: 只走 ChromaDB + BM25（现有方式）
  graph:  只走 LightRAG 知识图谱
  auto:   先走向量，再用图谱增强（如果两者都可用）

同步/异步双入口：
  search()       → 同步（CLI/脚本用）
  async_search() → 异步（MCP/FastAPI handler 用）
"""

import logging
from typing import Optional

logger = logging.getLogger("search_router")


class SearchRouter:
    """融合 ChromaDB 和 LightRAG 的检索路由"""

    def __init__(self, vector_engine, lightrag_engine):
        self.vec = vector_engine
        self.lr = lightrag_engine

    # ── 同步入口（CLI/脚本用） ──

    def search(self, query: str, n_results: int = 5,
               doc_type: str = None, mode: str = "auto") -> dict:
        """同步检索入口（CLI/脚本用）"""
        if mode == "chroma":
            return self._chroma_only(query, n_results, doc_type)
        if mode == "graph":
            return self._graph_only(query, n_results)

        chroma_results = self.vec.search(query, n_results, doc_type=doc_type)
        graph_result = self.lr.search(query, n_results) if self.lr.is_available() else {"ok": False}
        return self._merge_results(chroma_results, graph_result)

    # ── 异步入口（MCP/FastAPI handler 用） ──

    async def async_search(self, query: str, n_results: int = 5,
                           doc_type: str = None, mode: str = "auto") -> dict:
        """异步检索入口（MCP/FastAPI handler 用）"""
        if mode == "chroma":
            return self._chroma_only(query, n_results, doc_type)
        if mode == "graph":
            return await self._async_graph_only(query, n_results)

        chroma_results = self.vec.search(query, n_results, doc_type=doc_type)
        graph_result = await self.lr.async_search(query, n_results) if self.lr.is_available() else {"ok": False}
        return self._merge_results(chroma_results, graph_result)

    # ── 私有方法 ──

    def _merge_results(self, chroma_results: list, graph_result: dict) -> dict:
        """合并向量结果和图谱结果"""
        if not graph_result.get("ok"):
            return {
                "mode": "chroma",
                "results": chroma_results,
                "graph_hits": None,
                "total": len(chroma_results),
            }
        return {
            "mode": "auto",
            "results": chroma_results,
            "graph_hits": {
                "entities": graph_result.get("entities", []),
                "relationships": graph_result.get("relationships", []),
                "chunks": graph_result.get("chunks", []),
            },
            "total": len(chroma_results),
        }

    def _chroma_only(self, query, n_results, doc_type=None):
        results = self.vec.search(query, n_results, doc_type=doc_type)
        return {
            "mode": "chroma",
            "results": results,
            "graph_hits": None,
            "total": len(results),
        }

    def _graph_only(self, query, n_results):
        graph_result = self.lr.search(query, n_results)
        return self._graph_result(mode="graph", graph_result=graph_result)

    async def _async_graph_only(self, query, n_results):
        graph_result = await self.lr.async_search(query, n_results)
        return self._graph_result(mode="graph", graph_result=graph_result)

    def _graph_result(self, mode: str, graph_result: dict) -> dict:
        if not graph_result.get("ok"):
            return {
                "mode": mode,
                "results": [],
                "graph_hits": None,
                "total": 0,
                "error": graph_result.get("message", "图谱检索失败"),
            }
        return {
            "mode": mode,
            "results": [],
            "graph_hits": {
                "entities": graph_result.get("entities", []),
                "relationships": graph_result.get("relationships", []),
                "chunks": graph_result.get("chunks", []),
            },
            "total": 0,
        }
