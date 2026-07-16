"""
LightRAG 知识图谱引擎 — DeepSeek/Ollama 双模式封装

支持两种 LLM 后端：
  deepseek: 通过 OpenAI 兼容 API 调用（推荐，建图成本极低）
  ollama:   调用本地 Ollama 实例（内网部署用）

嵌入统一使用 sentence-transformers（CPU 即可）。
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
from collections.abc import AsyncIterator

logger = logging.getLogger("lightrag_engine")


def _build_llm_func(provider: str, api_key: str, base_url: str, model: str):
    """
    根据 provider 返回一个兼容 LightRAG 的 llm_model_func。

    LightRAG 期望的函数签名：
      async def func(
          prompt: str,
          system_prompt: str | None = None,
          history_messages: list[dict] | None = None,
          **kwargs,
      ) -> str | AsyncIterator[str]:
    """

    if provider == "deepseek":
        import openai

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )

        async def deepseek_llm(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            # 从 kwargs 中提取模型名（LightRAG 会传）
            model_name = model
            if "hashing_kv" in kwargs:
                try:
                    model_name = kwargs["hashing_kv"].global_config.get("llm_model_name", model)
                except Exception:
                    pass

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2048),
                stream=False,
            )
            return resp.choices[0].message.content

        return deepseek_llm

    elif provider == "ollama":
        import httpx

        async def ollama_llm(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            model_name = model
            if "hashing_kv" in kwargs:
                try:
                    model_name = kwargs["hashing_kv"].global_config.get("llm_model_name", model)
                except Exception:
                    pass

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": kwargs.get("temperature", 0.1)},
            }
            async with httpx.AsyncClient(timeout=300) as http:
                resp = await http.post(f"{base_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]

        return ollama_llm

    else:
        raise ValueError(f"不支持的 LLM provider: {provider}，仅支持 deepseek/ollama")


def _build_embed_func(model_name: str):
    """返回一个兼容 LightRAG 的 embedding_func

    需要用 @wrap_embedding_func_with_attrs 装饰，LightRAG 内部会访问 .func 属性。
    """
    from lightrag.utils import wrap_embedding_func_with_attrs
    from sentence_transformers import SentenceTransformer

    # 全局缓存模型，避免重复加载
    if not hasattr(_build_embed_func, "_model"):
        logger.info(f"加载嵌入模型: {model_name}")
        _build_embed_func._model = SentenceTransformer(model_name, device="cpu")
    model = _build_embed_func._model

    @wrap_embedding_func_with_attrs(
        embedding_dim=model.get_embedding_dimension(),
        max_token_size=512,
        model_name=model_name,
    )
    async def embed_func(texts: list[str]) -> "np.ndarray":
        import numpy as np
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    return embed_func


class LightRAGEngine:
    """LightRAG 知识图谱引擎封装"""

    def __init__(self, config):
        self.cfg = config
        self._rag = None
        self._ready = False
        self._error = None
        self._storages_initialized = False

    def _lazy_init(self):
        if self._rag is not None:
            return
        if not self.cfg.lightrag_enabled:
            self._ready = False
            self._error = "LightRAG 未启用（config.lightrag.enabled = false）"
            return

        try:
            from lightrag import LightRAG, QueryParam

            working_dir = self.cfg.lightrag_working_dir
            if not os.path.isabs(working_dir):
                working_dir = str(Path(__file__).parent.parent / working_dir)
            os.makedirs(working_dir, exist_ok=True)

            # LLM 函数
            llm_func = _build_llm_func(
                provider=self.cfg.llm_provider,
                api_key=self._resolve_api_key(),
                base_url=self.cfg.ollama_base_url,
                model=self.cfg.llm_model,
            )

            # 嵌入函数
            embed_func = _build_embed_func(self.cfg.lightrag_embed_model)

            self._rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                llm_model_name=self.cfg.llm_model,
                embedding_func=embed_func,
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                top_k=self.cfg.lightrag_top_k,
                max_parallel_insert=2,
            )

            self._QueryParam = QueryParam
            self._ready = True
            logger.info(f"LightRAG 初始化成功 (provider={self.cfg.llm_provider}, model={self.cfg.llm_model})")

        except Exception as e:
            self._ready = False
            self._error = str(e)
            logger.error(f"LightRAG 初始化失败: {e}")

    async def _init_storages_async(self):
        """异步初始化 LightRAG 存储（懒加载，首次 async_search 时自动调）"""
        if self._storages_initialized:
            return True
        if not self._ready or self._rag is None:
            return False
        try:
            await self._rag.initialize_storages()
            self._storages_initialized = True
            logger.info("LightRAG 存储初始化完成")
            return True
        except Exception as e:
            self._ready = False
            self._error = str(e)
            logger.error(f"LightRAG 存储初始化失败: {e}")
            return False

    async def _ensure_storages_async(self):
        """确保存储已初始化，async_search 入口调用"""
        if not self._storages_initialized:
            await self._init_storages_async()

    def _resolve_api_key(self) -> str:
        key = self.cfg.deepseek_api_key
        if key.startswith("${") and key.endswith("}"):
            env_name = key[2:-1]
            key = os.getenv(env_name, "")
        return key

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> Optional[str]:
        return self._error

    def is_available(self) -> bool:
        if not self.cfg.lightrag_enabled:
            return False
        self._lazy_init()
        return self._ready

    async def async_insert(self, texts: list[str], ids: list[str] = None) -> dict:
        """异步插入（从 async MCP handler 中直接 await）"""
        if not self.is_available():
            return {"ok": False, "message": self._error or "LightRAG 不可用"}
        await self._ensure_storages_async()
        if not self._ready:
            return {"ok": False, "message": self._error or "存储初始化失败"}
        try:
            track_id = await self._rag.ainsert(texts, ids=ids)
            return {"ok": True, "message": f"成功插入 {len(texts)} 条", "track_id": track_id}
        except Exception as e:
            logger.error(f"LightRAG async_insert 失败: {e}")
            return {"ok": False, "message": str(e)}

    def insert(self, texts: list[str], ids: list[str] = None) -> dict:
        if not self.is_available():
            return {"ok": False, "message": self._error or "LightRAG 不可用"}
        try:
            import asyncio

            async def _do_insert():
                return await self._rag.insert(texts, ids=ids)

            track_id = asyncio.run(_do_insert())
            return {"ok": True, "message": f"成功插入 {len(texts)} 条", "track_id": track_id}
        except Exception as e:
            logger.error(f"LightRAG insert 失败: {e}")
            return {"ok": False, "message": str(e)}

    def search(self, query: str, n_results: int = 5) -> dict:
        """同步搜索（从非异步上下文调用：迁移脚本、CLI 等）"""
        if not self.is_available():
            return {"ok": False, "message": self._error or "LightRAG 不可用", "entities": [], "relationships": [], "chunks": []}
        try:
            import asyncio
            return asyncio.run(self.async_search(query, n_results))
        except Exception as e:
            logger.error(f"LightRAG search 失败: {e}")
            return {"ok": False, "message": str(e), "entities": [], "relationships": [], "chunks": []}

    async def async_search(self, query: str, n_results: int = 5) -> dict:
        """异步搜索（从 async MCP handler 中直接 await）"""
        if not self.is_available():
            return {"ok": False, "message": self._error or "LightRAG 不可用", "entities": [], "relationships": [], "chunks": []}
        await self._ensure_storages_async()
        if not self._ready:
            return {"ok": False, "message": self._error or "存储初始化失败", "entities": [], "relationships": [], "chunks": []}
        try:
            param = self._QueryParam(
                mode=self.cfg.lightrag_mode,
                top_k=n_results * 2,
                chunk_top_k=n_results,
                only_need_context=True,
            )
            result = await self._rag.aquery_data(query, param=param)

            if result.get("status") != "success":
                return {"ok": False, "message": result.get("message", "未知错误"), "entities": [], "relationships": [], "chunks": []}

            data = result.get("data", {})
            entities = []
            for e in data.get("entities", []):
                entities.append({
                    "name": e.get("entity_name", ""),
                    "type": e.get("entity_type", ""),
                    "description": e.get("description", ""),
                })
            relationships = []
            for r in data.get("relationships", []):
                relationships.append({
                    "source": r.get("src_id", ""),
                    "target": r.get("tgt_id", ""),
                    "description": r.get("description", ""),
                    "weight": r.get("weight", 0),
                })
            chunks = []
            for c in data.get("chunks", []):
                chunks.append({
                    "content": c.get("content", "")[:500],
                    "doc_id": c.get("doc_id", ""),
                })

            return {
                "ok": True,
                "message": f"找到 {len(entities)} 个实体, {len(relationships)} 条关系, {len(chunks)} 个片段",
                "entities": entities[:n_results * 2],
                "relationships": relationships[:n_results],
                "chunks": chunks[:n_results],
            }

        except Exception as e:
            logger.error(f"LightRAG async_search 失败: {e}")
            return {"ok": False, "message": str(e), "entities": [], "relationships": [], "chunks": []}

    def get_status(self) -> dict:
        if not self.cfg.lightrag_enabled:
            return {"enabled": False, "ready": False, "message": "未启用"}
        self._lazy_init()
        if not self._ready:
            return {"enabled": True, "ready": False, "message": self._error or "初始化失败"}
        try:
            import asyncio

            async def _fetch_status():
                try:
                    status = await self._rag.get_processing_status()
                    graph = await self._rag.get_knowledge_graph("")
                    return status, graph
                except Exception as e:
                    return None, {"error": str(e)}

            # 如果有运行中的事件循环，用它跑；否则创建新循环
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(_fetch_status(), loop)
                    status_info, graph = future.result(timeout=15)
                else:
                    status_info, graph = asyncio.run(_fetch_status())
            except RuntimeError:
                # 没有运行中的事件循环
                status_info, graph = asyncio.run(_fetch_status())

            node_count = 0
            if graph and isinstance(graph, dict):
                try:
                    node_count = len(graph.get("nodes", []))
                except Exception:
                    pass
            return {
                "enabled": True,
                "ready": True,
                "provider": self.cfg.llm_provider,
                "model": self.cfg.llm_model,
                "node_count": node_count,
                "processing_status": status_info,
            }
        except Exception as e:
            # 降级：至少返回启用/就绪信息
            return {"enabled": True, "ready": True, "message": str(e)}

    async def async_get_graph_data(self) -> dict:
        """异步获取图谱完整数据（节点+关系），供前端可视化"""
        if not self.is_available():
            return {"ok": False, "message": self._error or "LightRAG 不可用", "nodes": [], "edges": []}
        await self._ensure_storages_async()
        if not self._ready:
            return {"ok": False, "message": self._error or "存储初始化失败", "nodes": [], "edges": []}
        try:
            graph = await self._rag.get_knowledge_graph("")
            nodes = []
            edges = []
            if graph and isinstance(graph, dict):
                for n in graph.get("nodes", []):
                    nodes.append({
                        "id": n.get("id", ""),
                        "name": n.get("name", n.get("id", "")),
                        "type": n.get("type", n.get("entity_type", "entity")),
                        "description": n.get("description", "")[:200],
                    })
                for e in graph.get("edges", []):
                    edges.append({
                        "source": e.get("source", e.get("src_id", "")),
                        "target": e.get("target", e.get("tgt_id", "")),
                        "label": e.get("label", e.get("description", "")),
                        "weight": e.get("weight", 1),
                    })
            return {"ok": True, "nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"get_graph_data 失败: {e}")
            return {"ok": False, "message": str(e), "nodes": [], "edges": []}
