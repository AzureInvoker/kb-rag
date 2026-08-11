"""
核心引擎 — ChromaDB 向量库 + sentence-transformers 嵌入 + BM25 + RRF + Rerank
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .models import KnowledgeItem
from .retrieval import Reranker, dedupe_by_parent, parent_id_from_meta, reciprocal_rank_fusion, split_content

try:
    from .config import get_config
except ImportError:
    from config import get_config

cfg = get_config()
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = cfg.chroma_dir
EMBED_MODEL = cfg.embed_model
COLLECTION_NAME = cfg.collection_name

logger = logging.getLogger("engine")

_CHROMA_RESERVED = {
    "id", "doc_type", "title", "content", "tags", "metadata_json", "created_at",
    "parent_id", "chunk_index", "is_chunk", "parent_content",
}


class VectorEngine:
    """ChromaDB 引擎，管理知识条目的向量化存储和检索"""

    def __init__(
        self,
        chroma_dir=None,
        embed_model=None,
        collection_name=None,
        *,
        enable_chunking: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
    ):
        self._chroma_dir = Path(chroma_dir) if chroma_dir else CHROMA_DIR
        self._chroma_dir = Path(self._chroma_dir).resolve()
        self._embed_model = embed_model or EMBED_MODEL
        self._collection_name = collection_name or COLLECTION_NAME
        self._collection = None
        self._embedder = None
        self._init_lock = threading.Lock()

        _cfg = get_config()
        self._enable_chunking = _cfg.chunk_enabled if enable_chunking is None else enable_chunking
        self._enable_rerank = _cfg.rerank_enabled if enable_rerank is None else enable_rerank
        self._chunk_min_chars = _cfg.chunk_min_chars
        self._chunk_size = _cfg.chunk_size
        self._chunk_overlap = _cfg.chunk_overlap
        self._rrf_k = _cfg.rrf_k
        self._vec_candidates = _cfg.vec_candidates
        self._bm25_candidates = _cfg.bm25_candidates
        self._rerank_top_n = _cfg.rerank_top_n
        self._reranker = Reranker(_cfg.rerank_model) if self._enable_rerank else None

        self._bm25 = None
        self._bm25_metadata = None
        self._bm25_all_ids = None
        self._bm25_documents = None
        self._bm25_size = 0

    def _invalidate_bm25(self) -> None:
        self._bm25 = None
        self._bm25_all_ids = None
        self._bm25_documents = None
        self._bm25_size = 0

    def _prepare_chroma_dir(self) -> None:
        """确保 ChromaDB 目录存在且可写（清空库后首次访问需要）。"""
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(self._chroma_dir, os.W_OK):
            raise PermissionError(f"ChromaDB 目录不可写: {self._chroma_dir}")

    def _create_chroma_client(self):
        import chromadb

        self._prepare_chroma_dir()
        try:
            return chromadb.PersistentClient(path=str(self._chroma_dir))
        except Exception as exc:
            msg = str(exc).lower()
            if "unable to open database file" not in msg:
                raise
            logger.warning("ChromaDB 打开失败，尝试清理损坏的 sqlite 后重建: %s", self._chroma_dir)
            for name in ("chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"):
                stale = self._chroma_dir / name
                if stale.exists():
                    stale.unlink()
            return chromadb.PersistentClient(path=str(self._chroma_dir))

    def _lazy_init(self):
        if self._collection is not None:
            return
        with self._init_lock:
            if self._collection is not None:
                return
            import chromadb
            from sentence_transformers import SentenceTransformer

            logger.info(
                "加载嵌入模型: %s  |  库: %s / %s (chunk=%s rerank=%s)",
                self._embed_model, self._chroma_dir, self._collection_name,
                self._enable_chunking, self._enable_rerank,
            )
            self._embedder = SentenceTransformer(self._embed_model, device="cpu")

            client = self._create_chroma_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB 就绪")
            self._validate_dimension()

    def _validate_dimension(self):
        try:
            existing = self._collection.get(limit=1, include=["embeddings"])
            if not existing.get("ids"):
                logger.info("  库为空，跳过维度校验")
                return

            model_dim = self._embedder.get_sentence_embedding_dimension()
            stored_dim = len(existing["embeddings"][0])
            if model_dim != stored_dim:
                msg = (
                    f"\n{'='*60}\n"
                    f"❌ 嵌入模型维度不匹配！\n"
                    f"   模型 {self._embed_model}: {model_dim} 维\n"
                    f"   库 {self._collection_name}:   {stored_dim} 维\n\n"
                    f"   修复命令：\n"
                    f"     uv run python -c \"from server.engine import get_engine; "
                    f"e = get_engine(); e.reembed()\"\n"
                    f"{'='*60}"
                )
                logger.error(msg)
                raise ValueError(msg)
            logger.info("✅ 维度校验通过: 模型=%dd, 库=%dd", model_dim, stored_dim)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning("维度校验跳过（新库或无数据）: %s", e)

    def reembed(self):
        import shutil
        import chromadb
        from sentence_transformers import SentenceTransformer
        from datetime import datetime

        logger.warning("加载嵌入模型: %s", self._embed_model)
        embedder = SentenceTransformer(self._embed_model, device="cpu")
        logger.warning("⚠️  开始重新嵌入所有数据...")

        old_client = chromadb.PersistentClient(path=str(self._chroma_dir))
        all_data = {}
        total_items = 0
        for col in old_client.list_collections():
            all_data[col.name] = col.get(include=["documents", "metadatas"])
            total_items += len(all_data[col.name]["ids"])
            logger.info("  读取 collection '%s': %d 条", col.name, len(all_data[col.name]["ids"]))

        if not all_data:
            logger.warning("  无数据，无需重建")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_p = Path(f"{self._chroma_dir}_bak_{ts}")
        shutil.copytree(str(self._chroma_dir), str(backup_p))
        shutil.rmtree(str(self._chroma_dir))
        logger.info("  💾 备份: %s", backup_p)

        new_client = chromadb.PersistentClient(path=str(self._chroma_dir))
        for col_name, data in all_data.items():
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            ids = data.get("ids") or []
            if not docs:
                continue
            logger.info("  ⚡ 重嵌入 '%s' (%d 条)...", col_name, len(docs))
            embs = embedder.encode(docs, show_progress_bar=True).tolist()
            new_col = new_client.get_or_create_collection(name=col_name)
            new_col.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)

        self._collection = None
        self._embedder = None
        self._invalidate_bm25()
        logger.info("  ✅ reembed 完成！共 %d 条，备份在 %s", total_items, backup_p)

    @property
    def collection(self):
        self._lazy_init()
        return self._collection

    @property
    def embedder(self):
        self._lazy_init()
        return self._embedder

    @staticmethod
    def _parse_tags(raw_tags) -> list:
        try:
            return json.loads(raw_tags) if raw_tags else []
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _extract_brain_meta(meta: dict) -> dict:
        nested = {}
        raw_json = meta.get("metadata_json")
        if isinstance(raw_json, str) and raw_json.strip():
            try:
                nested = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError):
                nested = {}
        if not isinstance(nested, dict):
            nested = {}

        flat = {k: v for k, v in meta.items() if k not in _CHROMA_RESERVED}
        result = dict(nested)
        for k, v in flat.items():
            if v is not None and v != "" and v != 0:
                result[k] = v
        return result

    @staticmethod
    def _stored_content(meta: dict, doc_text: str) -> str:
        for key in ("parent_content", "content"):
            stored = meta.get(key)
            if isinstance(stored, str) and stored.strip():
                return stored
        return doc_text or ""

    def _build_chroma_metadata(
        self,
        item: KnowledgeItem,
        *,
        parent_id: str,
        chunk_index: int = 0,
        is_chunk: bool = False,
        full_content: str = "",
    ) -> dict:
        content = full_content or item.content
        store_parent_content = content if (not is_chunk or chunk_index == 0) else ""
        return {
            "id": item.id,
            "doc_type": item.doc_type,
            "title": item.title,
            "content": store_parent_content,
            "parent_content": store_parent_content,
            "tags": json.dumps(item.tags, ensure_ascii=False),
            "metadata_json": json.dumps(item.metadata, ensure_ascii=False),
            "created_at": item.created_at,
            "parent_id": parent_id,
            "chunk_index": int(chunk_index),
            "is_chunk": bool(is_chunk),
        }

    def _add_single(
        self,
        item: KnowledgeItem,
        *,
        parent_id: str,
        chunk_index: int = 0,
        is_chunk: bool = False,
        full_content: str = "",
    ) -> str:
        if not item.id:
            item.id = item.gen_id()
        embed_text = item.get_embedding_text()
        full = full_content or item.content
        if is_chunk:
            doc_store = embed_text
        else:
            doc_store = item.get_bm25_text()
        emb = self.embedder.encode([embed_text]).tolist()
        meta = self._build_chroma_metadata(
            item,
            parent_id=parent_id,
            chunk_index=chunk_index,
            is_chunk=is_chunk,
            full_content=full_content,
        )
        self.collection.upsert(
            ids=[item.id],
            embeddings=emb,
            metadatas=[meta],
            documents=[doc_store],
        )
        self._invalidate_bm25()
        return item.id

    def _delete_by_parent(self, parent_id: str) -> int:
        try:
            existing = self.collection.get(where={"parent_id": parent_id})
            ids = list(existing.get("ids") or [])
            if parent_id not in ids:
                probe = self.collection.get(ids=[parent_id])
                if probe.get("ids"):
                    ids.append(parent_id)
            if not ids:
                return 0
            self.collection.delete(ids=ids)
            self._invalidate_bm25()
            return len(ids)
        except Exception:
            return 0

    def add(self, item: KnowledgeItem) -> str:
        parent_id = item.id or item.gen_id()
        item.id = parent_id
        full_content = item.content or ""

        use_chunking = (
            self._enable_chunking
            and item.doc_type != "brain_memory"
            and len(full_content) > self._chunk_min_chars
        )
        if not use_chunking:
            self._delete_by_parent(parent_id)
            return self._add_single(
                item, parent_id=parent_id, full_content=full_content,
            )

        chunks = split_content(full_content, self._chunk_size, self._chunk_overlap)
        self._delete_by_parent(parent_id)
        if len(chunks) <= 1:
            return self._add_single(
                item, parent_id=parent_id, full_content=full_content,
            )

        for i, chunk_text in enumerate(chunks):
            chunk_item = KnowledgeItem(
                id=f"{parent_id}_c{i}",
                doc_type=item.doc_type,
                title=item.title,
                content=chunk_text,
                metadata=item.metadata,
                tags=item.tags,
                created_at=item.created_at,
            )
            self._add_single(
                chunk_item,
                parent_id=parent_id,
                chunk_index=i,
                is_chunk=True,
                full_content=full_content,
            )
        return parent_id

    def add_many(self, items: list[KnowledgeItem]) -> int:
        if not items:
            return 0
        count = 0
        for item in items:
            self.add(item)
            count += 1
        return count

    def delete(self, item_id: str) -> bool:
        deleted = self._delete_by_parent(item_id)
        if deleted:
            return True
        try:
            self.collection.delete(ids=[item_id])
            self._invalidate_bm25()
            return True
        except Exception:
            return False

    def delete_many(self, doc_type: str = None) -> int:
        where = {"doc_type": doc_type} if doc_type else None
        try:
            existing = self.collection.get(where=where)
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
                self._invalidate_bm25()
            return len(ids)
        except Exception:
            return 0

    def _bm25_corpus_text(self, meta: dict, doc_text: str) -> str:
        title = meta.get("title", "")
        doc_type = meta.get("doc_type", "")
        tags_str = ", ".join(self._parse_tags(meta.get("tags", "[]")))
        if meta.get("is_chunk"):
            body = doc_text
        else:
            body = self._stored_content(meta, doc_text)
        return f"{title} {doc_type} {tags_str} {body}".strip()

    def _bm25_search(self, query: str, where_clause: dict = None) -> list[tuple[str, float]]:
        import jieba

        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return []

        current_size = len(all_docs["ids"])
        if (
            self._bm25 is not None
            and self._bm25_size == current_size
            and self._bm25_all_ids == all_docs["ids"]
        ):
            bm25 = self._bm25
            metadata = self._bm25_metadata
            all_ids = self._bm25_all_ids
        else:
            from rank_bm25 import BM25Okapi

            corpus = []
            documents = all_docs.get("documents") or []
            for i, meta in enumerate(all_docs["metadatas"]):
                doc_text = documents[i] if i < len(documents) else ""
                text = self._bm25_corpus_text(meta, doc_text)
                corpus.append(jieba.lcut(text)[:400])
            bm25 = BM25Okapi(corpus)
            self._bm25 = bm25
            self._bm25_metadata = all_docs["metadatas"]
            self._bm25_all_ids = all_docs["ids"]
            self._bm25_documents = documents
            self._bm25_size = current_size
            metadata = all_docs["metadatas"]
            all_ids = all_docs["ids"]

        query_tokens = jieba.lcut(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        results = []
        for i, doc_id in enumerate(all_ids):
            if scores[i] <= 0:
                continue
            if where_clause:
                skip = any(metadata[i].get(k) != v for k, v in where_clause.items())
                if skip:
                    continue
            results.append((doc_id, float(scores[i])))

        results.sort(key=lambda x: -x[1])
        return results[: self._bm25_candidates]

    def _hit_to_result(self, hit_id: str, meta: dict, doc_text: str, score: float) -> dict:
        full_content = self._stored_content(meta, doc_text)
        parent_id = parent_id_from_meta(meta, hit_id)
        is_chunk = bool(meta.get("is_chunk"))
        summary_src = doc_text or full_content
        return {
            "id": parent_id if is_chunk else hit_id,
            "parent_id": parent_id,
            "doc_type": meta.get("doc_type", ""),
            "title": meta.get("title", ""),
            "tags": self._parse_tags(meta.get("tags", "[]")),
            "metadata": self._extract_brain_meta(meta),
            "created_at": meta.get("created_at", ""),
            "score": round(float(score), 4),
            "summary": summary_src[:200] + "..." if len(summary_src) > 200 else summary_src,
            "content": full_content,
            "rerank_text": f"{meta.get('title', '')}\n{summary_src[:1200]}",
            "is_chunk": is_chunk,
            "chunk_index": int(meta.get("chunk_index") or 0),
        }

    def search(self, query: str, n_results: int = 10, doc_type: str = None) -> list[dict]:
        """RRF(向量, BM25) → 可选 Rerank → 按 parent 去重"""
        query_emb = self.embedder.encode([query]).tolist()
        where_clause = {"doc_type": doc_type} if doc_type else None

        try:
            vec_results = self.collection.query(
                query_embeddings=query_emb,
                n_results=max(self._vec_candidates, n_results * 3),
                where=where_clause,
            )
        except Exception:
            vec_results = {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}

        vec_ranked: list[str] = []
        hit_cache: dict[str, dict] = {}
        if vec_results["ids"] and vec_results["ids"][0]:
            for i, hit_id in enumerate(vec_results["ids"][0]):
                meta = vec_results["metadatas"][0][i]
                doc = vec_results["documents"][0][i] if vec_results["documents"] else ""
                vec_ranked.append(hit_id)
                hit_cache[hit_id] = {"meta": meta, "doc": doc}

        bm25_pairs = self._bm25_search(query, where_clause=where_clause)
        bm25_ranked = [doc_id for doc_id, _ in bm25_pairs]
        for doc_id, _ in bm25_pairs:
            if doc_id in hit_cache:
                continue
            try:
                doc_data = self.collection.get(ids=[doc_id])
                if doc_data["ids"]:
                    hit_cache[doc_id] = {
                        "meta": doc_data["metadatas"][0],
                        "doc": doc_data["documents"][0] if doc_data["documents"] else "",
                    }
            except Exception:
                pass

        if not vec_ranked and not bm25_ranked:
            return []

        rrf_scores = reciprocal_rank_fusion(
            [vec_ranked, bm25_ranked],
            k=self._rrf_k,
        )
        ranked_ids = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])
        candidates: list[dict] = []
        for hit_id in ranked_ids[: max(self._rerank_top_n, n_results * 3)]:
            cached = hit_cache.get(hit_id)
            if not cached:
                continue
            candidates.append(
                self._hit_to_result(hit_id, cached["meta"], cached["doc"], rrf_scores[hit_id])
            )

        if self._reranker and candidates:
            try:
                candidates = self._reranker.rerank(
                    query, candidates, top_k=max(self._rerank_top_n, n_results),
                )
            except Exception as exc:
                logger.warning("Reranker 失败，回退 RRF 排序: %s", exc)

        deduped = dedupe_by_parent(candidates)
        return deduped[:n_results]

    def get_by_id(self, item_id: str) -> Optional[dict]:
        results = self.collection.get(ids=[item_id])
        if not results["ids"]:
            probe = self.collection.get(where={"parent_id": item_id}, limit=1)
            if not probe.get("ids"):
                return None
            results = probe

        i = 0
        meta = results["metadatas"][i] if results["metadatas"] else {}
        doc = results["documents"][i] if results["documents"] else ""
        parent_id = parent_id_from_meta(meta, results["ids"][i])
        full_content = self._stored_content(meta, doc)

        if meta.get("is_chunk") or results["ids"][i] != parent_id:
            siblings = self.collection.get(where={"parent_id": parent_id})
            for j, sm in enumerate(siblings.get("metadatas") or []):
                full_content = self._stored_content(sm, siblings["documents"][j] if siblings.get("documents") else "")
                if full_content:
                    break

        return {
            "id": parent_id,
            "doc_type": meta.get("doc_type", ""),
            "title": meta.get("title", ""),
            "tags": self._parse_tags(meta.get("tags", "[]")),
            "metadata": self._extract_brain_meta(meta),
            "created_at": meta.get("created_at", ""),
            "content": full_content,
        }

    def get_stats(self) -> dict:
        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return {"total": 0, "by_type": {}, "chunks": 0}

        parents: set[str] = set()
        by_type: dict[str, set[str]] = {}
        for i, hit_id in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i]
            pid = parent_id_from_meta(meta, hit_id)
            parents.add(pid)
            dt = meta.get("doc_type", "unknown")
            by_type.setdefault(dt, set()).add(pid)

        return {
            "total": len(parents),
            "chunks": len(all_docs["ids"]),
            "by_type": {dt: len(ids) for dt, ids in sorted(by_type.items(), key=lambda x: -len(x[1]))},
        }

    def _parent_documents(self, doc_type: str = None) -> list[dict]:
        """按逻辑文档（parent_id）去重后的完整列表，用于浏览分页。"""
        where_clause = {"doc_type": doc_type} if doc_type else None
        all_docs = self.collection.get(where=where_clause)
        if not all_docs["ids"]:
            return []

        parents: dict[str, tuple[int, dict]] = {}
        documents = all_docs.get("documents") or []

        for i, hit_id in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i]
            is_chunk = bool(meta.get("is_chunk"))
            chunk_idx = int(meta.get("chunk_index") or 0)
            if is_chunk and chunk_idx > 0:
                continue

            pid = parent_id_from_meta(meta, hit_id)
            doc = documents[i] if i < len(documents) else ""
            priority = 1 if is_chunk else 0
            item = {
                "id": pid,
                "doc_type": meta.get("doc_type", ""),
                "title": meta.get("title", ""),
                "tags": self._parse_tags(meta.get("tags", "[]")),
                "metadata": self._extract_brain_meta(meta),
                "created_at": meta.get("created_at", ""),
                "content": self._stored_content(meta, doc),
            }
            prev = parents.get(pid)
            if prev is None or priority < prev[0]:
                parents[pid] = (priority, item)

        items = [row for _, row in parents.values()]
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def list_parents(self, doc_type: str = None, offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        """按文档分页浏览，返回 (当前页条目, 文档总数)。"""
        all_items = self._parent_documents(doc_type)
        total = len(all_items)
        if limit <= 0:
            return [], total
        return all_items[offset: offset + limit], total

    def count_parents(self, doc_type: str = None) -> int:
        return len(self._parent_documents(doc_type))

    def get_all(self, doc_type: str = None, offset: int = 0, limit: int = 50) -> list[dict]:
        items, _ = self.list_parents(doc_type=doc_type, offset=offset, limit=limit)
        return items

    def get_all_texts(self) -> list[dict]:
        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return []

        seen_parents: set[str] = set()
        results = []
        documents = all_docs.get("documents") or []
        for i, hit_id in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i]
            parent_id = parent_id_from_meta(meta, hit_id)
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            doc = documents[i] if i < len(documents) else ""
            results.append({
                "id": parent_id,
                "title": meta.get("title", ""),
                "doc_type": meta.get("doc_type", ""),
                "tags": meta.get("tags", "[]"),
                "text": self._stored_content(meta, doc),
            })
        return results

    def count(self) -> int:
        return self.count_parents()

    def count_by_type(self, doc_type: str = None) -> int:
        return self.count_parents(doc_type)


_engine_instance = None


def get_engine() -> VectorEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = VectorEngine()
    return _engine_instance
