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
from .taxonomy import (
    RETIRED_STATUSES,
    compute_next_verify_at,
    infer_source_quality,
    normalize_doc_type,
    normalize_governance_status,
    now_str,
    require_ui_control_app,
    should_skip_chunking,
)

try:
    from .config import get_config
except ImportError:
    from config import get_config

DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("engine")

_CHROMA_RESERVED = {
    "id", "doc_type", "title", "content", "tags", "metadata_json", "created_at",
    "parent_id", "chunk_index", "is_chunk", "parent_content",
    "category", "game", "app", "governance_status", "version",
    "updated_at", "next_verify_at", "source_quality",
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
        _cfg = get_config()
        self._chroma_dir = Path(chroma_dir) if chroma_dir else Path(_cfg.chroma_dir)
        self._chroma_dir = Path(self._chroma_dir).resolve()
        self._embed_model = embed_model or _cfg.embed_model
        self._collection_name = collection_name or _cfg.collection_name
        self._collection = None
        self._embedder = None
        self._init_lock = threading.Lock()
        self._enable_chunking = _cfg.chunk_enabled if enable_chunking is None else enable_chunking
        self._enable_rerank = _cfg.rerank_enabled if enable_rerank is None else enable_rerank
        self._chunk_min_chars = 1000 if _cfg is None else _cfg.chunk_min_chars
        self._chunk_size = _cfg.chunk_size
        self._chunk_overlap = _cfg.chunk_overlap
        self._rrf_k = _cfg.rrf_k
        self._vec_candidates = _cfg.vec_candidates
        self._bm25_candidates = _cfg.bm25_candidates
        self._rerank_top_n = _cfg.rerank_top_n
        self._reranker = Reranker(_cfg.rerank_model) if self._enable_rerank else None
        self._freshness_sla_days = dict(getattr(_cfg, "freshness_sla_days", {}) or {})
        self._auto_curated_penalty = float(getattr(_cfg, "auto_curated_penalty", 0.5))
        self._stale_penalty = float(getattr(_cfg, "stale_penalty", 0.85))
        self._archive_after_stale_days = int(getattr(_cfg, "archive_after_stale_days", 180))

        self._bm25 = None
        self._bm25_metadata = None
        self._bm25_all_ids = None
        self._bm25_documents = None
        self._bm25_size = 0
        self._bm25_lock = threading.Lock()
        self._embed_cache: dict[str, list[float]] = {}
        self._embed_cache_lock = threading.Lock()
        self._embed_cache_max = 512

    def count_parents(self, doc_type: str = None) -> int:
        return len(self._parent_documents(doc_type))

    def _invalidate_bm25(self) -> None:
        with getattr(self, "_bm25_lock", threading.Lock()):
            self._bm25 = None
            self._bm25_all_ids = None
            self._bm25_documents = None
            self._bm25_metadata = None
            self._bm25_size = 0

    def _encode_query(self, query: str) -> list[float]:
        q = (query or "").strip()
        if not q:
            return []
        with self._embed_cache_lock:
            cached = self._embed_cache.get(q)
            if cached is not None:
                return cached
        emb = self.embedder.encode([q]).tolist()[0]
        with self._embed_cache_lock:
            if len(self._embed_cache) >= self._embed_cache_max:
                # 简单淘汰一半
                keys = list(self._embed_cache.keys())[: self._embed_cache_max // 2]
                for k in keys:
                    self._embed_cache.pop(k, None)
            self._embed_cache[q] = emb
        return emb

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

    def _prepare_item_for_write(self, item: KnowledgeItem) -> KnowledgeItem:
        """规范化类型/治理字段；ui_control 校验 app。"""
        item.doc_type = normalize_doc_type(item.doc_type)
        item.governance_status = normalize_governance_status(item.governance_status)
        item.source_quality = infer_source_quality(item.tags, item.source_quality)
        item.app = require_ui_control_app(item.doc_type, item.app, item.metadata)
        if item.app and isinstance(item.metadata, dict):
            item.metadata["app"] = item.app
        if not item.created_at:
            item.created_at = now_str()
        item.updated_at = item.updated_at or now_str()
        if not item.next_verify_at:
            item.next_verify_at = compute_next_verify_at(
                item.doc_type, self._freshness_sla_days,
            )
        if not item.version:
            item.version = 1
        return item

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
            "metadata_json": json.dumps(item.metadata or {}, ensure_ascii=False),
            "created_at": item.created_at or "",
            "parent_id": parent_id,
            "chunk_index": int(chunk_index),
            "is_chunk": bool(is_chunk),
            "category": item.category or "",
            "game": item.game or "",
            "app": item.app or "",
            "governance_status": item.governance_status or "published",
            "version": int(item.version or 1),
            "updated_at": item.updated_at or item.created_at or "",
            "next_verify_at": item.next_verify_at or "",
            "source_quality": item.source_quality or "curated",
        }

    def _item_from_stored(self, parent_id: str, meta: dict, content: str) -> KnowledgeItem:
        tags = self._parse_tags(meta.get("tags", "[]"))
        nested = self._extract_brain_meta(meta)
        return KnowledgeItem(
            id=parent_id,
            doc_type=meta.get("doc_type", "doc"),
            title=meta.get("title", ""),
            content=content or "",
            metadata=nested if isinstance(nested, dict) else {},
            tags=tags,
            created_at=meta.get("created_at", ""),
            category=meta.get("category", "") or "",
            game=meta.get("game", "") or "",
            app=meta.get("app", "") or "",
            governance_status=meta.get("governance_status", "published") or "published",
            version=int(meta.get("version") or 1),
            updated_at=meta.get("updated_at", "") or "",
            next_verify_at=meta.get("next_verify_at", "") or "",
            source_quality=meta.get("source_quality", "curated") or "curated",
        )

    def _parent_fields_from_meta(self, meta: dict) -> dict:
        return {
            "category": meta.get("category", "") or "",
            "game": meta.get("game", "") or "",
            "app": meta.get("app", "") or "",
            "governance_status": meta.get("governance_status", "published") or "published",
            "version": int(meta.get("version") or 1),
            "updated_at": meta.get("updated_at", "") or "",
            "next_verify_at": meta.get("next_verify_at", "") or "",
            "source_quality": meta.get("source_quality", "curated") or "curated",
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
        item = self._prepare_item_for_write(item)
        parent_id = item.id or item.gen_id()
        item.id = parent_id
        full_content = item.content or ""

        use_chunking = (
            self._enable_chunking
            and not should_skip_chunking(item.doc_type)
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
                category=item.category,
                game=item.game,
                app=item.app,
                governance_status=item.governance_status,
                version=item.version,
                updated_at=item.updated_at,
                next_verify_at=item.next_verify_at,
                source_quality=item.source_quality,
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

    def update_metadata_only(self, parent_id: str, item: KnowledgeItem) -> bool:
        """只更新 parent 下所有 chunk 的标量 meta，不 reembed。"""
        try:
            existing = self.collection.get(where={"parent_id": parent_id})
            ids = list(existing.get("ids") or [])
            if parent_id not in ids:
                probe = self.collection.get(ids=[parent_id])
                if probe.get("ids"):
                    ids.append(parent_id)
                    if not existing.get("ids"):
                        existing = probe
            if not ids:
                return False

            # 重新 get 保证顺序一致
            row = self.collection.get(ids=ids)
            new_metas = []
            for i, hit_id in enumerate(row["ids"]):
                old = dict(row["metadatas"][i] or {})
                chunk_index = int(old.get("chunk_index") or 0)
                is_chunk = bool(old.get("is_chunk"))
                full = self._stored_content(old, (row.get("documents") or [""])[i] if row.get("documents") else "")
                built = self._build_chroma_metadata(
                    item,
                    parent_id=parent_id,
                    chunk_index=chunk_index,
                    is_chunk=is_chunk,
                    full_content=full if (not is_chunk or chunk_index == 0) else old.get("parent_content") or "",
                )
                # 保留原 content/parent_content（不改正文）
                if old.get("content") is not None:
                    built["content"] = old.get("content")
                if old.get("parent_content") is not None:
                    built["parent_content"] = old.get("parent_content")
                built["id"] = hit_id
                new_metas.append(built)
            self.collection.update(ids=row["ids"], metadatas=new_metas)
            self._invalidate_bm25()
            return True
        except Exception as exc:
            logger.warning("update_metadata_only failed for %s: %s", parent_id, exc)
            return False

    def update(self, item_id: str, **fields) -> Optional[dict]:
        """合并更新条目。改正文/标题则重建向量；只改 meta 则 update。"""
        current = self.get_by_id(item_id)
        if not current:
            return None

        item = KnowledgeItem(
            id=current["id"],
            doc_type=current.get("doc_type") or "doc",
            title=current.get("title") or "",
            content=current.get("content") or "",
            metadata=dict(current.get("metadata") or {}),
            tags=list(current.get("tags") or []),
            created_at=current.get("created_at") or "",
            category=current.get("category") or "",
            game=current.get("game") or "",
            app=current.get("app") or "",
            governance_status=current.get("governance_status") or "published",
            version=int(current.get("version") or 1),
            updated_at=current.get("updated_at") or "",
            next_verify_at=current.get("next_verify_at") or "",
            source_quality=current.get("source_quality") or "curated",
        )

        content_keys = {"title", "content"}
        touch_content = False
        explicit_next_verify = fields.get("next_verify_at")
        for key, value in fields.items():
            if value is None:
                continue
            if key == "tags":
                item.tags = list(value) if isinstance(value, list) else item.tags
            elif key == "metadata" and isinstance(value, dict):
                merged = dict(item.metadata or {})
                merged.update(value)
                item.metadata = merged
            elif hasattr(item, key):
                setattr(item, key, value)
            if key in content_keys:
                touch_content = True

        item.version = int(item.version or 1) + 1
        item.updated_at = now_str()
        # 避免 prepare 覆盖调用方指定的 next_verify_at
        if not item.next_verify_at:
            item.next_verify_at = compute_next_verify_at(item.doc_type, self._freshness_sla_days)
        item = self._prepare_item_for_write(item)
        if explicit_next_verify:
            item.next_verify_at = explicit_next_verify

        if touch_content:
            item.id = current["id"]
            self.add(item)
        else:
            ok = self.update_metadata_only(current["id"], item)
            if not ok:
                return None
        return self.get_by_id(current["id"])

    def set_status(self, item_id: str, status: str, *, refresh_verify: bool = False) -> Optional[dict]:
        current = self.get_by_id(item_id)
        if not current:
            return None
        if refresh_verify:
            return self.update(
                item_id,
                governance_status="published",
                next_verify_at=compute_next_verify_at(
                    current.get("doc_type") or "doc", self._freshness_sla_days,
                ),
            )
        return self.update(item_id, governance_status=normalize_governance_status(status))

    def mark_stale_batch(self, *, dry_run: bool = True) -> dict:
        """published 且超过 next_verify_at → stale。"""
        from datetime import datetime

        items = self._parent_documents()
        now = datetime.now()
        candidates = []
        for it in items:
            if (it.get("governance_status") or "published") != "published":
                continue
            nva = it.get("next_verify_at") or ""
            if not nva:
                continue
            try:
                due = datetime.strptime(nva, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if now > due:
                candidates.append(it["id"])
        if dry_run:
            return {"dry_run": True, "count": len(candidates), "ids": candidates[:50]}
        updated = 0
        for iid in candidates:
            if self.set_status(iid, "stale"):
                updated += 1
        return {"dry_run": False, "count": updated, "ids": candidates[:50]}

    def archive_stale_batch(self, *, dry_run: bool = True) -> dict:
        """stale 超过 archive_after_stale_days → archived。"""
        from datetime import datetime, timedelta

        items = self._parent_documents()
        cutoff = datetime.now() - timedelta(days=self._archive_after_stale_days)
        candidates = []
        for it in items:
            if (it.get("governance_status") or "") != "stale":
                continue
            ts = it.get("updated_at") or it.get("created_at") or ""
            try:
                when = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if when < cutoff:
                candidates.append(it["id"])
        if dry_run:
            return {"dry_run": True, "count": len(candidates), "ids": candidates[:50]}
        updated = 0
        for iid in candidates:
            if self.set_status(iid, "archived"):
                updated += 1
        return {"dry_run": False, "count": updated, "ids": candidates[:50]}

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

        with self._bm25_lock:
            if self._bm25 is None:
                all_docs = self.collection.get()
                if not all_docs["ids"]:
                    return []
                from rank_bm25 import BM25Okapi

                corpus = []
                documents = all_docs.get("documents") or []
                metas = all_docs.get("metadatas") or []
                all_ids = all_docs.get("ids") or []
                for i, meta in enumerate(metas):
                    doc_text = documents[i] if i < len(documents) else ""
                    text = self._bm25_corpus_text(meta, doc_text)
                    corpus.append(jieba.lcut(text)[:400])
                if not corpus:
                    return []
                self._bm25 = BM25Okapi(corpus)
                self._bm25_metadata = metas
                self._bm25_all_ids = all_ids
                self._bm25_documents = documents
                self._bm25_size = len(all_ids)

            bm25 = self._bm25
            metadata = self._bm25_metadata
            all_ids = self._bm25_all_ids

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

    def _hit_to_result(self, hit_id: str, meta: dict, doc_text: str, rrf_score: float) -> dict:
        full_content = self._stored_content(meta, doc_text)
        parent_id = parent_id_from_meta(meta, hit_id)
        is_chunk = bool(meta.get("is_chunk"))
        summary_src = doc_text or full_content
        tags = self._parse_tags(meta.get("tags", "[]"))
        return {
            "id": parent_id if is_chunk else hit_id,
            "parent_id": parent_id,
            "doc_type": meta.get("doc_type", ""),
            "title": meta.get("title", ""),
            "tags": tags,
            "metadata": self._extract_brain_meta(meta),
            "created_at": meta.get("created_at", ""),
            "rrf_score": round(float(rrf_score), 6),
            "score": round(float(rrf_score), 4),  # 归一化前先占位
            "summary": summary_src[:200] + "..." if len(summary_src) > 200 else summary_src,
            "content": full_content,
            "rerank_text": f"{meta.get('title', '')}\n{summary_src[:1200]}",
            "is_chunk": is_chunk,
            "chunk_index": int(meta.get("chunk_index") or 0),
            **self._parent_fields_from_meta(meta),
        }

    @staticmethod
    def _build_where(
        doc_type: str = None,
        game: str = None,
        app: str = None,
        category: str = None,
    ) -> Optional[dict]:
        clauses = []
        if doc_type:
            clauses.append({"doc_type": normalize_doc_type(doc_type)})
        if game:
            clauses.append({"game": game})
        if app:
            clauses.append({"app": app})
        if category:
            clauses.append({"category": category})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _postprocess_results(
        self,
        results: list[dict],
        *,
        n_results: int,
        game: str = None,
        app: str = None,
        category: str = None,
        tags_any: list = None,
        include_retired: bool = False,
        filter_relaxed_hint: str = "",
    ) -> list[dict]:
        if not results:
            return []

        def _apply_penalties(rows: list[dict]) -> list[dict]:
            out = []
            for r in rows:
                score = float(r.get("rrf_score") or r.get("score") or 0)
                status = (r.get("governance_status") or "published").lower()
                if not include_retired and status in RETIRED_STATUSES:
                    continue
                sq = (r.get("source_quality") or "").lower()
                tags = [str(t).lower() for t in (r.get("tags") or [])]
                if sq == "auto_curated" or "auto_curated" in tags:
                    score *= self._auto_curated_penalty
                if status == "stale":
                    score *= self._stale_penalty
                r = dict(r)
                r["rrf_score"] = round(score, 6)
                out.append(r)
            out.sort(key=lambda x: -float(x.get("rrf_score") or 0))
            return out

        def _hard_filter(rows: list[dict]) -> list[dict]:
            filtered = []
            tag_set = {str(t).lower() for t in (tags_any or []) if t}
            for r in rows:
                if game and (r.get("game") or "") != game:
                    # 也看 metadata
                    if (r.get("metadata") or {}).get("game") != game:
                        continue
                if app and (r.get("app") or "") != app:
                    if (r.get("metadata") or {}).get("app") != app:
                        continue
                if category and (r.get("category") or "") != category:
                    if (r.get("metadata") or {}).get("category") != category:
                        continue
                if tag_set:
                    row_tags = {str(t).lower() for t in (r.get("tags") or [])}
                    if not tag_set.intersection(row_tags):
                        continue
                filtered.append(r)
            return filtered

        penalized = _apply_penalties(results)
        filtered = _hard_filter(penalized)
        relaxed = False
        if (game or app or category or tags_any) and not filtered and penalized:
            filtered = penalized
            relaxed = True

        max_rrf = max((float(r.get("rrf_score") or 0) for r in filtered), default=0.0)
        final = []
        for i, r in enumerate(filtered[:n_results]):
            row = dict(r)
            raw = float(row.get("rrf_score") or 0)
            row["rank"] = i + 1
            row["score"] = round((raw / max_rrf) if max_rrf > 0 else 0.0, 4)
            if relaxed:
                row["filter_relaxed"] = True
                if filter_relaxed_hint:
                    row["filter_hint"] = filter_relaxed_hint
            final.append(row)
        return final

    def search(
        self,
        query: str,
        n_results: int = 10,
        doc_type: str = None,
        *,
        game: str = None,
        app: str = None,
        category: str = None,
        include_retired: bool = False,
        tags_any: list = None,
    ) -> list[dict]:
        """RRF(向量, BM25) → 可选 Rerank → parent 去重 → 过滤降权 → 归一化 score"""
        query_emb = self._encode_query(query)
        where_clause = self._build_where(doc_type=doc_type, game=game, app=app, category=category)
        # Chroma where 对缺失字段的旧数据不友好：硬过滤放到后处理，where 只用 doc_type
        chroma_where = {"doc_type": normalize_doc_type(doc_type)} if doc_type else None

        try:
            vec_results = self.collection.query(
                query_embeddings=[query_emb] if query_emb else None,
                n_results=max(self._vec_candidates, n_results * 3),
                where=chroma_where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            vec_results = {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}

        vec_ranked: list[str] = []
        hit_cache: dict[str, dict] = {}
        if vec_results["ids"] and vec_results["ids"][0]:
            for i, hit_id in enumerate(vec_results["ids"][0]):
                meta = vec_results["metadatas"][0][i]
                doc = vec_results["documents"][0][i] if vec_results["documents"] else ""
                dist = None
                if vec_results.get("distances") and vec_results["distances"][0]:
                    dist = vec_results["distances"][0][i]
                vec_ranked.append(hit_id)
                hit_cache[hit_id] = {"meta": meta, "doc": doc, "distance": dist}

        bm25_pairs = self._bm25_search(query, where_clause=chroma_where)
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
                        "distance": None,
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
        # 多取一些供后处理过滤
        fetch_n = max(self._rerank_top_n, n_results * 5)
        candidates: list[dict] = []
        for hit_id in ranked_ids[:fetch_n]:
            cached = hit_cache.get(hit_id)
            if not cached:
                continue
            row = self._hit_to_result(hit_id, cached["meta"], cached["doc"], rrf_scores[hit_id])
            if cached.get("distance") is not None:
                row["vec_distance"] = round(float(cached["distance"]), 6)
            candidates.append(row)

        if self._reranker and candidates:
            try:
                candidates = self._reranker.rerank(
                    query, candidates, top_k=max(self._rerank_top_n, n_results * 3),
                )
                # rerank 后用 rerank score 作为 rrf_score 继续后处理
                for c in candidates:
                    if "rerank_score" in c:
                        c["rrf_score"] = float(c["rerank_score"])
            except Exception as exc:
                logger.warning("Reranker 失败，回退 RRF 排序: %s", exc)

        deduped = dedupe_by_parent(candidates)
        return self._postprocess_results(
            deduped,
            n_results=n_results,
            game=game,
            app=app,
            category=category,
            tags_any=tags_any,
            include_retired=include_retired,
            filter_relaxed_hint="精确过滤无结果，已回退到未过滤排序",
        )

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
            **self._parent_fields_from_meta(meta),
        }

    def get_stats(self) -> dict:
        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return {"total": 0, "by_type": {}, "by_status": {}, "ui_control_by_app": {}, "chunks": 0}

        parents: set[str] = set()
        by_type: dict[str, set[str]] = {}
        by_status: dict[str, set[str]] = {}
        ui_by_app: dict[str, set[str]] = {}
        seen_meta: dict[str, dict] = {}
        for i, hit_id in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i]
            pid = parent_id_from_meta(meta, hit_id)
            parents.add(pid)
            if pid not in seen_meta or not bool(meta.get("is_chunk")):
                seen_meta[pid] = meta
            dt = meta.get("doc_type", "unknown")
            by_type.setdefault(dt, set()).add(pid)

        for pid, meta in seen_meta.items():
            status = meta.get("governance_status") or "published"
            by_status.setdefault(status, set()).add(pid)
            if meta.get("doc_type") == "ui_control":
                app = meta.get("app") or (self._extract_brain_meta(meta).get("app") if meta else "") or "unknown"
                ui_by_app.setdefault(str(app), set()).add(pid)

        return {
            "total": len(parents),
            "chunks": len(all_docs["ids"]),
            "by_type": {dt: len(ids) for dt, ids in sorted(by_type.items(), key=lambda x: -len(x[1]))},
            "by_status": {st: len(ids) for st, ids in sorted(by_status.items(), key=lambda x: -len(x[1]))},
            "ui_control_by_app": {app: len(ids) for app, ids in sorted(ui_by_app.items(), key=lambda x: -len(x[1]))},
        }

    def _parent_documents(
        self,
        doc_type: str = None,
        *,
        game: str = None,
        app: str = None,
        category: str = None,
        governance_status: str = None,
        include_retired: bool = True,
    ) -> list[dict]:
        """按逻辑文档（parent_id）去重后的完整列表，用于浏览分页。"""
        where_clause = {"doc_type": normalize_doc_type(doc_type)} if doc_type else None
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
                **self._parent_fields_from_meta(meta),
            }
            prev = parents.get(pid)
            if prev is None or priority < prev[0]:
                parents[pid] = (priority, item)

        items = [row for _, row in parents.values()]
        filtered = []
        for it in items:
            status = (it.get("governance_status") or "published").lower()
            if not include_retired and status in RETIRED_STATUSES:
                continue
            if governance_status and status != governance_status.lower():
                continue
            if game and (it.get("game") or "") != game:
                continue
            if app and (it.get("app") or "") != app:
                continue
            if category and (it.get("category") or "") != category:
                continue
            filtered.append(it)
        filtered.sort(key=lambda x: x.get("updated_at") or x.get("created_at", ""), reverse=True)
        return filtered

    def list_parents(
        self,
        doc_type: str = None,
        offset: int = 0,
        limit: int = 50,
        *,
        game: str = None,
        app: str = None,
        category: str = None,
        governance_status: str = None,
        include_retired: bool = True,
    ) -> tuple[list[dict], int]:
        """按文档分页浏览，返回 (当前页条目, 文档总数)。"""
        all_items = self._parent_documents(
            doc_type,
            game=game,
            app=app,
            category=category,
            governance_status=governance_status,
            include_retired=include_retired,
        )
        total = len(all_items)
        if limit <= 0:
            return [], total
        return all_items[offset: offset + limit], total

    def get_all(
        self,
        doc_type: str = None,
        offset: int = 0,
        limit: int = 50,
        *,
        game: str = None,
        app: str = None,
        category: str = None,
        governance_status: str = None,
        include_retired: bool = True,
    ) -> list[dict]:
        items, _ = self.list_parents(
            doc_type=doc_type,
            offset=offset,
            limit=limit,
            game=game,
            app=app,
            category=category,
            governance_status=governance_status,
            include_retired=include_retired,
        )
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
