"""
核心引擎 — ChromaDB 向量库 + sentence-transformers 嵌入 + BM25 混合搜索

通用版，支持多种文档类型（doc_type）。
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .models import KnowledgeItem

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


# ── 向量库引擎 ──


class VectorEngine:
    """ChromaDB 引擎，管理知识条目的向量化存储和检索"""

    def __init__(self):
        self._collection = None
        self._embedder = None
        self._init_lock = threading.Lock()
        # BM25 缓存
        self._bm25 = None
        self._bm25_metadata = None
        self._bm25_all_ids = None
        self._bm25_size = 0

    def _lazy_init(self):
        if self._collection is not None:
            return
        with self._init_lock:
            # 双重检查锁：拿到锁后再次检查
            if self._collection is not None:
                return
            import chromadb
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载嵌入模型: {EMBED_MODEL}")
            self._embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB 就绪: {CHROMA_DIR} / {COLLECTION_NAME}")

            # ── 维度校验：防止嵌入模型与 ChromaDB 维度不匹配 ──
            self._validate_dimension()

    def _validate_dimension(self):
        """校验嵌入模型输出维度与 ChromaDB 存储的 embedding 维度一致"""
        try:
            sample_emb = self._embedder.encode(["验证维度"]).tolist()
            model_dim = len(sample_emb[0])

            existing = self._collection.get(limit=1, include=["embeddings"])
            if existing["ids"] and existing.get("embeddings"):
                stored_dim = len(existing["embeddings"][0])
                if model_dim != stored_dim:
                    msg = (
                        f"\n{'='*60}\n"
                        f"❌ 嵌入模型维度不匹配！\n"
                        f"   模型 {EMBED_MODEL}: {model_dim} 维\n"
                        f"   库 {COLLECTION_NAME}:   {stored_dim} 维\n\n"
                        f"   原因：ChromaDB 是用另一个模型建的，改模型后没重建。\n\n"
                        f"   修复命令：\n"
                        f"     uv run python -c \"from server.engine import get_engine; "
                        f"e = get_engine(); e.reembed()\"\n"
                        f"{'='*60}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    logger.info(f"✅ 维度校验通过: 模型={model_dim}d, 库={stored_dim}d")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning(f"维度校验跳过（新库或无数据）: {e}")

    def reembed(self):
        """
        用当前模型重新嵌入所有已有数据（修复模型切换后的维度不匹配）。
        自动备份旧库，安全可恢复。
        """
        import shutil
        from pathlib import Path
        import chromadb
        from sentence_transformers import SentenceTransformer

        # 只加载模型（不连 ChromaDB，避免维度校验报错）
        logger.warning(f"加载嵌入模型: {EMBED_MODEL}")
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

        logger.warning("⚠️  开始重新嵌入所有数据...")

        old_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        old_cols = old_client.list_collections()
        all_data = {}
        for col in old_cols:
            all_data[col.name] = col.get(include=["documents", "metadatas"])
            logger.info(f"  读取 collection '{col.name}': {col.count()} 条")

        backup_dir = f"{CHROMA_DIR}_reembed_bak"
        backup_p = Path(backup_dir)
        if backup_p.exists():
            shutil.rmtree(backup_p)
        shutil.copytree(str(CHROMA_DIR), str(backup_p))
        logger.info(f"  💾 备份: {backup_p}")

        shutil.rmtree(str(CHROMA_DIR))
        logger.info(f"  🗑️  旧库已删除")

        new_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        for col_name, data in all_data.items():
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            ids = data.get("ids") or []
            if not docs:
                continue
            logger.info(f"  ⚡ 重嵌入 '{col_name}' ({len(docs)} 条)...")
            embs = embedder.encode(docs, show_progress_bar=True).tolist()
            new_col = new_client.get_or_create_collection(name=col_name)
            new_col.add(
                documents=docs,
                metadatas=metas,
                ids=ids,
                embeddings=embs,
            )

        self._collection = None
        self._bm25 = None
        self._bm25_all_ids = None
        self._bm25_size = 0
        self._lazy_init()
        logger.info(f"  ✅ reembed 完成！备份在 {backup_p}")

    @property
    def collection(self):
        self._lazy_init()
        return self._collection

    @property
    def embedder(self):
        self._lazy_init()
        return self._embedder

    # ── 增删改 ──

    def add(self, item: KnowledgeItem) -> str:
        """添加单条知识条目，返回 ID"""
        if not item.id:
            item.id = item.gen_id()
        text = item.get_embedding_text()
        emb = self.embedder.encode([text]).tolist()

        # metadata 中 tags 和 metadata 字段都转 JSON 字符串存入
        self.collection.add(
            ids=[item.id],
            embeddings=emb,
            metadatas=[{
                "id": item.id,
                "doc_type": item.doc_type,
                "title": item.title,
                "tags": json.dumps(item.tags, ensure_ascii=False),
                "metadata_json": json.dumps(item.metadata, ensure_ascii=False),
                "created_at": item.created_at,
            }],
            documents=[text],
        )
        self._bm25 = None
        return item.id

    def add_many(self, items: list[KnowledgeItem]) -> int:
        """批量添加，返回添加数量"""
        if not items:
            return 0
        documents = []
        ids = []
        metadatas = []
        for item in items:
            if not item.id:
                item.id = item.gen_id()
            documents.append(item.get_embedding_text())
            ids.append(item.id)
            metadatas.append({
                "id": item.id,
                "doc_type": item.doc_type,
                "title": item.title,
                "tags": json.dumps(item.tags, ensure_ascii=False),
                "metadata_json": json.dumps(item.metadata, ensure_ascii=False),
                "created_at": item.created_at,
            })

        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch_texts = documents[i:i + batch_size]
            batch_emb = self.embedder.encode(batch_texts).tolist()
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=batch_emb,
                metadatas=metadatas[i:i + batch_size],
                documents=batch_texts,
            )
        self._bm25 = None
        return len(items)

    def delete(self, item_id: str) -> bool:
        """删除指定条目"""
        try:
            self.collection.delete(ids=[item_id])
            self._bm25 = None
            return True
        except Exception:
            return False

    def delete_many(self, doc_type: str = None) -> int:
        """按 doc_type 批量删除"""
        where = {}
        if doc_type:
            where["doc_type"] = doc_type
        try:
            existing = self.collection.get(where=where if where else None)
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                self._bm25 = None
            return len(existing["ids"])
        except Exception:
            return 0

    # ── 检索 ──

    def search(self, query: str, n_results: int = 10,
               doc_type: str = None) -> list[dict]:
        """
        混合搜索（向量 0.6 + BM25 关键词 0.4）

        搜索流程：
        1. 向量搜索：用 ChromaDB 做语义匹配
        2. BM25 搜索：用 jieba 分词 + rank_bm25 做关键词精确匹配
        3. 融合排序：min-max 归一化后加权合并

        参数:
          query:      搜索关键词
          n_results:  返回结果数量
          doc_type:   按文档类型筛选 (如 "test_case")

        返回: [{id, doc_type, title, tags, score, summary, ...}]
        """
        query_emb = self.embedder.encode([query]).tolist()

        where_clause = None
        if doc_type:
            where_clause = {"doc_type": doc_type}

        # 先 try query+where，失败则回退
        try:
            vec_results = self.collection.query(
                query_embeddings=query_emb,
                n_results=n_results * 3,
                where=where_clause,
            )
        except Exception:
            vec_results = {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}

        # 构建 ID → 结果映射
        hit_map = {}
        if vec_results["ids"] and vec_results["ids"][0]:
            for i, id_ in enumerate(vec_results["ids"][0]):
                meta = vec_results["metadatas"][0][i]
                dist = vec_results["distances"][0][i]
                doc = vec_results["documents"][0][i] if vec_results["documents"] else ""
                hit_map[id_] = {
                    "meta": meta,
                    "doc": doc,
                    "vec_score": 1.0 - dist,
                    "bm25_score": 0.0,
                }

        # BM25 关键词搜索
        bm25_pairs = self._bm25_search(query, where_clause=where_clause)
        for id_, bm25_score in bm25_pairs:
            if id_ in hit_map:
                hit_map[id_]["bm25_score"] = bm25_score
            else:
                # BM25 命中但向量没命中
                try:
                    doc_data = self.collection.get(ids=[id_])
                    if doc_data["ids"]:
                        meta = doc_data["metadatas"][0]
                        doc = doc_data["documents"][0] if doc_data["documents"] else ""
                        hit_map[id_] = {
                            "meta": meta,
                            "doc": doc,
                            "vec_score": 0.0,
                            "bm25_score": bm25_score,
                        }
                except Exception:
                    pass

        if not hit_map:
            return []

        # 分数归一化 + 融合
        bm25_all_zero = all(h["bm25_score"] == 0.0 for h in hit_map.values())
        vec_scores = [h["vec_score"] for h in hit_map.values()]
        bm25_scores = [h["bm25_score"] for h in hit_map.values()]

        if bm25_scores and not bm25_all_zero:
            vec_min, vec_max = min(vec_scores), max(vec_scores)
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        else:
            vec_min, vec_max = min(vec_scores), max(vec_scores)
            bm25_min, bm25_max = 0, 1

        vec_range = vec_max - vec_min if vec_max > vec_min else 1.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        results = []
        for id_, data in hit_map.items():
            norm_vec = (data["vec_score"] - vec_min) / vec_range
            norm_bm25 = (data["bm25_score"] - bm25_min) / bm25_range if not bm25_all_zero else 0.0

            if bm25_all_zero:
                final_score = norm_vec
            else:
                final_score = 0.6 * norm_vec + 0.4 * norm_bm25

            meta = data["meta"]
            doc_text = data["doc"]

            # 解析 tags 和 metadata_json
            raw_tags = meta.get("tags", "[]")
            try:
                tags = json.loads(raw_tags) if raw_tags else []
            except (json.JSONDecodeError, TypeError):
                tags = []

            raw_meta_json = meta.get("metadata_json", "{}")
            try:
                item_metadata = json.loads(raw_meta_json) if raw_meta_json else {}
            except (json.JSONDecodeError, TypeError):
                item_metadata = {}

            results.append({
                "id": id_,
                "doc_type": meta.get("doc_type", ""),
                "title": meta.get("title", ""),
                "tags": tags,
                "metadata": item_metadata,
                "created_at": meta.get("created_at", ""),
                "score": round(final_score, 4),
                "summary": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
            })

        results.sort(key=lambda x: -x["score"])
        return results[:n_results]

    def _bm25_search(self, query: str, where_clause: dict = None) -> list[tuple[str, float]]:
        """
        BM25 关键词搜索（内部方法）

        使用 jieba 分词 + rank_bm25 对 title/doc_type/tags 做关键词匹配。
        返回: [(id, score), ...] 按 score 降序
        """
        import jieba

        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return []

        # 检查 BM25 缓存
        current_size = len(all_docs["ids"])
        if (self._bm25 is not None and self._bm25_size == current_size
                and self._bm25_all_ids == all_docs["ids"]):
            bm25 = self._bm25
            metadata = self._bm25_metadata
            all_ids = self._bm25_all_ids
        else:
            from rank_bm25 import BM25Okapi
            corpus = []
            for meta in all_docs["metadatas"]:
                title = meta.get("title", "")
                doc_type = meta.get("doc_type", "")
                tags_raw = meta.get("tags", "[]")
                try:
                    tags_str = ", ".join(json.loads(tags_raw)) if tags_raw else ""
                except (json.JSONDecodeError, TypeError):
                    tags_str = ""
                text = f"{title} {doc_type} {tags_str}"
                tokens = jieba.lcut(text)[:200]
                corpus.append(tokens)
            bm25 = BM25Okapi(corpus)
            self._bm25 = bm25
            self._bm25_metadata = all_docs["metadatas"]
            self._bm25_all_ids = all_docs["ids"]
            self._bm25_size = current_size
            metadata = all_docs["metadatas"]
            all_ids = all_docs["ids"]

        query_tokens = jieba.lcut(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        results = []
        for i in range(len(all_ids)):
            if scores[i] <= 0:
                continue
            if where_clause:
                skip = False
                for key, val in where_clause.items():
                    if metadata[i].get(key) != val:
                        skip = True
                        break
                if skip:
                    continue
            results.append((all_ids[i], scores[i]))

        results.sort(key=lambda x: -x[1])
        return results[:50]

    def get_by_id(self, item_id: str) -> Optional[dict]:
        """按 ID 获取单条"""
        results = self.collection.get(ids=[item_id])
        if not results["ids"]:
            return None
        i = 0
        meta = results["metadatas"][i] if results["metadatas"] else {}
        doc = results["documents"][i] if results["documents"] else ""

        raw_tags = meta.get("tags", "[]")
        try:
            tags = json.loads(raw_tags) if raw_tags else []
        except (json.JSONDecodeError, TypeError):
            tags = []

        raw_meta_json = meta.get("metadata_json", "{}")
        try:
            item_metadata = json.loads(raw_meta_json) if raw_meta_json else {}
        except (json.JSONDecodeError, TypeError):
            item_metadata = {}

        return {
            "id": item_id,
            "doc_type": meta.get("doc_type", ""),
            "title": meta.get("title", ""),
            "tags": tags,
            "metadata": item_metadata,
            "created_at": meta.get("created_at", ""),
            "content": doc,
        }

    def get_stats(self) -> dict:
        """获取统计信息——按 doc_type 分布"""
        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return {"total": 0, "by_type": {}}

        by_type = {}
        for meta in all_docs["metadatas"]:
            dt = meta.get("doc_type", "unknown")
            by_type[dt] = by_type.get(dt, 0) + 1

        return {
            "total": len(all_docs["ids"]),
            "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        }

    def get_all(self, doc_type: str = None,
                offset: int = 0, limit: int = 50) -> list[dict]:
        """分页列出条目"""
        where_clause = None
        if doc_type:
            where_clause = {"doc_type": doc_type}

        results = self.collection.get(
            where=where_clause,
            offset=offset,
            limit=limit,
        )
        items = []
        if results["ids"]:
            for i, id_ in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                raw_tags = meta.get("tags", "[]")
                try:
                    tags = json.loads(raw_tags) if raw_tags else []
                except (json.JSONDecodeError, TypeError):
                    tags = []
                raw_meta_json = meta.get("metadata_json", "{}")
                try:
                    item_metadata = json.loads(raw_meta_json) if raw_meta_json else {}
                except (json.JSONDecodeError, TypeError):
                    item_metadata = {}
                items.append({
                    "id": id_,
                    "doc_type": meta.get("doc_type", ""),
                    "title": meta.get("title", ""),
                    "tags": tags,
                    "metadata": item_metadata,
                    "created_at": meta.get("created_at", ""),
                })
        return items

    def get_all_texts(self) -> list[dict]:
        """导出全量条目的原始文本和元数据（供 LightRAG 迁移用）"""
        all_docs = self.collection.get()
        if not all_docs["ids"]:
            return []
        results = []
        for i, id_ in enumerate(all_docs["ids"]):
            meta = all_docs["metadatas"][i]
            doc = all_docs["documents"][i] if all_docs["documents"] else ""
            results.append({
                "id": id_,
                "title": meta.get("title", ""),
                "doc_type": meta.get("doc_type", ""),
                "tags": meta.get("tags", "[]"),
                "text": doc,
            })
        return results

    def count(self) -> int:
        """快速获取总数"""
        all_docs = self.collection.get()
        return len(all_docs["ids"])

    def count_by_type(self, doc_type: str = None) -> int:
        """按文档类型统计条目数"""
        if not doc_type:
            return self.count()
        all_docs = self.collection.get(where={"doc_type": doc_type})
        return len(all_docs["ids"]) if all_docs and all_docs.get("ids") else 0


# ── 单例 ──

_engine_instance = None


def get_engine() -> VectorEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = VectorEngine()
    return _engine_instance
