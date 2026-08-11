"""Retrieval helpers — RRF fusion, chunking, optional reranking."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("retrieval")

RRF_K_DEFAULT = 60


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K_DEFAULT) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked ID lists."""
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def split_content(content: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split content into overlapping character-based chunks (Chinese-friendly)."""
    text = (content or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks or [text]


def parent_id_from_meta(meta: dict, hit_id: str) -> str:
    pid = (meta.get("parent_id") or "").strip()
    return pid if pid else hit_id


def dedupe_by_parent(results: list[dict]) -> list[dict]:
    """Keep the highest-scoring hit per logical parent document."""
    best: dict[str, dict] = {}
    for row in results:
        pid = row.get("parent_id") or row["id"]
        prev = best.get(pid)
        if prev is None or float(row["score"]) > float(prev["score"]):
            out = dict(row)
            out["parent_id"] = pid
            if row.get("is_chunk"):
                out["id"] = pid
            best[pid] = out
    return sorted(best.values(), key=lambda x: -float(x["score"]))


class Reranker:
    """Lazy-loaded CrossEncoder reranker."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("加载 Reranker 模型: %s", self._model_name)
        self._model = CrossEncoder(self._model_name, device="cpu")

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        self._lazy_init()
        pairs = [
            (query, f"{c.get('title', '')}\n{c.get('rerank_text', c.get('summary', ''))}"[:2000])
            for c in candidates
        ]
        scores = self._model.predict(pairs)
        ranked = []
        for i, cand in enumerate(candidates):
            row = dict(cand)
            row["score"] = round(float(scores[i]), 4)
            ranked.append(row)
        ranked.sort(key=lambda x: -float(x["score"]))
        return ranked[:top_k]
