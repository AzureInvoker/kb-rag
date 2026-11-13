"""通用知识库数据模型"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .taxonomy import (
    infer_source_quality,
    normalize_doc_type,
    normalize_governance_status,
    now_str,
)


@dataclass
class KnowledgeItem:
    """通用知识条目

    doc_type: doc / test_case / rule / ui_control
    title:    标题
    content:  正文内容（嵌入主要基于此字段）
    metadata: 类型专属的字段，灵活 key-value 结构
    tags:     标签列表
    """
    id: str = ""
    doc_type: str = "doc"
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    created_at: str = ""
    category: str = ""
    game: str = ""
    app: str = ""
    governance_status: str = "published"
    version: int = 1
    updated_at: str = ""
    next_verify_at: str = ""
    source_quality: str = "curated"

    def __post_init__(self):
        self.doc_type = normalize_doc_type(self.doc_type)
        self.governance_status = normalize_governance_status(self.governance_status)
        self.source_quality = infer_source_quality(self.tags, self.source_quality)
        if not self.created_at:
            self.created_at = now_str()
        if not self.updated_at:
            self.updated_at = self.created_at
        # 顶层与 metadata 对齐常用维度
        meta = self.metadata if isinstance(self.metadata, dict) else {}
        if not self.app and meta.get("app"):
            self.app = str(meta.get("app") or "").strip()
        if not self.game and meta.get("game"):
            self.game = str(meta.get("game") or "").strip()
        if not self.category and meta.get("category"):
            self.category = str(meta.get("category") or "").strip()
        if self.app and not meta.get("app"):
            meta["app"] = self.app
        if self.game and not meta.get("game"):
            meta["game"] = self.game
        if self.category and not meta.get("category"):
            meta["category"] = self.category
        self.metadata = meta

    def get_embedding_text(self) -> str:
        """生成用于向量化的文本。"""
        if self.doc_type == "ui_control":
            meta = self.metadata or {}
            parts = [
                f"标题: {self.title}",
                f"类型: {self.doc_type}",
                f"应用: {self.app or meta.get('app', '')}",
                f"页面: {meta.get('screen', '')}",
                f"控件ID: {meta.get('control_id', '')}",
                f"文案: {meta.get('name') or meta.get('text') or ''}",
            ]
            if self.game:
                parts.append(f"游戏: {self.game}")
            if self.content:
                parts.append(f"内容: {self.content[:400]}")
            return "\n".join(p for p in parts if p and not p.endswith(": "))

        parts = [
            f"标题: {self.title}",
            f"类型: {self.doc_type}",
        ]
        if self.content:
            parts.append(f"内容: {self.content[:800]}")
        return "\n".join(p for p in parts if p)

    def get_bm25_text(self, *, include_content: bool = True) -> str:
        """生成用于 BM25 关键词搜索的文本。"""
        parts = [self.title, self.doc_type]
        if self.app:
            parts.append(self.app)
        if self.game:
            parts.append(self.game)
        if self.category:
            parts.append(self.category)
        if self.tags:
            parts.append(", ".join(self.tags))
        if self.doc_type == "ui_control" and self.metadata:
            for key in ("screen", "control_id", "name", "text", "class_name", "path"):
                val = self.metadata.get(key)
                if val:
                    parts.append(str(val))
        if include_content and self.content:
            parts.append(self.content)
        return " ".join(p for p in parts if p)

    def get_lightrag_text(self, *, max_chars: int = 12000) -> str:
        """LightRAG 建图用全文（比 get_embedding_text 包含更多正文）。"""
        parts = [
            f"标题: {self.title}",
            f"类型: {self.doc_type}",
        ]
        if self.tags:
            parts.append(f"标签: {', '.join(self.tags)}")
        if self.content:
            parts.append(f"内容: {self.content[:max_chars]}")
        return "\n".join(p for p in parts if p)

    def gen_id(self) -> str:
        """基于内容生成唯一 ID"""
        raw = f"{self.doc_type}:{self.title}:{self.content[:100]}:{self.created_at}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_type": self.doc_type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "category": self.category,
            "game": self.game,
            "app": self.app,
            "governance_status": self.governance_status,
            "version": self.version,
            "updated_at": self.updated_at,
            "next_verify_at": self.next_verify_at,
            "source_quality": self.source_quality,
        }
