"""通用知识库数据模型"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class KnowledgeItem:
    """通用知识条目

    doc_type: 文档类型标识，如 "test_case" / "doc" / "faq" / "wiki"
    title:    标题
    content:  正文内容（嵌入主要基于此字段）
    metadata: 类型专属的字段，灵活 key-value 结构（存入前确保可 JSON 序列化）
    tags:     标签列表
    created_at: 创建时间（自动生成）
    """
    id: str = ""
    doc_type: str = "doc"
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    created_at: str = ""

    def get_embedding_text(self) -> str:
        """生成用于向量化的文本——只拼 title + content 前 800 字 + doc_type

        避免像旧版那样把十几个字段全拼进去稀释信号。
        """
        parts = [
            f"标题: {self.title}",
            f"类型: {self.doc_type}",
        ]
        if self.content:
            parts.append(f"内容: {self.content[:800]}")
        return "\n".join(p for p in parts if p)

    def get_bm25_text(self) -> str:
        """生成用于 BM25 关键词搜索的文本（聚焦高信号字段）"""
        parts = [self.title, self.doc_type]
        if self.tags:
            parts.append(", ".join(self.tags))
        return " ".join(p for p in parts if p)

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
        }
