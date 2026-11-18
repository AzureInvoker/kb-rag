"""知识库类型与治理常量。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

DOC_TYPES = ("doc", "test_case", "rule", "ui_control")

DOC_TYPE_ALIASES = {
    "wiki": "doc",
    "faq": "doc",
    "document": "doc",
    "testcase": "test_case",
    "test": "test_case",
    "规范": "rule",
    "rule_doc": "rule",
}

GOVERNANCE_STATUS = ("published", "stale", "deprecated", "archived")
ACTIVE_STATUSES = ("published", "stale")
RETIRED_STATUSES = ("deprecated", "archived")

SOURCE_QUALITIES = ("curated", "auto_curated")

# 默认可检索 / 建图类型
GRAPH_ELIGIBLE_TYPES = frozenset({"doc", "test_case", "rule"})
NO_CHUNK_TYPES = frozenset({"brain_memory", "ui_control"})

# ui_control metadata 常用键（均为可选，除 app 在顶层必填）
UI_CONTROL_META_KEYS = (
    "screen",
    "control_id",
    "name",
    "text",
    "class_name",
    "path",
    "bounds",
    "parent_control_id",  # 父级控件 id
    "purpose",            # 节点作用
    "page_id",            # 所在页面 id
    "page_type",          # native | h5
)

PAGE_TYPES = ("native", "h5")
PAGE_TYPE_ALIASES = {
    "native": "native",
    "android": "native",
    "ios": "native",
    "rn": "native",
    "flutter": "native",
    "h5": "h5",
    "web": "h5",
    "html": "h5",
    "webview": "h5",
}

DEFAULT_SLA_DAYS = {
    "rule": 60,
    "test_case": 90,
    "doc": 120,
    "ui_control": 60,
}

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def normalize_doc_type(raw: Any, *, strict: bool = False) -> str:
    """规范化 doc_type。未知类型默认映射为 doc；strict 时抛 ValueError。"""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "doc"
    key = str(raw).strip().lower()
    if key in DOC_TYPES:
        return key
    alias = DOC_TYPE_ALIASES.get(key) or DOC_TYPE_ALIASES.get(str(raw).strip())
    if alias:
        return alias
    if strict:
        raise ValueError(f"不支持的 doc_type: {raw}，允许: {', '.join(DOC_TYPES)}")
    return "doc"


def normalize_governance_status(raw: Any, *, default: str = "published") -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    key = str(raw).strip().lower()
    if key in GOVERNANCE_STATUS:
        return key
    return default


def infer_source_quality(tags: Optional[list] = None, explicit: Any = None) -> str:
    if explicit in SOURCE_QUALITIES:
        return str(explicit)
    tags = tags or []
    if any(str(t).strip().lower() == "auto_curated" for t in tags):
        return "auto_curated"
    return "curated"


def should_index_graph(doc_type: str) -> bool:
    return normalize_doc_type(doc_type) in GRAPH_ELIGIBLE_TYPES


def should_skip_chunking(doc_type: str) -> bool:
    return normalize_doc_type(doc_type) in NO_CHUNK_TYPES or doc_type == "brain_memory"


def now_str() -> str:
    return datetime.now().strftime(TIME_FMT)


def compute_next_verify_at(doc_type: str, sla_days: Optional[dict] = None, *, now: Optional[datetime] = None) -> str:
    dt = normalize_doc_type(doc_type)
    table = sla_days or DEFAULT_SLA_DAYS
    days = int(table.get(dt, table.get("doc", 120)))
    base = now or datetime.now()
    return (base + timedelta(days=days)).strftime(TIME_FMT)


def require_ui_control_app(doc_type: str, app: Any, metadata: Optional[dict] = None) -> str:
    """ui_control 必须有 app；返回规范化 app 字符串。"""
    dt = normalize_doc_type(doc_type)
    if dt != "ui_control":
        return (str(app).strip() if app else "") or str((metadata or {}).get("app") or "").strip()
    value = (str(app).strip() if app else "") or str((metadata or {}).get("app") or "").strip()
    if not value:
        raise ValueError("ui_control 必须提供 app（应用包名或产品名）")
    return value


def normalize_page_type(raw: Any) -> str:
    """规范化页面类型为 native/h5；空或未知返回空串（可选字段）。"""
    if raw is None:
        return ""
    key = str(raw).strip().lower()
    if not key:
        return ""
    if key in PAGE_TYPES:
        return key
    return PAGE_TYPE_ALIASES.get(key, "")


def normalize_ui_control_metadata(metadata: Optional[dict] = None) -> dict:
    """规范化 ui_control metadata：补齐别名、清洗可选字段。

    可选字段：
    - parent_control_id: 父级控件 id（兼容 parent_id / parent_node_id）
    - purpose: 节点作用（兼容 node_purpose / role）
    - page_id: 所在页面 id
    - page_type: native | h5
    """
    meta = dict(metadata) if isinstance(metadata, dict) else {}

    def _pick(*keys: str) -> str:
        for k in keys:
            val = meta.get(k)
            if val is None:
                continue
            s = str(val).strip()
            if s:
                return s
        return ""

    parent = _pick("parent_control_id", "parent_id", "parent_node_id", "parentControlId")
    purpose = _pick("purpose", "node_purpose", "role", "action")
    page_id = _pick("page_id", "pageId")
    page_type = normalize_page_type(_pick("page_type", "pageType", "page_kind"))

    if parent:
        meta["parent_control_id"] = parent
    if purpose:
        meta["purpose"] = purpose
    if page_id:
        meta["page_id"] = page_id
    if page_type:
        meta["page_type"] = page_type

    # 去掉纯空白字符串，避免污染检索
    for key in ("parent_control_id", "purpose", "page_id", "page_type",
                "screen", "control_id", "name", "text", "class_name", "path"):
        if key in meta and isinstance(meta[key], str) and not meta[key].strip():
            meta.pop(key, None)

    return meta
