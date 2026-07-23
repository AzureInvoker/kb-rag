"""
MCP 工具定义 + Handler — api 和 stdio server 共享复用

使用方式：
  from server.mcp_handler import TOOLS, handle_tool, async_handle_tool
"""

import json
import re
import logging
from datetime import datetime
from collections import Counter

from .models import KnowledgeItem

logger = logging.getLogger("mcp_handler")


# ── 输入清洗（共享） ──


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\\n", " ").replace("\\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clean_list(items: list) -> list:
    if not isinstance(items, list):
        return []
    return [_clean_text(i) for i in items if isinstance(i, str) and _clean_text(i)]


def _make_item(args: dict) -> KnowledgeItem:
    """从参数字典构造 KnowledgeItem"""
    title = _clean_text(args.get("title", ""))
    if not title:
        raise ValueError("标题不能为空")

    doc_type = _clean_text(args.get("doc_type", "doc")) or "doc"
    content = _clean_text(args.get("content", ""))

    # 解析 metadata（支持 dict 或 JSON 字符串）
    raw_meta = args.get("metadata", {})
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}
    if not isinstance(raw_meta, dict):
        raw_meta = {}

    return KnowledgeItem(
        title=title,
        doc_type=doc_type,
        content=content,
        metadata=raw_meta,
        tags=_clean_list(args.get("tags", [])),
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _format_item_table(item: dict) -> str:
    """格式化单条条目为 markdown 表格"""
    tags_str = ", ".join(item.get("tags", [])) if item.get("tags") else "-"
    meta_display = ""
    if item.get("metadata"):
        meta_display = "\n" + json.dumps(item["metadata"], ensure_ascii=False, indent=2)
    return (
        f"| ID | `{item['id']}` |\n"
        f"| 类型 | {item['doc_type']} |\n"
        f"| 标题 | {item['title']} |\n"
        f"| 标签 | {tags_str} |\n"
        f"| 创建时间 | {item.get('created_at', '-')} |\n"
        + (f"| metadata | {meta_display} |\n" if meta_display else "")
    )


# ── MCP 工具定义 ──

TOOLS = [
    {
        "name": "kb_search",
        "description": "基础语义搜索（ChromaDB + BM25 向量引擎）。\n【使用流程】① 不确定搜什么时先调 kb_stats 看有哪些 doc_type → ② 输入关键词搜索 → ③ 结果不够精准时加 doc_type 缩小范围 → ④ 需要跨文档关联时换 kb_agentic_search 或 kb_graph_search",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或自然语言描述"},
                "n_results": {"type": "number", "description": "返回结果数量（默认5，最多20）", "default": 5},
                "doc_type": {"type": "string", "description": "按文档类型筛选（如 test_case/doc/faq）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kb_list",
        "description": "浏览知识条目列表（支持按 doc_type 筛选和分页）。\n【使用流程】① 先调 kb_stats 看总览 → ② 用 kb_list 按 type 浏览 → ③ 看到感兴趣条目用 kb_get 看详情",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string", "description": "按文档类型筛选"},
                "offset": {"type": "number", "description": "分页偏移"},
                "limit": {"type": "number", "description": "每页数量（默认50，最多200）", "default": 50},
            },
            "required": [],
        },
    },
    {
        "name": "kb_get",
        "description": "按 ID 获取知识条目的完整内容（含 content + metadata 全部字段）。\n【使用场景】先调 kb_search / kb_list 找到目标条目的 ID，再用本工具看详情",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "条目 ID"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "kb_stats",
        "description": "获取知识库统计信息（总数、各文档类型分布）。\n【使用场景】① 第一次用先调此工具了解知识库规模 ② 确定有哪些 doc_type 后再做针对性搜索或添加",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "kb_add",
        "description": "添加单条知识条目。\n【新增字段说明】\n  - title：必填，条目标题\n  - doc_type：文档类型，可自定义（如 test_case/doc/faq/wiki），默认 doc\n  - content：正文内容（嵌入主要基于此字段，前800字）\n  - metadata：类型专属的灵活 JSON 字段。不同类型建议的字段：\n    · test_case → {\"module\":\"登录\", \"priority\":\"P0\", \"preconditions\":\"已登录\", \"expected\":\"跳转首页\"}\n    · doc       → {\"author\":\"张三\", \"source\":\"内部文档\", \"version\":\"1.0\"}\n    · faq       → {\"category\":\"账户问题\", \"answer\":\"具体回答\"}\n  - tags：标签列表\n【使用流程】先在 kb_stats 中确认是否存在目标 doc_type → 选类型填写添加",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题（必填）"},
                "doc_type": {"type": "string", "description": "文档类型，如 test_case/doc/faq（默认 doc）"},
                "content": {"type": "string", "description": "正文内容"},
                "metadata": {"type": "object", "description": "类型专属的灵活字段，如 {\"module\": \"登录\", \"priority\": \"P0\"}"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "kb_add_batch",
        "description": "批量添加知识条目（逐条清洗，单条失败不阻塞整体）。字段规则同 kb_add。返回成功/失败统计 + 新增条目 ID 列表。\n【使用场景】需要一次性录入多条同类型数据时使用",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "条目数组，每条清洗规则同 kb_add",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "标题（必填）"},
                            "doc_type": {"type": "string", "description": "文档类型（默认 doc）"},
                            "content": {"type": "string", "description": "正文"},
                            "metadata": {"type": "object", "description": "类型专属字段"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "标签"},
                        },
                        "required": ["title"],
                    },
                }
            },
            "required": ["items"],
        },
    },
    {
        "name": "kb_delete",
        "description": "删除知识条目。支持两种模式：按 ID 删除单条，或按 doc_type 批量删除。\n【使用场景】① 清理测试数据 ② 删除错误的录入 ③ 整批替换某类型数据\n注意：批量删除不可撤销，删除不同步清除 LightRAG 图谱中的对应实体",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "条目 ID（按 ID 删单条时填写）"},
                "doc_type": {"type": "string", "description": "文档类型，删除该类型下所有条目（批量模式）"},
            },
            "required": [],
        },
    },
    {
        "name": "kb_graph_search",
        "description": "【需 LightRAG 启用】知识图谱检索——通过实体-关系图做跨文档关联推理。\\n【适用场景】① 跨文档关联查询（如 XX模块关联哪些文档）② 多跳推理 ③ 概念关系发现\\n【使用流程】先调 kb_graph_status 确认图谱就绪 → 用本工具搜索 → 结果空洞时简化查询词\\n【注意】只返回实体和关系，不返回完整文档内容。想看详情用 kb_get",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "n_results": {"type": "number", "description": "返回结果数量（默认5，最多20）", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kb_agentic_search",
        "description": "【推荐】自适应检索——自动融合向量搜索 + 知识图谱增强。先走 ChromaDB 做语义匹配，再调用 LightRAG 图谱补充实体关系。\n【适用场景】① 复杂问题不确定怎么精确表达关键词 ② 需要同时看语义匹配和相关实体关系 ③ kb_search 首轮不够理想时的深入检索\n【使用流程】先用 kb_search 试 → 结果不够好时换本工具看有没有图谱增强信息 → 想看纯推理用 kb_graph_search\n【内容控制】summary_only=true 只显示摘要（200 字符），false（默认）显示完整内容（最多 4000 字符/条）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "n_results": {"type": "number", "description": "返回结果数量（默认5，最多20）", "default": 5},
                "doc_type": {"type": "string", "description": "按文档类型筛选"},
                "summary_only": {"type": "boolean", "description": "只显示摘要（200 字符）而非完整内容，节省上下文", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kb_graph_status",
        "description": "诊断 LightRAG 知识图谱状态：是否启用、是否已建图、实体数量、LLM 提供商、处理状态等。\n【使用场景】① kb_graph_search 无结果时先调此工具诊断 ② 确认图谱就绪后再做图谱检索 ③ 建图过程中查看处理进度",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mb_check",
        "description": "【脑记忆】查询某条路径的经验记录。输入路径签名（target|method|source|params 四级，越完整越精确），返回该路径及上级路径的探索历史、平均爽感值和推荐策略。"
            "用于 agent 在动手前先判断：这条路走过吗？结果是痛还是爽？应不应该再走一次？\n"
            "【返回说明】\n"
            "  - ✅ 推荐: avg_pleasure > 3，多次成功，放心走\n"
            "  - ⚡ 还行: avg_pleasure > 0，总体偏正面\n"
            "  - ⚠️ 谨慎: avg_pleasure ≈ 0，有成功有失败\n"
            "  - ❌ 避开: avg_pleasure < -3，多次失败，换方案\n"
            "  - 🔄 换方法: 目标下有痛的方法但有别的路子没试\n"
            "  - 🆕 没试过: 无记录，放心探索\n"
            "【格式】format=text（默认）返回人类可读文本；format=json 返回结构化 JSON 对象，便于程序化处理",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pathsig": {"type": "string", "description": "路径签名。格式 'target|method|source|params'。越完整越精确。"
                    " 也可以只有 target（查这个目标整体经验）。\n"
                    "  示例: '查天气|web_search|baidu' 或 '查天气|工具调用' 或 '查天气'"},
                "format": {"type": "string", "description": "输出格式。text（默认）= 人类可读文本；json = 结构化 JSON", "default": "text"},
            },
            "required": ["pathsig"],
        },
    },
    {
        "name": "mb_remember",
        "description": "【脑记忆】记录一条路径探索经验。告诉 agent 这条路走完后的感受和结果。\n"
            "【爽感值规则】\n"
            "  - 正值: 顺利、高效、得到想要的结果（+1~+10）\n"
            "  - 负值: 失败、踩坑、浪费 token（-1~-10）\n"
            "  - 绝对值越大越强烈\n"
            "【注意】已有路径会自动累计：tries+1、更新 avg_pleasure、更新 min/max、\n"
            "  成功次数/失败次数+1、可靠性 = 成功/总次数",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "目标（必填），如 '查天气'、'搜公司官网'、'写测试用例'"},
                "method": {"type": "string", "description": "方法（推荐），如 'web_search'、'tool_call'、'文件读取'"},
                "source": {"type": "string", "description": "工具/来源（可选），如 'baidu'、'bing'、'curl'"},
                "params": {"type": "string", "description": "参数/配置（可选），如 'lang=zh'、'timeout=30'"},
                "pleasure": {"type": "number", "description": "爽感值（必填），-10 ~ +10"},
                "note": {"type": "string", "description": "经验描述（必填），记录当时发生了什么、为什么爽/痛"},
            },
            "required": ["target", "pleasure", "note"],
        },
    },
    {
        "name": "mb_avoid",
        "description": "【脑记忆】标记某条路径为疼痛路径。等价于 mb_remember + pleasure=-8，快捷方式。\n"
            "用于 agent 明确知道这条路不该再走时快速标记。\n"
            "【注意】如果只是普通失败请用 mb_remember，mb_avoid 默认给强力负分。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "目标（必填）"},
                "method": {"type": "string", "description": "方法（推荐）"},
                "source": {"type": "string", "description": "工具/来源（可选）"},
                "params": {"type": "string", "description": "参数/配置（可选）"},
                "reason": {"type": "string", "description": "避开原因（必填）"},
            },
            "required": ["target", "reason"],
        },
    },
    {
        "name": "mb_prune",
        "description": "【脑记忆】清理和合并脑记忆记录。自动执行三项任务：\n"
            "1. 去重：完全相同的 pathsig → 合并统计后保留一条（保留最新 updated_at）\n"
            "2. 归档：avg_pleasure < -5 且超过 90 天未更新的 → 删除\n"
            "3. 矛盾合并：同 target 下多条矛盾的记录 → 保留 tries 最高的那条\n"
            "【注意】本工具会实际修改数据，建议先用 mb_check 确认状态后再调用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "description": "设为 true 时只预览不实际删除（默认 true，安全的预览模式）", "default": True},
            },
        },
    },
]


# ── 同步 Handler ──


def handle_tool(name: str, args: dict, engine, lightrag_engine, mem_engine=None) -> dict:
    """同步 MCP 工具处理函数"""
    # ── kb_search ──
    if name == "kb_search":
        query = args.get("query", "").strip()
        if not query:
            return {"content": [{"type": "text", "text": "请提供搜索关键词"}]}
        results = engine.search(
            query=query,
            n_results=min(int(args.get("n_results", 5)), 20),
            doc_type=args.get("doc_type"),
        )
        if not results:
            return {"content": [{"type": "text", "text": f"未找到与「{query}」相关的条目"}]}
        text = f"## 🔍 搜索「{query}」共找到 {len(results)} 条\n\n"
        for r in results:
            score_bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
            tags_str = f" [{', '.join(r['tags'])}]" if r.get("tags") else ""
            text += (
                f"### {r['title']}  [{score_bar}] {r['score']:.2f}\n\n"
                f"| 字段 | 值 |\n"
                f"|------|-----|\n"
                f"| ID | `{r['id']}` |\n"
                f"| 类型 | {r['doc_type']} |\n"
                f"| 标签 | {tags_str or '-'} |\n"
                f"\n摘要: {r.get('summary', '')}\n\n---\n\n"
            )
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── kb_list ──
    elif name == "kb_list":
        items = engine.get_all(
            doc_type=args.get("doc_type"),
            offset=int(args.get("offset", 0)),
            limit=min(int(args.get("limit", 50)), 200),
        )
        if not items:
            return {"content": [{"type": "text", "text": "暂无条目（或筛选条件无匹配）"}]}
        text = f"## 📋 条目列表（共 {len(items)} 条）\n\n"
        for i, item in enumerate(items):
            tags_str = f" [{', '.join(item['tags'])}]" if item.get("tags") else ""
            text += f"{i+1}. **{item['title']}**\n"
            text += f"   `{item['id']}` | {item['doc_type']}{tags_str}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── kb_get ──
    elif name == "kb_get":
        item = engine.get_by_id(args.get("id", ""))
        if not item:
            return {"content": [{"type": "text", "text": f"❌ 条目 {args.get('id')} 不存在"}]}
        text = f"# {item['title']}\n\n{_format_item_table(item)}\n\n{item.get('content', '')}"
        return {"content": [{"type": "text", "text": text}]}

    # ── kb_stats ──
    elif name == "kb_stats":
        stats = engine.get_stats()
        text = f"## 📊 知识库统计\n\n总条目数: {stats['total']}\n\n"
        if stats.get("by_type"):
            text += "### 按类型\n\n"
            for dt, count in stats["by_type"].items():
                bar = "█" * count + "░" * max(0, min(40 - count, 40))
                text += f"- {dt}: {count} 条  {bar}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── kb_add ──
    elif name == "kb_add":
        try:
            item = _make_item(args)
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"❌ {e}"}]}
        item.id = item.gen_id()
        engine.add(item)
        if lightrag_engine.is_available():
            lightrag_engine.insert([item.get_embedding_text()], ids=[item.id])
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"✅ 条目已添加\n\n"
                    f"{_format_item_table(item.to_dict())}"
                    f"\n可用 kb_get 传入 ID `{item.id}` 查看详情"
                ),
            }]
        }

    # ── kb_add_batch ──
    elif name == "kb_add_batch":
        raw_items = args.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            return {"content": [{"type": "text", "text": "❌ items 必须是数组"}]}
        added = []
        errors = []
        for i, c in enumerate(raw_items):
            try:
                if not isinstance(c, dict):
                    errors.append(f"第 {i+1} 条：参数格式错误")
                    continue
                item = _make_item(c)
                item.id = item.gen_id()
                engine.add(item)
                added.append(item)
            except ValueError as e:
                errors.append(f"第 {i+1} 条：{e}")
        if added and lightrag_engine.is_available():
            texts = [it.get_embedding_text() for it in added]
            ids = [it.id for it in added]
            lightrag_engine.insert(texts, ids=ids)
        summary = f"✅ 成功添加 {len(added)} 条"
        if errors:
            summary += f"，{len(errors)} 条失败:\n" + "\n".join(errors)
        if added:
            types = Counter(it.doc_type for it in added)
            summary += "\n\n**按类型分布:**\n"
            for dt, count in types.most_common():
                summary += f"- {dt}: {count} 条\n"
            summary += "\n**新增条目 ID:**\n"
            for it in added[:10]:
                summary += f"- `{it.id}` — {it.title}\n"
            if len(added) > 10:
                summary += f"  ... 还有 {len(added) - 10} 条\n"
        return {"content": [{"type": "text", "text": summary}]}

    # ── kb_delete ──
    elif name == "kb_delete":
        item_id = args.get("id", "").strip()
        doc_type = args.get("doc_type", "").strip()
        if item_id:
            ok = engine.delete(item_id)
            if ok:
                return {"content": [{"type": "text", "text": f"✅ 条目 `{item_id}` 已删除"}]}
            else:
                return {"content": [{"type": "text", "text": f"❌ 条目 `{item_id}` 不存在"}]}
        elif doc_type:
            count = engine.delete_many(doc_type=doc_type)
            return {"content": [{"type": "text", "text": f"✅ 已删除 {count} 条（类型={doc_type}）"}]}
        else:
            return {"content": [{"type": "text", "text": "❌ 请提供 id（删单条）或 doc_type（批量删除）"}]}

    # ── kb_graph_search ──
    elif name == "kb_graph_search":
        if not lightrag_engine.is_available():
            return {"content": [{"type": "text", "text": "❌ LightRAG 图谱未启用或初始化失败。可调 kb_graph_status 查看详情"}]}
        query = args.get("query", "").strip()
        if not query:
            return {"content": [{"type": "text", "text": "请提供搜索关键词"}]}
        result = lightrag_engine.search(query, n_results=min(int(args.get("n_results", 5)), 20))
        if not result.get("ok"):
            return {"content": [{"type": "text", "text": f"❌ 图谱检索失败: {result.get('message', '')}"}]}
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        text = f"## 🕸️ 知识图谱检索「{query}」\n\n共找到 {len(entities)} 个实体, {len(relationships)} 条关系\n\n"
        if entities:
            text += "### 📍 实体\n\n"
            for e in entities:
                text += f"- **{e['name']}**（{e.get('type', '-')}）\n  {e.get('description', '')[:150]}\n"
        if relationships:
            text += "\n### 🔗 关系\n\n"
            for r in relationships[:10]:
                text += f"- {r['source']} → {r['target']}: {r.get('description', '')[:100]}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── kb_agentic_search ──
    elif name == "kb_agentic_search":
        query = args.get("query", "").strip()
        if not query:
            return {"content": [{"type": "text", "text": "请提供搜索关键词"}]}
        n_results = min(int(args.get("n_results", 5)), 20)
        summary_only = args.get("summary_only", False)
        chroma_results = engine.search(
            query=query,
            n_results=n_results,
            doc_type=args.get("doc_type"),
        )
        text = f"## 🔍 自适应检索「{query}」\n\n"
        if chroma_results:
            text += f"### 📋 向量匹配结果（{len(chroma_results)} 条）\n\n"
            for r in chroma_results:
                text += f"**{r['title']}** [{r['score']:.2f}]\n`{r['id']}` | {r['doc_type']}\n"
                if summary_only:
                    text += f"> 摘要: {r.get('summary', '')}\n\n"
                else:
                    # 获取完整内容（从 summary 已有 200 字符，想要完整内容需要单独拉）
                    content = r.get("summary", "")
                    # 尝试从 engine 获取完整内容
                    full = engine.get_by_id(r["id"])
                    if full and full.get("content"):
                        content = full["content"]
                    if len(content) > 4000:
                        content = content[:4000] + "\n... [内容截断，超出 4000 字符]"
                    text += f"> {content}\n\n"
        else:
            text += "无可用的向量搜索结果\n\n"
        if lightrag_engine.is_available():
            graph_result = lightrag_engine.search(query, n_results)
            if graph_result.get("ok") and graph_result.get("entities"):
                text += f"### 🕸️ 图谱增强（{len(graph_result['entities'])} 实体）\n"
                for e in graph_result["entities"][:5]:
                    text += f"- {e['name']}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── kb_graph_status ──
    elif name == "kb_graph_status":
        status = lightrag_engine.get_status()
        text = "## 📊 LightRAG 状态\n\n"
        text += f"| 字段 | 值 |\n|------|-----|\n"
        text += f"| 启用 | {'✅ 是' if status.get('enabled') else '❌ 否'} |\n"
        text += f"| 就绪 | {'✅ 是' if status.get('ready') else '❌ 否'} |\n"
        text += f"| LLM 提供商 | {status.get('provider', '-')} |\n"
        text += f"| LLM 模型 | {status.get('model', '-')} |\n"
        if status.get("node_count") is not None:
            text += f"| 实体数量 | {status['node_count']} |\n"
        if status.get("processing_status"):
            text += f"| 处理状态 | {status['processing_status']} |\n"
        if status.get("message"):
            text += f"| 消息 | {status['message']} |\\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── 脑记忆: mb_check ──
    elif name == "mb_check":
        raw_pathsig = _clean_text(args.get("pathsig", ""))
        output_format = args.get("format", "text")
        if not raw_pathsig:
            return {"content": [{"type": "text", "text": "请提供路径签名（pathsig）"}]}
        # 归一化：统一 "target|method" → "target | method"
        norm_parts = [p.strip() for p in raw_pathsig.split("|")]
        pathsig = " | ".join(norm_parts)
        target = norm_parts[0]

        # 查询所有 brain_memory（精确匹配 + target 级别前缀匹配）
        exact_results = (mem_engine or engine).collection.get(
            where={"$and": [{"doc_type": "brain_memory"}, {"title": pathsig}]}
        )
        # 全量 brain_memory（单条件不用 $and）
        target_results = (mem_engine or engine).collection.get(
            where={"doc_type": "brain_memory"}
        )

        # 构建全部匹配列表
        seen = set()
        all_results = []

        def _fmt_exact(ids, metas, docs):
            items = []
            for i, eid in enumerate(ids):
                m = metas[i] if metas else {}
                doc = docs[i] if docs else ""
                items.append({
                    "id": eid,
                    "title": m.get("title", ""),
                    "pathsig": m.get("title", ""),
                    "content": doc or "",
                    "pleasure": m.get("pleasure", 0),
                    "avg_pleasure": m.get("avg_pleasure", 0),
                    "tries": m.get("tries", 1),
                    "min_pleasure": m.get("min_pleasure", 0),
                    "max_pleasure": m.get("max_pleasure", 0),
                    "success_count": m.get("success_count", 0),
                    "fail_count": m.get("fail_count", 0),
                    "reliability": m.get("reliability", 0),
                    "updated_at": m.get("updated_at", ""),
                    "metadata": {
                        "target": m.get("target", ""),
                        "method": m.get("method", ""),
                        "source": m.get("source", ""),
                        "params": m.get("params", ""),
                    },
                })
            return items

        # 精确匹配
        if exact_results and exact_results["ids"]:
            all_results.extend(_fmt_exact(
                exact_results["ids"], exact_results["metadatas"], exact_results["documents"]
            ))
            seen.update(exact_results["ids"])

        # target 前缀匹配（从全量中筛选 title 以 target 开头的）
        if target_results and target_results["ids"]:
            for i, eid in enumerate(target_results["ids"]):
                if eid in seen:
                    continue
                m = target_results["metadatas"][i] if target_results["metadatas"] else {}
                title = m.get("title", "")
                if title.startswith(target):
                    doc = target_results["documents"][i] if target_results["documents"] else ""
                    all_results.append({
                        "id": eid,
                        "title": title,
                        "pathsig": title,
                        "content": doc or "",
                        "pleasure": m.get("pleasure", 0),
                        "avg_pleasure": m.get("avg_pleasure", 0),
                        "tries": m.get("tries", 1),
                        "min_pleasure": m.get("min_pleasure", 0),
                        "max_pleasure": m.get("max_pleasure", 0),
                        "success_count": m.get("success_count", 0),
                        "fail_count": m.get("fail_count", 0),
                        "reliability": m.get("reliability", 0),
                        "updated_at": m.get("updated_at", ""),
                        "metadata": {
                            "target": m.get("target", ""),
                            "method": m.get("method", ""),
                            "source": m.get("source", ""),
                            "params": m.get("params", ""),
                        },
                    })

        if not all_results:
            empty_result = {
                "pathsig": pathsig,
                "found": False,
                "recommendation": "🆕 没试过",
                "exact_hit": None,
                "related_hits": [],
                "stats": {"total_records": 0, "avg_pleasure": 0, "success_count": 0, "pain_count": 0},
            }
            if output_format == "json":
                return {"content": [{"type": "text", "text": json.dumps(empty_result, ensure_ascii=False, indent=2)}]}
            return {"content": [{"type": "text", "text": f"🆕 **没试过**: 「{pathsig}」无探索记录\n\n放心尝试，或先在 mb_check 中搜索是否有相近经验"}]}

        # 计算推荐
        exact_hit = next((r for r in all_results if r["pathsig"] == pathsig), None)
        target_hits = [r for r in all_results if r["metadata"]["target"] == target]

        total_pleasure = sum(r["pleasure"] for r in target_hits)
        pain_count = sum(1 for r in target_hits if r["pleasure"] < 0)
        success_count = sum(1 for r in target_hits if r["pleasure"] >= 0)
        avg_p = total_pleasure / len(target_hits) if target_hits else 0

        has_other_methods = any(
            r["pathsig"].startswith(target + "|") and r["pathsig"] != pathsig
            for r in all_results
        )

        if avg_p > 3 and success_count >= pain_count:
            recommendation = "✅ 推荐"
        elif avg_p > 0:
            recommendation = "⚡ 还行"
        elif avg_p > -3:
            recommendation = "⚠️ 谨慎"
        else:
            recommendation = "❌ 避开"
        if pain_count > success_count and has_other_methods:
            recommendation += " 🔄 换方法"

        if output_format == "json":
            result = {
                "pathsig": pathsig,
                "found": True,
                "recommendation": recommendation,
                "exact_hit": exact_hit,
                "related_hits": [r for r in all_results if r["pathsig"] != pathsig],
                "stats": {
                    "total_records": len(all_results),
                    "target_total": len(target_hits),
                    "avg_pleasure": round(avg_p, 2),
                    "success_count": success_count,
                    "pain_count": pain_count,
                    "has_other_methods": has_other_methods,
                },
            }
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]}

        # ── text 格式输出（保持原有可读风格） ──
        text = f"## 🧠 脑记忆查询: {pathsig}\n\n"
        text += f"共找到 {len(all_results)} 条相关记录\n\n"

        # 推荐策略
        display_recommendation = recommendation.replace("✅", "✅ **").replace("⚡", "⚡ **").replace("⚠️", "⚠️ **").replace("❌", "❌ **")
        if "换方法" in recommendation:
            display_recommendation = display_recommendation.replace(" 🔄 换方法", "**\n🔄 **换方法** — 当前路径痛，但目标下有其他路子没试")
        else:
            display_recommendation += "**"

        text += f"**推荐**: {display_recommendation}\n\n"
        text += f"**整体统计**: 总记录 {len(target_hits)} 条 | 平均爽感 {avg_p:.1f} | 😊 {success_count}  | 😣 {pain_count}\n\n"
        text += "### 📋 详细记录\n\n"

        # 按 pathsig 排序展示
        for r in sorted(all_results, key=lambda x: -x["tries"]):
            ap = r["avg_pleasure"]
            tr = r["tries"]
            text += f"**{r['pathsig']}**"
            if ap > 0:
                text += " 😊"
            elif ap < 0:
                text += " 😣"
            text += f" | 爽感 {ap:.1f} | 尝试 {tr} 次"
            if r.get("updated_at"):
                text += f" | 最后更新 {r['updated_at']}"
            text += "\n"
            note = r.get("content", "")[:200]
            if note:
                text += f"> {note}\n"
            text += "\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    # ── 脑记忆: mb_remember (upsert) ──
    elif name == "mb_remember":
        target = _clean_text(args.get("target", ""))
        if not target:
            return {"content": [{"type": "text", "text": "❌ target（目标）不能为空"}]}
        method = _clean_text(args.get("method", ""))
        source = _clean_text(args.get("source", ""))
        params = _clean_text(args.get("params", ""))
        try:
            pleasure = int(args.get("pleasure", 0))
        except (ValueError, TypeError):
            pleasure = 0
        pleasure = max(-10, min(10, pleasure))
        note = _clean_text(args.get("note", ""))

        # 构建路径签名作为 title
        sig_parts = [target]
        if method: sig_parts.append(method)
        if source: sig_parts.append(source)
        if params: sig_parts.append(params)
        pathsig = " | ".join(sig_parts)

        # 精确匹配 pathsig（用 ChromaDB 的 where 过滤，不用语义搜索）
        exact_matches = (mem_engine or engine).collection.get(
            where={"$and": [{"doc_type": "brain_memory"}, {"title": pathsig}]}
        )
        existing_ids = exact_matches["ids"] if exact_matches and exact_matches["ids"] else []

        if existing_ids:
            # ── 有重复时合并：保留最新元数据，汇总 tries/success/fail ──
            all_metas = []
            for i, eid in enumerate(existing_ids):
                em = exact_matches["metadatas"][i] if exact_matches["metadatas"] else {}
                all_metas.append(em)

            # 合并统计
            total_tries = sum(m.get("tries", 1) for m in all_metas)
            total_success = sum(m.get("success_count", 0) for m in all_metas)
            total_fail = sum(m.get("fail_count", 0) for m in all_metas)
            all_avg_sum = sum(m.get("avg_pleasure", 0) * m.get("tries", 1) for m in all_metas)
            overall_min = min(m.get("min_pleasure", pleasure) for m in all_metas)
            overall_max = max(m.get("max_pleasure", pleasure) for m in all_metas)

            # 加上本次新值
            new_tries = total_tries + 1
            new_success = total_success + (1 if pleasure >= 0 else 0)
            new_fail = total_fail + (1 if pleasure < 0 else 0)
            new_avg = round((all_avg_sum + pleasure) / new_tries, 2)

            merged_meta = {
                "target": target,
                "method": method,
                "source": source,
                "params": params,
                "pleasure": pleasure,
                "last_pleasure": pleasure,
                "tries": new_tries,
                "avg_pleasure": new_avg,
                "min_pleasure": min(overall_min, pleasure),
                "max_pleasure": max(overall_max, pleasure),
                "success_count": new_success,
                "fail_count": new_fail,
                "reliability": round(new_success / new_tries, 2),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 删除重复记录（保留第一条，合并到它）
            keep_id = existing_ids[0]
            dup_ids = existing_ids[1:]
            if dup_ids:
                (mem_engine or engine).collection.delete(ids=dup_ids)

            # 更新笔记（新 note 覆盖旧 note）
            # 注意：不传 documents 避免触发 ChromaDB 重嵌入（降维不匹配问题）
            (mem_engine or engine).collection.update(
                ids=[keep_id],
                metadatas=[merged_meta],
            )
            return {"content": [{"type": "text", "text": (
                f"✅ 经验已更新: 「{pathsig}」\n"
                f"最新爽感: {pleasure:+d} | 累计探索 {new_tries} 次 | 平均爽感 {new_avg:.1f} | 可靠性 {merged_meta.get('reliability', 0):.0%}"
                + (f"\n🔄 合并了 {len(dup_ids)} 条重复记录" if dup_ids else "")
            )}]}
        else:
            # ── 新建记录 ──
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            item = KnowledgeItem(
                title=pathsig,
                doc_type="brain_memory",
                content=note,
                metadata={
                    "target": target,
                    "method": method,
                    "source": source,
                    "params": params,
                    "pleasure": pleasure,
                    "last_pleasure": pleasure,
                    "tries": 1,
                    "avg_pleasure": float(pleasure),
                    "min_pleasure": pleasure,
                    "max_pleasure": pleasure,
                    "success_count": 1 if pleasure >= 0 else 0,
                    "fail_count": 1 if pleasure < 0 else 0,
                    "reliability": 1.0 if pleasure >= 0 else 0.0,
                    "updated_at": now,
                },
                tags=[target, method] if method else [target],
                created_at=now,
            )
            item.id = item.gen_id()
            (mem_engine or engine).add(item)
            return {"content": [{"type": "text", "text": (
                f"✅ 经验已记录: 「{pathsig}」\n"
                f"爽感: {pleasure:+d} | 首次探索"
            )}]}

    # ── 脑记忆: mb_avoid ──
    elif name == "mb_avoid":
        # mb_avoid 内部调用 mb_remember 的逻辑，但 pleasure=-8
        args["pleasure"] = -8
        note_parts = ["🚫 标记为疼痛路径"]
        reason = _clean_text(args.get("reason", ""))
        if reason:
            note_parts.append(f"原因: {reason}")
        args["note"] = " | ".join(note_parts)
        return handle_tool("mb_remember", args, engine, lightrag_engine, mem_engine)

    # ── 脑记忆: mb_prune ──
    elif name == "mb_prune":
        dry_run = args.get("dry_run", True)

        # 拉取所有 brain_memory
        all_docs = (mem_engine or engine).collection.get(
            where={"doc_type": "brain_memory"}
        )
        if not all_docs or not all_docs["ids"]:
            return {"content": [{"type": "text", "text": "🧹 无脑记忆记录需要清理"}]}

        total_before = len(all_docs["ids"])

        # 按 title(pathsig) 分组
        from collections import defaultdict
        groups = defaultdict(list)
        for i, eid in enumerate(all_docs["ids"]):
            m = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            doc = all_docs["documents"][i] if all_docs["documents"] else ""
            groups[m.get("title", "unknown")].append({
                "id": eid,
                "meta": m,
                "doc": doc,
            })

        now_ts = datetime.now()
        to_delete = set()
        merge_log = []

        # 1. 去重：完全相同的 pathsig → 合并（保留 tries 最多的）
        for title, records in groups.items():
            if len(records) <= 1:
                continue
            best = max(records, key=lambda r: r["meta"].get("tries", 1))
            for r in records:
                if r["id"] != best["id"]:
                    to_delete.add(r["id"])
            merge_log.append(f"  「{title}」: 合并 {len(records)} 条 → 保留 `{best['id']}` (tries={best['meta'].get('tries', 1)})")

        # 2. 归档：avg_pleasure < -5 且超过 90 天未更新
        for title, records in groups.items():
            for r in records:
                if r["id"] in to_delete:
                    continue
                avg_p = r["meta"].get("avg_pleasure", 0)
                if avg_p >= -5:
                    continue
                updated_at = r["meta"].get("updated_at", "")
                if not updated_at:
                    continue
                try:
                    updated_ts = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
                    days_since = (now_ts - updated_ts).days
                    if days_since >= 90:
                        to_delete.add(r["id"])
                        merge_log.append(f"  🗑️ 归档「{title}」: avg_pleasure={avg_p:.1f}，{days_since} 天未更新")
                except ValueError:
                    pass

        total_to_delete = len(to_delete)

        if dry_run:
            msg = f"🧹 **mb_prune 预览**（dry_run=true，未实际删除）\n\n"
            msg += f"清理前: **{total_before}** 条\n"
            msg += f"将删除: **{total_to_delete}** 条\n"
            msg += f"预计剩余: **{total_before - total_to_delete}** 条\n\n"
            if merge_log:
                msg += "**操作明细:**\n\n" + "\n".join(merge_log)
            else:
                msg += "无需要清理的记录 🎉"
            return {"content": [{"type": "text", "text": msg}]}

        # 实际执行删除
        if to_delete:
            (mem_engine or engine).collection.delete(ids=list(to_delete))
            msg = f"🧹 **mb_prune 已完成**\n\n"
            msg += f"清理前: {total_before} 条\n"
            msg += f"已删除: {total_to_delete} 条\n"
            msg += f"当前剩余: {total_before - total_to_delete} 条\n\n"
            if merge_log:
                msg += "**操作明细:**\n\n" + "\n".join(merge_log)
        else:
            msg = "🧹 无需要清理的记录 🎉"

        return {"content": [{"type": "text", "text": msg}]}

    else:
        return {"content": [{"type": "text", "text": f"未知工具: {name}"}]}


# ── 异步 Handler（供 api.py 的异步 MCP handler 使用） ──


async def async_handle_tool(name: str, args: dict, engine, lightrag_engine, mem_engine=None) -> dict:
    """异步 MCP 工具处理函数"""
    # 图谱工具用 async
    if name in ("kb_graph_search", "kb_agentic_search", "kb_graph_status"):
        return await _async_graph_tool(name, args, engine, lightrag_engine)
    # 写入工具用 async_insert（避免在 async 上下文调用 asyncio.run()）
    if name == "kb_add":
        return await _async_add(args, engine, lightrag_engine)
    if name == "kb_add_batch":
        return await _async_add_batch(args, engine, lightrag_engine)
    # 其他工具直接走同步版（纯读取，不涉及 asyncio.run）
    return handle_tool(name, args, engine, lightrag_engine)


async def _async_add(args: dict, engine, lightrag_engine) -> dict:
    """异步添加单条（用 async_insert 避免 asyncio.run 崩溃）"""
    try:
        item = _make_item(args)
    except ValueError as e:
        return {"content": [{"type": "text", "text": f"❌ {e}"}]}
    item.id = item.gen_id()
    engine.add(item)
    if lightrag_engine.is_available():
        await lightrag_engine.async_insert([item.get_embedding_text()], ids=[item.id])
    return {
        "content": [{
            "type": "text",
            "text": (
                f"✅ 条目已添加\n\n"
                f"{_format_item_table(item.to_dict())}"
                f"\n可用 kb_get 传入 ID `{item.id}` 查看详情"
            ),
        }]
    }


async def _async_add_batch(args: dict, engine, lightrag_engine) -> dict:
    """异步批量添加（用 async_insert 避免 asyncio.run 崩溃）"""
    raw_items = args.get("items", [])
    if not isinstance(raw_items, list) or not raw_items:
        return {"content": [{"type": "text", "text": "❌ items 必须是数组"}]}
    added = []
    errors = []
    for i, c in enumerate(raw_items):
        try:
            if not isinstance(c, dict):
                errors.append(f"第 {i+1} 条：参数格式错误")
                continue
            item = _make_item(c)
            item.id = item.gen_id()
            engine.add(item)
            added.append(item)
        except ValueError as e:
            errors.append(f"第 {i+1} 条：{e}")
    if added and lightrag_engine.is_available():
        texts = [it.get_embedding_text() for it in added]
        ids = [it.id for it in added]
        await lightrag_engine.async_insert(texts, ids=ids)

    summary = f"✅ 成功添加 {len(added)} 条"
    if errors:
        summary += f"，{len(errors)} 条失败:\n" + "\n".join(errors)
    if added:
        types = Counter(it.doc_type for it in added)
        summary += "\n\n**按类型分布:**\n"
        for dt, count in types.most_common():
            summary += f"- {dt}: {count} 条\n"
        summary += "\n**新增条目 ID:**\n"
        for it in added[:10]:
            summary += f"- `{it.id}` — {it.title}\n"
        if len(added) > 10:
            summary += f"  ... 还有 {len(added) - 10} 条\n"
    return {"content": [{"type": "text", "text": summary}]}


async def _async_graph_tool(name: str, args: dict, engine, lightrag_engine) -> dict:
    """异步处理的图谱相关工具"""
    if name == "kb_graph_search":
        if not lightrag_engine.is_available():
            return {"content": [{"type": "text", "text": "❌ LightRAG 图谱未启用或初始化失败"}]}
        query = args.get("query", "").strip()
        if not query:
            return {"content": [{"type": "text", "text": "请提供搜索关键词"}]}
        result = await lightrag_engine.async_search(query, n_results=min(int(args.get("n_results", 5)), 20))
        if not result.get("ok"):
            return {"content": [{"type": "text", "text": f"❌ 图谱检索失败: {result.get('message', '')}"}]}
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        text = f"## 🕸️ 知识图谱检索「{query}」\n\n共找到 {len(entities)} 个实体, {len(relationships)} 条关系\n\n"
        if entities:
            text += "### 📍 实体\n\n"
            for e in entities:
                text += f"- **{e['name']}**（{e.get('type', '-')}）\n  {e.get('description', '')[:150]}\n"
        if relationships:
            text += "\n### 🔗 关系\n\n"
            for r in relationships[:10]:
                text += f"- {r['source']} → {r['target']}: {r.get('description', '')[:100]}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    elif name == "kb_agentic_search":
        query = args.get("query", "").strip()
        if not query:
            return {"content": [{"type": "text", "text": "请提供搜索关键词"}]}
        n_results = min(int(args.get("n_results", 5)), 20)
        summary_only = args.get("summary_only", False)
        chroma_results = engine.search(query=query, n_results=n_results, doc_type=args.get("doc_type"))
        text = f"## 🔍 自适应检索「{query}」\n\n"
        if chroma_results:
            text += f"### 📋 向量匹配结果（{len(chroma_results)} 条）\n\n"
            for r in chroma_results:
                text += f"**{r['title']}** [{r['score']:.2f}]\n`{r['id']}` | {r['doc_type']}\n"
                if summary_only:
                    text += f"> 摘要: {r.get('summary', '')}\n\n"
                else:
                    content = r.get("summary", "")
                    full = engine.get_by_id(r["id"])
                    if full and full.get("content"):
                        content = full["content"]
                    if len(content) > 4000:
                        content = content[:4000] + "\n... [内容截断，超出 4000 字符]"
                    text += f"> {content}\n\n"
        else:
            text += "无可用的向量搜索结果\n\n"
        if lightrag_engine.is_available():
            graph_result = await lightrag_engine.async_search(query, n_results)
            if graph_result.get("ok") and graph_result.get("entities"):
                text += f"### 🕸️ 图谱增强（{len(graph_result['entities'])} 实体）\n"
                for e in graph_result["entities"][:5]:
                    text += f"- {e['name']}\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    elif name == "kb_graph_status":
        status = lightrag_engine.get_status()
        text = "## 📊 LightRAG 状态\n\n"
        text += f"| 字段 | 值 |\n|------|-----|\n"
        text += f"| 启用 | {'✅ 是' if status.get('enabled') else '❌ 否'} |\n"
        text += f"| 就绪 | {'✅ 是' if status.get('ready') else '❌ 否'} |\n"
        text += f"| LLM 提供商 | {status.get('provider', '-')} |\n"
        text += f"| LLM 模型 | {status.get('model', '-')} |\n"
        if status.get("node_count") is not None:
            text += f"| 实体数量 | {status['node_count']} |\n"
        if status.get("processing_status"):
            text += f"| 处理状态 | {status['processing_status']} |\n"
        if status.get("message"):
            text += f"| 消息 | {status['message']} |\n"
        return {"content": [{"type": "text", "text": text.strip()}]}

    return {"content": [{"type": "text", "text": f"未知工具: {name}"}]}
