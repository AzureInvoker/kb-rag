"""kb-rag 核心单元测试"""

import sys
import json
import tempfile
import os
from pathlib import Path

# 添加项目根
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

# ── 1. 数据模型测试 ──


def test_knowledge_item_basic():
    """测试 KnowledgeItem 基本创建和属性"""
    from server.models import KnowledgeItem

    item = KnowledgeItem(
        title="用户登录测试",
        doc_type="test_case",
        content="验证用户输入正确用户名密码后能成功登录",
        metadata={"module": "登录", "priority": "P0"},
        tags=["冒烟测试", "核心功能"],
    )
    assert item.title == "用户登录测试"
    assert item.doc_type == "test_case"
    assert item.content == "验证用户输入正确用户名密码后能成功登录"
    assert item.metadata == {"module": "登录", "priority": "P0"}
    assert item.tags == ["冒烟测试", "核心功能"]
    print("  ✅ test_knowledge_item_basic")


def test_knowledge_item_gen_id():
    """测试 ID 生成：相同内容生成相同 ID"""
    from server.models import KnowledgeItem

    item1 = KnowledgeItem(title="搜索测试", doc_type="test_case", created_at="2025-01-01")
    item2 = KnowledgeItem(title="搜索测试", doc_type="test_case", created_at="2025-01-01")
    item3 = KnowledgeItem(title="搜索测试", doc_type="doc", created_at="2025-01-01")

    assert item1.gen_id() == item2.gen_id(), "相同内容应生成相同 ID"
    assert item1.gen_id() != item3.gen_id(), "不同 doc_type 应生成不同 ID"
    print("  ✅ test_knowledge_item_gen_id")


def test_knowledge_item_embedding_text():
    """测试嵌入文本生成：只包含标题+类型+内容前800字"""
    from server.models import KnowledgeItem

    item = KnowledgeItem(
        title="测试标题",
        doc_type="test_case",
        content="A" * 1000,  # 超长内容
    )
    text = item.get_embedding_text()
    assert "标题: 测试标题" in text
    assert "类型: test_case" in text
    assert "内容: " in text
    assert len(text) < 900  # 标题~20 + 类型~20 + 内容800 ≈ 840
    print("  ✅ test_knowledge_item_embedding_text")


def test_knowledge_item_bm25_text():
    """测试 BM25 文本：拼接标题+类型+标签"""
    from server.models import KnowledgeItem

    item = KnowledgeItem(title="登录", doc_type="test_case", tags=["冒烟", "核心"])
    text = item.get_bm25_text()
    assert "登录" in text
    assert "test_case" in text
    assert "冒烟" in text
    assert "核心" in text
    print("  ✅ test_knowledge_item_bm25_text")


def test_knowledge_item_to_dict():
    """测试 to_dict 输出"""
    from server.models import KnowledgeItem

    item = KnowledgeItem(
        id="abc123", title="测试", doc_type="test_case",
        content="内容", metadata={"k": "v"}, tags=["t1"],
    )
    d = item.to_dict()
    assert d["id"] == "abc123"
    assert d["title"] == "测试"
    assert d["metadata"]["k"] == "v"
    assert d["tags"] == ["t1"]
    print("  ✅ test_knowledge_item_to_dict")


# ── 2. MCP Handler 工具测试 ──


def test_clean_text():
    """测试文本清洗"""
    from server.mcp_handler import _clean_text

    assert _clean_text("  hello  ") == "hello"
    assert _clean_text("hello\\nworld") == "hello world"
    assert _clean_text("") == ""
    assert _clean_text(None) == ""
    assert _clean_text(123) == ""
    print("  ✅ test_clean_text")


def test_clean_list():
    """测试列表清洗"""
    from server.mcp_handler import _clean_list

    assert _clean_list([" a ", "", "b", None, " c "]) == ["a", "b", "c"]
    assert _clean_list(None) == []
    assert _clean_list([]) == []
    print("  ✅ test_clean_list")


def test_make_item_basic():
    """测试从参数字典构造 KnowledgeItem"""
    from server.mcp_handler import _make_item

    item = _make_item({
        "title": " 登录测试 ",
        "doc_type": "test_case",
        "content": " 验证登录 ",
        "metadata": {"module": "登录", "priority": "P0"},
        "tags": ["冒烟"],
    })
    assert item.title == "登录测试"
    assert item.doc_type == "test_case"
    assert item.content == "验证登录"
    assert item.metadata["module"] == "登录"
    assert item.tags == ["冒烟"]
    print("  ✅ test_make_item_basic")


def test_make_item_defaults():
    """测试默认值"""
    from server.mcp_handler import _make_item

    item = _make_item({"title": "测试"})
    assert item.doc_type == "doc"
    assert item.metadata == {}
    assert item.tags == []
    assert item.content == ""
    print("  ✅ test_make_item_defaults")


def test_make_item_empty_title():
    """测试空标题应报错"""
    from server.mcp_handler import _make_item

    try:
        _make_item({"title": ""})
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "标题不能为空" in str(e)
    print("  ✅ test_make_item_empty_title")


# ── 3. 配置加载测试 ──


def test_config_defaults():
    """测试配置默认值"""
    # 清除所有环境变量
    for key in list(os.environ.keys()):
        if key.startswith("KB_") or key.startswith("TC_"):
            del os.environ[key]
    # 强制覆盖 lightrag enabled，覆盖 config.yaml 的值
    os.environ["KB_LIGHTRAG_ENABLED"] = "false"

    import config as cfg_mod
    cfg_mod._config_cache = None
    from config import Config
    cfg = Config()

    assert cfg.api_host == "0.0.0.0"
    assert cfg.api_port == 8766
    assert cfg.collection_name == "knowledge_items"
    assert cfg.lightrag_enabled == False

    # 清理
    del os.environ["KB_LIGHTRAG_ENABLED"]
    cfg_mod._config_cache = None
    print("  ✅ test_config_defaults")


def test_config_env_override():
    """测试环境变量覆盖"""
    os.environ["KB_API_PORT"] = "9999"
    os.environ["KB_EMBED_MODEL"] = "test-model"

    # 清除缓存
    import config as cfg_mod
    cfg_mod._config_cache = None
    from config import Config
    cfg = Config()

    assert cfg.api_port == 9999
    assert cfg.embed_model == "test-model"

    # 清理
    del os.environ["KB_API_PORT"]
    del os.environ["KB_EMBED_MODEL"]
    cfg_mod._config_cache = None
    print("  ✅ test_config_env_override")


# ── 4. Engine 集成测试（用内存 ChromaDB） ──


def test_engine_add_and_search():
    """测试引擎添加和搜索"""
    # 用临时目录避免污染
    import tempfile
    tmpdir = tempfile.mkdtemp()

    # 替换配置
    from config import _config_cache
    _config_cache = None

    # 手动设置 env 指向临时目录
    os.environ["KB_CHROMA_DIR"] = tmpdir
    os.environ["KB_COLLECTION_NAME"] = "test_collection"

    # 重新加载配置
    from config import get_config
    cfg = get_config()

    from server.engine import VectorEngine
    from server.models import KnowledgeItem

    engine = VectorEngine()

    # 添加
    item = KnowledgeItem(title="登录测试", doc_type="test_case",
                         content="验证用户登录功能", tags=["冒烟"],
                         metadata={"module": "登录"})
    item_id = engine.add(item)
    assert item_id is not None
    assert len(item_id) == 16

    # 搜索
    results = engine.search("登录", n_results=5)
    assert len(results) >= 1
    assert results[0]["title"] == "登录测试"
    assert results[0]["doc_type"] == "test_case"

    # 按 doc_type 筛选
    results = engine.search("登录", n_results=5, doc_type="test_case")
    assert len(results) >= 1

    results = engine.search("登录", n_results=5, doc_type="doc")
    assert len(results) == 0

    # stats
    stats = engine.get_stats()
    assert stats["total"] >= 1
    assert stats["by_type"]["test_case"] >= 1

    # get_by_id
    got = engine.get_by_id(item_id)
    assert got is not None
    assert got["title"] == "登录测试"

    # get_all
    all_items = engine.get_all()
    assert len(all_items) >= 1

    # delete
    assert engine.delete(item_id) == True
    assert engine.get_by_id(item_id) is None

    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    _config_cache = None
    for key in ["KB_CHROMA_DIR", "KB_COLLECTION_NAME"]:
        os.environ.pop(key, None)
    print("  ✅ test_engine_add_and_search")


# ── 运行 ──

if __name__ == "__main__":
    print("\n🧪 kb-rag 单元测试\n")

    tests = [
        test_knowledge_item_basic,
        test_knowledge_item_gen_id,
        test_knowledge_item_embedding_text,
        test_knowledge_item_bm25_text,
        test_knowledge_item_to_dict,
        test_clean_text,
        test_clean_list,
        test_make_item_basic,
        test_make_item_defaults,
        test_make_item_empty_title,
        test_config_defaults,
        test_config_env_override,
        test_engine_add_and_search,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
            import traceback as tb
            tb.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if failed:
        sys.exit(1)
    print("✅ 全部通过")
