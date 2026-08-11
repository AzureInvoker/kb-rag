"""
统一配置加载模块

优先级: 环境变量 > 配置文件 (config.yaml) > 默认值

环境变量：
  KB_API_HOST           API 监听地址 (默认 0.0.0.0)
  KB_API_PORT           API 监听端口 (默认 8766)
  KB_EMBED_MODEL        嵌入模型 (默认 intfloat/multilingual-e5-small)
  KB_CHROMA_DIR         ChromaDB 目录 (默认 .chroma_db)
  KB_COLLECTION_NAME    ChromaDB 集合名 (默认 knowledge_items)
  KB_DATA_DIR           项目数据根目录 (默认 auto)

  # LightRAG 环境变量覆盖
  KB_LIGHTRAG_ENABLED   是否启用 LightRAG
  KB_LLM_PROVIDER       deepseek | ollama
  KB_DEEPSEEK_API_KEY   DeepSeek API Key
  KB_OLLAMA_BASE_URL    Ollama 地址 (默认 http://localhost:11434)
  KB_LLM_MODEL          LLM 模型名

  # 检索增强
  KB_RERANK_ENABLED     是否启用 Reranker
  KB_CHUNK_ENABLED      是否启用长文档分块

兼容旧变量（TC_* 前缀亦可用，KB_* 优先）
"""

import os
import yaml
from pathlib import Path


def _find_project_root() -> Path:
    return Path(__file__).parent.resolve()


def _load_config_file() -> dict:
    config_file = _find_project_root() / "config.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


_config_cache = None


class Config:
    def __init__(self):
        file_config = _load_config_file()
        api_cfg = file_config.get("api", {}) if file_config else {}
        engine_cfg = file_config.get("engine", {}) if file_config else {}
        retrieval_cfg = file_config.get("retrieval", {}) if file_config else {}
        chunk_cfg = retrieval_cfg.get("chunking", {}) if retrieval_cfg else {}
        rerank_cfg = retrieval_cfg.get("rerank", {}) if retrieval_cfg else {}
        lightrag_cfg = file_config.get("lightrag", {}) if file_config else {}
        llm_cfg = lightrag_cfg.get("llm", {}) if lightrag_cfg else {}

        # ── API ──
        self.api_host = os.getenv("KB_API_HOST", os.getenv("TC_API_HOST",
                                  api_cfg.get("host", "0.0.0.0")))
        api_port_env = os.getenv("KB_API_PORT", os.getenv("TC_API_PORT"))
        self.api_port = int(api_port_env) if api_port_env else api_cfg.get("port", 8766)

        # ── Engine ──
        self.embed_model = os.getenv("KB_EMBED_MODEL", os.getenv("TC_EMBED_MODEL",
                                    engine_cfg.get("embed_model", "BAAI/bge-small-zh-v1.5")))
        self.collection_name = engine_cfg.get("collection_name", "knowledge_items")

        chroma_dir = engine_cfg.get("chroma_dir", ".chroma_db")
        chroma_env = os.getenv("KB_CHROMA_DIR", os.getenv("TC_CHROMA_DIR"))
        if chroma_env:
            self.chroma_dir = Path(chroma_env)
        else:
            data_dir_env = os.getenv("KB_DATA_DIR", os.getenv("TC_DATA_DIR"))
            if data_dir_env:
                data_dir = Path(data_dir_env)
            else:
                data_dir = _find_project_root()
            self.chroma_dir = data_dir / chroma_dir

        # ── 检索增强 ──
        self.rrf_k = int(retrieval_cfg.get("rrf_k", 60))
        self.vec_candidates = int(retrieval_cfg.get("vec_candidates", 30))
        self.bm25_candidates = int(retrieval_cfg.get("bm25_candidates", 50))

        _chunk_env = os.getenv("KB_CHUNK_ENABLED")
        if _chunk_env is not None:
            self.chunk_enabled = _chunk_env.lower() in ("1", "true", "yes")
        else:
            self.chunk_enabled = bool(chunk_cfg.get("enabled", True))
        self.chunk_min_chars = int(chunk_cfg.get("min_chars", 1000))
        self.chunk_size = int(chunk_cfg.get("chunk_size", 500))
        self.chunk_overlap = int(chunk_cfg.get("overlap", 80))

        _rerank_env = os.getenv("KB_RERANK_ENABLED")
        if _rerank_env is not None:
            self.rerank_enabled = _rerank_env.lower() in ("1", "true", "yes")
        else:
            self.rerank_enabled = bool(rerank_cfg.get("enabled", True))
        self.rerank_model = rerank_cfg.get("model", "BAAI/bge-reranker-v2-m3")
        self.rerank_top_n = int(rerank_cfg.get("top_n", 20))

        # ── 脑记忆库 ──
        memory_cfg = file_config.get("memory", {}) if file_config else {}
        self.memory_embed_model = os.getenv("KB_MEMORY_EMBED_MODEL",
                                            memory_cfg.get("embed_model", self.embed_model))
        self.memory_collection_name = memory_cfg.get("collection_name", "brain_memory")
        memory_chroma_dir = memory_cfg.get("chroma_dir", ".chroma_db_memory")
        memory_env = os.getenv("KB_MEMORY_CHROMA_DIR")
        if memory_env:
            self.memory_chroma_dir = Path(memory_env)
        else:
            data_dir_env = os.getenv("KB_DATA_DIR", os.getenv("TC_DATA_DIR"))
            if data_dir_env:
                data_dir = Path(data_dir_env)
            else:
                data_dir = _find_project_root()
            self.memory_chroma_dir = data_dir / memory_chroma_dir

        # ── LightRAG ──
        _lr_enabled = os.getenv("KB_LIGHTRAG_ENABLED", os.getenv("TC_LIGHTRAG_ENABLED"))
        if _lr_enabled is not None:
            self.lightrag_enabled = _lr_enabled.lower() in ("1", "true", "yes")
        else:
            self.lightrag_enabled = lightrag_cfg.get("enabled", False)

        self.lightrag_working_dir = lightrag_cfg.get("working_dir", ".lightrag_storage")
        self.lightrag_embed_model = lightrag_cfg.get("embed_model", self.embed_model)
        self.lightrag_top_k = lightrag_cfg.get("top_k", 20)
        self.lightrag_mode = lightrag_cfg.get("mode", "mix")

        # ── LLM ──
        self.llm_provider = os.getenv("KB_LLM_PROVIDER", os.getenv("TC_LLM_PROVIDER",
                                     llm_cfg.get("provider", "deepseek")))
        self.deepseek_api_key = os.getenv("KB_DEEPSEEK_API_KEY",
                              os.getenv("TC_DEEPSEEK_API_KEY",
                              os.getenv("DEEPSEEK_API_KEY", llm_cfg.get("api_key", ""))))
        self.ollama_base_url = os.getenv("KB_OLLAMA_BASE_URL", os.getenv("TC_OLLAMA_BASE_URL",
                                         llm_cfg.get("base_url", "http://localhost:11434")))
        self.llm_model = os.getenv("KB_LLM_MODEL", os.getenv("TC_LLM_MODEL",
                                   llm_cfg.get("model", "deepseek-chat")))


def get_config() -> Config:
    global _config_cache
    if _config_cache is None:
        _config_cache = Config()
    return _config_cache
