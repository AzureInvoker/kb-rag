#!/usr/bin/env python3
"""内置召回评测 — 检查 Top-K 是否命中期望关键词。

用法:
  python scripts/eval_recall.py
  python scripts/eval_recall.py --no-rerank
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_CASES = [
    {"id": "login", "query": "用户登录测试", "expect_tokens": ["登录"]},
    {"id": "bridge", "query": "Bridge 游戏接入定义", "expect_tokens": ["bridge", "接入"]},
    {"id": "tunnel", "query": "tunnel 抓包 spinbet", "expect_tokens": ["tunnel", "抓包"]},
    {"id": "makebet", "query": "makeBet 下注接口", "expect_tokens": ["makebet", "下注"]},
    {"id": "mr_rule", "query": "审查 MR 铁律", "expect_tokens": ["mr", "审查"]},
]


def _hit(title: str, body: str, tokens: list[str]) -> bool:
    blob = f"{title} {body}".lower()
    return any(tok.lower() in blob for tok in tokens)


def _seed_engine(enable_rerank: bool):
    tmpdir = tempfile.mkdtemp()
    os.environ["KB_CHROMA_DIR"] = tmpdir
    os.environ["KB_COLLECTION_NAME"] = "eval_collection"
    os.environ["KB_RERANK_ENABLED"] = "true" if enable_rerank else "false"
    os.environ["KB_CHUNK_ENABLED"] = "true"
    os.environ["KB_LIGHTRAG_ENABLED"] = "false"

    import config as cfg_mod
    cfg_mod._config_cache = None

    from server.engine import VectorEngine
    from server.models import KnowledgeItem

    engine = VectorEngine(enable_rerank=enable_rerank)

    samples = [
        KnowledgeItem(title="用户登录测试", doc_type="test_case", content="验证登录流程与密码校验", tags=["登录"]),
        KnowledgeItem(title="Bridge 接入说明", doc_type="wiki", content="Bridge 是中台与游戏 H5 的 JS 接入桥", tags=["bridge"]),
        KnowledgeItem(title="tunnel 抓包指南", doc_type="doc", content="使用 eagis tunnel 抓包分析 spinbet 请求", tags=["tunnel"]),
        KnowledgeItem(
            title="makeBet 接口",
            doc_type="doc",
            content="中台 makeBet 下注扣币接口，参数含 uid、betAmount",
            tags=["makebet"],
        ),
        KnowledgeItem(title="MR 审查铁律", doc_type="doc", content="合并请求审查必须检查资损与幂等", tags=["mr"]),
        KnowledgeItem(
            title="长文档示例",
            doc_type="doc",
            content=("接口说明 " * 300) + " 文末关键字 special_tail_token",
            tags=["long"],
        ),
    ]
    for item in samples:
        engine.add(item)
    return engine, tmpdir


def run_eval(enable_rerank: bool) -> dict:
    engine, tmpdir = _seed_engine(enable_rerank)
    hits = 0
    details = []

    for case in EVAL_CASES:
        results = engine.search(case["query"], n_results=3)
        ok = False
        top_titles = []
        for row in results:
            top_titles.append(row.get("title", ""))
            if _hit(row.get("title", ""), row.get("content", "") or row.get("summary", ""), case["expect_tokens"]):
                ok = True
        hits += int(ok)
        details.append({"id": case["id"], "query": case["query"], "hit": ok, "top": top_titles})

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    for key in ["KB_CHROMA_DIR", "KB_COLLECTION_NAME", "KB_RERANK_ENABLED", "KB_CHUNK_ENABLED", "KB_LIGHTRAG_ENABLED"]:
        os.environ.pop(key, None)

    import config as cfg_mod
    cfg_mod._config_cache = None

    total = len(EVAL_CASES)
    return {
        "rerank": enable_rerank,
        "recall_at_3": round(hits / total, 3) if total else 0.0,
        "hits": hits,
        "total": total,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = {
        "baseline": run_eval(enable_rerank=False),
        "with_rerank": None if args.no_rerank else run_eval(enable_rerank=True),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
