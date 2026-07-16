#!/usr/bin/env python3
"""
通用知识库 MCP Server — 复用 server/mcp_handler 的 TOOLS 和 handle_tool

使用方式：
  # 独立测试:
  echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python3 mcp/server.py

  # Hermes config.yaml:
  mcp_servers:
    kb-rag:
      command: "python3"
      args: ["/home/admin/kb-rag/mcp/server.py"]
"""

import json
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from server.mcp_handler import TOOLS, handle_tool
from server.engine import get_engine
from server.config import get_config
from server.lightrag_engine import LightRAGEngine

engine = get_engine()
cfg = get_config()
lightrag_engine = LightRAGEngine(cfg)


# ── MCP 协议 ──


def send_json(obj: dict):
    msg = json.dumps(obj, ensure_ascii=False)
    raw = f"Content-Length: {len(msg.encode('utf-8'))}\r\n\r\n{msg}".encode("utf-8")
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def read_json() -> dict | None:
    buf = sys.stdin.buffer
    headers = b""
    while True:
        chunk = buf.readline()
        if not chunk:
            return None
        headers += chunk
        if chunk in (b"\r\n", b"\n", b"\r"):
            break
    length = 0
    for header_line in headers.split(b"\r\n"):
        header_line = header_line.strip()
        if header_line.lower().startswith(b"content-length:"):
            length = int(header_line.split(b":")[1].strip())
            break
    if length <= 0:
        return None
    raw_bytes = buf.read(length)
    if not raw_bytes:
        return None
    raw = raw_bytes.decode("utf-8")
    return json.loads(raw) if raw else None


# ── 主循环 ──


def main():
    while True:
        try:
            msg = read_json()
        except Exception:
            continue
        if msg is None:
            break

        msg_id = msg.get("id")
        method = msg.get("method")

        if method == "initialize":
            send_json({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "kb-rag-mcp", "version": "1.0.0"},
                },
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send_json({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = msg.get("params", {}).get("name", "")
            args = msg.get("params", {}).get("arguments", {})
            # 统一参数格式（兼容部分客户端误传 list）
            if isinstance(args, list) and len(args) == 1 and isinstance(args[0], dict):
                args = args[0]
            result = handle_tool(name, args, engine, lightrag_engine)
            send_json({"jsonrpc": "2.0", "id": msg_id, "result": result})
        elif method == "ping":
            send_json({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        else:
            send_json({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"不支持的方法: {method}"},
            })


if __name__ == "__main__":
    main()
