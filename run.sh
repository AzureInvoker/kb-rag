#!/bin/bash
cd "$(dirname "$0")"

case "${1:-help}" in
  start)
    echo "🚀 启动 KB-RAG API..."
    if [ -f /tmp/kb-rag.pid ]; then
      OLD_PID=$(cat /tmp/kb-rag.pid)
      if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "检测到旧进程 PID=$OLD_PID，先停止..."
        kill "$OLD_PID" 2>/dev/null
        sleep 1
      fi
    fi
    nohup uv run uvicorn server.api:app --host 0.0.0.0 --port 8767 > /tmp/kb-rag.log 2>&1 &
    PID=$!
    echo "$PID" > /tmp/kb-rag.pid
    for i in $(seq 1 15); do
      if curl -s http://localhost:8767/api/v1/health > /dev/null 2>&1; then
        echo "✅ 启动成功！（PID: $PID，耗时 ${i}s）"
        exit 0
      fi
      sleep 1
    done
    echo "❌ 启动失败（15s内未就绪），请查看日志：tail /tmp/kb-rag.log"
    exit 1
    ;;
  stop)
    echo "🛑 停止 KB-RAG API..."
    if [ -f /tmp/kb-rag.pid ]; then
      PID=$(cat /tmp/kb-rag.pid)
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && echo "已停止 PID: $PID" || echo "停止失败"
      else
        echo "PID $PID 不存在"
      fi
      rm -f /tmp/kb-rag.pid
    else
      echo "未找到 PID 文件"
    fi
    ;;
  restart)
    $0 stop
    sleep 1
    $0 start
    ;;
  test)
    echo "🧪 运行测试..."
    uv run python3 -m pytest tests/ -v 2>/dev/null || uv run python3 tests/test_core.py
    ;;
  update)
    echo "🔄 拉取最新代码..."
    OLD_HASH=$(git rev-parse HEAD)
    if ! git pull; then
      echo "❌ git pull 失败"
      exit 1
    fi
    NEW_HASH=$(git rev-parse HEAD)
    if [ "$OLD_HASH" != "$NEW_HASH" ]; then
      if git diff "$OLD_HASH".."$NEW_HASH" -- pyproject.toml | grep -q .; then
        echo "📦 检测到依赖变更，同步中..."
        uv sync && echo "✅ 依赖更新完成" || echo "⚠️ 依赖更新失败"
      elif git diff "$OLD_HASH".."$NEW_HASH" -- requirements.txt | grep -q .; then
        echo "📦 检测到依赖变更（requirements.txt），安装中..."
        uv pip install -r requirements.txt && echo "✅ 依赖更新完成" || echo "⚠️ 依赖更新失败"
      else
        echo "✅ 已是最新，无需更新依赖"
      fi
    else
      echo "✅ 已是最新"
    fi
    ;;
  *)
    echo "用法: ./run.sh start|stop|restart|update"
    ;;
esac
