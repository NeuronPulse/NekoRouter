#!/usr/bin/env bash
# 一键启动 NekoRouter 所需的外部依赖：Qdrant + Neo4j + NapCat
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 可在运行前 export NAPCAT_ACCOUNT=你的QQ号 覆盖默认值
NAPCAT_ACCOUNT="${NAPCAT_ACCOUNT:-2913335827}"
NAPCAT_IMAGE="${NAPCAT_IMAGE:-docker.1ms.run/mlikiowa/napcat-docker:latest}"

cd "$PROJECT_ROOT"

echo "[1/3] Starting Qdrant and Neo4j..."
docker compose -f docker-compose.dev.yml up -d --wait

echo "[2/3] Starting NapCat (QQ: $NAPCAT_ACCOUNT)..."
if docker ps -a --format '{{.Names}}' | grep -qx napcat; then
    echo "    Container 'napcat' already exists, starting it..."
    docker start napcat >/dev/null
else
    docker run -d \
        --name napcat \
        --restart unless-stopped \
        -p 3000:3000 \
        -p 3001:3001 \
        -p 6099:6099 \
        -e "ACCOUNT=$NAPCAT_ACCOUNT" \
        -e WS_ENABLE=true \
        -e "NAPCAT_UID=$(id -u)" \
        -e "NAPCAT_GID=$(id -g)" \
        -e TZ=Asia/Shanghai \
        -v "$PROJECT_ROOT/napcat-data:/app/.config/QQ" \
        -v "$PROJECT_ROOT/napcat-config:/app/napcat/config" \
        "$NAPCAT_IMAGE" >/dev/null
fi

echo "[3/3] Waiting for NapCat to be running..."
sleep 2
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E 'napcat|neko-qdrant|neko-neo4j' || true

echo ""
echo "Done. Next steps:"
echo "  1. Scan QR code: docker logs -f napcat"
echo "  2. Open NapCat WebUI: http://127.0.0.1:6099"
echo "  3. Add a WebSocket server on port 3001 in the WebUI"
echo "  4. Run NekoRouter: cargo run -p neko-router"
