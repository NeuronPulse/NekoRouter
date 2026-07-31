#!/usr/bin/env bash
# Configure Docker daemon to use a Chinese registry mirror.
# Run with: bash scripts/setup-docker-mirror.sh [mirror_url]
set -euo pipefail

DEFAULT_MIRROR="https://docker.m.daocloud.io"
MIRROR="${1:-$DEFAULT_MIRROR}"

echo "Configuring Docker registry mirror: ${MIRROR}"

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "${MIRROR}"
  ]
}
EOF

echo "Done. Restart Docker daemon to apply:"
echo "  sudo systemctl restart docker"
echo "Or on macOS/Windows, restart Docker Desktop."
