#!/usr/bin/env bash
set -uo pipefail
PING_URL="${1:-}"
REPO_DIR="${2:-.}"
REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"
echo "Pinging $PING_URL/reset..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30)
echo "HTTP: $HTTP_CODE"
[ "$HTTP_CODE" = "200" ] && echo "PASSED -- HF Space live" || echo "FAILED"
echo "Running docker build..."
docker build server/ && echo "PASSED -- Docker build" || echo "FAILED"
echo "Running openenv validate..."
openenv validate && echo "PASSED -- openenv validate" || echo "FAILED"
