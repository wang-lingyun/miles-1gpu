#!/usr/bin/env bash
# Sync between the Mac checkout and the GPU box (no git needed there).
#
#   scripts/sync.sh push            # Mac -> GPU: code only (data/ is never touched)
#   scripts/sync.sh pull            # GPU -> Mac: run logs, status and configs (no checkpoints, models or caches)
#   Host: gpu5090-lan if it answers, else gpu5090 (Tailscale); GPU_HOST=... overrides
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${GPU_HOST:-}" ]; then
  HOST="$GPU_HOST"
elif ssh -o ConnectTimeout=3 -o BatchMode=yes gpu5090-lan true 2>/dev/null; then
  HOST=gpu5090-lan                # same LAN
else
  HOST=gpu5090                    # Tailscale
fi
REMOTE_DIR="${REMOTE_DIR:-src/miles}"

case "${1:-}" in
  push)
    rsync -az --delete \
      --exclude .git/ --exclude .venv/ --exclude data/ --exclude __pycache__/ --exclude .pytest_cache/ \
      --exclude .DS_Store \
      "$REPO/" "$HOST:$REMOTE_DIR/"
    ;;
  pull)
    mkdir -p "$REPO/data"
    # rsync takes the first matching rule: directory excludes, then descend, then the file types to keep
    rsync -az --prune-empty-dirs \
      --exclude 'checkpoints/' --exclude 'hf-model/' --exclude 'models/' --exclude 'converted/' \
      --exclude 'datasets/' --exclude 'cache/' \
      --include '*/' --include '*.log' --include '*.json' --include '*.jsonl' --include '*.csv' --include '*.out' \
      --exclude '*' \
      "$HOST:$REMOTE_DIR/data/" "$REPO/data/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
