#!/usr/bin/env bash
# Entry point inside the Miles container (see scripts/miles_docker.sh): shared environment, run the command, then give
# everything under /data back to the host user (the container runs as root).
set -uo pipefail

export PYTHONPATH="/opt/experiment:/root/miles:/root/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export RAY_USAGE_STATS_ENABLED=0 NCCL_NVLS_ENABLE=0
if [ "${MILES_WANDB:-0}" = 1 ]; then unset WANDB_MODE; else export WANDB_MODE=disabled; unset WANDB_API_KEY; fi
export TRITON_CACHE_DIR=/root/.cache/triton

give_back() {
  if [ -n "${HOST_UID:-}" ]; then chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" /data 2>/dev/null || true; fi
}
trap give_back EXIT
trap 'exit 143' TERM INT

"$@"
