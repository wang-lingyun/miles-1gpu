#!/usr/bin/env bash
# Run a command inside the pinned Miles image on this box. The repo is mounted read-only at /opt/experiment and the
# data dir at /data, the same paths Modal uses (image files and Volume mount), so in-container code is identical on
# both platforms.
#
#   scripts/miles_docker.sh python gpu_pipeline.py --profile rtx5090 --run-id test   # no network (default)
#   NET=bridge scripts/miles_docker.sh python assets.py                               # downloads need a network
#   scripts/miles_docker.sh bash                                                      # interactive shell
#
# NET=none: Ray, the SGLang servers and the router listen without authentication, so the training container gets no
# network at all (loopback only). Training needs none: assets are on disk and HF_HUB_OFFLINE=1.
# Files the container writes under the data dir are handed back to your user when the command ends.
# WANDB_ENV_FILE=<file with WANDB_API_KEY=...> passes the key with --env-file and turns W&B on (run_5090.sh does it).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${MILES_DATA:-$REPO/data}"
NET="${NET:-none}"
NAME="${NAME:-miles-$(date +%H%M%S)}"
IMAGE="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pins"]["image"])' "$REPO/experiment-config.json")"

mkdir -p "$DATA/cache"
wandb=()
if [ -n "${WANDB_ENV_FILE:-}" ]; then wandb=(--env-file "$WANDB_ENV_FILE" -e MILES_WANDB=1); fi
tty=()
if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi

# --init + TINI_KILL_PROCESS_GROUP: `docker stop` / Ctrl-C reaches the whole command, not just the entry script.
exec docker run --rm "${tty[@]}" --init -e TINI_KILL_PROCESS_GROUP=1 \
  --name "$NAME" --gpus all --network "$NET" \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO:/opt/experiment:ro" -v "$DATA:/data" -v "$DATA/cache:/root/.cache" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "${wandb[@]}" \
  -w /opt/experiment --entrypoint /opt/experiment/scripts/in_container.sh \
  "$IMAGE" "$@"
