#!/usr/bin/env bash
# The local experiment on the RTX 5090: pinned assets (first time only, with network), then conversion, training and
# HF export in one network-less container. Arguments go to gpu_pipeline.py; flags after `--` go to train.py.
#
#   scripts/run_5090.sh                                            # smoke: 3 rollouts, eval on 2 problems, save, export
#   RUN_ID=r20 scripts/run_5090.sh --no-timeouts -- --num-rollout 20 --save-interval 10 --eval-interval 10
#   scripts/run_5090.sh --stages conversion                       # just convert
#   RECIPE=gsm8k RUN_ID=gsm8k-1 scripts/run_5090.sh --no-timeouts  # Miles CI baseline: eval/gsm8k >= 0.71 in 60 rollouts
#   WANDB=0 scripts/run_5090.sh                                    # no W&B even if ~/.config/miles-1gpu/wandb.env exists
#
# Detached (survives SSH drops):  setsid nohup scripts/run_5090.sh > data/logs/last.out 2>&1 &
# Stop a run:                     docker stop miles-run-<RUN_ID>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
DATA="${MILES_DATA:-$REPO/data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$DATA/logs"

busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
if [ -n "$busy" ]; then
  echo "the GPU is in use; stop these first:" >&2
  echo "$busy" >&2
  exit 1
fi

# (re)download when the manifest is missing or doesn't match every pinned asset in experiment-config.json
if ! python3 - "$REPO/experiment-config.json" "$DATA/assets-manifest.json" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
try:
    have = json.load(open(sys.argv[2]))["assets"]
except (OSError, KeyError, ValueError):
    sys.exit(1)
sys.exit(any(have.get(k, {}).get("revision") != v["revision"] for k, v in config["assets"].items()))
EOF
then
  WANDB_ENV_FILE= NET=bridge NAME="miles-assets-$RUN_ID" scripts/miles_docker.sh python assets.py --data-root /data
fi

# W&B: on when the key file exists (WANDB=0 turns it off). Miles logs from several processes into one W&B run in
# online "shared" mode, so the training container then needs a network (Docker bridge) instead of none.
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/miles-1gpu/wandb.env}"
if [ "${WANDB:-1}" != 0 ] && [ -f "$WANDB_ENV_FILE" ]; then
  export WANDB_ENV_FILE
  TRAIN_NET="${NET:-bridge}"
  echo "W&B: on (key file $WANDB_ENV_FILE, network $TRAIN_NET)"
else
  unset WANDB_ENV_FILE
  TRAIN_NET="${NET:-none}"
fi

echo "run: $DATA/runs/$RUN_ID  recipe ${RECIPE:-dapo-smoke}  profile ${PROFILE:-rtx5090}  (container miles-run-$RUN_ID)"
NET="$TRAIN_NET" NAME="miles-run-$RUN_ID" scripts/miles_docker.sh \
  python gpu_pipeline.py --recipe "${RECIPE:-dapo-smoke}" --profile "${PROFILE:-rtx5090}" --run-id "$RUN_ID" \
  --gpu-log "$@"
