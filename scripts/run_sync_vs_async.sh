#!/usr/bin/env bash
# Sync vs fully-async GRPO on one RTX 5090, judged by Miles CI's pass bar for Qwen3-0.6B GSM8K
# (eval/gsm8k >= 0.71 within 60 rollouts, plus the --ci-test invariants). Arms and seeds run back to back, arms
# alternating within each seed so slow drift (thermals, background load) hits every arm alike. Each run is an ordinary
# scripts/run_5090.sh run (W&B on if configured); the table at the end comes from scripts/summarize_runs.py.
#
#   arms:  sync      train.py, rollout and training take turns           (recipe gsm8k)
#          async     train_async.py --fully-async --colocate               (recipe gsm8k-async)
#          async-s1  async with staleness bounded to 1 at submission        (--max-weight-staleness 1
#                    and in the buffer (AReaL-style)                         --async-submit-within-staleness)
#
#   TAG=sva1 scripts/run_sync_vs_async.sh                         # default: SEEDS="1 2" ARMS="sync async"
#   TAG=sva1 SEEDS="3" ARMS="async-s1" scripts/run_sync_vs_async.sh
#   setsid nohup env TAG=sva1 scripts/run_sync_vs_async.sh > data/logs/sva1.out 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
TAG="${TAG:-sva-$(date +%m%d-%H%M)}"
SEEDS="${SEEDS:-1 2}"
ARMS="${ARMS:-sync async}"
mkdir -p data/logs

for seed in $SEEDS; do
  for arm in $ARMS; do
    case "$arm" in
      sync) recipe=gsm8k; extra=() ;;
      async) recipe=gsm8k-async; extra=() ;;
      async-s1) recipe=gsm8k-async; extra=(--max-weight-staleness 1 --async-submit-within-staleness) ;;
      *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    run_id="$TAG-$arm-s$seed"
    echo "$(date +%T) start $run_id"
    RUN_ID="$run_id" RECIPE="$recipe" scripts/run_5090.sh --no-timeouts --stages training \
      -- --seed $((1233 + seed)) --rollout-seed $((41 + seed)) "${extra[@]}" > "data/logs/$run_id.out" 2>&1
    echo "$(date +%T) end   $run_id exit $?"
  done
done

echo
python3 scripts/summarize_runs.py data/runs/"$TAG"-*
