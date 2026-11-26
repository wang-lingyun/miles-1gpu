#!/usr/bin/env bash
# Whole-model gradient check on this GPU with Miles' own replay and CI tools. Record two GSM8K rollouts once, then
# train on exactly those rollouts with three attention paths and compare Megatron's grad norm at every step, with
# Miles CI's check (--ci-load-grad-norm: rel_tol = abs_tol = 0.03):
#
#   unfused   reference: TE's plain attention, no flash kernels                -> saves the grad norms
#   fa2       what the rtx5090 profile trains with (flash + sm120.prefer_fa2)   -> must match the reference
#   fa4       the handoff recipe unchanged (h100 profile; flash = FA4 here)     -> negative control
#
#   scripts/check_gradients.sh          # about 15 min; summary at the end, logs in data/logs/<tag>-*.out
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
TAG="${TAG:-gradcheck-$(date +%Y%m%d-%H%M%S)}"
REC="/data/runs/$TAG-record"
GRAD="$REC/grad_norm_{role}_{rollout_id}_{step_id}.pt"
NO_EVAL_SAVE=eval-interval,eval-prompt-data,n-samples-per-eval-prompt,eval-max-response-len,eval-top-k,save,save-interval,ci-metric-checker-key,ci-metric-checker-threshold
REPLAY=(--num-rollout 2 --load-debug-rollout-data "$REC/rollout_{rollout_id}.pt")
declare -A RC

run() {  # run <name> <profile> <drop> <extra train.py flags...>
  local name=$1 profile=$2 drop=$3
  shift 3
  echo "$(date +%T) $name"
  WANDB=0 RUN_ID="$TAG-$name" PROFILE="$profile" RECIPE=gsm8k scripts/run_5090.sh --stages training --no-timeouts \
    --drop "$drop" -- "$@" > "data/logs/$TAG-$name.out" 2>&1
  RC[$name]=$?
  echo "$(date +%T) $name exit ${RC[$name]}"
}

run record rtx5090 "$NO_EVAL_SAVE,ci-test,no-enable-sample-ownership-checker" \
  --num-rollout 2 --debug-rollout-only --save-debug-rollout-data "$REC/rollout_{rollout_id}.pt"
[ "${RC[record]}" -eq 0 ] || { echo "recording failed, see data/logs/$TAG-record.out"; exit 1; }
run unfused rtx5090 "$NO_EVAL_SAVE" "${REPLAY[@]}" --attention-backend unfused --ci-save-grad-norm "$GRAD"
[ "${RC[unfused]}" -eq 0 ] || { echo "reference failed, see data/logs/$TAG-unfused.out"; exit 1; }
run fa2 rtx5090 "$NO_EVAL_SAVE" "${REPLAY[@]}" --ci-load-grad-norm "$GRAD"
run fa4 h100 "$NO_EVAL_SAVE" "${REPLAY[@]}" --ci-load-grad-norm "$GRAD"

echo
echo "== $TAG: grad norm per step on identical rollouts (Miles CI tolerance 3%)"
for name in unfused fa2 fa4; do
  python3 - "data/logs/$TAG-$name.out" "$name" "${RC[$name]}" <<'EOF'
import re, sys
log, name, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(log, errors="replace").read()
norms = {}
for m in re.finditer(r"model\.py:\d+ - step (\d+): \{.*?'train/grad_norm': ([0-9.eE+-]+)", text):
    norms[int(m.group(1))] = float(m.group(2))
attn = re.search(r"\[sm120\] .*|FlashAttention backend \(version [^)]*\)", text)
mismatch = re.search(r"grad norm mismatch: [^\n']*", text)
verdict = "reference" if name == "unfused" else ("MATCH" if rc == 0 else "MISMATCH" if mismatch else f"FAILED (exit {rc})")
print(f"{name:8s} {verdict:10s} " + "  ".join(f"step {k}: {v:.5f}" for k, v in sorted(norms.items())))
if mismatch:
    print(f"{'':19s}{mismatch.group(0)}")
EOF
done
