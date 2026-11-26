"""Single-GPU Miles GRPO launcher: Qwen3-0.6B, Megatron trainer + SGLang rollouts colocated on one GPU.

train.py arguments = COMMON + a task RECIPE + a hardware PROFILE, then `--drop`ped flags removed, then extra flags.
Miles' own command backend runs it: it starts a local Ray head, submits /root/miles/train.py with the Qwen3-0.6B
model args, follows the job and stops Ray.

Recipes
  dapo-smoke   the Modal handoff: official Megatron GRPO/deepscaler recipe sized to 3 steps on one GPU
  gsm8k        Miles CI's Qwen3-0.6B baseline (tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py) with the
               Megatron trainer on one GPU, including its pass bar: eval/gsm8k >= 0.71 within 60 rollouts
Profiles
  h100         nothing changed
  rtx5090      sm_120 attention fix (see sm120.py)

    python run_single_gpu.py --recipe gsm8k --profile rtx5090 --run-dir /data/runs/<id> --conversion-dir /data/converted/<id>
    python run_single_gpu.py ... --print-args                      # print the argv and exit (no Miles import)
    python run_single_gpu.py ... --drop eval-interval,save -- --num-rollout 2    # later flags win
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment-config.json").read_text())

# Flag -> values (an empty list is a bare switch). {model} {data} {run} {conversion} are filled in by build_argv().
COMMON: dict[str, list[str]] = {
    "--train-backend": ["megatron"],
    "--ckpt-format": ["torch_dist"],
    "--hf-checkpoint": ["{model}"],
    "--ref-load": ["{conversion}"],  # the converted initialization; fresh training, no --load
    "--save": ["{run}/checkpoints"],
    "--apply-chat-template": [],
    "--rollout-shuffle": [],
    "--rollout-temperature": ["1"],
    # parallelism: everything 1 on one GPU
    "--tensor-model-parallel-size": ["1"],
    "--pipeline-model-parallel-size": ["1"],
    "--context-parallel-size": ["1"],
    "--expert-model-parallel-size": ["1"],
    "--expert-tensor-parallel-size": ["1"],
    "--recompute-granularity": ["full"],
    "--recompute-method": ["uniform"],
    "--recompute-num-layers": ["1"],
    "--use-dynamic-batch-size": [],
    "--max-tokens-per-gpu": ["10240"],
    # GRPO
    "--advantage-estimator": ["grpo"],
    "--kl-loss-coef": ["0.00"],
    "--kl-loss-type": ["low_var_kl"],
    "--entropy-coef": ["0.00"],
    "--eps-clip": ["0.2"],
    "--eps-clip-high": ["0.28"],
    # optimizer
    "--optimizer": ["adam"],
    "--lr": ["1e-6"],
    "--lr-decay-style": ["constant"],
    "--weight-decay": ["0.1"],
    "--adam-beta1": ["0.9"],
    "--adam-beta2": ["0.98"],
    # SGLang
    "--rollout-num-gpus-per-engine": ["1"],
    "--sglang-mem-fraction-static": ["0.45"],
    # Megatron numerics
    "--attention-dropout": ["0.0"],
    "--hidden-dropout": ["0.0"],
    "--accumulate-allreduce-grads-in-fp32": [],
    "--attention-softmax-in-fp32": [],
    "--attention-backend": ["flash"],
    # one GPU, trainer and engine take turns on it
    "--colocate": [],
    "--actor-num-nodes": ["1"],
    "--actor-num-gpus-per-node": ["1"],
    "--num-gpus-per-node": ["1"],
}

RECIPES: dict[str, dict[str, list[str]]] = {
    "dapo-smoke": {
        "--prompt-data": ["{data}/datasets/dapo-math-17k/dapo-math-17k.jsonl"],
        "--input-key": ["prompt"],
        "--label-key": ["label"],
        "--rm-type": ["deepscaler"],
        "--balance-data": [],
        "--num-rollout": ["3"],
        "--rollout-batch-size": ["2"],
        "--n-samples-per-prompt": ["4"],
        "--global-batch-size": ["8"],
        "--num-steps-per-rollout": ["1"],
        "--rollout-max-response-len": ["8192"],
        "--sglang-context-length": ["16384"],
        "--use-kl-loss": [],
        # eval once, after the last rollout, on the fixed 2-problem subset
        "--eval-interval": ["3"],
        "--skip-eval-before-train": [],
        "--eval-prompt-data": ["aime-smoke", "{run}/eval-subset.jsonl"],
        "--eval-top-p": ["1"],
        "--eval-max-response-len": ["8192"],
        "--n-samples-per-eval-prompt": ["1"],
        "--save-interval": ["3"],
    },
    "gsm8k": {
        "--prompt-data": ["{data}/datasets/gsm8k/train.parquet"],
        "--input-key": ["messages"],
        "--label-key": ["label"],
        "--rm-type": ["math"],
        "--num-rollout": ["60"],
        "--rollout-batch-size": ["32"],
        "--n-samples-per-prompt": ["8"],
        "--rollout-max-response-len": ["1024"],
        "--over-sampling-batch-size": ["64"],
        "--dynamic-sampling-filter-path": [
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"],
        "--global-batch-size": ["256"],
        "--kl-coef": ["0.00"],
        # eval before training (the base model) and every 20 rollouts: GSM8K test, greedy
        "--eval-interval": ["20"],
        "--eval-prompt-data": ["gsm8k", "{data}/datasets/gsm8k/test.parquet"],
        "--n-samples-per-eval-prompt": ["1"],
        "--eval-max-response-len": ["1024"],
        "--eval-top-k": ["1"],
        "--save-interval": ["20"],
        # Miles CI's in-process invariants and its pass bar for this recipe
        "--ci-test": [],
        "--no-enable-sample-ownership-checker": [],
        "--ci-metric-checker-key": ["eval/gsm8k"],
        "--ci-metric-checker-threshold": ["0.71"],
    },
}

# Per-GPU overrides on top of COMMON + recipe. Keep each one justified.
PROFILES: dict[str, dict[str, list[str]]] = {
    "h100": {},
    # RTX 5090 (sm_120): same `--attention-backend flash`, but TE would pick FlashAttention 4 here, which gives wrong
    # dK/dV for Miles' packed batches on sm_120. The hook makes TE use the image's flash-attn 2 instead (sm120.py;
    # probe_attention.py and scripts/check_gradients.sh measure it).
    "rtx5090": {
        "--custom-megatron-init-path": ["sm120.prefer_fa2"],
    },
}


def build_argv(
    recipe: str,
    profile: str,
    *,
    data_root: Path,
    run_dir: Path,
    conversion_dir: Path,
    drop: list[str] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    if recipe not in RECIPES:
        raise SystemExit(f"unknown recipe {recipe!r}; choose from {sorted(RECIPES)}")
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    flags = {**COMMON, **RECIPES[recipe], **PROFILES[profile]}
    for name in drop or []:
        flag = "--" + name.removeprefix("--")
        if flag not in flags:
            raise SystemExit(f"--drop {name}: {flag} is not in recipe {recipe!r} / profile {profile!r}")
        del flags[flag]
    paths = {
        "model": str(data_root / CONFIG["assets"]["model"]["dest"]),
        "data": str(data_root),
        "run": str(run_dir),
        "conversion": str(conversion_dir),
    }
    argv: list[str] = []
    for flag, values in flags.items():
        argv += [flag, *(v.format(**paths) for v in values)]
    return argv + list(extra or [])


def train_env() -> dict[str, str]:
    """Environment for the Ray job (train.py and everything it spawns)."""
    return {
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "RAY_USAGE_STATS_ENABLED": "0",
        "NCCL_NVLS_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def split_csv(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default="dapo-smoke", choices=sorted(RECIPES))
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--conversion-dir", type=Path, required=True)
    ap.add_argument("--drop", type=split_csv, default=[], help="comma-separated flags to remove, without dashes")
    ap.add_argument("--print-args", action="store_true", help="print the train.py argv and exit")
    ap.add_argument("extra", nargs="*", help="extra train.py flags (after --)")
    args = ap.parse_args()

    argv = build_argv(args.recipe, args.profile, data_root=args.data_root, run_dir=args.run_dir,
                      conversion_dir=args.conversion_dir, drop=args.drop, extra=args.extra)
    if args.print_args:
        print(shlex.join(argv))
        return

    # Deferred: only importable inside the Miles image.
    from miles.utils.external_utils import command_utils as U

    for key in ("WANDB_API_KEY", "MILES_SCRIPT_EXTERNAL_RAY"):
        os.environ.pop(key, None)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.update(train_env())

    config = U.ExecuteTrainConfig(output_dir=str(args.run_dir), num_nodes=1, graceful_ray_stop=True)
    config.create_backend().execute_train(
        train_args=shlex.join(argv),
        num_gpus_per_node=1,
        megatron_model_type=CONFIG["megatron_model_type"],
        megatron_path="/root/Megatron-LM",
        job_lifetime="launcher",
        extra_env_vars=train_env(),
    )


if __name__ == "__main__":
    sys.exit(main())
