# Miles + SGLang GRPO on one GPU: RTX 5090 now, Modal H100 later

[Miles](https://github.com/radixark/miles) GRPO on Qwen3-0.6B with the Megatron trainer and SGLang rollouts
**colocated on a single GPU**. One pipeline, two places to run it:

| | RTX 5090 (this box) | Modal (later) |
|---|---|---|
| Image | `radixark/miles@sha256:bab7c3a5…` (Miles `fe7037f`, Megatron `8e43d38`) | same digest |
| Entry | `scripts/run_5090.sh` → Docker | `modal_miles.py` → `modal run` |
| Repo inside the container | `/opt/experiment` (read-only bind mount) | `/opt/experiment` (`add_local_file`) |
| Data | `./data` → `/data` | Volume `miles-qwen3-smoke-v2` → `/data` |
| Profile | `rtx5090` | `h100` (the handoff recipe unchanged) |
| Conversion | reused per image | fresh per run |

Everything inside the container (`assets.py`, `gpu_pipeline.py`, `run_single_gpu.py`, `checkpoint_compat.py`) is
the same code on both. The recipe is the official Megatron GRPO/deepscaler one (Miles' `run_qwen3_dense.py`) sized
for one GPU, as written in the Modal handoff. The two profiles differ in one flag, see [Why the 5090 needs a
profile](#why-the-5090-needs-a-profile).

## Layout

```
experiment-config.json     pins (image digest, Miles/Megatron commits, HF revisions), eval subset, timeouts, Modal resources
assets.py                  CPU: download the pinned model + datasets, write the 2-problem eval subset and a manifest
run_single_gpu.py          the recipe + per-GPU profiles; launches train.py through Miles' command backend (Ray)
gpu_pipeline.py            one GPU job: HF -> torch_dist conversion, training (eval + save), HF export; logs + status.json
checkpoint_compat.py       exports the latest checkpoint with Miles' official exporter (common.pt view, see below)
probe_attention.py         sm_120 check: Megatron/TE attention backends on packed sequences, fwd + bwd vs fp32
scripts/check_gradients.sh whole-model gradient check: replay identical rollouts through 3 attention paths
modal_miles.py             Modal app: download() on CPU, train(run_id) on one H100, same image and pipeline
scripts/host/install_docker.sh   one-time sudo: Docker Engine + NVIDIA Container Toolkit on the box
scripts/miles_docker.sh    run any command in the pinned image (repo at /opt/experiment, data at /data)
scripts/in_container.sh    container entry: shared env, run, chown /data back to your user
scripts/run_5090.sh        the local experiment: assets (once), then the pipeline in a network-less container
scripts/sync.sh            rsync code Mac -> box, logs box -> Mac
tests/                     CPU tests: recipe/profile argv, stage runner (logs, exit codes, timeouts), checkpoint view
```

## RTX 5090

One-time host setup (needs sudo; adds your user to the `docker` group, which is root-equivalent on the box):

```bash
scripts/sync.sh push                                              # on the Mac
ssh -t gpu5090-lan 'sudo bash ~/src/miles/scripts/host/install_docker.sh'
ssh gpu5090-lan 'docker pull radixark/miles@sha256:bab7c3a516f34ddb430d820087259822d438e9f74594614d61e98bb645627d00'
```

Run (on the box, in `~/src/miles`):

```bash
scripts/miles_docker.sh python probe_attention.py         # optional: attention kernels on this GPU
scripts/run_5090.sh                                      # smoke: assets once, convert, 3 steps, eval, save, export
setsid nohup scripts/run_5090.sh > data/logs/last.out 2>&1 &    # detached, survives SSH drops
RUN_ID=r50 scripts/run_5090.sh --no-timeouts -- --num-rollout 50 --save-interval 25 --eval-interval 25
docker stop miles-run-<RUN_ID>                           # stop a run
scripts/miles_docker.sh bash                             # a shell in the image
```

Arguments to `run_5090.sh` go to `gpu_pipeline.py` (`--stages`, `--no-timeouts`, `--conversion-id`,
`--fresh-conversion`); flags after `--` go to Miles' `train.py` and override the recipe (later flags win).

Outputs, the same layout as on Modal:

```
data/models/Qwen3-0.6B  data/datasets/{dapo-math-17k,aime-2024}  data/eval-subset.jsonl  data/assets-manifest.json
data/converted/<conversion-id>/              torch_dist initialization (--ref-load), kept apart from trained checkpoints
data/runs/<run-id>/run-config.json           profile, pins, full train.py argv
data/runs/<run-id>/status.json               stage states and times
data/runs/<run-id>/logs/{conversion,training,hf_export}.log, gpu.csv
data/runs/<run-id>/checkpoints/iter_0000002/ torch_dist checkpoint (rollout ids are zero-based)
data/runs/<run-id>/hf-model/                 HF safetensors + config/tokenizer
```

`scripts/sync.sh pull` copies logs, status and configs (no weights) back to the Mac.

The box suspends itself after 30 idle minutes (`gpu-autosleep.timer`: GPU at 0%, low CPU load, no SSH, no desktop
use). A running job keeps the GPU busy, so this doesn't interrupt runs, but the box is offline afterwards until it is
woken up.

### First smoke run on the 5090 (`smoke1`, 2026-11-25)

All three stages finished in 326 s: conversion 44 s, training 264 s (Ray, trainer and SGLang startup, 3 rollouts +
steps, eval, save), HF export 18 s. Peak GPU memory 27.8 of 32.6 GB, mean utilization 38%.

| rollout / step | raw reward (8 samples) | truncated at 8192 | mean response | `train_rollout_logprob_abs_diff` | `train_rollout_kl` | grad norm |
|---|---|---|---|---|---|---|
| 0 | 0.25 | 62.5% | 7255 | 0.0151 | 0.00062 | 0.73 |
| 1 | 0.25 | 62.5% | 7438 | 0.0186 | 0.00071 | 0.79 |
| 2 | 0.0 | 37.5% | 6872 | 0.0178 | 0.00068 | 0.0 |

- The trainer log shows `[sm120] TE FlashAttention: FA4 4.0.0b37 disabled, using flash-attn 2.7.4.post1`, and
  rollout/trainer logprobs agree at bf16 level (abs diff 0.015-0.019) across two weight syncs.
- SGLang: 1.2 GB of weights plus a 112k-token KV cache (12 GB) at `--sglang-mem-fraction-static 0.45`.
- Eval after rollout 2: 0/2. Both AIME answers hit the 8192-token cap.
- Checkpoint `checkpoints/iter_0000002` (7.8 GB with optimizer state). The HF export
  (`hf-model/`, 1.19 GB) has all 310 tensors of the original with the same names, shapes and dtypes. `lm_head.weight`
  is absent because the embeddings are tied (`tie_word_embeddings: true`). No reload test was run.

Three steps with 8 samples each show that the plumbing works, not that the model learns. Most responses hit the 8k
cap, so for real runs raise `--rollout-max-response-len` (and `--sglang-context-length` / `--max-tokens-per-gpu`
with it) or train on easier prompts. About 4.7 GB of GPU memory is still free at peak.

The training container runs with `--network none`. Ray, the SGLang servers and the router listen without
authentication, and another account logs in to this box, so nothing outside the container should reach them. Only
the asset download gets a network.

## Why the 5090 needs a profile

The image targets H100/H200/B200. Measured on the 5090 (sm_120) in the pinned image (TE 2.17.0+miles, cuDNN 9.24,
torch 2.14.1), packed `thd` attention like Megatron's, GQA 16/8, head_dim 128, bf16, relative error vs fp32:

| `--attention-backend` | what TE runs on sm_120 | fwd | dK/dV, doc lengths 37,50,123,300 | dK/dV, 64-aligned lengths |
|---|---|---|---|---|
| `flash` (handoff recipe) | **FlashAttention 4** 4.0.0b37 | 0.002 | **1.2 (wrong)** | 0.002 |
| `fused` | cuDNN (sub-backend 1) | 0.002 | **~130 (wrong)** | **8-70 (wrong)** |
| `unfused` | UnfusedDotProductAttention | 0.003 | 0.004 | |
| flash-attn 2.7.4 directly (`flash_attn_varlen_func`) | FA2, has sm_120 SASS | 0.002 | 0.003 (also 8000+2000) | 0.003 |
| **`flash` + `sm120.prefer_fa2` (rtx5090 profile)** | **FlashAttention 2.7.4.post1** | 0.002 | **0.003** (also 8000+2000, 1000+2000+37+4000) | 0.003 |

Miles packs documents at their natural lengths (only the stream end is padded to 128), so the recipe as written
would train on wrong gradients here without any error: the same FA4 sm_120 bug trl_rl found. On H100, TE prefers
FA3 and none of this applies, so the `h100` profile is the handoff recipe unchanged.

The `rtx5090` profile keeps `--attention-backend flash` and adds Miles' official trainer hook
`--custom-megatron-init-path sm120.prefer_fa2`, which hides FA4 from TE's backend selection in the trainer process so
TE uses the image's FA2 (`sm120.py`). SGLang is untouched. `unfused` is the hook-free fallback (correct, but it
materializes the score matrices, a memory risk at 8k-token responses).

`probe_attention.py` reruns the table. Set `PROBE_LENGTHS` inside the container, since `docker run` doesn't forward
host variables: `scripts/miles_docker.sh bash -c 'PROBE_LENGTHS=8000,2000 python probe_attention.py flash-fa2 flash'`.

## Verification: how we know training on the 5090 is right

Four layers, each using Miles' own tooling where it exists:

1. **Kernels** (`probe_attention.py`): the table above, per backend, forward and backward vs fp32.
2. **Whole-model gradients on real data** (`scripts/check_gradients.sh`, about 10 min). The script records two GSM8K
   rollouts once (`--debug-rollout-only --save-debug-rollout-data`), then trains on exactly those rollouts three
   ways (`--load-debug-rollout-data`). Megatron's grad norm is compared at every step with Miles CI's check
   (`--ci-save-grad-norm` / `--ci-load-grad-norm`, rel/abs tolerance 3%). Result `gradcheck1`, 2026-11-25:

   | attention path | grad norm step 0 | step 1 | verdict |
   |---|---|---|---|
   | `unfused` (reference, no flash kernels) | 0.64468 | 0.59032 | reference |
   | **rtx5090 profile (flash + FA2 hook)** | 0.64237 (−0.4%) | 0.58741 (−0.5%) | **MATCH** |
   | handoff recipe unchanged (FA4 on sm_120) | **202.7** (314×) | stopped | MISMATCH, the negative control |

3. **Invariants during training** (`--ci-test`, on in the `gsm8k` recipe): Miles CI's in-process assertions.
   At step 0: KL = 0 and clip fraction = 0. Trainer vs SGLang logprobs must be within 0.03. After each weight sync,
   trainer and engine weights must be equal. A violation stops the run where it happens.
4. **Learning vs an external baseline** (`RECIPE=gsm8k`): Miles CI's Qwen3-0.6B test
   (`tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py`, FSDP on 2× H200) with the same data, GRPO
   settings and pass bar. `--ci-metric-checker-key eval/gsm8k --ci-metric-checker-threshold 0.71` fails the run unless
   GSM8K test accuracy (greedy, 1319 problems) reaches 0.71 within 60 rollouts. The base-model eval before training
   gives the starting point on the same harness. Differences from the CI test: Megatron instead of FSDP, one GPU
   instead of two, `--max-tokens-per-gpu 10240` (a micro-batch size; same global batch of 256), and the KL checker
   left on.

   Result `gsm8k-1`, 2026-11-25, RTX 5090, 40 min of training: **passed** (`[MetricChecker] pass dispose check`, no
   `--ci-test` assertion fired in 60 steps):

   | GSM8K test (1319, greedy, 1024 tokens) | correct | accuracy | truncated | mean length |
   |---|---|---|---|---|
   | base model (eval before training) | 843 | 0.639 | 30.4% | 699 |
   | after 20 rollouts | 905 | 0.686 | 17.0% | 547 |
   | after 40 rollouts | 936 | 0.710 | 12.4% | 500 |
   | **after 60 rollouts** | **982** | **0.745** | **9.2%** | **485** |

   During training, trainer-vs-SGLang `train_rollout_logprob_abs_diff` stayed at 0.0136-0.0144 and `train_rollout_kl`
   at about 0.0006 from step 0 to step 59, with grad norm 0.60-0.79. The weight syncs delivered what the trainer
   held, without drift. Most of the gain comes from finishing within the 1024-token budget: answers that completed
   rose from 918 to 1198, and the share of those that were correct fell from 92% to 82%. That's what this recipe
   rewards. It doesn't show better math beyond fitting the budget.

```bash
scripts/check_gradients.sh                                        # layer 2
RECIPE=gsm8k RUN_ID=gsm8k-1 setsid nohup scripts/run_5090.sh --no-timeouts > data/logs/gsm8k-1.out 2>&1 &   # layers 3+4
```

Watch `train/train_rollout_logprob_abs_diff` in the training log: it compares SGLang's sampling logprobs with the
trainer's recomputed ones, so a wrong kernel on either side shows up there.

## Modal (not run yet)

`modal_miles.py` follows the Modal handoff brief (not included in this repo): same digest with `ENTRYPOINT []`,
files attached at `/opt/experiment` only when `modal.is_local()`, Volume `miles-qwen3-smoke-v2` (v2,
`create_if_missing=False`), CPU `download()` (2 cores, 32 GiB, 30 min) then GPU `train(run_id)` (`H100!`, 8 cores,
64 GiB, 90 min), no retries, one single-use container each. `train` runs `gpu_pipeline.run(profile="h100",
reuse_conversion=False, commit=volume.commit)`, so the H100 job converts fresh, trains 3 steps, evaluates on the
2 fixed problems, saves and exports, committing the Volume at stage transitions and every 30 s.

Nothing has been created or submitted on Modal. Before the first run, per the handoff: verify the workspace and
profile, the Volume status and current prices, then get the cost estimate and the exact commands approved.

```bash
uv sync --group modal                                     # isolated modal==1.6.4 CLI in .venv
.venv/bin/modal run --env main --detach --timestamps --write-result artifacts/download-result.json modal_miles.py::download
.venv/bin/modal run --env main --detach --timestamps --write-result artifacts/train-result.json \
  modal_miles.py::train --run-id <unique-run-id>
```

## HF export and `common.pt`

The pinned Megatron writes common state inside the torch_dist checkpoint (`common_state`), while the pinned Miles
exporter reads `<ckpt>/common.pt`. `checkpoint_compat.py` resolves the latest checkpoint from the tracker, builds a
temporary symlink view, writes `common.pt` there from Megatron's public `load_common_state_dict`, and runs the
unmodified `tools/convert_torch_dist_to_hf.py`. The checkpoint itself is never modified.

## Tests

```bash
uv sync --group dev && .venv/bin/python -m pytest -q      # on the Mac, CPU only
```
