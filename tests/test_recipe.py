"""The train.py argv: profiles, paths, batch arithmetic, overrides. CPU only, no Miles import."""

from pathlib import Path

import pytest

from run_single_gpu import COMMON, PROFILES, RECIPES, build_argv


def as_dict(argv: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    flag = None
    for tok in argv:
        if tok.startswith("--"):
            flag = tok
            out[flag] = []
        else:
            out[flag].append(tok)
    return out


def argv_for(profile: str, extra=None, recipe="dapo-smoke", drop=None) -> list[str]:
    return build_argv(recipe, profile, data_root=Path("/data"), run_dir=Path("/data/runs/r1"),
                      conversion_dir=Path("/data/converted/c1"), drop=drop, extra=extra)


def test_profiles_differ_only_in_their_overrides():
    h100, rtx = as_dict(argv_for("h100")), as_dict(argv_for("rtx5090"))
    differing = {f for f in h100.keys() | rtx.keys() if h100.get(f) != rtx.get(f)}
    assert differing == set(PROFILES["rtx5090"])
    assert h100["--attention-backend"] == rtx["--attention-backend"] == ["flash"]  # handoff recipe unchanged
    assert rtx["--custom-megatron-init-path"] == ["sm120.prefer_fa2"]  # FA2 instead of FA4 on sm_120
    assert "--custom-megatron-init-path" not in h100


def test_paths():
    a = as_dict(argv_for("rtx5090"))
    assert a["--hf-checkpoint"] == ["/data/models/Qwen3-0.6B"]
    assert a["--ref-load"] == ["/data/converted/c1"]
    assert a["--save"] == ["/data/runs/r1/checkpoints"]
    assert a["--prompt-data"] == ["/data/datasets/dapo-math-17k/dapo-math-17k.jsonl"]
    assert a["--eval-prompt-data"] == ["aime-smoke", "/data/runs/r1/eval-subset.jsonl"]
    assert "--load" not in a  # fresh training, not resume


def test_batch_arithmetic_and_final_eval():
    a = {k: int(v[0]) for k, v in as_dict(argv_for("h100")).items()
         if k in ("--rollout-batch-size", "--n-samples-per-prompt", "--global-batch-size", "--num-steps-per-rollout",
                  "--num-rollout", "--eval-interval", "--save-interval")}
    assert a["--rollout-batch-size"] * a["--n-samples-per-prompt"] == (
        a["--global-batch-size"] * a["--num-steps-per-rollout"])
    assert a["--num-rollout"] % a["--eval-interval"] == 0
    assert a["--num-rollout"] % a["--save-interval"] == 0


def test_single_gpu_colocated():
    a = as_dict(argv_for("rtx5090"))
    assert "--colocate" in a
    for flag in ("--actor-num-nodes", "--actor-num-gpus-per-node", "--num-gpus-per-node",
                 "--rollout-num-gpus-per-engine", "--tensor-model-parallel-size"):
        assert a[flag] == ["1"], flag


def test_extra_args_come_last_so_they_win():
    argv = argv_for("rtx5090", extra=["--num-rollout", "50"])
    assert argv[-2:] == ["--num-rollout", "50"]
    assert argv.index("--num-rollout") < len(argv) - 2  # recipe value is still there, earlier


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_every_flag_is_emitted_once(recipe):
    flags = [t for t in argv_for("h100", recipe=recipe) if t.startswith("--")]
    assert len(flags) == len(set(flags))
    assert set(COMMON) | set(RECIPES[recipe]) == set(flags)
    assert not any("{" in t for t in argv_for("rtx5090", recipe=recipe))  # every path placeholder filled


def test_handoff_smoke_recipe_is_unchanged():
    a = as_dict(argv_for("h100"))
    assert a["--attention-backend"] == ["flash"] and a["--sglang-mem-fraction-static"] == ["0.45"]
    assert a["--rollout-max-response-len"] == ["8192"] and a["--max-tokens-per-gpu"] == ["10240"]
    assert "--use-kl-loss" in a and "--skip-eval-before-train" in a


def test_gsm8k_matches_miles_ci_recipe():
    a = as_dict(argv_for("rtx5090", recipe="gsm8k"))
    assert a["--prompt-data"] == ["/data/datasets/gsm8k/train.parquet"]
    assert a["--eval-prompt-data"] == ["gsm8k", "/data/datasets/gsm8k/test.parquet"]
    for flag, value in {"--input-key": "messages", "--rm-type": "math", "--num-rollout": "60",
                        "--rollout-batch-size": "32", "--n-samples-per-prompt": "8", "--global-batch-size": "256",
                        "--rollout-max-response-len": "1024", "--over-sampling-batch-size": "64",
                        "--eval-interval": "20", "--eval-top-k": "1", "--lr": "1e-6",
                        "--ci-metric-checker-key": "eval/gsm8k", "--ci-metric-checker-threshold": "0.71"}.items():
        assert a[flag] == [value], flag
    assert "--use-kl-loss" not in a  # commented out in the CI test
    assert "--skip-eval-before-train" not in a  # base-model eval at rollout 0 is the start of the curve


def test_drop_removes_flags_and_rejects_unknown():
    a = as_dict(argv_for("rtx5090", recipe="gsm8k", drop=["eval-interval", "--save", "ci-test"]))
    assert not {"--eval-interval", "--save", "--ci-test"} & set(a)
    with pytest.raises(SystemExit):
        argv_for("rtx5090", recipe="gsm8k", drop=["skip-eval-before-train"])


def test_unknown_profile_or_recipe():
    with pytest.raises(SystemExit):
        argv_for("a100")
    with pytest.raises(SystemExit):
        argv_for("h100", recipe="mmlu")
