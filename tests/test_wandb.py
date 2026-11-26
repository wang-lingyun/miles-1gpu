"""W&B wiring: on only with MILES_WANDB=1 and a key, key only in the training stage's environment, never in argv."""

from pathlib import Path

import pytest

import gpu_pipeline as gp
from run_single_gpu import build_argv, train_env

KEY = "wandb_v1_" + "x" * 77


@pytest.fixture
def wandb_on(monkeypatch):
    monkeypatch.setenv("MILES_WANDB", "1")
    monkeypatch.setenv("WANDB_API_KEY", KEY)
    monkeypatch.setenv("WANDB_MODE", "disabled")  # in_container.sh's default when W&B is off; must not leak through
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)


def test_off_without_flag_or_key(monkeypatch):
    monkeypatch.delenv("MILES_WANDB", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", KEY)
    assert gp.wandb_args("r1", Path("/data/runs/r1")) == []
    monkeypatch.setenv("MILES_WANDB", "1")
    monkeypatch.delenv("WANDB_API_KEY")
    assert gp.wandb_args("r1", Path("/data/runs/r1")) == []


def test_flags_name_the_run_and_keep_files_in_the_run_dir(wandb_on):
    flags = gp.wandb_args("gsm8k-2", Path("/data/runs/gsm8k-2"))
    assert flags[:3] == ["--use-wandb", "--wandb-mode", "online"]
    assert flags[flags.index("--wandb-project") + 1] == "miles-1gpu"
    assert flags[flags.index("--wandb-group") + 1] == "gsm8k-2"
    assert "--disable-wandb-random-suffix" in flags
    assert flags[flags.index("--wandb-dir") + 1] == "/data/runs/gsm8k-2"
    assert "--wandb-team" not in flags  # default entity comes from the key


def test_entity_and_project_from_env(wandb_on, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "some-team")
    monkeypatch.setenv("WANDB_PROJECT", "other")
    flags = gp.wandb_args("r1", Path("/r"))
    assert flags[flags.index("--wandb-team") + 1] == "some-team"
    assert flags[flags.index("--wandb-project") + 1] == "other"


def test_key_never_in_argv(wandb_on):
    argv = build_argv("gsm8k", "rtx5090", data_root=Path("/data"), run_dir=Path("/data/runs/r1"),
                      conversion_dir=Path("/c"), extra=gp.wandb_args("r1", Path("/data/runs/r1")))
    assert "--wandb-key" not in argv and not any(KEY in t for t in argv)


def test_key_only_in_wandb_training_env(wandb_on):
    off, on = gp.stage_env(), gp.stage_env(wandb=True)
    assert "WANDB_API_KEY" not in off and off["WANDB_MODE"] == "disabled"
    assert on["WANDB_API_KEY"] == KEY and "WANDB_MODE" not in on


def test_ray_job_env_never_carries_the_key():
    assert "WANDB_API_KEY" not in train_env(True) and "WANDB_MODE" not in train_env(True)
    assert train_env(False)["WANDB_MODE"] == "disabled"


@pytest.mark.parametrize("url", ["https://wandb.ai/team-x/miles-1gpu/runs/ab12cd34",
                                 "https://forge.coreweave.com/wandb/team-x/miles-1gpu/runs/7xbqu14w"])
def test_wandb_url_from_log(tmp_path, url):
    log = tmp_path / "training.log"
    log.write_text("(MegatronTrainRayActor pid=1) wandb: ⭐️ View project at https://example/team-x/miles-1gpu\n"
                   f"(MegatronTrainRayActor pid=1) wandb: 🚀 View run at {url}\n")
    assert gp.wandb_url(log) == url
    assert gp.wandb_url(tmp_path / "missing.log") is None
