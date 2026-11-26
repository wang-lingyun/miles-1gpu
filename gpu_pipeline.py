"""One GPU job, inside the Miles image: HF -> torch_dist conversion, GRPO training (eval + save), HF export.

Same code on the RTX 5090 (scripts/run_5090.sh, Docker) and on Modal (modal_miles.py::train). Each stage is a
subprocess whose output streams to the console and to <run-dir>/logs/<stage>.log; <run-dir>/status.json records
stage states and times. A nonzero exit or a timeout stops the pipeline and the exit code propagates.

    python gpu_pipeline.py --profile rtx5090 --run-id 20261125-1200            # conversion reused if complete
    python gpu_pipeline.py --profile h100 --run-id <id> --fresh-conversion     # what Modal runs
    python gpu_pipeline.py --recipe gsm8k --profile rtx5090 --run-id g1 --no-timeouts   # Miles CI baseline recipe
    python gpu_pipeline.py ... --stages training -- --num-rollout 20           # extra flags go to train.py
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment-config.json").read_text())
MILES = Path("/root/miles")
MEGATRON = Path("/root/Megatron-LM")
STAGES = ("conversion", "training", "hf_export")
DONE_MARKER = ".conversion-complete"


class StageFailed(RuntimeError):
    pass


def model_args() -> list[str]:
    script = MILES / "scripts" / "models" / f"{CONFIG['megatron_model_type']}.py"
    return shlex.split(runpy.run_path(str(script))["model_args"]())


def stage_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("WANDB_API_KEY", "MILES_SCRIPT_EXTERNAL_RAY")}
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(MILES), str(MEGATRON)]),
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
        RAY_USAGE_STATS_ENABLED="0",
        NCCL_NVLS_ENABLE="0",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MASTER_ADDR="127.0.0.1",
    )
    return env


class Status:
    def __init__(self, path: Path, base: dict, commit: Callable[[], None]):
        self.path, self.commit = path, commit
        self.data = {**base, "state": "running", "stages": {}, "started": time.time()}
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        tmp.replace(self.path)

    def stage(self, name: str, **fields) -> None:
        self.data["stages"].setdefault(name, {}).update(fields)
        self.write()
        self.commit()

    def finish(self, state: str, **fields) -> None:
        self.data.update(state=state, elapsed_s=round(time.time() - self.data["started"], 1), **fields)
        self.write()
        self.commit()


def run_stage(
    name: str,
    cmd: list[str],
    log_path: Path,
    timeout_s: float | None,
    commit: Callable[[], None],
    commit_every_s: float = 30.0,
) -> None:
    """Run cmd in its own process group, tee its output, kill the group on timeout, raise on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] {name}: {shlex.join(cmd)}", flush=True)
    with open(log_path, "a", buffering=1) as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=stage_env(), start_new_session=True,
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        last_commit = time.monotonic()
        while proc.poll() is None:
            time.sleep(1)
            if deadline is not None and time.monotonic() > deadline:
                print(f"[pipeline] {name}: timeout after {timeout_s:.0f} s, stopping process group", flush=True)
                _stop_group(proc)
                reader.join(timeout=10)
                raise StageFailed(f"{name} timed out after {timeout_s:.0f} s")
            if time.monotonic() - last_commit > commit_every_s:
                commit()
                last_commit = time.monotonic()
        reader.join(timeout=10)
    if proc.returncode != 0:
        raise StageFailed(f"{name} exited with {proc.returncode} (log: {log_path})")


def _stop_group(proc: subprocess.Popen, grace_s: float = 60.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def gpu_monitor(path: Path, interval_s: int = 5) -> subprocess.Popen | None:
    if shutil.which("nvidia-smi") is None:
        return None
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw",
         "--format=csv", "-l", str(interval_s)],
        stdout=open(path, "w"), stderr=subprocess.DEVNULL,
    )


def conversion_complete(conv_dir: Path) -> bool:
    tracker = conv_dir / "latest_checkpointed_iteration.txt"
    return (conv_dir / DONE_MARKER).exists() and tracker.exists() and tracker.read_text().strip() == "release"


def run(
    *,
    profile: str,
    run_id: str,
    recipe: str = "dapo-smoke",
    data_root: Path = Path("/data"),
    conversion_id: str | None = None,
    reuse_conversion: bool = True,
    stages: tuple[str, ...] = STAGES,
    extra_train_args: list[str] | None = None,
    drop: list[str] | None = None,
    timeouts: bool = True,
    gpu_log: bool = False,
    commit: Callable[[], None] = lambda: None,
) -> dict:
    sys.path.insert(0, str(ROOT))
    from run_single_gpu import build_argv

    t0 = time.monotonic()
    limits = CONFIG["stage_timeouts_s"]
    model_dir = data_root / CONFIG["assets"]["model"]["dest"]
    conversion_id = conversion_id or (run_id if not reuse_conversion else f"qwen3-0.6B-{image_tag()}")
    conv_dir = data_root / "converted" / conversion_id
    run_dir = data_root / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists; pick a new --run-id")
    (run_dir / "logs").mkdir(parents=True)
    shutil.copy(data_root / CONFIG["eval_subset"]["dest"], run_dir / CONFIG["eval_subset"]["dest"])

    drop = list(drop or [])
    train_argv = build_argv(recipe, profile, data_root=data_root, run_dir=run_dir, conversion_dir=conv_dir,
                            drop=drop, extra=extra_train_args)
    base = {"run_id": run_id, "recipe": recipe, "profile": profile, "conversion_dir": str(conv_dir),
            "run_dir": str(run_dir)}
    (run_dir / "run-config.json").write_text(json.dumps(
        {**base, "pins": CONFIG["pins"], "stages": list(stages), "train_argv": train_argv,
         "drop": drop, "extra_train_args": extra_train_args or []}, indent=2) + "\n")
    status = Status(run_dir / "status.json", base, commit)
    monitor = gpu_monitor(run_dir / "logs" / "gpu.csv") if gpu_log else None

    def budget(stage: str) -> float | None:
        if not timeouts:
            return None
        remaining = limits["overall"] - (time.monotonic() - t0)
        if remaining <= 0:
            raise StageFailed(f"overall deadline of {limits['overall']} s reached before {stage}")
        return min(limits[stage], remaining)

    def stage(name: str, cmd: list[str]) -> None:
        started = time.monotonic()
        status.stage(name, state="running", log=f"logs/{name}.log")
        try:
            run_stage(name, cmd, run_dir / "logs" / f"{name}.log", budget(name), commit)
        except BaseException as exc:
            status.stage(name, state="failed", error=str(exc), elapsed_s=round(time.monotonic() - started, 1))
            raise
        status.stage(name, state="done", elapsed_s=round(time.monotonic() - started, 1))

    try:
        if "conversion" in stages:
            if reuse_conversion and conversion_complete(conv_dir):
                print(f"[pipeline] conversion: reusing {conv_dir}", flush=True)
                status.stage("conversion", state="reused")
            elif conv_dir.exists():
                raise SystemExit(f"{conv_dir} exists but is not a complete conversion; remove it or pass another "
                                 "--conversion-id")
            else:
                stage("conversion", [
                    sys.executable, str(MILES / "tools" / "convert_hf_to_torch_dist.py"), *model_args(),
                    "--hf-checkpoint", str(model_dir), "--save", str(conv_dir),
                ])
                (conv_dir / DONE_MARKER).write_text(json.dumps({"run_id": run_id, "pins": CONFIG["pins"]}) + "\n")
        if "training" in stages:
            stage("training", [
                sys.executable, str(ROOT / "run_single_gpu.py"), "--recipe", recipe, "--profile", profile,
                "--data-root", str(data_root), "--run-dir", str(run_dir), "--conversion-dir", str(conv_dir),
                "--drop", ",".join(drop), "--", *(extra_train_args or []),
            ])
        result = {}
        if "hf_export" in stages:
            stage("hf_export", [
                sys.executable, str(ROOT / "checkpoint_compat.py"), "--checkpoints", str(run_dir / "checkpoints"),
                "--output-dir", str(run_dir / "hf-model"), "--origin-hf-dir", str(model_dir),
            ])
            result["hf_model"] = str(run_dir / "hf-model")
        tracker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            result["latest_checkpoint_iteration"] = tracker.read_text().strip()
        status.finish("done", result=result)
        return status.data
    except BaseException as exc:
        status.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if monitor is not None:
            monitor.terminate()


def image_tag() -> str:
    return CONFIG["pins"]["image"].rsplit(":", 1)[-1][:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--recipe", default="dapo-smoke")
    ap.add_argument("--drop", default="", help="comma-separated train.py flags to remove (see run_single_gpu.py)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    ap.add_argument("--conversion-id", help="default: shared per image when reusing, else the run id")
    ap.add_argument("--fresh-conversion", action="store_true", help="always convert into a new directory")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--no-timeouts", action="store_true", help="for long local runs")
    ap.add_argument("--gpu-log", action="store_true", help="sample nvidia-smi into logs/gpu.csv every 5 s")
    ap.add_argument("extra", nargs="*", help="extra train.py flags (after --)")
    args = ap.parse_args()
    stages = tuple(s for s in args.stages.split(",") if s)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}")
    try:
        run(profile=args.profile, run_id=args.run_id, recipe=args.recipe, data_root=args.data_root,
            conversion_id=args.conversion_id, reuse_conversion=not args.fresh_conversion, stages=stages,
            extra_train_args=args.extra, drop=[d for d in args.drop.split(",") if d],
            timeouts=not args.no_timeouts, gpu_log=args.gpu_log)
    except StageFailed as exc:
        print(f"[pipeline] FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
