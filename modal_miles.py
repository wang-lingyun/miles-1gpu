"""Modal: the same experiment on one H100, sharing everything with the RTX 5090 path.

Same pinned image digest, same assets.py, same gpu_pipeline.py and run_single_gpu.py, same in-container paths
(/opt/experiment for this repo, /data for the Volume). Only the profile (h100: the handoff recipe unchanged) and
fresh conversion per run differ.

Importing this module builds nothing and creates nothing remotely. Running it costs money: see the README's Modal
section and get the cost and commands approved first. The CPU job runs first, then the GPU job:

    .venv/bin/modal run --env main --detach --timestamps --write-result artifacts/download-result.json \
        modal_miles.py::download
    .venv/bin/modal run --env main --detach --timestamps --write-result artifacts/train-result.json \
        modal_miles.py::train --run-id <unique-run-id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/experiment")
CONFIG = json.loads((ROOT / "experiment-config.json").read_text())
PLATFORM = CONFIG["platforms"]["modal"]
REMOTE_DIR = "/opt/experiment"
FILES = ["experiment-config.json", "assets.py", "run_single_gpu.py", "gpu_pipeline.py", "checkpoint_compat.py", "sm120.py"]

GPU_ENV = {
    "PYTHONPATH": f"{REMOTE_DIR}:/root/miles:/root/Megatron-LM",
    "PYTHONUNBUFFERED": "1",
    "WANDB_MODE": "disabled",
    "RAY_USAGE_STATS_ENABLED": "0",
    "NCCL_NVLS_ENABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}


def _with_experiment_files(image: modal.Image) -> modal.Image:
    # Only the submitting machine has the files; a remote import must not try to upload anything.
    if modal.is_local():
        for name in FILES:
            image = image.add_local_file(ROOT / name, f"{REMOTE_DIR}/{name}")
    return image


cpu_image = _with_experiment_files(
    modal.Image.debian_slim(python_version=CONFIG["pins"]["downloader"]["python"]).pip_install(
        f"huggingface_hub=={CONFIG['pins']['downloader']['huggingface_hub']}"
    )
)
# The Miles image as published: its own Python (/opt/sglang/bin/python), CUDA stack, /root/miles, /root/Megatron-LM.
gpu_image = _with_experiment_files(
    modal.Image.from_registry(CONFIG["pins"]["image"]).dockerfile_commands("ENTRYPOINT []").env(GPU_ENV)
)

volume = modal.Volume.from_name(
    PLATFORM["volume"], environment_name=PLATFORM["environment"], create_if_missing=False, version=2
)
app = modal.App("miles-qwen3-smoke")


def _resources(kind: str) -> dict:
    r = PLATFORM[kind]
    return dict(
        cpu=tuple(r["cpu"]),
        memory=tuple(r["memory_mib"]),
        ephemeral_disk=r["ephemeral_disk_mib"],
        timeout=r["timeout_s"],
        startup_timeout=r["startup_timeout_s"],
        retries=0,
        max_containers=1,
        single_use_containers=True,
        volumes={PLATFORM["data_root"]: volume},
    )


@app.function(image=cpu_image, **_resources("download"))
def download() -> dict:
    sys.path.insert(0, REMOTE_DIR)
    import assets

    manifest = assets.download_all(Path(PLATFORM["data_root"]))
    volume.commit()
    return manifest


@app.function(image=gpu_image, gpu=PLATFORM["train"]["gpu"], **_resources("train"))
def train(run_id: str, recipe: str = PLATFORM["recipe"]) -> dict:
    sys.path.insert(0, REMOTE_DIR)
    import gpu_pipeline

    return gpu_pipeline.run(
        profile=PLATFORM["profile"],
        run_id=run_id,
        recipe=recipe,
        data_root=Path(PLATFORM["data_root"]),
        reuse_conversion=PLATFORM["reuse_existing_conversion"],
        commit=volume.commit,
    )
