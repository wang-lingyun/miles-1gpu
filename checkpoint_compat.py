"""Export the latest Megatron torch_dist checkpoint of a run to HF format with Miles' official exporter.

The pinned Megatron keeps the common (non-sharded) state inside the torch_dist checkpoint as a `common_state`
ShardedObject, while the pinned /root/miles/tools/convert_torch_dist_to_hf.py still reads `<ckpt>/common.pt`. This
builds a temporary view of the checkpoint (a symlink per entry, plus a common.pt written from Megatron's public
load_common_state_dict when the checkpoint has none) and runs the unmodified exporter on it. The checkpoint itself is
never written to.

    python checkpoint_compat.py --checkpoints /data/runs/<id>/checkpoints \
        --output-dir /data/runs/<id>/hf-model --origin-hf-dir /data/models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

EXPORTER = "/root/miles/tools/convert_torch_dist_to_hf.py"


def resolve_latest(checkpoints: Path) -> Path:
    from megatron.training.checkpointing import get_checkpoint_name

    tracker = (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip()
    release = tracker == "release"
    iteration = 0 if release else int(tracker)
    return Path(get_checkpoint_name(str(checkpoints), iteration, release=release, return_base_dir=True))


def build_view(checkpoint: Path, view: Path) -> None:
    for entry in checkpoint.iterdir():
        (view / entry.name).symlink_to(entry.resolve())
    if not (checkpoint / "common.pt").exists():
        import torch
        from megatron.core.dist_checkpointing.serialization import load_common_state_dict

        torch.save(load_common_state_dict(str(checkpoint)), view / "common.pt")


def export(checkpoints: Path, output_dir: Path, origin_hf_dir: Path, exporter: str = EXPORTER) -> Path:
    checkpoint = resolve_latest(checkpoints)
    print(f"[export] checkpoint {checkpoint}", flush=True)
    view = Path(tempfile.mkdtemp(prefix="ckpt-view-"))
    try:
        build_view(checkpoint, view)
        argv = sys.argv
        sys.argv = [exporter, "--input-dir", str(view), "--output-dir", str(output_dir),
                    "--origin-hf-dir", str(origin_hf_dir)]
        try:
            runpy.run_path(exporter, run_name="__main__")
        finally:
            sys.argv = argv
    finally:
        shutil.rmtree(view, ignore_errors=True)
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--origin-hf-dir", type=Path, required=True)
    args = ap.parse_args()
    checkpoint = export(args.checkpoints, args.output_dir, args.origin_hf_dir)
    print(f"[export] {checkpoint} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
