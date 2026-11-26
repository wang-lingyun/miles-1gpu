"""Download the pinned model and datasets into <data-root>, then write the fixed eval subset and a manifest.

CPU only, anonymous reads (token=False). huggingface_hub manages its cache and resumes interrupted downloads.

    python assets.py --data-root /data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment-config.json").read_text())


def download_all(data_root: Path) -> dict:
    from huggingface_hub import snapshot_download

    data_root = Path(data_root)
    manifest: dict = {"assets": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for name, spec in CONFIG["assets"].items():
        dest = data_root / spec["dest"]
        print(f"[assets] {spec['repo_id']}@{spec['revision'][:12]} -> {dest}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            revision=spec["revision"],
            allow_patterns=spec["files"],
            local_dir=str(dest),
            token=False,
        )
        manifest["assets"][name] = {k: spec[k] for k in ("repo_id", "repo_type", "revision", "files")} | {
            "path": str(dest)
        }

    subset = CONFIG["eval_subset"]
    source = data_root / CONFIG["assets"][subset["source"]]["dest"] / subset["file"]
    rows = source.read_text().splitlines()
    picked = [rows[i] for i in subset["indices"]]
    (data_root / subset["dest"]).write_text("\n".join(picked) + "\n")
    manifest["eval_subset"] = {"source": str(source), "indices": subset["indices"], "path": str(data_root / subset["dest"])}

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (data_root / CONFIG["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[assets] done; manifest {data_root / CONFIG['manifest']}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    download_all(ap.parse_args().data_root)


if __name__ == "__main__":
    main()
