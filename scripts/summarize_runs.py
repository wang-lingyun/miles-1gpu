"""Compare finished runs from their logs: eval curve, Miles CI verdict, speed, and on/off-policy health.

    python3 scripts/summarize_runs.py data/runs/sva1-*          # host python, stdlib only
    python3 scripts/summarize_runs.py --threshold 0.71 --json data/runs/sva1-* > sva1.json

Reads <run>/logs/training.log (Miles' own "eval N:", "rollout N:", "step N:" lines) and <run>/run-config.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

TS = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
LINE = {
    "eval": re.compile(r"metrics\.py:\d+ - eval (\d+): (\{.*\})\s*$"),
    "rollout": re.compile(r"log_utils\.py:\d+ - rollout (\d+): (\{.*\})\s*$"),
    "step": re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})\s*$"),
    "perf": re.compile(r"\.py:\d+ - perf (\d+): (\{.*\})\s*$"),  # several processes; merged per id
}


def parse(run: Path) -> dict:
    series: dict[str, dict[int, tuple[datetime | None, dict]]] = {k: {} for k in LINE}
    text = (run / "logs" / "training.log").read_text(errors="replace")
    for line in text.splitlines():
        for kind, rx in LINE.items():
            m = rx.search(line)
            if not m:
                continue
            ts = TS.search(line)
            try:
                d = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
            when = datetime.fromisoformat(ts.group(1)) if ts else None
            key = int(m.group(1))
            if kind == "perf" and key in series[kind]:
                d = {**series[kind][key][1], **d}
            series[kind][key] = (when, d)
    cfg = json.loads((run / "run-config.json").read_text()) if (run / "run-config.json").exists() else {}
    verdict = ("pass" if "[MetricChecker] pass" in text else
               "FAIL" if re.search(r"MetricChecker|metric_checker", text) and "AssertionError" in text else
               "no gate" if "--ci-metric-checker-key" not in cfg.get("train_argv", []) else "unfinished")
    mem = []
    if (gpu := run / "logs" / "gpu.csv").exists():
        for row in gpu.read_text().splitlines()[1:]:
            parts = [c.strip() for c in row.split(",")]
            if len(parts) > 1 and parts[1].split()[0].isdigit():
                mem.append(int(parts[1].split()[0]))
    return {"run": run.name, "recipe": cfg.get("recipe"), "argv": cfg.get("train_argv", []), "verdict": verdict,
            "gpu_peak_mib": max(mem) if mem else None, **series}


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None


def summarize(p: dict, threshold: float) -> dict:
    evals = {k: d.get("eval/gsm8k") for k, (_, d) in sorted(p["eval"].items())}
    rollouts, steps = p["rollout"], p["step"]
    times = [t for t, _ in (rollouts[k] for k in sorted(rollouts)) if t]
    train_s = (times[-1] - times[0]).total_seconds() if len(times) > 1 else None
    per_step = train_s / (len(times) - 1) if train_s else None
    hit = next((k for k, v in evals.items() if k > 0 and v is not None and v >= threshold), None)
    t_hit = None
    if hit is not None and times and p["eval"][hit][0]:
        t_hit = (p["eval"][hit][0] - times[0]).total_seconds()
    rd = [d for _, d in rollouts.values()]
    sd = [d for _, d in steps.values()]
    pd = [d for _, d in p["perf"].values()]
    return {
        "run": p["run"], "recipe": p["recipe"], "verdict": p["verdict"],
        "evals": evals, "final_eval": evals[max(evals)] if evals else None,
        "rollouts": len(rollouts), "train_minutes": round(train_s / 60, 1) if train_s else None,
        "sec_per_step": round(per_step, 1) if per_step else None,
        "first_eval_at_threshold": hit, "minutes_to_threshold": round(t_hit / 60, 1) if t_hit else None,
        "raw_reward": mean(d.get("rollout/raw_reward_filtered", d.get("rollout/raw_reward")) for d in rd),
        "truncated": mean(d.get("rollout/truncated") for d in rd),
        "logprob_abs_diff": mean(d.get("train/train_rollout_logprob_abs_diff") for d in sd),
        "train_rollout_kl": mean(d.get("train/train_rollout_kl") for d in sd),
        "avg_staleness": mean(d.get("rollout/fully_async/avg_staleness") for d in pd),
        "max_staleness": max((d.get("rollout/fully_async/max_staleness") or 0 for d in pd), default=None),
        "stale_groups_filtered": sum(d.get("rollout/fully_async/stale_groups_filtered") or 0 for d in pd),
        "gpu_peak_gb": round(p["gpu_peak_mib"] / 1024, 1) if p["gpu_peak_mib"] else None,
    }


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.71)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = [summarize(parse(r), args.threshold) for r in args.runs if (r / "logs" / "training.log").exists()]
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return
    steps = sorted({k for r in rows for k in r["evals"]})
    head = ["run", "CI", *[f"eval@{k}" for k in steps], "min", "s/step", f"min→{args.threshold}", "reward",
            "trunc", "lp_diff", "staleness", "GPU peak GiB"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [r["run"], r["verdict"], *[fmt(r["evals"].get(k)) for k in steps], fmt(r["train_minutes"], 1),
                 fmt(r["sec_per_step"], 1), fmt(r["minutes_to_threshold"], 1), fmt(r["raw_reward"]),
                 fmt(r["truncated"]), fmt(r["logprob_abs_diff"], 4),
                 "-" if r["avg_staleness"] is None else f"{r['avg_staleness']:.2f} (max {r['max_staleness']})",
                 fmt(r["gpu_peak_gb"], 1)]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
