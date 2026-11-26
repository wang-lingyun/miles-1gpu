"""Stage runner and checkpoint view: logs, exit codes, timeouts. CPU only, no Miles or Megatron import."""

import os
import sys
import time

import pytest

import checkpoint_compat
from gpu_pipeline import StageFailed, conversion_complete, run_stage


def test_stage_output_is_logged(tmp_path):
    log = tmp_path / "logs" / "s.log"
    run_stage("s", [sys.executable, "-c", "print('hello'); print('world')"], log, timeout_s=30, commit=lambda: None)
    assert log.read_text().splitlines() == ["hello", "world"]


def test_nonzero_exit_raises(tmp_path):
    with pytest.raises(StageFailed, match="exited with 3"):
        run_stage("s", [sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "s.log", 30, lambda: None)


def test_timeout_kills_the_whole_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = f"import subprocess, time; p = subprocess.Popen(['sleep', '100']); open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(100)"
    t0 = time.monotonic()
    with pytest.raises(StageFailed, match="timed out"):
        run_stage("s", [sys.executable, "-c", script], tmp_path / "s.log", timeout_s=2, commit=lambda: None)
    assert time.monotonic() - t0 < 30
    child = int(pidfile.read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the grandchild went down with the group


def test_conversion_complete_needs_tracker_and_marker(tmp_path):
    assert not conversion_complete(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    assert not conversion_complete(tmp_path)  # tracker alone could be a crashed conversion
    (tmp_path / ".conversion-complete").write_text("{}")
    assert conversion_complete(tmp_path)


def test_checkpoint_view_links_everything_and_keeps_existing_common_pt(tmp_path):
    ckpt, view = tmp_path / "iter_0000002", tmp_path / "view"
    ckpt.mkdir(), view.mkdir()
    for name in (".metadata", "__0_0.distcp", "common.pt"):
        (ckpt / name).write_text(name)
    checkpoint_compat.build_view(ckpt, view)  # common.pt present: no Megatron import needed
    assert sorted(p.name for p in view.iterdir()) == [".metadata", "__0_0.distcp", "common.pt"]
    assert all((view / p).is_symlink() for p in (".metadata", "__0_0.distcp", "common.pt"))
    assert (view / "common.pt").read_text() == "common.pt"
