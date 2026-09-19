"""Validate transfer timing semantics without GPU synchronization or RDMA."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "transfer_profile",
    ROOT / "python/sglang/srt/disaggregation/common/transfer_profile.py",
)
transfer_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_profile)


def test_profile_separates_build_submit_queue_and_bytes(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        transfer_profile, "time", SimpleNamespace(perf_counter=lambda: now[0])
    )
    profile = transfer_profile.DCPTransferProfile(
        room=123,
        worker=2,
        source_rank=0,
        chunk=SimpleNamespace(
            index_slice=slice(0, 10), num_kv_tokens=640, is_last_chunk=True
        ),
        enqueued_at=95,
    )

    def batches():
        now[0] += 2
        yield [(100, 200, 128), (300, 400, 4)]
        now[0] += 3
        yield [(500, 600, 64)]

    def send(session, blocks):
        assert session == "peer"
        now[0] += 7
        return 0

    for blocks in profile.batches(batches()):
        assert profile.transfer("dsa", send, "peer", blocks) == 0
    logs = []
    profile.log(
        SimpleNamespace(info=lambda fmt, data: logs.append(json.loads(data))),
        error=False,
    )
    result = logs[0]
    assert result["queue_wait_s"] == 5
    assert result["dsa_build_s"] == 5
    assert result["dsa_submit_s"] == 14
    assert result["wall_s"] == 19
    assert result["dsa_bytes"] == 196
    assert result["dsa_blocks"] == 3
    assert result["dsa_calls"] == 2
    assert result["room"] == 123
    assert result["error"] is False


@pytest.mark.parametrize("raises", [False, True])
def test_profile_preserves_transfer_failures(raises):
    profile = transfer_profile.DCPTransferProfile(
        room=1,
        worker=0,
        source_rank=0,
        chunk=SimpleNamespace(
            index_slice=slice(0, 1), num_kv_tokens=1, is_last_chunk=False
        ),
        enqueued_at=0,
    )

    def fail(session, blocks):
        if raises:
            raise RuntimeError("engine failed")
        return -1

    if raises:
        with pytest.raises(RuntimeError, match="engine failed"):
            profile.transfer("mla", fail, "peer", [(1, 2, 64)])
    else:
        assert profile.transfer("mla", fail, "peer", [(1, 2, 64)]) == -1
    assert profile.values["mla_failures"] == 1
    assert profile.values["mla_bytes"] == 64
    assert "mla_submit_s" in profile.values
