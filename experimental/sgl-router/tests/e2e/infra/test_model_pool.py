import os
import signal
import types

import pytest
from infra import model_pool


class _Process:
    pid = 1234

    def __init__(self):
        self.wait_timeouts = []

    def poll(self):
        return None

    def send_signal(self, sig):
        raise AssertionError(f"signaled only the parent process: {sig}")

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return 0


def test_shutdown_waits_for_the_worker_process_group(monkeypatch):
    process = _Process()
    signals = []
    probes = iter([True, True, False])

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == 0 and not next(probes):
            raise ProcessLookupError

    monkeypatch.setattr(model_pool.os, "killpg", killpg)
    monkeypatch.setattr(model_pool.time, "sleep", lambda _: None)

    instance = model_pool.ModelInstance(
        url="http://127.0.0.1:30000",
        port=30000,
        process=process,
        model_id="qwen3-0.6b",
    )
    instance.shutdown()

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, 0),
        (process.pid, 0),
        (process.pid, 0),
    ]
    assert process.wait_timeouts == [60]


def test_resolve_device_ids_maps_onto_this_processs_allotment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert model_pool.resolve_device_ids([0, 1]) == ["2", "3"]
    assert model_pool.resolve_device_ids([1]) == ["3"]


def test_resolve_device_ids_keeps_uuid_entries_and_trims_whitespace(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " GPU-abc , MIG-def ")

    assert model_pool.resolve_device_ids([1, 0]) == ["MIG-def", "GPU-abc"]


def test_resolve_device_ids_passes_indices_through_when_unrestricted(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert model_pool.resolve_device_ids([0, 3]) == ["0", "3"]


def test_resolve_device_ids_rejects_a_gpu_outside_the_allotment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES=2,3"):
        model_pool.resolve_device_ids([2])


def _spawn_capture(monkeypatch) -> dict:
    """Stub out the worker process and report how it was launched."""
    spawned: dict = {}

    class _Popen:
        pid = 4321
        returncode = None

        def __init__(self, cmd, env=None, **kwargs):
            spawned["cmd"] = cmd
            spawned["env"] = env
            spawned["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(model_pool.subprocess, "Popen", _Popen)
    monkeypatch.setattr(
        model_pool.httpx,
        "get",
        lambda *a, **kw: types.SimpleNamespace(status_code=200),
    )
    return spawned


def test_spawn_worker_binds_the_resolved_devices(monkeypatch, tmp_path):
    """The child's ``CUDA_VISIBLE_DEVICES`` must carry resolved devices, never
    the raw logical indices — those are absolute, and land on another CI job's
    GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(model_pool.tempfile, "gettempdir", lambda: str(tmp_path))
    spawned = _spawn_capture(monkeypatch)

    inst = model_pool.spawn_worker("qwen3-0.6b", gpu_ids=[1], port=31000)

    assert spawned["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    # Logical, so the caller can hand them straight back to the allocator.
    assert inst.gpu_ids == [1]


@pytest.mark.parametrize(
    ("cuda_visible_devices", "expected"),
    [
        ("2,3", ["2", "3"]),
        ("", []),
        ("-1", []),
        ("0,-1,1", ["0"]),
        ("0,0", ["0"]),
        ("0,1,0", ["0", "1"]),
        ("0,", ["0"]),
        ("none", []),
    ],
)
def test_visible_devices_stops_where_cuda_stops(
    monkeypatch, cuda_visible_devices, expected
):
    """CUDA truncates enumeration at the first invalid or repeated entry. Any
    extra token counted here becomes a logical index that resolves onto a card
    another worker already holds, or onto no card at all."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)

    assert model_pool.visible_devices() == expected


def test_visible_devices_is_none_when_unrestricted(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert model_pool.visible_devices() is None


def test_spawn_worker_leaves_this_process_s_own_allotment_alone(monkeypatch, tmp_path):
    """Resolving must not narrow the parent's view: the next worker would then
    resolve its logical index onto the card this one is already using."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("HF_TOKEN", "sentinel")
    monkeypatch.setattr(model_pool.tempfile, "gettempdir", lambda: str(tmp_path))
    spawned = _spawn_capture(monkeypatch)

    model_pool.spawn_worker("qwen3-0.6b", gpu_ids=[1, 0], port=31001)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert spawned["env"]["CUDA_VISIBLE_DEVICES"] == "3,2"
    # The child inherits the rest of the environment; CI injects HF_TOKEN.
    assert spawned["env"]["HF_TOKEN"] == "sentinel"
    # ModelInstance.shutdown signals the process group, which only exists
    # because the child starts its own session.
    assert spawned["kwargs"]["start_new_session"] is True
