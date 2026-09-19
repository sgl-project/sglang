"""Tests for the CPU-only crash snapshot (``SGLANG_DEBUG_CRASH_SNAPSHOT``).

The snapshot exists for the case where the CUDA context is already gone by the
time the process notices an asynchronous fault, so the tests here also assert
that recording and dumping never reach for the GPU.
"""

import json
import logging
import os
import types

import pytest
import torch

from sglang.srt.debug_utils import crash_snapshot
from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.io_log import log_host_pool_io
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 32
TOKEN_STRIDE_SIZE = 256
LAYER_NUM = 16  # 32 * 16 * 256 = 131072 == the staged write-back batch threshold


class _FakePool:
    """Pool-shaped metadata holder; the real classes need device pools."""

    def __init__(self):
        self.page_size = PAGE_SIZE
        self.layer_num = LAYER_NUM
        self.token_stride_size = TOKEN_STRIDE_SIZE
        self.layout = "page_first"
        self.can_use_jit = True
        self.can_use_write_back_jit = True
        self.device_pool = types.SimpleNamespace(device="cuda")

    @log_host_pool_io
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        return "ok"

    @log_host_pool_io
    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        raise ValueError(f"Unsupported IO backend: {io_backend}")


@pytest.fixture(autouse=True)
def clean_state():
    crash_snapshot.reset()
    yield
    crash_snapshot.reset()


def _transfer(pool, io_backend="kernel"):
    pool.backup_from_device_all_layer(
        None,
        torch.arange(PAGE_SIZE, dtype=torch.int64),
        torch.arange(PAGE_SIZE, dtype=torch.int64),
        io_backend,
    )


def test_nothing_recorded_unless_enabled(tmp_path):
    pool = _FakePool()
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(False):
        _transfer(pool)
        assert crash_snapshot.records() == []
        assert (
            crash_snapshot.dump_crash_snapshot(
                "test",
                server_args=types.SimpleNamespace(crash_dump_folder=str(tmp_path)),
            )
            is None
        )
    assert list(tmp_path.iterdir()) == []


def test_records_describe_the_transfer(tmp_path):
    pool = _FakePool()
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        _transfer(pool)

        recorded = crash_snapshot.records()
        assert len(recorded) == 1
        entry = recorded[0]
        assert entry["kind"] == "host_pool_io"
        assert entry["op"] == "backup_from_device_all_layer"
        assert entry["direction"] == "D2H"
        assert entry["per_page_bytes"] == 131072
        assert entry["batch_threshold"] is True
        assert entry["host_alloc"] == "alloc_with_host_register"
        assert entry["indices"]["host_indices"].endswith("@cpu")
        assert entry["io_backend"] == "kernel"

        path = crash_snapshot.dump_crash_snapshot(
            "scheduler_exception",
            server_args=types.SimpleNamespace(crash_dump_folder=str(tmp_path)),
            traceback_text="Traceback: boom",
        )
        payload = json.loads(open(path).read())
        assert payload["reason"] == "scheduler_exception"
        assert payload["traceback"] == "Traceback: boom"
        assert payload["pid"] == os.getpid()
        assert payload["recent_host_pool_io"] == recorded


def test_raising_transfer_is_recorded_too():
    pool = _FakePool()
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        with pytest.raises(ValueError):
            pool.load_to_device_per_layer(None, None, None, 0, "bogus")
        assert [entry["op"] for entry in crash_snapshot.records()] == [
            "load_to_device_per_layer"
        ]


def test_ring_keeps_the_newest_and_stays_bounded():
    pool = _FakePool()
    with (
        envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True),
        envs.SGLANG_DEBUG_CRASH_SNAPSHOT_SIZE.override(3),
    ):
        for _ in range(5):
            _transfer(pool)
        recorded = crash_snapshot.records()
        assert len(recorded) == 3
        # Oldest first, so the tail of the ring is the most recent transfer.
        assert [entry["t"] for entry in recorded] == sorted(
            entry["t"] for entry in recorded
        )


def test_ring_is_resized_when_the_configured_size_changes():
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        for _ in range(4):
            crash_snapshot.record("unit", {"i": 1})
        assert len(crash_snapshot.records()) == 4
        with envs.SGLANG_DEBUG_CRASH_SNAPSHOT_SIZE.override(2):
            assert len(crash_snapshot.records()) == 2


def test_dump_is_written_once_per_process(tmp_path):
    args = types.SimpleNamespace(crash_dump_folder=str(tmp_path))
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        first = crash_snapshot.dump_crash_snapshot("first", server_args=args)
        second = crash_snapshot.dump_crash_snapshot("second", server_args=args)
        assert first == second
        assert len(list(tmp_path.glob("*/*.json"))) == 1


def test_dump_never_raises_on_an_unwritable_folder(caplog):
    caplog.set_level(logging.ERROR, logger=crash_snapshot.__name__)
    args = types.SimpleNamespace(crash_dump_folder="/proc/does/not/exist")
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        assert crash_snapshot.dump_crash_snapshot("test", server_args=args) is None
    assert any(
        "Failed to write crash snapshot" in r.getMessage() for r in caplog.records
    )


def test_traceback_is_capped(tmp_path):
    args = types.SimpleNamespace(crash_dump_folder=str(tmp_path))
    long_traceback = "x" * (crash_snapshot.MAX_TRACEBACK_CHARS * 2)
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        path = crash_snapshot.dump_crash_snapshot(
            "test", server_args=args, traceback_text=long_traceback
        )
    payload = json.loads(open(path).read())
    assert len(payload["traceback"]) == crash_snapshot.MAX_TRACEBACK_CHARS
    assert payload["traceback"].endswith("x")


def test_scheduler_context_is_cpu_only_and_partial():

    batch = types.SimpleNamespace(reqs=[1, 2, 3])

    class Raising:
        @property
        def forward_ct(self):
            raise RuntimeError("no attribute on this build")

    scheduler = types.SimpleNamespace(
        forward_ct=42, running_batch=batch, waiting_queue=[1], max_total_num_tokens=1000
    )
    context = crash_snapshot.scheduler_context(scheduler)
    assert context["forward_ct"] == 42
    assert context["num_running_reqs"] == 3
    assert context["waiting_queue_len"] == 1
    assert context["max_total_num_tokens"] == 1000
    # Missing attributes are dropped, not fatal.
    assert "tp_rank" not in context
    assert crash_snapshot.scheduler_context(Raising()) == {}
    assert crash_snapshot.scheduler_context(None) == {}


def test_recording_never_touches_the_gpu(monkeypatch):
    """After an asynchronous fault every CUDA call raises; recording must not."""

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    for name in ("synchronize", "current_stream", "memory_allocated", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, name, _boom, raising=False)

    pool = _FakePool()
    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        _transfer(pool)
        assert len(crash_snapshot.records()) == 1


def test_scheduler_crash_path_writes_snapshot(tmp_path, monkeypatch):
    """The real crash path of the scheduler process must dump the ring.

    Everything before the failing ``Scheduler(...)`` construction is stubbed, and
    the parent-process signal is captured instead of delivered, so this exercises
    the production code path without touching a GPU or killing the test runner.

    ``server_args`` answers every attribute with a neutral value: the startup steps
    ahead of the ``try`` keep growing upstream (``resolve_spawn_dp_rank`` reads
    config fields), and this test only cares that control reaches the crash handler.
    """
    import signal as signal_module

    import sglang.srt.managers.scheduler as scheduler_module

    class _PermissiveArgs(types.SimpleNamespace):
        """Answers any config read with 0 instead of raising."""

        def __getattr__(self, name):
            return 0

    monkeypatch.setattr(scheduler_module, "load_plugins", lambda: None)
    monkeypatch.setattr(scheduler_module, "publish", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler_module, "configure_scheduler_process", lambda *a, **k: None
    )
    monkeypatch.setattr(
        scheduler_module,
        "get_observability",
        lambda: types.SimpleNamespace(
            enable_trace=False, crash_dump_folder=str(tmp_path)
        ),
    )

    def _fail(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(scheduler_module, "Scheduler", _fail)

    signalled = []
    fake_parent = types.SimpleNamespace(send_signal=signalled.append)
    monkeypatch.setattr(
        scheduler_module,
        "psutil",
        types.SimpleNamespace(
            Process=lambda *a, **k: types.SimpleNamespace(parent=lambda: fake_parent)
        ),
    )

    with envs.SGLANG_DEBUG_CRASH_SNAPSHOT.override(True):
        scheduler_module.run_scheduler_process(
            server_args=_PermissiveArgs(crash_dump_folder=str(tmp_path)),
            port_args=None,
            gpu_id=0,
            tp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            dp_rank=None,
            pipe_writer=None,
        )

    files = list(tmp_path.glob("*/*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["reason"] == "scheduler_exception"
    assert "illegal memory access" in payload["traceback"]
    assert signalled == [signal_module.SIGQUIT]
