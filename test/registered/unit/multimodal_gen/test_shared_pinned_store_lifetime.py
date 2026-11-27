"""Cross-process ownership, registration and cleanup of shared pinned stores."""

import fcntl
import gc
import multiprocessing
import os
import signal
import sys
import time
import uuid
from contextlib import closing, suppress
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    shared_pinned_store as store,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

SEGMENT_BYTES = 1 << 20
DEADLINE_S = 60.0

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/dev/shm"), reason="named segments are a POSIX shm feature"
)


def _pool_name() -> str:
    return f"pytest-shared-pinned-{uuid.uuid4().hex[:12]}"


def _keys(pool: str, count: int = 3) -> list[str]:
    return [f"{pool}:{index}" for index in range(count)]


def _segment_path(name: str) -> str:
    return os.path.join("/dev/shm", name)


def _serve(pool, keys, accept, report, done):
    role = store.join_pool(pool)
    outcomes = []
    # Exercise the real publish path in CPU-only workers.
    with patch.object(store, "_register", return_value=True):
        for key in keys:
            held = store.begin(key, SEGMENT_BYTES, lambda _: accept, pool=pool)
            if held is None:
                outcomes.append("private")
            else:
                assert store.commit(held)
                outcomes.append("created" if held.created else "joined")
    report.send((role, outcomes))
    report.close()
    assert done.wait(DEADLINE_S)


def _release(proc, done):
    done.set()
    proc.join(timeout=DEADLINE_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("child did not exit when released")
    assert proc.exitcode == 0


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MARKER_DIR", str(tmp_path))
    name = _pool_name()
    yield name
    # Remove test-owned segments even if a failed assertion left no manifest.
    for key in [*_keys(name), f"{name}:unpublished"]:
        segment, _ = store._names(key)
        with (
            suppress(FileNotFoundError),
            closing(shared_memory.SharedMemory(name=segment)) as shm,
        ):
            shm.unlink()


@pytest.fixture
def start(pool, request):
    ctx = multiprocessing.get_context("fork")

    def launch(keys=(), *, accept=True):
        reader, writer = ctx.Pipe(duplex=False)
        done = ctx.Event()
        proc = ctx.Process(target=_serve, args=(pool, keys, accept, writer, done))
        proc.start()
        writer.close()

        def cleanup():
            if proc.is_alive():
                _release(proc, done)

        request.addfinalizer(cleanup)
        with reader:
            assert reader.poll(DEADLINE_S), "child did not report its stores"
            return proc, done, reader.recv()

    return launch


class _CpuDevice:
    class Stream:
        pass

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 0


def _mutable_server(pool: str, marker_dir: str, conn) -> None:
    """Run the real manager in an independent process; fake only device calls."""
    from sglang.multimodal_gen.runtime.managers.memory_managers import (
        layerwise_offload as offload,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
        HostPinBudget,
    )

    try:
        with (
            patch.object(store, "MARKER_DIR", marker_dir),
            patch.object(store, "_register", return_value=True),
            patch.object(store, "_unregister"),
            patch.object(torch, "get_device_module", return_value=_CpuDevice),
            patch.object(offload.current_platform, "device_type", "cpu"),
            patch.object(offload.envs, "SGLANG_DIFFUSION_SHARE_PINNED_STORES", True),
            patch.object(
                offload,
                "_pinned_empty",
                side_effect=lambda *size, dtype: torch.empty(*size, dtype=dtype),
            ),
        ):
            model = torch.nn.Module()
            model.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 3, bias=False) for _ in range(2)]
            )
            with torch.no_grad():
                for block in model.blocks:
                    block.weight.fill_(1)
            budget = HostPinBudget(available_bytes=1 << 30)
            budget.reserve_bytes = 0
            manager = offload.LayerwiseOffloadManager(
                model=model,
                layers_attr_str="blocks",
                num_layers=2,
                enabled=True,
                pin_cpu_memory=True,
                initialize=False,
                pin_budget=budget,
                pin_component_name=pool,
            )
            manager._initialize_layer_weights()
            conn.send(bool(manager._shared_cpu_weights))
            while True:
                command = conn.recv()
                if command == "update":
                    manager.update_cpu_weights(
                        {"blocks.0.weight": torch.full((3, 4), 17.0)}
                    )
                elif command == "rejoin":
                    # The writer still owns the pool through layer 1, but may
                    # have dropped its layer-0 mapping after detaching. Layer 0
                    # must keep the same name and pages for the peer reading it.
                    key = manager._shared_store_key(0, torch.float32, 12)
                    held = store.begin(key, 48, lambda _: True, pool=pool)
                    tensor = held.tensor(torch.float32)
                    assert not held.created
                    assert store.commit(held)
                    conn.send(tensor.tolist())
                    del tensor
                    continue
                elif command == "release":
                    manager.enabled = False
                    manager.release_host_stores()
                    gc.collect()
                    conn.send(budget.committed_bytes)
                    break
                conn.send(manager.get_cpu_weight("blocks.0.weight").tolist())
    finally:
        conn.close()


@pytest.mark.parametrize("writer", [0, 1])
def test_independent_servers_isolate_updates_and_release(tmp_path, writer):
    pool = _pool_name()
    children = []

    def receive(conn):
        assert conn.poll(DEADLINE_S), "server did not respond"
        return conn.recv()

    try:
        for _ in range(2):
            parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe()
            proc = multiprocessing.get_context("spawn").Process(
                target=_mutable_server, args=(pool, str(tmp_path), child_conn)
            )
            proc.start()
            child_conn.close()
            children.append((proc, parent_conn))
            assert receive(parent_conn), "server did not acquire shared weights"
        children[writer][1].send("update")
        assert receive(children[writer][1]) == [[17.0] * 4] * 3
        children[writer][1].send("rejoin")
        assert receive(children[writer][1]) == [1.0] * 12
        reader = children[1 - writer][1]
        reader.send("read")
        assert receive(reader) == [[1.0] * 4] * 3
        children[writer][1].send("release")
        assert receive(children[writer][1]) == 0
        reader.send("read")
        assert receive(reader) == [[1.0] * 4] * 3
        reader.send("release")
        assert receive(reader) == 0
    finally:
        for proc, conn in children:
            conn.close()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()
        assert all(proc.exitcode == 0 for proc, _ in children)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA registration")
def test_registration_lives_until_the_last_tensor_view(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MARKER_DIR", str(tmp_path))
    monkeypatch.setattr(store, "_note_leave", lambda pool: None)
    pool = _pool_name()
    key = _keys(pool, count=1)[0]
    held = store.begin(key, SEGMENT_BYTES, lambda _: True, pool=pool)
    tensor = held.tensor(torch.float32)
    tensor.fill_(3)
    assert store.commit(held)
    assert tensor.is_pinned()
    assert torch.equal(tensor.cuda().cpu(), tensor)
    view = tensor[:4]
    del tensor
    gc.collect()
    assert not held.closed
    assert torch.equal(view, torch.full((4,), 3.0))
    del view
    gc.collect()
    assert held.closed
    assert not held.registered
    assert pool not in store._POOL_REFS
    assert pool not in store._LOCKS
    assert not os.path.exists(_segment_path(store._names(key)[0]))


@pytest.mark.parametrize("has_owner", [False, True], ids=["owner", "follower"])
def test_leave_timeout_drops_pool_seat(pool, start, monkeypatch, has_owner):
    monkeypatch.setattr(store, "DECIDE_TIMEOUT_S", 0)
    peer = start() if has_owner else None
    monkeypatch.setattr(store, "_note_leave", lambda pool: None)
    try:
        assert store.join_pool(pool) is (not has_owner)
        _, _, decide, _, _ = store._pool_paths(pool)
        with open(decide, "r+") as busy:
            fcntl.flock(busy, fcntl.LOCK_EX)
            assert not store.leave_pool(pool)
        if peer is not None:
            _release(*peer[:2])
        assert store.join_pool(pool) is True
    finally:
        store.leave_pool(pool)


@pytest.mark.parametrize("terminate", [False, True], ids=["return", "sigterm"])
def test_only_last_participant_reclaims_generation(pool, start, terminate):
    keys = _keys(pool)
    owner, owner_done, report = start(keys)
    assert report == (True, ["created"] * len(keys))
    follower, follower_done, report = start(keys)
    assert report == (False, ["joined"] * len(keys))

    _release(owner, owner_done)
    names = [store._names(key)[0] for key in keys]
    assert all(os.path.exists(_segment_path(name)) for name in names)
    if terminate:
        follower.terminate()
        follower.join(timeout=DEADLINE_S)
        assert follower.exitcode == -signal.SIGTERM
    else:
        _release(follower, follower_done)

    for name in names:
        assert not os.path.exists(_segment_path(name))
        assert not os.path.exists(store._marker_path(name))
    assert not os.path.exists(store._pool_paths(pool)[-1])


def test_next_creator_reclaims_after_sigkill(pool, start):
    keys = _keys(pool)
    owner, _, report = start(keys)
    assert report == (True, ["created"] * len(keys))
    follower, _, report = start(keys)
    assert report == (False, ["joined"] * len(keys))
    for proc in (owner, follower):
        proc.kill()
        proc.join(timeout=DEADLINE_S)
        assert proc.exitcode == -signal.SIGKILL

    segments = [Path(_segment_path(store._names(key)[0])) for key in keys]
    assert all(path.exists() for path in segments)
    _, _, report = start()
    assert report == (True, [])
    assert not any(path.exists() for path in segments)


def test_unpublished_key_falls_back_at_stall_timeout(pool, start, monkeypatch):
    monkeypatch.setattr(store, "STALL_TIMEOUT_S", 2.0)
    monkeypatch.setattr(store, "MAX_WAIT_S", 30.0)
    keys = _keys(pool)
    _, _, report = start(keys)
    assert report == (True, ["created"] * len(keys))
    started = time.monotonic()
    _, _, report = start([f"{pool}:unpublished"])
    assert report == (False, ["private"])
    assert time.monotonic() - started < 10.0


@pytest.mark.parametrize("accept", [True, False])
def test_join_requires_matching_weights(pool, start, accept):
    keys = _keys(pool, count=1)
    _, _, report = start(keys)
    assert report == (True, ["created"])
    _, _, report = start(keys, accept=accept)
    assert report == (False, ["joined" if accept else "private"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
