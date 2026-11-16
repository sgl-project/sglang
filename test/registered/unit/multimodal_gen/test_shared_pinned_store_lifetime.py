"""Lifetime of the named layerwise pinned stores, without a device.

`begin()` and `commit()` are the device half of the pool: they fill a segment
and register it. This drives the lifetime half -- who owns a generation, who
may replace it, and who releases it -- through `join_pool` / `leave_pool` in
real child processes, which needs no GPU.

Each child takes a store per key the way a serving process does, and is held
open by the parent until the parent says otherwise, so every check below sees
one arrangement at a time:

  * the first process owns the pool and the second joins it;
  * a participant that leaves while another is still serving reclaims nothing;
  * the participant that leaves last releases the whole generation;
  * a participant killed outright reaches no exit hook, so its generation is
    left behind -- and the next creator reclaims it.
"""

import gc
import multiprocessing
import os
import sys
import time
import uuid
from multiprocessing import shared_memory
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


def _present(names: list[str]) -> bool:
    return all(os.path.exists(_segment_path(name)) for name in names)


def _wait_for_gate(gate: str) -> None:
    deadline = time.monotonic() + DEADLINE_S
    while time.monotonic() < deadline:
        if os.path.exists(gate):
            return
        time.sleep(0.05)


def _hold(pool: str, report: str, gate: str) -> None:
    """Join the pool without taking a store, and report the role it was given."""
    role = store.join_pool(pool)
    with open(report, "w") as handle:
        handle.write({True: "creator", False: "follower", None: "none"}[role])
    _wait_for_gate(gate)


def _serve(pool: str, keys: list[str], report: str, gate: str) -> None:
    """Own or join the pool, take a store per key, then wait to be released.

    Returning from here is what a serving process does when it is asked to shut
    down, so it is also what runs the finalizer that releases the pool.

    `commit()` is what publishes a creator's marker, and it needs a device to
    register the segment first. There is none here, so the marker -- the only
    part of publishing that a follower waits on -- is written directly.
    """
    role = store.join_pool(pool)
    outcomes = []
    for key in keys:
        held = store.begin(key, SEGMENT_BYTES, lambda buf: True, pool=pool)
        if held is None:
            outcomes.append("private")
        else:
            outcomes.append("created" if held.created else "joined")
            if held.created:
                name, marker = store._names(key)
                with open(marker, "w") as handle:
                    handle.write(name)
    with open(report, "w") as handle:
        handle.write(f"{'creator' if role else 'follower'}:{','.join(outcomes)}")
    _wait_for_gate(gate)


def _join_one(key: str, pool: str, accept: bool, report: str, gate: str) -> None:
    """Take a single store, reporting what the pool was willing to give."""
    held = store.begin(key, SEGMENT_BYTES, lambda buf: accept, pool=pool)
    if held is None:
        outcome = "private"
    else:
        outcome = "created" if held.created else "joined"
    with open(report, "w") as handle:
        handle.write(outcome)
    _wait_for_gate(gate)


def _serve_pools(pools: list[str], report: str, gate: str) -> None:
    """Serve several pools at once, the way one worker serves several components."""
    for pool in pools:
        store.join_pool(pool)
        for index in range(3):
            key = f"{pool}:{index}"
            held = store.begin(key, SEGMENT_BYTES, lambda buf: True, pool=pool)
            if held is not None and held.created:
                name, marker = store._names(key)
                with open(marker, "w") as handle:
                    handle.write(name)
    with open(report, "w") as handle:
        handle.write("serving")
    _wait_for_gate(gate)


def _start(tmp_path, target, *args):
    token = uuid.uuid4().hex[:8]
    report = tmp_path / f"{token}.report"
    gate = tmp_path / f"{token}.gate"
    proc = multiprocessing.get_context("fork").Process(
        target=target, args=(*args, str(report), str(gate))
    )
    proc.start()
    deadline = time.monotonic() + DEADLINE_S
    while time.monotonic() < deadline:
        if report.exists():
            return proc, gate, report.read_text()
        if not proc.is_alive():
            break
        time.sleep(0.05)
    proc.kill()
    proc.join()
    raise AssertionError(f"child never reported (exit {proc.exitcode})")


def _release(proc, gate) -> None:
    """Let a child return from its target, so its exit hook runs."""
    gate.touch()
    proc.join(timeout=DEADLINE_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise AssertionError("child did not exit when released")
    assert proc.exitcode == 0, f"child exited with {proc.exitcode}"


def _clear(pool: str) -> None:
    """Take down whatever the test left behind, however it ended."""
    owner, users, decide, progress, manifest = store._pool_paths(pool)
    try:
        with open(manifest) as handle:
            names = [line.strip() for line in handle if line.strip()]
    except OSError:
        names = []
    for name in names:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except (FileNotFoundError, OSError):
            pass
        else:
            try:
                shm.unlink()
            finally:
                shm.close()
        try:
            os.remove(store._marker_path(name))
        except OSError:
            pass
    for path in (manifest, progress, owner, users, decide):
        try:
            os.remove(path)
        except OSError:
            pass


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


class TestSharedPinnedStoreLifetime:
    def test_first_process_owns_and_the_second_joins(self, tmp_path):
        pool = _pool_name()
        keys = _keys(pool)
        children = []
        try:
            children.append(_start(tmp_path, _serve, pool, keys))
            children.append(_start(tmp_path, _serve, pool, keys))
            assert children[0][2] == "creator:created,created,created"
            assert children[1][2] == "follower:joined,joined,joined"
        finally:
            for proc, gate, _ in children:
                if proc.is_alive():
                    _release(proc, gate)
            _clear(pool)

    def test_a_participant_that_leaves_early_reclaims_nothing(self, tmp_path):
        pool = _pool_name()
        keys = _keys(pool)
        names = [store._names(key)[0] for key in keys]
        owner = follower = None
        try:
            owner = _start(tmp_path, _serve, pool, keys)
            follower = _start(tmp_path, _serve, pool, keys)
            assert owner[2].startswith("creator:")
            assert follower[2].startswith("follower:")

            _release(*owner[:2])
            owner = None
            assert _present(names), (
                "the pool was reclaimed while a follower was still serving it"
            )
        finally:
            if owner is not None:
                _release(*owner[:2])
            if follower is not None:
                _release(*follower[:2])
            _clear(pool)

    def test_the_last_participant_releases_the_generation(self, tmp_path):
        pool = _pool_name()
        keys = _keys(pool)
        names = [store._names(key)[0] for key in keys]
        owner = follower = None
        try:
            owner = _start(tmp_path, _serve, pool, keys)
            follower = _start(tmp_path, _serve, pool, keys)
            _release(*owner[:2])
            owner = None
            _release(*follower[:2])
            follower = None

            for name in names:
                assert not os.path.exists(_segment_path(name))
                assert not os.path.exists(store._marker_path(name))
            _, _, _, _, manifest = store._pool_paths(pool)
            assert not os.path.exists(manifest)
        finally:
            if owner is not None:
                _release(*owner[:2])
            if follower is not None:
                _release(*follower[:2])
            _clear(pool)

    def test_a_killed_participant_leaves_the_generation_to_the_next_creator(
        self, tmp_path
    ):
        pool = _pool_name()
        keys = _keys(pool)
        names = [store._names(key)[0] for key in keys]
        children = []
        try:
            children.append(_start(tmp_path, _serve, pool, keys))
            children.append(_start(tmp_path, _serve, pool, keys))
            assert children[0][2].startswith("creator:")
            assert children[1][2].startswith("follower:")

            for proc, _, _ in children:
                proc.kill()
                proc.join(timeout=DEADLINE_S)
            assert _present(names), "a killed participant reclaimed on its way out"

            revisit = _start(tmp_path, _hold, pool)
            try:
                assert revisit[2] == "creator"
                assert not any(os.path.exists(_segment_path(n)) for n in names), (
                    "the next creator did not reclaim the killed generation"
                )
            finally:
                _release(*revisit[:2])
        finally:
            for proc, gate, _ in children:
                if proc.is_alive():
                    _release(proc, gate)
            _clear(pool)

    def test_a_sigterm_releases_the_generation(self, tmp_path):
        """The exit that actually happens in a deployment: a signal, not a return.

        A unit stops with SIGTERM to every process in its cgroup, so the worker
        never reaches `atexit` or a finalizer. The last participant still has to
        release the generation from the handler, or the whole component stays in
        /dev/shm until some later start.
        """
        pool = _pool_name()
        keys = _keys(pool)
        names = [store._names(key)[0] for key in keys]
        owner = follower = None
        try:
            owner = _start(tmp_path, _serve, pool, keys)
            follower = _start(tmp_path, _serve, pool, keys)
            _release(*owner[:2])
            owner = None
            assert _present(names), "a clean return released the pool too early"

            follower[0].terminate()
            follower[0].join(timeout=DEADLINE_S)
            follower = None
            assert not any(os.path.exists(_segment_path(n)) for n in names), (
                "the last participant was signalled but did not release the generation"
            )
        finally:
            if owner is not None:
                _release(*owner[:2])
            if follower is not None:
                _release(*follower[:2])
            _clear(pool)

    def test_a_key_the_creator_never_publishes_goes_private_quickly(
        self, tmp_path, monkeypatch
    ):
        """Two co-resident instances do not always have the same key set.

        A store key is structural, so an instance running a different adapter
        asks for keys the creator never made -- and nothing will ever publish
        their markers. Waiting on those has to end at the stall timeout, not at
        the wait limit, or the instance spends the whole load on markers that
        are not coming.
        """
        monkeypatch.setattr(store, "STALL_TIMEOUT_S", 2.0)
        monkeypatch.setattr(store, "MAX_WAIT_S", 30.0)
        pool = _pool_name()
        holder = None
        try:
            holder = _start(tmp_path, _serve, pool, _keys(pool))
            assert holder[2] == "creator:created,created,created"

            started = time.monotonic()
            joiner = _start(tmp_path, _join_one, f"{pool}:unpublished", pool, True)
            elapsed = time.monotonic() - started
            try:
                assert joiner[2] == "private"
                assert elapsed < 10.0, (
                    f"waited {elapsed:.1f}s on a marker the creator never published"
                )
            finally:
                _release(*joiner[:2])
        finally:
            if holder is not None:
                _release(*holder[:2])
            _clear(pool)

    @pytest.mark.parametrize("accept", [True, False])
    def test_begin_trusts_a_store_only_after_the_bytes_agree(self, tmp_path, accept):
        pool = _pool_name()
        key = _keys(pool, count=1)[0]
        holder = None
        try:
            holder = _start(tmp_path, _serve, pool, [key])
            assert holder[2] == "creator:created"

            joiner = _start(tmp_path, _join_one, key, pool, accept)
            try:
                assert joiner[2] == ("joined" if accept else "private")
            finally:
                _release(*joiner[:2])
        finally:
            if holder is not None:
                _release(*holder[:2])
            _clear(pool)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
