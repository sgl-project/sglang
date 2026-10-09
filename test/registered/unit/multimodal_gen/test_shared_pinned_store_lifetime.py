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

import multiprocessing
import os
import sys
import time
import uuid
from multiprocessing import shared_memory

import pytest

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    shared_pinned_store as store,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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
