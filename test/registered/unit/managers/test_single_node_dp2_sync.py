from __future__ import annotations

import ctypes
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_WORLD_SIZE = 2
_PAYLOAD_WIDTH = 7
_ERROR_BUFFER_SIZE = 512
_EXPECTED_ABI_VERSION = 2


class _SyncStats(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("peer_wait_ns", ctypes.c_uint64),
        ("arrival_skew_ns", ctypes.c_uint64),
        ("post_latest_arrival_ns", ctypes.c_uint64),
    ]


def _load_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    library.sglang_dp2_sync_abi_version.argtypes = []
    library.sglang_dp2_sync_abi_version.restype = ctypes.c_uint32
    library.sglang_dp2_sync_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sglang_dp2_sync_open.restype = ctypes.c_int
    library.sglang_dp2_sync_exchange.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(_SyncStats),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sglang_dp2_sync_exchange.restype = ctypes.c_int
    library.sglang_dp2_sync_close.argtypes = [ctypes.c_void_p]
    library.sglang_dp2_sync_close.restype = None
    if library.sglang_dp2_sync_abi_version() != _EXPECTED_ABI_VERSION:
        raise RuntimeError("unexpected native DP2 ABI")
    return library


def _open_handle(
    library: ctypes.CDLL,
    session_id: str,
    rank: int,
    timeout_ns: int,
) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    result = library.sglang_dp2_sync_open(
        session_id.encode(),
        rank,
        timeout_ns,
        ctypes.byref(handle),
        error,
        len(error),
    )
    if result != 0:
        raise RuntimeError(error.value.decode())
    if handle.value is None:
        raise RuntimeError("native DP2 open returned a null handle")
    return handle


def _exchange_worker(
    library_path: str,
    session_id: str,
    rank: int,
    iterations: int,
    ready,
    result_queue,
) -> None:
    try:
        library = _load_library(library_path)
        handle = _open_handle(
            library,
            session_id,
            rank,
            timeout_ns=5_000_000_000,
        )
        ready.wait(timeout=10)
        local = (ctypes.c_int64 * _PAYLOAD_WIDTH)()
        gathered = (ctypes.c_int64 * (_WORLD_SIZE * _PAYLOAD_WIDTH))()
        stats = _SyncStats()
        error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        total_ns = []
        try:
            for sequence in range(1, iterations + 1):
                for index in range(_PAYLOAD_WIDTH):
                    local[index] = rank * 1_000_000_000 + sequence * 10 + index
                result = library.sglang_dp2_sync_exchange(
                    handle,
                    local,
                    gathered,
                    ctypes.byref(stats),
                    error,
                    len(error),
                )
                if result != 0:
                    raise RuntimeError(error.value.decode())
                if stats.sequence != sequence:
                    raise AssertionError(
                        f"sequence mismatch: {stats.sequence} != {sequence}"
                    )
                for gathered_rank in range(_WORLD_SIZE):
                    for index in range(_PAYLOAD_WIDTH):
                        expected = gathered_rank * 1_000_000_000 + sequence * 10 + index
                        actual = gathered[gathered_rank * _PAYLOAD_WIDTH + index]
                        if actual != expected:
                            raise AssertionError(f"{actual} != {expected}")
                if sequence > 100:
                    total_ns.append(stats.total_ns)
        finally:
            library.sglang_dp2_sync_close(handle)
        result_queue.put((rank, total_ns, None))
    except BaseException as error:
        result_queue.put((rank, [], repr(error)))


class TestSingleNodeDP2Sync(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("a C compiler is required")
        cls._temp_dir = tempfile.TemporaryDirectory(prefix="sglang-dp2-sync-test-")
        cls.library_path = str(Path(cls._temp_dir.name) / "sglang_dp2_sync.so")
        source = (
            Path(__file__).resolve().parents[4]
            / "sgl-kernel"
            / "csrc"
            / "cpu"
            / "dp2_sync.c"
        )
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c11",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-o",
                cls.library_path,
                str(source),
                "-lrt",
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def test_invalid_rank_is_rejected(self) -> None:
        library = _load_library(self.library_path)
        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        result = library.sglang_dp2_sync_open(
            f"invalid-rank-{uuid.uuid4()}".encode(),
            2,
            1_000_000_000,
            ctypes.byref(handle),
            error,
            len(error),
        )
        self.assertNotEqual(result, 0)
        self.assertIn(b"rank must be 0 or 1", error.value)
        self.assertIsNone(handle.value)

    def test_timeout_is_fail_closed(self) -> None:
        library = _load_library(self.library_path)
        handle = _open_handle(
            library,
            f"timeout-{uuid.uuid4()}",
            rank=0,
            timeout_ns=50_000_000,
        )
        local = (ctypes.c_int64 * _PAYLOAD_WIDTH)(*range(_PAYLOAD_WIDTH))
        gathered = (ctypes.c_int64 * (_WORLD_SIZE * _PAYLOAD_WIDTH))()
        stats = _SyncStats()
        error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        started = time.monotonic()
        try:
            result = library.sglang_dp2_sync_exchange(
                handle,
                local,
                gathered,
                ctypes.byref(stats),
                error,
                len(error),
            )
        finally:
            library.sglang_dp2_sync_close(handle)
        self.assertNotEqual(result, 0)
        self.assertIn(b"timed out", error.value)
        self.assertGreaterEqual(time.monotonic() - started, 0.04)

    def test_exchange_correctness_and_latency(self) -> None:
        context = multiprocessing.get_context("spawn")
        ready = context.Barrier(_WORLD_SIZE)
        result_queue = context.Queue()
        session_id = f"correctness-{uuid.uuid4()}"
        processes = [
            context.Process(
                target=_exchange_worker,
                args=(
                    self.library_path,
                    session_id,
                    rank,
                    2_000,
                    ready,
                    result_queue,
                ),
            )
            for rank in range(_WORLD_SIZE)
        ]
        for process in processes:
            process.start()

        durations = []
        for _ in processes:
            rank, total_ns, error = result_queue.get(timeout=30)
            self.assertIsNone(error, f"rank {rank}: {error}")
            durations.extend(total_ns)
        for process in processes:
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)

        durations.sort()
        p50_us = durations[len(durations) // 2] / 1_000
        self.assertLess(p50_us, 1_000)

    def test_python_runtime_validation_loads_packaged_abi(self) -> None:
        from sglang.srt.managers.scheduler_components import single_node_dp2_sync

        environment = {
            "SGLANG_DSPARK_DP2_SHM_MLP_SYNC": "1",
            "SGLANG_DSPARK_DP2_SHM_SESSION_ID": f"validate-{uuid.uuid4()}",
            "SGLANG_DSPARK_DP2_SHM_TIMEOUT_MS": "5000",
            "SGLANG_DSPARK_DP2_SHM_METRICS": "1",
            "SGLANG_DSPARK_DP2_SHM_LIBRARY": self.library_path,
            "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "0",
            "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH": "0",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(single_node_dp2_sync, "_enabled", None),
        ):
            single_node_dp2_sync.validate_single_node_dp2_sync_runtime()


if __name__ == "__main__":
    unittest.main()
