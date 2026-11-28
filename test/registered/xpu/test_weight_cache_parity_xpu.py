# SPDX-License-Identifier: Apache-2.0
"""Tensor-level disk-vs-daemon weight parity on Intel XPU.

The Phase 5 correctness gate for the XPU quantization allowlist. The daemon
tests next door assert that a client-mode server answers and that its greedy
text matches a disk-loading server's; neither looks at a tensor, so a transport
that rebuilds a view at the wrong storage offset or loses a tied name can pass
them. This loads the same checkpoint twice -- once from disk, once from a daemon
over Level Zero vmm_fd -- with the same ServerArgs, ModelConfig and distributed
setup, and compares every parameter's and buffer's shape, stride, dtype, storage
offset, storage size, aliasing group and logical byte digest.

Each load runs in its own process, and the disk reference runs before the daemon
starts: one card cannot hold a daemon's shard and a disk-loaded copy of the same
shard at the sizes the MoE case uses, and with no daemon alive the reference
provably did not come off one.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from sglang.srt.platforms import current_platform
from sglang.srt.weight_cache.protocol import (
    XPU_VMM_QUANT_SUPPORT,
    get_ready_path,
    get_socket_path,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    terminate_and_kill_process_tree,
)
from sglang.test.weight_cache_parity_utils import compare_manifests, manifest_summary

register_xpu_ci(est_time=90, suite="extra-test-1-gpu-xpu")

BASE_GPU_ID = 0
WORKER_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as handle:
        return handle.read()


def _is_cached(model_id: str) -> bool:
    """Whether the checkpoint is already downloaded, without reaching the hub."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


class TestWeightCacheParityDenseXpu(CustomTestCase):
    """Qwen3-0.6B, TP1, unquantized BF16: the XPU allowlist's dense entry."""

    MODEL = "Qwen/Qwen3-0.6B"
    TP_SIZE = 1
    LOAD_FORMAT = "auto"
    IS_MOE = False
    # Skip unless the checkpoint is already in the local HF cache. Subclasses
    # that need a checkpoint too large to pull on a CI runner set this.
    REQUIRE_CACHED_CHECKPOINT = False
    # Set before anything can fail, so tearDownClass can always clean up.
    daemon = None
    uuids: list = []

    @classmethod
    def setUpClass(cls):
        if not current_platform.is_xpu():
            raise unittest.SkipTest("requires an Intel XPU device")
        if torch.xpu.device_count() < cls.TP_SIZE:
            raise unittest.SkipTest(
                f"requires {cls.TP_SIZE} XPU device(s), found "
                f"{torch.xpu.device_count()}"
            )
        if cls.IS_MOE and not XPU_VMM_QUANT_SUPPORT[""].moe:
            raise unittest.SkipTest(
                "the XPU weight-cache allowlist does not enable MoE, so the "
                "daemon refuses this checkpoint"
            )
        if cls.REQUIRE_CACHED_CHECKPOINT and not _is_cached(cls.MODEL):
            raise unittest.SkipTest(
                f"{cls.MODEL} is not in the local Hugging Face cache, and it is "
                f"too large to download as part of a test run"
            )
        cls.workdir = tempfile.mkdtemp(prefix="wcd_parity_xpu_")
        cls.gpu_ids = [BASE_GPU_ID + rank for rank in range(cls.TP_SIZE)]
        cls.uuids = [current_platform.get_device_uuid(i) for i in cls.gpu_ids]
        for uuid in cls.uuids:
            for path in (get_ready_path(uuid), get_socket_path(uuid)):
                if os.path.exists(path):
                    os.unlink(path)
        cls.daemon_log = os.path.join(cls.workdir, "daemon.log")
        try:
            # Disk reference first, with no daemon running. A daemon holding a
            # TP4 shard of a 57 GiB checkpoint plus a disk-loaded copy of the
            # same shard does not fit on one 23.9 GiB card, and with no daemon
            # alive the reference provably cannot have come off one.
            cls.disk_manifests = cls._run_workers("disk")
            cls._start_daemon()
            cls.daemon_manifests = cls._run_workers("daemon")
        except BaseException:
            cls._stop_daemon()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._stop_daemon()

    @classmethod
    def _start_daemon(cls):
        with open(cls.daemon_log, "w") as log:
            cls.daemon = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.weight_cache.daemon",
                    "--model-path",
                    cls.MODEL,
                    "--device",
                    "xpu",
                    "--tp-size",
                    str(cls.TP_SIZE),
                    "--base-gpu-id",
                    str(BASE_GPU_ID),
                    "--load-format",
                    cls.LOAD_FORMAT,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        ready_paths = [get_ready_path(uuid) for uuid in cls.uuids]
        while not all(os.path.exists(path) for path in ready_paths):
            if cls.daemon.poll() is not None:
                raise RuntimeError(
                    f"weight cache daemon exited with {cls.daemon.returncode}:\n"
                    f"{_read(cls.daemon_log)[-4000:]}"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"weight cache daemon not ready in time:\n"
                    f"{_read(cls.daemon_log)[-4000:]}"
                )
            time.sleep(2)

    @classmethod
    def _stop_daemon(cls):
        if cls.daemon is not None and cls.daemon.poll() is None:
            # SIGTERM first: a SIGKILLed daemon can still hold its weights while
            # the kernel unwinds its context, and the next case on this card
            # would size itself against memory that is already gone.
            terminate_and_kill_process_tree(cls.daemon)
        cls.daemon = None
        for uuid in cls.uuids:
            for path in (get_ready_path(uuid), get_socket_path(uuid)):
                if os.path.exists(path):
                    os.unlink(path)

    @classmethod
    def _run_workers(cls, mode: str) -> list:
        """One manifest per TP rank, each written by its own process."""
        dist_init_method = f"tcp://127.0.0.1:{find_available_port(31500)}"
        processes = []
        outputs = []
        logs = []
        for tp_rank in range(cls.TP_SIZE):
            out = os.path.join(cls.workdir, f"{mode}_rank{tp_rank}.json")
            log_path = os.path.join(cls.workdir, f"{mode}_rank{tp_rank}.log")
            outputs.append(out)
            logs.append(log_path)
            command = [
                sys.executable,
                "-m",
                "sglang.test.weight_cache_parity_utils",
                "--mode",
                mode,
                "--model-path",
                cls.MODEL,
                "--out",
                out,
                "--device",
                "xpu",
                "--tp-size",
                str(cls.TP_SIZE),
                "--tp-rank",
                str(tp_rank),
                "--base-gpu-id",
                str(BASE_GPU_ID),
                "--dist-init-method",
                dist_init_method,
                "--load-format",
                cls.LOAD_FORMAT,
            ]
            if mode == "daemon":
                command += ["--socket-path", get_socket_path(cls.uuids[tp_rank])]
            with open(log_path, "w") as log:
                processes.append(
                    subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                )
        for tp_rank, process in enumerate(processes):
            if process.wait(timeout=WORKER_TIMEOUT) != 0:
                raise RuntimeError(
                    f"{mode} manifest worker rank {tp_rank} exited with "
                    f"{process.returncode}:\n{_read(logs[tp_rank])[-4000:]}"
                )
        manifests = []
        for out in outputs:
            with open(out) as handle:
                manifests.append(json.load(handle))
        return manifests

    def test_daemon_side_really_came_off_the_daemon(self):
        # Without this, a disk fallback on the daemon side would compare the
        # disk path against itself and pass unconditionally.
        for tp_rank, manifest in enumerate(self.daemon_manifests):
            with self.subTest(tp_rank=tp_rank):
                self.assertTrue(
                    manifest["zero_copy_mapped"],
                    f"rank {tp_rank} fell back to disk instead of mapping the "
                    f"daemon's weights",
                )
                self.assertGreater(manifest["loader_preloaded_weights_bytes"], 0)

    def test_disk_side_really_came_off_disk(self):
        for tp_rank, manifest in enumerate(self.disk_manifests):
            with self.subTest(tp_rank=tp_rank):
                self.assertFalse(
                    manifest["zero_copy_mapped"],
                    f"rank {tp_rank} reached a daemon; the reference must be a "
                    f"plain disk load",
                )

    def test_manifests_are_not_empty(self):
        for tp_rank, manifest in enumerate(self.disk_manifests):
            with self.subTest(tp_rank=tp_rank):
                self.assertGreater(len(manifest["tensors"]), 0)
                self.assertEqual(manifest["duplicate_names"], [])

    def test_every_tensor_matches_the_disk_load(self):
        for tp_rank, (disk, daemon) in enumerate(
            zip(self.disk_manifests, self.daemon_manifests)
        ):
            with self.subTest(tp_rank=tp_rank):
                problems = compare_manifests(
                    reference=disk,
                    actual=daemon,
                    reference_label="disk",
                    actual_label="daemon",
                )
                self.assertEqual(
                    problems,
                    [],
                    f"rank {tp_rank} disk={manifest_summary(disk)} "
                    f"daemon={manifest_summary(daemon)}\n" + "\n".join(problems),
                )


class TestWeightCacheParityDenseTp2Xpu(TestWeightCacheParityDenseXpu):
    """The same dense entry sharded over two cards.

    TP is what decides which slice of a weight each rank holds, so a transport
    that got the sharding right at TP1 can still mismatch at TP2 -- an earlier
    build of this work mapped rank 1's slabs onto card 0.
    """

    TP_SIZE = 2


class TestWeightCacheParityMoeTp8DummyXpu(TestWeightCacheParityDenseXpu):
    """Qwen3-30B-A3B's architecture at TP8, with dummy weights.

    MoE expert weights are why the XPU allowlist gates MoE separately from
    dense: they take their own create_weights branch, which on XPU can return a
    narrowed view of an over-allocated buffer so the row stride avoids L3 set
    aliasing.

    ``--load-format dummy`` reseeds a generator per parameter from a fixed seed,
    so separate processes produce byte-identical weights and the 57 GiB
    checkpoint is never read. That covers the expert tensors' layout, sharding,
    aliasing and transport on the real model code; it does NOT cover loading the
    checkpoint's values, which is why this result alone does not enable MoE in
    the allowlist. TP8 rather than the cookbook's TP4 because a TP4 shard
    (15.2 GiB) does not leave the daemon room to pack on a 23.9 GiB card.

    Skipped while the XPU allowlist refuses MoE, since the daemon would refuse
    to start; enable the entry locally to run it.
    """

    MODEL = "Qwen/Qwen3-30B-A3B"
    TP_SIZE = 8
    LOAD_FORMAT = "dummy"
    IS_MOE = True
    # Only config.json is read, but it still has to be local: this must not
    # reach the hub mid-test.
    REQUIRE_CACHED_CHECKPOINT = True


class TestWeightCacheParityMoeTp4Xpu(TestWeightCacheParityMoeTp8DummyXpu):
    """The cookbook's TP4 deployment with the real checkpoint: the full gate
    for enabling MoE in the XPU allowlist.

    Needs the checkpoint on storage fast enough to read 57 GiB twice, and a
    daemon that can pack a 15.2 GiB shard on a 23.9 GiB card.
    """

    TP_SIZE = 4
    LOAD_FORMAT = "auto"


if __name__ == "__main__":
    unittest.main()
