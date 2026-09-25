"""Unit tests for the NIXL HiCache storage backend -- no server, no model loading."""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd")

import contextlib
import ctypes
import json
import mmap
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo, IndexKeyLayout
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

# Stress tests are opt-in: CI never sets this; set locally to exercise them.
STRESS_ENABLED = bool(os.environ.get("SGLANG_RUN_NIXL_STRESS"))
# Mirrors _OS_PAGE_BYTES in hicache_nixl; kept local so a revert of that module
# cannot turn a probe assertion into a collection error.
_PAGE_BYTES = 4096


class MockHybridPool:
    def __init__(
        self,
        num_pages: int = 4,
        page_size: int = 1,
        component_bytes: int = 8,
        expose_zero_copy: bool = True,
        stride_page_aligned: bool = True,
        layout: str = "page_first",
    ):
        self.page_size = page_size
        self.layout = layout
        self.stride_page_aligned = stride_page_aligned
        self.dtype = torch.uint8
        self.device = "cpu"
        self.pin_memory = False
        self.temporal_buffer = torch.zeros(
            (num_pages * page_size, component_bytes), dtype=self.dtype
        )
        self.conv_buffer = [
            torch.zeros((num_pages * page_size, component_bytes), dtype=self.dtype)
        ]
        if expose_zero_copy:
            self.get_hybrid_pool_buffer = self._get_hybrid_pool_buffer

    def _get_hybrid_pool_buffer(self):
        return [self.temporal_buffer, *self.conv_buffer]

    def get_page_buffer_meta(self, indices):
        # Mirrors the real mamba pool, which only exposes per-page slots page-first.
        if self.layout not in ("page_first", "page_first_direct"):
            raise ValueError(
                f"Mamba storage zero-copy requires page_first layout, got {self.layout}"
            )
        ptr_list = []
        size_list = []
        for index in indices.tolist():
            ptr_list.append(self.temporal_buffer[index].data_ptr())
            size_list.append(self.temporal_buffer[index].numel())
            ptr_list.append(self.conv_buffer[0][index].data_ptr())
            size_list.append(self.conv_buffer[0][index].numel())
        return ptr_list, size_list

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.temporal_buffer.shape[1] * 2, dtype=self.dtype)

    def get_data_page(self, index, flat=True):
        data = torch.cat([self.temporal_buffer[index], self.conv_buffer[0][index]])
        return data.flatten() if flat else data

    def set_from_flat_data_page(self, index, data_page):
        split = self.temporal_buffer.shape[1]
        self.temporal_buffer[index].copy_(data_page[:split])
        self.conv_buffer[0][index].copy_(data_page[split:])

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        # Every real pool reports False off page-first, so the flag only applies there.
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        return self.stride_page_aligned


class MockMemPoolHost:
    """Minimal MHA-style HostKVCache stand-in supporting the v1 paths.

    zero_copy mode uses ``page_first`` so ``get_page_buffer_meta`` returns
    valid (k, v) pointers into ``kv_buffer``. Non-zero-copy uses
    ``layer_first`` so the slow path uses ``get_data_page`` /
    ``set_from_flat_data_page`` against the same buffer.
    """

    def __init__(
        self,
        is_zero_copy_mode: bool,
        page_size: int = 2,
        layer_num: int = 2,
        head_num: int = 2,
        head_dim: int = 4,
        num_pages: int = 4,
        dtype: torch.dtype = torch.float32,
        stride_page_aligned: bool = False,
    ):
        self.layout = "page_first" if is_zero_copy_mode else "layer_first"
        self.page_size = page_size
        self.stride_page_aligned = stride_page_aligned
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_pages = num_pages
        self.size = page_size * num_pages
        self.pin_memory = False
        if is_zero_copy_mode:
            # page_first: (2, size, layer, head, head_dim)
            self.kv_buffer = torch.zeros(
                (2, self.size, layer_num, head_num, head_dim), dtype=dtype
            )
        else:
            # layer_first: (2, layer, size, head, head_dim)
            self.kv_buffer = torch.zeros(
                (2, layer_num, self.size, head_num, head_dim), dtype=dtype
            )

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        base = self.kv_buffer.data_ptr()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        idx_list = indices.tolist()
        for i in range(0, len(idx_list), self.page_size):
            k_ptr = base + idx_list[i] * (
                self.layer_num * self.head_num * self.head_dim * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
            ptr_list.append(k_ptr + v_offset)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return ptr_list, [element_size] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def get_data_page(self, index, flat=True):
        if hasattr(index, "item"):
            index = int(index.item())
        page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        if hasattr(index, "item"):
            index = int(index.item())
        self.kv_buffer[:, :, index : index + self.page_size, :, :] = data_page.reshape(
            2, self.layer_num, self.page_size, self.head_num, self.head_dim
        )

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        # Defaults to False because test tensors are too small to satisfy 4 KiB
        # stride alignment; the O_DIRECT path then falls back to copy mode.
        return self.stride_page_aligned


class MinioFixture:
    """Spin up a single-node MinIO server on localhost and create a bucket.

    Relies on MinIO's default ``minioadmin``/``minioadmin`` root credentials
    so no env vars need to be plumbed through.
    """

    user = "minioadmin"
    password = "minioadmin"

    def __init__(self, bucket: str = "hicache-test"):
        self.bucket = bucket
        self.api_port = self._find_free_port()
        self.data_dir = tempfile.mkdtemp(prefix="nixl_minio_")
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.api_port}"

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _minio_bin() -> str | None:
        path = shutil.which("minio") or "/usr/local/bin/minio"
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    @classmethod
    def is_available(cls) -> bool:
        """True iff a minio binary and boto3 are both importable."""
        if cls._minio_bin() is None:
            return False
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self) -> None:
        minio_bin = self._minio_bin()
        if minio_bin is None:
            raise FileNotFoundError("minio binary not available")

        self.proc = subprocess.Popen(
            [minio_bin, "server", "--address", self.endpoint, self.data_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"minio exited early with rc={self.proc.returncode}")
            try:
                with socket.create_connection(
                    ("127.0.0.1", self.api_port), timeout=0.5
                ):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            self.stop()
            raise RuntimeError("minio did not become ready within 15s")

        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{self.endpoint}",
            aws_access_key_id=self.user,
            aws_secret_access_key=self.password,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )
        s3.create_bucket(Bucket=self.bucket)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        shutil.rmtree(self.data_dir, ignore_errors=True)


class TestNixlUnified(CustomTestCase):
    """Unified test suite for all NIXL components."""

    def setUp(self):
        """Set up test environment."""
        # Per-process directory: cleanup removes it, so a fixed path would let two
        # concurrent test processes delete each other's files mid-test.
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_unified_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

        # Disable O_DIRECT here: these tests use small, arbitrarily-aligned
        # tensors that do not satisfy the sector-alignment constraints required
        # by O_DIRECT. O_DIRECT-specific behaviour is exercised in
        # TestNixlDirectIO below.
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
            enable_storage_metrics=False,
            extra_config={
                "plugin": {"posix": {"active": True}},
                "use_direct_io": False,
                # Round trips require retained files regardless of runner disk
                # pressure. Cleaner behavior has its own dedicated test suite.
                "l3_cleaner_enabled": False,
            },
        )

        try:
            self.hicache = HiCacheNixl(
                storage_config=self.storage_config,
                file_path=self.test_dir,
            )
            self.addCleanup(self.hicache.close)
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    @staticmethod
    def _open_fds() -> int:
        return len(os.listdir("/proc/self/fd"))

    def test_storage_register_failure_closes_fds(self):
        """If NIXL register_memory raises after fds are opened, all fds are still closed."""
        files = [os.path.join(self.test_dir, f"fail_{i}.bin") for i in range(3)]
        buffers = [(0, 64) for _ in range(3)]

        fds_before = self._open_fds()

        orig = self.hicache.agent.register_memory

        def boom(*args, **kwargs):
            raise RuntimeError("simulated register_memory failure")

        self.hicache.agent.register_memory = boom
        try:
            with self.hicache.registry.storage(buffers, files, "WRITE") as descs:
                self.assertIsNone(
                    descs, "storage CM should yield None on register failure"
                )
        finally:
            self.hicache.agent.register_memory = orig

        self.assertEqual(
            self._open_fds(),
            fds_before,
            "fd leak after register_memory failure mid-storage",
        )

    def _assert_host_addrs_pre_registered(
        self, is_zero_copy_mode: bool, hicache: HiCacheNixl = None
    ):
        """Exercise the v1 path and assert every host xfer addr lies within a
        currently-registered host (DRAM/tensor) region.

        Spies are installed BEFORE ``register_mem_pool_host`` so the up-front
        pre-registration is captured too.
        """
        if hicache is None:
            hicache = self.hicache
        agent = hicache.agent

        # Map registration-handle id -> [(addr, size, mem_type), ...]
        active_regs: dict = {}
        # Capture items list per get_reg_descs call so we can attribute them
        # to the registration handle returned by the next register_memory call.
        pending: list = []

        orig_get_reg = agent.get_reg_descs

        def spy_get_reg(items, mem_type=None):
            # NIXL's register_memory calls get_reg_descs internally with an
            # already-built nixlRegDList; iterating that pybind11 type is
            # unsafe, so only record entries when the input is a plain list.
            if isinstance(items, list) and items:
                entries = []
                for it in items:
                    if isinstance(it, torch.Tensor):
                        entries.append(
                            (it.data_ptr(), it.numel() * it.element_size(), None)
                        )
                    elif isinstance(it, tuple):
                        entries.append((it[0], it[1], mem_type))
                pending.append(entries)
            return orig_get_reg(items, mem_type)

        orig_register = agent.register_memory

        def spy_register(reg_descs):
            reg = orig_register(reg_descs)
            entries = pending.pop(0) if pending else []
            active_regs[id(reg)] = entries
            return reg

        orig_dereg = agent.deregister_memory

        def spy_dereg(reg):
            active_regs.pop(id(reg), None)
            return orig_dereg(reg)

        last_host_xfer: list = []

        orig_get_xfer = agent.get_xfer_descs

        def spy_get_xfer(items, mem_type=None):
            if mem_type in (None, "DRAM"):
                ranges = []
                for it in items:
                    if isinstance(it, torch.Tensor):
                        ranges.append((it.data_ptr(), it.numel() * it.element_size()))
                    elif isinstance(it, tuple):
                        ranges.append((it[0], it[1]))
                last_host_xfer.clear()
                last_host_xfer.extend(ranges)
            return orig_get_xfer(items, mem_type)

        violations: list = []
        orig_init = agent.initialize_xfer

        def spy_init(direction, local, remote, agent_name):
            host_regs = [
                (a, s)
                for entries in active_regs.values()
                for (a, s, mt) in entries
                if mt in (None, "DRAM")
            ]
            for a, s in last_host_xfer:
                if not any(ra <= a and a + s <= ra + rs for (ra, rs) in host_regs):
                    violations.append((a, s, dict(host_regs=host_regs)))
            last_host_xfer.clear()
            return orig_init(direction, local, remote, agent_name)

        agent.get_reg_descs = spy_get_reg
        agent.register_memory = spy_register
        agent.deregister_memory = spy_dereg
        agent.get_xfer_descs = spy_get_xfer
        agent.initialize_xfer = spy_init
        try:
            mock_host = MockMemPoolHost(is_zero_copy_mode)
            hicache.register_mem_pool_host(mock_host)
            # Force the requested mode regardless of how register_mem_pool_host derives it.
            hicache.is_zero_copy = is_zero_copy_mode

            num_pages = 3
            keys = [
                f"compliance_{int(is_zero_copy_mode)}_{i}" for i in range(num_pages)
            ]
            host_indices = torch.arange(
                num_pages * mock_host.page_size, dtype=torch.int64
            )

            # Distinct data per key so a round trip that mixes keys up (e.g. a
            # FILE registration that collapses every desc onto one file) is
            # caught, not just the pass/fail return codes. Non-zero-copy only:
            # the layer_first layout makes per-page seeding straightforward.
            if not is_zero_copy_mode:
                ps = mock_host.page_size
                for p in range(num_pages):
                    mock_host.kv_buffer[:, :, p * ps : (p + 1) * ps] = float(p + 1)
                expected = mock_host.kv_buffer.clone()

            set_results = hicache.batch_set_v1(keys, host_indices)
            self.assertTrue(
                all(set_results),
                f"batch_set_v1 failed (zero_copy={is_zero_copy_mode}): {set_results}",
            )

            if not is_zero_copy_mode:
                mock_host.kv_buffer.zero_()

            get_results = hicache.batch_get_v1(keys, host_indices)
            self.assertTrue(
                all(get_results),
                f"batch_get_v1 failed (zero_copy={is_zero_copy_mode}): {get_results}",
            )

            if not is_zero_copy_mode:
                self.assertTrue(
                    torch.equal(mock_host.kv_buffer, expected),
                    "round trip corrupted or mixed up per-key data",
                )
        finally:
            agent.get_reg_descs = orig_get_reg
            agent.register_memory = orig_register
            agent.deregister_memory = orig_dereg
            agent.get_xfer_descs = orig_get_xfer
            agent.initialize_xfer = orig_init

        self.assertEqual(
            violations,
            [],
            f"Host xfer addrs not covered by registration (zero_copy={is_zero_copy_mode}): {violations}",
        )

    def test_nixl_api_contract_host_addrs_within_registered_region_zero_copy(self):
        """All host xfer addrs must lie within a registered region -- zero-copy."""
        self._assert_host_addrs_pre_registered(is_zero_copy_mode=True)

    def test_nixl_api_contract_host_addrs_within_registered_region_non_zero_copy(self):
        """All host xfer addrs must lie within a registered region -- non-zero-copy."""
        self._assert_host_addrs_pre_registered(is_zero_copy_mode=False)

    def _make_obj_hicache(self) -> HiCacheNixl:
        """Start a MinIO server (cleaned up via addCleanup) and return a
        HiCacheNixl wired to its OBJ backend. Skips the test if the backend
        cannot be constructed."""
        minio = MinioFixture()
        minio.start()
        self.addCleanup(minio.stop)

        obj_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
            enable_storage_metrics=False,
            extra_config={
                "plugin": {
                    "obj": {
                        "active": True,
                        "endpoint_override": f"http://{minio.endpoint}",
                        "use_virtual_addressing": "false",
                        "access_key": minio.user,
                        "secret_key": minio.password,
                        "bucket": minio.bucket,
                    }
                }
            },
        )
        try:
            hicache = HiCacheNixl(storage_config=obj_config, file_path="")
            self.addCleanup(hicache.close)
            return hicache
        except Exception as e:
            self.skipTest(f"NIXL OBJ backend unavailable: {e}")

    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_nixl_api_contract_host_addrs_within_registered_region_obj(self):
        """Same property over the OBJ backend (MinIO fixture)."""
        self._assert_host_addrs_pre_registered(
            is_zero_copy_mode=False, hicache=self._make_obj_hicache()
        )

    def _batch_set_v1_log(self, xfer_results: list) -> str:
        mock_host = MockMemPoolHost(is_zero_copy_mode=False)
        self.hicache.register_mem_pool_host(mock_host)
        self.hicache._batch_xfer = lambda keys, key_strs, host_buffers, direction: list(
            xfer_results
        )

        keys = [f"stats_k{i}" for i in range(len(xfer_results))]
        host_indices = torch.arange(len(keys) * mock_host.page_size, dtype=torch.int64)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.hicache_nixl", level="DEBUG"
        ) as logs:
            results = self.hicache.batch_set_v1(keys, host_indices)

        self.assertEqual(results, xfer_results)
        return "\n".join(logs.output)

    def test_successful_batch_set_v1_still_reports_bandwidth(self):
        """The outcome flag must not be inverted: successes keep their stats line."""
        log = self._batch_set_v1_log([True, True])

        self.assertIn("batch_set_v1 transferred", log)
        self.assertRegex(log, r" ([1-9]\d*) of \1 bytes transferred")

    def _failed_xfer_log(self, run) -> str:
        self.hicache._batch_xfer = lambda keys, key_strs, host_buffers, direction: (
            [False] * len(key_strs)
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.hicache_nixl", level="DEBUG"
        ) as logs:
            run()
        return "\n".join(logs.output)

    def test_every_stats_call_site_reports_zero_bytes_for_a_failed_batch(self):
        """A failed batch must report zero bytes moved, not the requested size, on all
        four transfer paths that share the stats helper."""
        mock_host = MockMemPoolHost(is_zero_copy_mode=False)
        self.hicache.register_mem_pool_host(mock_host)
        self.hicache.register_mem_host_pool_v2(MockHybridPool(), PoolName.MAMBA)

        keys = ["stats_k0", "stats_k1"]
        host_indices = torch.arange(len(keys) * mock_host.page_size, dtype=torch.int64)
        transfers = [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=keys,
                host_indices=torch.arange(len(keys), dtype=torch.int64),
            )
        ]
        ops = [
            ("batch_set_v1", lambda: self.hicache.batch_set_v1(keys, host_indices)),
            ("batch_get_v1", lambda: self.hicache.batch_get_v1(keys, host_indices)),
            (
                f"batch_get_v2[{PoolName.MAMBA}]",
                lambda: self.hicache.batch_get_v2(transfers),
            ),
            (
                f"batch_set_v2[{PoolName.MAMBA}]",
                lambda: self.hicache.batch_set_v2(transfers),
            ),
        ]
        for op_name, run in ops:
            with self.subTest(op=op_name):
                log = self._failed_xfer_log(run)

                self.assertIn(f"{op_name} failed", log)
                self.assertRegex(log, r" 0 of [1-9]\d* bytes transferred")
                self.assertIn("effective bandwidth: 0.00 MB/s", log)

    def test_batch_set_v1_skips_on_nonzero_mla_rank(self):
        """batch_set_v1 is a no-op on nonzero MLA backup ranks.

        With backup_skip=True the early-return must fire before the host-regs
        check, so calling without register_mem_pool_host still returns all-True
        (the host-regs check would otherwise return all-False).
        """
        self.hicache.backup_skip = True
        results = self.hicache.batch_set_v1(
            ["key1", "key2"], torch.tensor([0, 1], dtype=torch.int64)
        )
        self.assertEqual(results, [True, True])

    def test_batch_exists_zero_copy_mla_uses_single_key_denominator(self):
        """Zero-copy MLA batch_exists counts one storage key per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = True
        self.hicache.agent.query_memory = lambda *a, **kw: [object(), None]

        self.assertEqual(self.hicache.batch_exists(["key1", "key2"]), 1)

    def test_batch_exists_zero_copy_mha_uses_two_key_denominator(self):
        """Zero-copy non-MLA batch_exists counts k/v pairs per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = False
        self.hicache.agent.query_memory = lambda *a, **kw: [
            object(),
            object(),
            None,
            None,
        ]

        self.assertEqual(self.hicache.batch_exists(["key1", "key2"]), 1)

    def test_dsa_indexer_bounce_round_trip_preserves_every_byte(self):
        """Index pages must survive copy fallback when the main KV dtype is bf16."""
        from sglang.srt.mem_cache.device_pool_info import (
            IndexKeyBufferInfo,
            IndexKeyLayout,
        )
        from sglang.srt.mem_cache.pool_host.dsa import (
            DSAIndexerPoolHost,
        )

        pool = SimpleNamespace(
            layer_num=2,
            device="cpu",
            layer_shard_enabled=False,
            store_dtype=torch.bfloat16,
            start_layer=0,
            end_layer=2,
            index_head_dim=128,
            quant_block_size=128,
            page_size=64,
            index_page_size=64,
            skip_topk_layers=[False, False],
            index_k_with_scale_buffer=[
                torch.empty((4, 64 * 132), dtype=torch.uint8) for _ in range(2)
            ],
        )
        anchor = SimpleNamespace(
            page_size=64, layout="page_first", size=256, page_num=4
        )
        host = DSAIndexerPoolHost(
            IndexKeyBufferInfo(
                page_size=64,
                compress_ratio=1,
                key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES,
                buffers=tuple(pool.index_k_with_scale_buffer),
            ),
            layer_mapping={0: 0, 1: 1},
            target_device_layer_num=pool.layer_num,
            num_host_pages=anchor.page_num,
            transfer_page_size=anchor.page_size,
            layout=anchor.layout,
            buffer_device=pool.device,
            dtype=pool.store_dtype,
            pin_memory=False,
        )
        self.addCleanup(host.destroy)
        with mock.patch.object(
            self.hicache, "_hybrid_pool_supports_zero_copy", return_value=False
        ):
            self.hicache.register_mem_host_pool_v2(host, PoolName.INDEXER)
        self.addCleanup(self.hicache.close)
        ctx = self.hicache._hybrid_pool_ctx[PoolName.INDEXER]
        self.assertFalse(ctx.is_zero_copy)
        expected = (
            torch.arange(host.get_dummy_flat_data_page().numel())
            .remainder(251)
            .to(torch.uint8)
        )
        host.set_from_flat_data_page(64, expected)
        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            keys=["dsa-index-bytes"],
            host_indices=torch.arange(64, 128),
        )
        self.assertEqual(
            self.hicache.batch_set_v2([transfer]), {PoolName.INDEXER: [True]}
        )
        host.set_from_flat_data_page(64, torch.zeros_like(expected))
        ctx.bounce_get.zero_()
        self.assertEqual(
            self.hicache.batch_get_v2([transfer]), {PoolName.INDEXER: [True]}
        )
        self.assertTrue(torch.equal(host.get_data_page(64), expected))

    def test_failed_write_after_file_creation_reads_back_as_a_miss(self):
        """A FILE WRITE creates its files before moving data, in both path mode (NIXL
        creates them at registration) and the legacy fd mode (the registry opens them
        with O_CREAT and only closes them); a write that then fails must not leave
        files that batch_exists reports as cached pages."""
        mock_host = MockMemPoolHost(is_zero_copy_mode=False)
        self.hicache.register_mem_pool_host(mock_host)
        host_indices = torch.arange(2 * mock_host.page_size, dtype=torch.int64)
        modes = [False, True] if self.hicache.registry.path_mode else [False]
        for path_mode in modes:
            with self.subTest(path_mode=path_mode):
                self.hicache.registry.path_mode = path_mode
                keys = [f"torn_{path_mode}_k0", f"torn_{path_mode}_k1"]
                created = []

                def fail_after_create(host_descs, storage_descs, direction):
                    created.extend(self._files_under(self.test_dir))
                    return False

                with mock.patch.object(
                    self.hicache, "_xfer_and_wait", side_effect=fail_after_create
                ):
                    results = self.hicache.batch_set_v1(keys, host_indices)

                self.assertEqual(results, [False, False])
                # Without this the checks below also pass if no file was created.
                self.assertEqual(len(created), len(keys))
                self.assertEqual(self._files_under(self.test_dir), [])
                self.assertEqual(self.hicache.batch_exists(keys), 0)

    @staticmethod
    def _files_under(directory: str) -> list:
        return [
            os.path.join(root, name)
            for root, _dirs, files in os.walk(directory)
            for name in files
        ]

    def test_register_mem_host_pool_v2_uses_zero_copy_when_supported(self):
        pool = MockHybridPool(expose_zero_copy=True)
        self.hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        ctx = self.hicache._hybrid_pool_ctx[PoolName.MAMBA]
        self.assertTrue(ctx.is_zero_copy)
        self.assertIs(ctx.host_pool, pool)

    def test_mla_mamba_round_trip_keeps_each_tp_rank(self):
        """MLA shares KV keys, but Mamba checkpoints must not overwrite another rank."""
        for zero_copy in (False, True):
            with self.subTest(zero_copy=zero_copy):
                caches, pools, expected = [], [], []
                transfer = PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=[f"same_prefix_{zero_copy}"],
                    host_indices=torch.tensor([0]),
                )
                for rank in range(2):
                    config = replace(
                        self.storage_config, tp_rank=rank, is_mla_model=True
                    )
                    cache = HiCacheNixl(storage_config=config, file_path=self.test_dir)
                    self.addCleanup(cache.close)
                    pool = MockHybridPool(expose_zero_copy=zero_copy)
                    cache.register_mem_host_pool_v2(pool, PoolName.MAMBA)
                    pool.temporal_buffer[0].fill_(rank + 11)
                    pool.conv_buffer[0][0].fill_(rank + 21)
                    expected.append(pool.get_data_page(0).clone())
                    caches.append(cache)
                    pools.append(pool)
                    self.assertEqual(
                        cache.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                    )

                for cache, pool, page in zip(caches, pools, expected):
                    pool.temporal_buffer.zero_()
                    pool.conv_buffer[0].zero_()
                    self.assertEqual(
                        cache.batch_get_v2([transfer])[PoolName.MAMBA], [True]
                    )
                    self.assertTrue(torch.equal(pool.get_data_page(0), page))

    def test_register_mem_host_pool_v2_uses_persistent_bounce_otherwise(self):
        pool = MockHybridPool(expose_zero_copy=False)
        self.hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        ctx = self.hicache._hybrid_pool_ctx[PoolName.MAMBA]
        self.assertFalse(ctx.is_zero_copy)
        self.assertIsNotNone(ctx.bounce_set)
        self.assertIsNotNone(ctx.bounce_get)
        self.assertEqual(ctx.bounce_page_bytes, pool.get_dummy_flat_data_page().numel())

    def test_batch_set_v2_expands_zero_copy_mamba_component_keys(self):
        pool = MockHybridPool(expose_zero_copy=True)
        self.hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        captured = {}

        def fake_batch_xfer(keys, key_strs, host_buffers, direction):
            captured["keys"] = key_strs
            captured["host_buffers"] = host_buffers
            captured["direction"] = direction
            return [True] * len(key_strs)

        self.hicache._batch_xfer = fake_batch_xfer
        results = self.hicache.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=["p0", "p1"],
                    host_indices=torch.tensor([0, 1], dtype=torch.int64),
                )
            ]
        )

        self.assertEqual(results[PoolName.MAMBA], [True, True])
        self.assertEqual(
            captured["keys"],
            [
                self.hicache._get_suffixed_key("p0") + "_mamba_temporal",
                self.hicache._get_suffixed_key("p0") + "_mamba_conv_0",
                self.hicache._get_suffixed_key("p1") + "_mamba_temporal",
                self.hicache._get_suffixed_key("p1") + "_mamba_conv_0",
            ],
        )
        self.assertEqual(len(captured["host_buffers"]), 4)
        self.assertEqual(captured["direction"], "WRITE")

    def test_batch_get_v2_uses_bounce_buffer_for_non_zero_copy_pool(self):
        pool = MockHybridPool(expose_zero_copy=False)
        self.hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        def fake_batch_xfer(keys, key_strs, host_buffers, direction):
            ctx = self.hicache._hybrid_pool_ctx[PoolName.MAMBA]
            ctx.bounce_get[0].fill_(3)
            return [True] * len(key_strs)

        self.hicache._batch_xfer = fake_batch_xfer
        results = self.hicache.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=["p0"],
                    host_indices=torch.tensor([0], dtype=torch.int64),
                )
            ]
        )

        self.assertEqual(results[PoolName.MAMBA], [True])
        self.assertTrue(torch.all(pool.get_data_page(0) == 3))

    def test_batch_set_get_v2_distinguishes_same_key_by_pool_name(self):
        mamba_pool = MockHybridPool(expose_zero_copy=False)
        swa_pool = MockHybridPool(expose_zero_copy=False)
        self.hicache.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
        self.hicache.register_mem_host_pool_v2(swa_pool, PoolName.SWA)

        mamba_pool.temporal_buffer[0].fill_(11)
        mamba_pool.conv_buffer[0][0].fill_(12)
        swa_pool.temporal_buffer[0].fill_(21)
        swa_pool.conv_buffer[0][0].fill_(22)
        expected_mamba = mamba_pool.get_data_page(0).clone()
        expected_swa = swa_pool.get_data_page(0).clone()

        key = "shared_key"
        host_indices = torch.tensor([0], dtype=torch.int64)
        set_results = self.hicache.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA, keys=[key], host_indices=host_indices
                ),
                PoolTransfer(name=PoolName.SWA, keys=[key], host_indices=host_indices),
            ]
        )
        self.assertEqual(set_results[PoolName.MAMBA], [True])
        self.assertEqual(set_results[PoolName.SWA], [True])

        mamba_pool.temporal_buffer.zero_()
        mamba_pool.conv_buffer[0].zero_()
        swa_pool.temporal_buffer.zero_()
        swa_pool.conv_buffer[0].zero_()

        get_results = self.hicache.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA, keys=[key], host_indices=host_indices
                ),
                PoolTransfer(name=PoolName.SWA, keys=[key], host_indices=host_indices),
            ]
        )

        self.assertEqual(get_results[PoolName.MAMBA], [True])
        self.assertEqual(get_results[PoolName.SWA], [True])
        self.assertTrue(torch.equal(mamba_pool.get_data_page(0), expected_mamba))
        self.assertTrue(torch.equal(swa_pool.get_data_page(0), expected_swa))


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available on this platform")
class TestNixlDirectIO(CustomTestCase):
    """Tests for the O_DIRECT file I/O path in NixlFileManager and HiCacheNixl."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_direct_io_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def _skip_unless_o_direct_opens(self) -> None:
        # Raw syscall so it cannot pass by agreeing with the code under test. The
        # directory follows TMPDIR, which need not be a filesystem with O_DIRECT.
        path = os.path.join(self.test_dir, "fs_support_probe")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
        except OSError as e:
            self.skipTest(f"{self.test_dir} rejects O_DIRECT opens: {e}")
        os.close(fd)
        os.unlink(path)

    def test_open_file_sets_o_direct(self):
        """open_file sets O_DIRECT on the file descriptor when use_direct_io=True."""
        import fcntl

        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        self._skip_unless_o_direct_opens()
        fm = NixlFileManager(self.test_dir, use_direct_io=True)
        test_file = os.path.join(self.test_dir, "test_odirect.bin")
        fd = fm.open_file(test_file, create=True)
        try:
            self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)

    def test_open_file_no_o_direct(self):
        """open_file does not set O_DIRECT when use_direct_io=False."""
        import fcntl

        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        fm = NixlFileManager(self.test_dir, use_direct_io=False)
        test_file = os.path.join(self.test_dir, "test_buffered.bin")
        fd = fm.open_file(test_file, create=True)
        try:
            self.assertFalse(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)

    def _make_direct_io_hicache(self, probe_error=None) -> HiCacheNixl:
        # Stubbing the probe keeps registration outcomes independent of whether
        # the test filesystem supports O_DIRECT.
        hicache = self._make_direct_io_hicache_with_real_probe()
        hicache.file_manager.direct_io_error = lambda addr, size: probe_error
        return hicache

    def _make_direct_io_hicache_with_real_probe(self) -> HiCacheNixl:
        # use_direct_io defaults to True (env var)
        return self._make_posix_hicache({"plugin": {"posix": {"active": True}}})

    def _make_posix_hicache(self, extra_config: dict) -> HiCacheNixl:
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
            enable_storage_metrics=False,
            extra_config=extra_config,
        )
        try:
            return HiCacheNixl(storage_config=storage_config, file_path=self.test_dir)
        except ImportError:
            self.skipTest("NIXL not available")

    def test_needs_page_alignment_true_for_file_backend(self):
        """File-based backend + use_direct_io=True must set needs_page_alignment."""
        hicache = self._make_direct_io_hicache()
        self.assertTrue(hicache.needs_page_alignment)

    @staticmethod
    def _aligned_addr(buf: torch.Tensor) -> int:
        addr = buf.data_ptr()
        return addr + (-addr) % _PAGE_BYTES

    @staticmethod
    def _recording_probe(error=None):
        calls = []

        def probe(addr, size):
            calls.append((addr, size))
            return error

        return probe, calls

    def test_odirect_unusable_host_pool_falls_back_to_buffered_io(self):
        """A bounce buffer O_DIRECT cannot pin makes every tier-3 write leave a
        zero-length file that a later existence query counts as cached.
        """
        hicache = self._make_direct_io_hicache()
        probe, probed = self._recording_probe("Bad address (errno 14)")
        hicache.file_manager.direct_io_error = probe
        hicache.register_mem_pool_host(MockMemPoolHost(is_zero_copy_mode=True))

        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)
        # The fallback flips use_direct_io on the manager the registry opens files
        # through, so a copy of it would silently keep O_DIRECT on.
        self.assertIs(hicache.registry.file_manager, hicache.file_manager)
        self.assertEqual(len(probed), 1)
        # An unaligned probe would fail on every filesystem, disabling O_DIRECT
        # unconditionally.
        addr, size = probed[0]
        self.assertEqual(addr % _PAGE_BYTES, 0)
        self.assertEqual(size % _PAGE_BYTES, 0)

    def test_probe_that_raises_falls_back_to_buffered_io(self):
        """A probe that cannot run must fall back to buffered I/O, not abort pool
        registration."""
        hicache = self._make_direct_io_hicache()

        def raising_probe(addr, size):
            raise RuntimeError("probe could not open a scratch file")

        hicache.file_manager.direct_io_error = raising_probe
        hicache.register_mem_pool_host(MockMemPoolHost(is_zero_copy_mode=True))

        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)

    def test_odirect_usable_host_pool_keeps_direct_io(self):
        """A successful probe must leave O_DIRECT and page alignment in place."""
        hicache = self._make_direct_io_hicache()
        hicache.register_mem_pool_host(MockMemPoolHost(is_zero_copy_mode=True))

        self.assertTrue(hicache.file_manager.use_direct_io)
        self.assertTrue(hicache.needs_page_alignment)

    def test_zero_copy_pool_probes_its_kv_buffer(self):
        """A page-aligned zero-copy pool sources every transfer straight out of
        kv_buffer, so that is the allocation O_DIRECT has to be able to pin.
        """
        hicache = self._make_direct_io_hicache()
        probe, probed = self._recording_probe("Bad address (errno 14)")
        hicache.file_manager.direct_io_error = probe
        # num_pages grows kv_buffer past a page so the probe is reached instead of
        # the too-small guard.
        pool = MockMemPoolHost(
            is_zero_copy_mode=True, num_pages=32, stride_page_aligned=True
        )
        hicache.register_mem_pool_host(pool)

        self.assertEqual(probed, [(self._aligned_addr(pool.kv_buffer), _PAGE_BYTES)])
        # Only the alignment requirement goes away; zero copy is still correct
        # against a buffered fd.
        self.assertTrue(hicache.is_zero_copy)
        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)

    def test_pools_registered_across_a_probe_fallback_still_round_trip(self):
        """A probe failure drops O_DIRECT partway through registration: the pool
        that failed was already sized with O_DIRECT-padded slots, and pools after it
        are not; each must keep its own geometry and round-trip under buffered I/O.
        """
        hicache = self._make_direct_io_hicache(probe_error="Bad address (errno 14)")
        kv_pool = MockMemPoolHost(is_zero_copy_mode=False)
        hicache.register_mem_pool_host(kv_pool)
        hybrid_pool = MockHybridPool()
        hicache.register_mem_host_pool_v2(hybrid_pool, PoolName.MAMBA)

        self.assertFalse(hicache.file_manager.use_direct_io)
        page_bytes = kv_pool.get_dummy_flat_data_page().numel() * 4
        self.assertLess(page_bytes, _PAGE_BYTES)
        # Sized before the probe ran, so still padded to a whole O_DIRECT block.
        self.assertEqual(hicache._bounce_page_bytes, _PAGE_BYTES)
        # Registered after the fallback: unaligned zero-copy is allowed again.
        self.assertTrue(hicache._hybrid_pool_ctx[PoolName.MAMBA].is_zero_copy)

        keys = ["fallback_k0", "fallback_k1"]
        host_indices = torch.arange(len(keys) * kv_pool.page_size, dtype=torch.int64)
        kv_pool.kv_buffer.copy_(torch.randn(kv_pool.kv_buffer.shape))
        kv_expected = kv_pool.kv_buffer.clone()
        self.assertEqual(hicache.batch_set_v1(keys, host_indices), [True, True])
        kv_pool.kv_buffer.zero_()
        self.assertEqual(hicache.batch_get_v1(keys, host_indices), [True, True])
        n = len(host_indices)
        self.assertTrue(torch.equal(kv_pool.kv_buffer[:, :, :n], kv_expected[:, :, :n]))

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["fallback_m0"],
            host_indices=torch.tensor([1]),
        )
        hybrid_pool.temporal_buffer[1].fill_(7)
        hybrid_pool.conv_buffer[0][1].fill_(9)
        hybrid_expected = hybrid_pool.get_data_page(1).clone()
        self.assertEqual(hicache.batch_set_v2([transfer]), {PoolName.MAMBA: [True]})
        hybrid_pool.temporal_buffer.zero_()
        hybrid_pool.conv_buffer[0].zero_()
        self.assertEqual(hicache.batch_get_v2([transfer]), {PoolName.MAMBA: [True]})
        self.assertTrue(torch.equal(hybrid_pool.get_data_page(1), hybrid_expected))

    def test_logical_anchor_pool_registers_under_direct_io(self):
        """A LogicalHostPool (V4 KV anchor, no KV tensor) must still register under
        O_DIRECT."""
        hicache = self._make_direct_io_hicache()
        hicache.register_mem_pool_host(LogicalHostPool(size=8, page_size=2))

        self.assertTrue(hicache._logical_anchor)
        self.assertFalse(hicache.is_zero_copy)
        self.assertTrue(hicache.file_manager.use_direct_io)
        # The anchor marker is what tier-3 writes read from, so it must be a full
        # page while O_DIRECT is on.
        self.assertEqual(hicache._bounce_page_bytes, _PAGE_BYTES)

    def test_logical_anchor_host_pool_group_registers(self):
        """PD decode offload hands NIXL a HostPoolGroup, not its anchor pool; a V4
        group anchored on a LogicalHostPool must register like the bare anchor."""
        hicache = self._make_direct_io_hicache()
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=LogicalHostPool(size=8, page_size=2),
                    device_pool=SimpleNamespace(),
                    layer_mapper=lambda layer: layer,
                    is_primary_index_anchor=True,
                )
            ]
        )
        hicache.register_mem_pool_host(group)

        self.assertTrue(hicache._logical_anchor)
        self.assertFalse(hicache.is_zero_copy)
        self.assertEqual(hicache._bounce_page_bytes, _PAGE_BYTES)

    def test_logical_anchor_pool_probes_its_marker_buffer(self):
        """The marker buffers are the anchor pool's only transfer source, so they
        are what the probe has to accept before O_DIRECT stays on.
        """
        hicache = self._make_direct_io_hicache(probe_error="Bad address (errno 14)")
        hicache.register_mem_pool_host(LogicalHostPool(size=8, page_size=2))

        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)

    def test_hybrid_pool_probe_falls_back_to_buffered_io(self):
        """Hybrid v2 pools own their own zero-copy buffers, so probing only the KV
        pool left mamba/SWA state writing through an O_DIRECT fd it cannot pin.
        """
        hicache = self._make_direct_io_hicache()
        probe, probed = self._recording_probe("Bad address (errno 14)")
        hicache.file_manager.direct_io_error = probe
        # A buffer smaller than two pages would trip the too-small guard instead,
        # which reports a fallback without ever reaching the probe.
        pool = MockHybridPool(component_bytes=_PAGE_BYTES)
        hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        buf = pool.get_hybrid_pool_buffer()[0]
        self.assertEqual(probed, [(self._aligned_addr(buf), _PAGE_BYTES)])
        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)
        # Buffered I/O has no alignment constraint, so the stride check that would
        # otherwise force bounce buffers must not fire once O_DIRECT is off.
        self.assertTrue(hicache._hybrid_pool_ctx[PoolName.MAMBA].is_zero_copy)

    def test_hybrid_pool_probes_every_component_buffer(self):
        """Each mamba component is its own allocation, so an unpinnable conv buffer
        must fall back even when the temporal buffer probes fine."""
        hicache = self._make_direct_io_hicache()
        pool = MockHybridPool(component_bytes=_PAGE_BYTES)
        conv_addr = self._aligned_addr(pool.conv_buffer[0])
        probe, probed = self._recording_probe()
        hicache.file_manager.direct_io_error = lambda addr, size: (
            probe(addr, size)
            or ("Bad address (errno 14)" if addr == conv_addr else None)
        )
        hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        self.assertEqual(
            [addr for addr, _ in probed],
            [self._aligned_addr(pool.temporal_buffer), conv_addr],
        )
        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)

    def test_conv_only_hybrid_pool_keeps_direct_io(self):
        """A conv-only mamba pool's 0-element temporal buffer sources no transfer, so
        it must not be probed and must not drop O_DIRECT."""
        hicache = self._make_direct_io_hicache()
        pool = MockHybridPool(component_bytes=_PAGE_BYTES)
        pool.temporal_buffer = torch.empty((0, _PAGE_BYTES), dtype=pool.dtype)
        probe, probed = self._recording_probe()
        hicache.file_manager.direct_io_error = probe
        hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        self.assertEqual(
            probed, [(self._aligned_addr(pool.conv_buffer[0]), _PAGE_BYTES)]
        )
        self.assertTrue(hicache.file_manager.use_direct_io)
        self.assertTrue(hicache.needs_page_alignment)
        self.assertTrue(hicache._hybrid_pool_ctx[PoolName.MAMBA].is_zero_copy)

    def test_unaligned_hybrid_pool_keeps_direct_io_with_bounce_buffers(self):
        """An unaligned hybrid pool transfers through bounce buffers, so its own
        buffers must not be probed and must not drop O_DIRECT."""
        hicache = self._make_direct_io_hicache()
        hicache.register_mem_host_pool_v2(
            MockHybridPool(component_bytes=32, stride_page_aligned=False),
            PoolName.MAMBA,
        )

        ctx = hicache._hybrid_pool_ctx[PoolName.MAMBA]
        self.assertFalse(ctx.is_zero_copy)
        # A bounce buffer below a page reports the same fallback from the too-small
        # guard, so pin the size this case depends on instead of its geometry.
        self.assertGreaterEqual(
            ctx.bounce_set.numel() * ctx.bounce_set.element_size(), _PAGE_BYTES
        )
        self.assertTrue(hicache.file_manager.use_direct_io)
        self.assertTrue(hicache.needs_page_alignment)

    def test_layer_first_hybrid_pool_stays_on_bounce_buffers_after_fallback(self):
        """Dropping O_DIRECT must not promote a layer_first hybrid pool to zero-copy:
        its per-page slots are not addressable, so transfers would raise.
        """
        hicache = self._make_direct_io_hicache(probe_error="Bad address (errno 14)")
        # The KV pool registers first and trips the fallback, so the hybrid pool
        # below is registered with O_DIRECT already off.
        hicache.register_mem_pool_host(LogicalHostPool(size=8, page_size=2))
        self.assertFalse(hicache.needs_page_alignment)

        hicache.register_mem_host_pool_v2(
            MockHybridPool(component_bytes=32, layout="layer_first"), PoolName.MAMBA
        )

        self.assertFalse(hicache._hybrid_pool_ctx[PoolName.MAMBA].is_zero_copy)

    @staticmethod
    def _mha_host_pool(layout: str) -> MHATokenToKVPoolHost:
        """A real MHA host pool, which is what a hybrid SWA pool is built as."""
        device = MHATokenToKVPool(
            size=256,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        return MHATokenToKVPoolHost(
            buffer_info=device.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=64,
            layout=layout,
            pin_memory=False,
            device_capacity_tokens=device.size,
            start_layer=device.start_layer,
            end_layer=device.end_layer,
            write_back_staging=device.hicache_write_back_staging,
        )

    def test_hybrid_zero_copy_follows_the_pools_own_layout_support(self):
        """An SWA hybrid pool is an MHA host pool, which moves page_head pages in
        place like page_first, so a page_first-only layout list sent it to bounce
        buffers. Its layer_first pages are per-layer slices and stay on bounce buffers.
        """
        cases = {"page_first": True, "page_head": True, "layer_first": False}
        for layout, zero_copy in cases.items():
            with self.subTest(layout=layout):
                hicache = self._make_posix_hicache(
                    {"plugin": {"posix": {"active": True}}, "use_direct_io": False}
                )
                self.assertFalse(hicache.needs_page_alignment)
                hicache.register_mem_host_pool_v2(
                    self._mha_host_pool(layout), PoolName.SWA
                )

                self.assertEqual(
                    hicache._hybrid_pool_ctx[PoolName.SWA].is_zero_copy, zero_copy
                )

    def test_k_only_hybrid_pool_keeps_zero_copy_without_o_direct(self):
        """The MiniMax-M3 index-K pool moves page_first pages in place; without its
        own layout check it fell to the base default, losing zero-copy with O_DIRECT
        off and logging a false O_DIRECT warning."""
        cases = {"page_first": True, "page_first_direct": True, "layer_first": False}
        for layout, zero_copy in cases.items():
            with self.subTest(layout=layout):
                hicache = self._make_posix_hicache(
                    {"plugin": {"posix": {"active": True}}, "use_direct_io": False}
                )
                index_host = MHATokenToKOnlyPoolHost(
                    buffer_info=IndexKeyBufferInfo(
                        page_size=64,
                        buffers=tuple(
                            torch.zeros((256, 1, 128), dtype=torch.bfloat16)
                            for _ in range(2)
                        ),
                        compress_ratio=1,
                        key_layout=IndexKeyLayout.KEY_ROWS,
                    ),
                    anchor_host=self._mha_host_pool(layout),
                    layout=layout,
                    pin_memory=False,
                )
                with self.assertNoLogs(
                    "sglang.srt.mem_cache.pool_host.base", level="WARNING"
                ):
                    hicache.register_mem_host_pool_v2(index_host, PoolName.INDEXER)

                self.assertEqual(
                    hicache._hybrid_pool_ctx[PoolName.INDEXER].is_zero_copy, zero_copy
                )

    def test_probe_rejects_a_buffer_too_small_for_an_aligned_page(self):
        """A buffer that cannot hold an aligned page cannot source an O_DIRECT
        write either, so it must not be probed past its end.
        """
        hicache = self._make_direct_io_hicache()
        probed = []
        hicache.file_manager.direct_io_error = lambda addr, size: probed.append(addr)

        hicache._fall_back_if_direct_io_unusable(torch.empty(8, dtype=torch.uint8))

        self.assertFalse(hicache.file_manager.use_direct_io)
        self.assertFalse(hicache.needs_page_alignment)
        self.assertEqual(probed, [])

    def _path_mode_write_spec(self, hicache: HiCacheNixl) -> str:
        hicache.registry.path_mode = True
        specs = []

        @contextlib.contextmanager
        def spy_registered(tuples, mem_type):
            specs.extend(metainfo.split(":", 1)[0] for *_, metainfo in tuples)
            yield None

        hicache.registry._registered = spy_registered
        with hicache.registry.storage([(0, 4096)], ["page-123"], "WRITE") as descs:
            self.assertIsNone(descs)

        self.assertEqual(len(specs), 1)
        return specs[0]

    def test_fallback_drops_the_direct_flag_from_the_path_mode_spec(self):
        """Path mode hands O_DIRECT to NIXL as a "direct" token in the file spec
        rather than an open flag, so the fallback has to reach that string too.
        """
        usable = self._make_direct_io_hicache()
        usable.register_mem_pool_host(MockMemPoolHost(is_zero_copy_mode=True))
        self.assertEqual(self._path_mode_write_spec(usable), "rw,create,direct")

        unusable = self._make_direct_io_hicache(probe_error="Bad address (errno 14)")
        unusable.register_mem_pool_host(MockMemPoolHost(is_zero_copy_mode=True))
        self.assertEqual(self._path_mode_write_spec(unusable), "rw,create")

    def test_odirect_unaligned_pool_falls_back_to_copy(self):
        """O_DIRECT with non-aligned pool strides falls back to copy mode."""
        hicache = self._make_direct_io_hicache()

        mock_host = MockMemPoolHost(is_zero_copy_mode=True)
        hicache.register_mem_pool_host(mock_host)

        # MockMemPoolHost.is_stride_page_aligned() returns False, so even though
        # the layout would otherwise enable zero-copy, the backend must fall back.
        self.assertFalse(hicache.is_zero_copy)
        self.assertIsNotNone(hicache._bounce_set)
        self.assertIsNotNone(hicache._bounce_get)

    def test_odirect_disabled_via_config(self):
        """Top-level use_direct_io=false in extra_config disables O_DIRECT."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
            enable_storage_metrics=False,
            extra_config={
                "plugin": {"posix": {"active": True}},
                "use_direct_io": False,
            },
        )
        try:
            hicache = HiCacheNixl(
                storage_config=storage_config, file_path=self.test_dir
            )
        except ImportError:
            self.skipTest("NIXL not available")
        self.assertFalse(hicache.needs_page_alignment)
        self.assertFalse(hicache.file_manager.use_direct_io)


class _ODirectDisk:
    """Stands in for NIXL's POSIX plugin on an O_DIRECT filesystem: a transfer
    whose address or length is not whole 4 KiB blocks is refused, as the kernel
    refuses it (EINVAL); any other keeps its bytes by key."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.refused: list[tuple[str, int, int]] = []

    def transfer(self, keys, key_strs, host_buffers, direction):
        results = []
        for key, (addr, size) in zip(key_strs, host_buffers):
            if addr % 4096 or size % 4096:
                self.refused.append((key, addr, size))
                results.append(False)
            elif direction == "WRITE":
                self.files[key] = ctypes.string_at(addr, size)
                results.append(True)
            elif len(self.files.get(key, b"")) >= size:
                ctypes.memmove(addr, self.files[key], size)
                results.append(True)
            else:
                results.append(False)
        return results


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available on this platform")
class TestNixlDirectIOPages(CustomTestCase):
    """Host pages that are not whole O_DIRECT blocks survive a round trip under
    O_DIRECT: in place when the pool pads them (page_first_direct with NIXL),
    otherwise through staging slots padded to whole blocks."""

    PAGE_TOKENS = 64
    LAYERS = 3
    # Pages of 25,728 B (KV: bf16 rows of 61 + 6) and 25,344 B (DSA index keys),
    # whole blocks of neither 512 B nor 4 KiB; both pad to 7 blocks.
    PADDED_PAGE_BYTES = 7 * 4096
    HOST_PAGES = 160  # more than the 128 staging slots

    def setUp(self):
        self.test_dir = "/tmp/test_nixl_direct_io"
        os.makedirs(self.test_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        env = mock.patch.dict(os.environ, {"SGLANG_HICACHE_NIXL_USE_DIRECT_IO": "1"})
        env.start()
        self.addCleanup(env.stop)
        # The L3 cleaner would delete the files at once on a disk past its watermark.
        self.extra_config = {
            "plugin": {"posix": {"active": True}},
            "l3_cleaner_enabled": False,
        }
        try:
            self.hicache = HiCacheNixl(
                HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    is_page_first_layout=True,
                    model_name="test_model",
                    enable_storage_metrics=False,
                    extra_config=dict(self.extra_config),
                ),
                file_path=self.test_dir,
            )
        except ImportError:
            self.skipTest("NIXL not available")
        self.addCleanup(self.hicache.close)
        self.assertTrue(self.hicache.needs_page_alignment)
        self.disk = _ODirectDisk()
        self.hicache._batch_xfer = self.disk.transfer

    def _host_pools(self, allocator_type):
        """An MLA KV host pool (v1 path) and its DSA indexer pool (v2 path)."""
        device = MLATokenToKVPool(
            size=self.PAGE_TOKENS * self.HOST_PAGES // 2,
            page_size=self.PAGE_TOKENS,
            dtype=torch.bfloat16,
            kv_lora_rank=61,
            qk_rope_head_dim=6,
            layer_num=self.LAYERS,
            device="cpu",
            enable_memory_saver=False,
        )
        index_device = SimpleNamespace(
            layer_num=self.LAYERS,
            device="cpu",
            layer_shard_enabled=False,
            store_dtype=torch.bfloat16,
            start_layer=0,
            end_layer=self.LAYERS,
            index_head_dim=128,
            quant_block_size=128,
            page_size=self.PAGE_TOKENS,
            index_page_size=self.PAGE_TOKENS,
            skip_topk_layers=[False] * self.LAYERS,
            index_k_with_scale_buffer=[
                torch.empty((4, self.PAGE_TOKENS * 132), dtype=torch.uint8)
                for _ in range(self.LAYERS)
            ],
        )
        with get_context().override_server_args(
            hicache_storage_backend_extra_config=json.dumps(self.extra_config)
        ):
            kv = MLATokenToKVPoolHost(
                device.get_device_pool_infos()[0].buffer_info,
                host_to_device_ratio=2,
                host_size=0,
                page_size=self.PAGE_TOKENS,
                layout="page_first_direct",
                pin_memory=False,
                allocator_type=allocator_type,
                device_capacity_tokens=device.size,
                start_layer=device.start_layer,
                end_layer=device.end_layer,
                target_layer_num=self.LAYERS,
            )
            self.addCleanup(kv.destroy)
            indexer = DSAIndexerPoolHost(
                IndexKeyBufferInfo(
                    page_size=self.PAGE_TOKENS,
                    compress_ratio=1,
                    key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES,
                    buffers=tuple(index_device.index_k_with_scale_buffer),
                ),
                layer_mapping={i: i for i in range(self.LAYERS)},
                target_device_layer_num=self.LAYERS,
                num_host_pages=kv.page_num,
                transfer_page_size=kv.page_size,
                layout=kv.layout,
                buffer_device=index_device.device,
                dtype=index_device.store_dtype,
                pin_memory=False,
                allocator_type=allocator_type,
            )
            self.addCleanup(indexer.destroy)
        self.hicache.register_mem_pool_host(kv)
        self.hicache.register_mem_host_pool_v2(indexer, PoolName.INDEXER)
        return kv, indexer

    def _assert_stored(self, keys):
        self.assertEqual(self.disk.refused, [])
        self.assertEqual(
            {len(data) for data in self.disk.files.values()}, {self.PADDED_PAGE_BYTES}
        )

    def _assert_round_trip(self, kv, indexer, pages):
        """Store pages of random bytes from both pools, zero them, load them back."""
        keys = [f"page-{p}" for p in pages]
        starts = [p * self.PAGE_TOKENS for p in pages]
        indices = torch.cat([torch.arange(s, s + self.PAGE_TOKENS) for s in starts])
        index_transfer = PoolTransfer(
            name=PoolName.INDEXER, keys=keys, host_indices=indices
        )
        generator = torch.Generator().manual_seed(0)
        expected = {}
        for host in (kv, indexer):
            for start in starts:
                page = host.get_dummy_flat_data_page()
                raw = page.view(torch.uint8)
                raw.copy_(
                    torch.randint(
                        0, 256, raw.shape, dtype=torch.uint8, generator=generator
                    )
                )
                host.set_from_flat_data_page(start, page)
                expected[host, start] = raw

        stored = [True] * len(pages)
        self.assertEqual(self.hicache.batch_set_v1(keys, indices), stored)
        self.assertEqual(
            self.hicache.batch_set_v2([index_transfer]), {PoolName.INDEXER: stored}
        )
        self._assert_stored(keys)

        for host, start in expected:
            host.set_from_flat_data_page(start, host.get_dummy_flat_data_page())
        self.assertEqual(self.hicache.batch_get_v1(keys, indices), stored)
        self.assertEqual(
            self.hicache.batch_get_v2([index_transfer]), {PoolName.INDEXER: stored}
        )
        for (host, start), raw in expected.items():
            got = host.get_data_page(start).view(torch.uint8)
            self.assertTrue(
                torch.equal(got, raw), f"{type(host).__name__} page at {start}"
            )

    def test_unpadded_pages_go_through_whole_block_staging_slots(self):
        kv, indexer = self._host_pools(allocator_type="default")
        self.assertFalse(self.hicache.is_zero_copy)
        self.assertFalse(self.hicache._hybrid_pool_ctx[PoolName.INDEXER].is_zero_copy)
        self._assert_round_trip(kv, indexer, pages=range(1, 129))

    def test_padded_pages_move_in_place_in_any_batch_size(self):
        kv, indexer = self._host_pools(allocator_type="nixl")
        self.assertTrue(self.hicache.is_zero_copy)
        self.assertTrue(self.hicache._hybrid_pool_ctx[PoolName.INDEXER].is_zero_copy)
        self._assert_round_trip(kv, indexer, pages=range(1, self.HOST_PAGES + 1))


class TestNixlDirectIOPagesOnDisk(TestNixlDirectIOPages):
    """The same round trips through NIXL's POSIX plugin and real O_DIRECT files."""

    def setUp(self):
        super().setUp()
        del self.hicache._batch_xfer
        probe = os.path.join(self.test_dir, "o_direct_probe")
        block = mmap.mmap(-1, 4096)
        try:
            fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            try:
                os.write(fd, block)
            finally:
                os.close(fd)
        except OSError as e:
            self.skipTest(f"{self.test_dir} does not take O_DIRECT writes: {e}")
        finally:
            block.close()
            if os.path.exists(probe):
                os.unlink(probe)

    def _assert_stored(self, keys):
        sizes = {
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(self.test_dir)
            for name in names
        }
        self.assertEqual(sizes, {self.PADDED_PAGE_BYTES})


class TestNixlFileLayout(CustomTestCase):
    """Tests for deterministic NIXL FILE storage path layout."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_layout_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_route_key_is_stable_and_bucketed(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_routing import (
            BUCKET_HEX_CHARS,
            route_key,
        )

        self.assertEqual(route_key("page-123", 4), route_key("page-123", 4))
        disk_idx, bucket = route_key("page-123", 4)
        self.assertGreaterEqual(disk_idx, 0)
        self.assertLess(disk_idx, 4)
        self.assertEqual(len(bucket), BUCKET_HEX_CHARS)
        self.assertRegex(bucket, rf"^[0-9a-f]{{{BUCKET_HEX_CHARS}}}$")

    def test_route_key_rejects_empty_disk_set(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_routing import route_key

        with self.assertRaises(ValueError):
            route_key("page-123", 0)

    def test_file_manager_routes_to_bucketed_base_dir(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_routing import (
            route_disk,
            route_key,
        )
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(3)]
        fm = NixlFileManager(base_dirs, use_direct_io=False)
        key = "page-123"

        disk_idx, bucket = route_key(key, len(base_dirs))
        self.assertEqual(route_disk(key, len(base_dirs)), disk_idx)
        self.assertEqual(
            fm.get_file_path(key), os.path.join(base_dirs[disk_idx], bucket, key)
        )
        self.assertEqual(fm.iter_all_base_dirs(), base_dirs)

    def test_open_file_creates_bucket_directory(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        fm = NixlFileManager(self.test_dir, use_direct_io=False)
        file_path = fm.get_file_path("page-123")
        fd = fm.open_file(file_path, create=True)
        try:
            self.assertIsNotNone(fd)
            self.assertTrue(os.path.exists(file_path))
        finally:
            if fd is not None:
                os.close(fd)

    def test_clear_removes_nested_bucket_files(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        fm = NixlFileManager(self.test_dir, use_direct_io=False)
        file_path = fm.get_file_path("page-123")
        fd = fm.open_file(file_path, create=True)
        os.close(fd)

        fm.clear()

        self.assertFalse(os.path.exists(file_path))


class TestNixlStorageMetrics(CustomTestCase):
    """get_stats() feeds sglang:{prefetch,backup}_bandwidth for the NIXL L3 tier."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_metrics_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.hicache = self._make_hicache(enable_storage_metrics=True)

    def _make_hicache(self, enable_storage_metrics: bool) -> HiCacheNixl:
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
            enable_storage_metrics=enable_storage_metrics,
            extra_config={
                "plugin": {"posix": {"active": True}},
                "use_direct_io": False,
            },
        )
        try:
            return HiCacheNixl(storage_config=storage_config, file_path=self.test_dir)
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def _run_batch_set_v1(self, hicache, xfer_results: list) -> str:
        mock_host = MockMemPoolHost(is_zero_copy_mode=False)
        hicache.register_mem_pool_host(mock_host)
        hicache._batch_xfer = lambda keys, key_strs, host_buffers, direction: list(
            xfer_results
        )
        keys = [f"metrics_k{i}" for i in range(len(xfer_results))]
        host_indices = torch.arange(len(keys) * mock_host.page_size, dtype=torch.int64)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.hicache_nixl", level="DEBUG"
        ) as logs:
            hicache.batch_set_v1(keys, host_indices)
        return "\n".join(logs.output)

    def test_recorded_bandwidth_is_gb_per_second_not_mb_per_second(self):
        """The histogram buckets are GB/s (max 100); recording the debug line's MB/s
        figure would put every sample in the overflow bucket."""
        log = self._run_batch_set_v1(self.hicache, [True, True])

        logged_mb_s = float(log.split("effective bandwidth: ")[1].split(" MB/s")[0])
        stats = self.hicache.get_stats()

        self.assertEqual(len(stats.backup_bandwidth), 1)
        self.assertAlmostEqual(stats.backup_bandwidth[0], logged_mb_s / 1024, places=4)

    def test_failed_batch_records_no_bandwidth_sample(self):
        """Bytes moved are unknown unless every entry succeeded, so a failed batch
        must not reach the histogram."""
        self._run_batch_set_v1(self.hicache, [True, False])

        stats = self.hicache.get_stats()

        self.assertEqual(stats.backup_bandwidth, [])
        self.assertEqual(stats.backup_pgs, [])

    def test_reads_and_writes_route_to_prefetch_and_backup_respectively(self):
        """Each of the four transfer paths passes its own is_read, so a flipped flag
        at one call site swaps that path's tier without failing anything else.
        """
        mock_host = MockMemPoolHost(is_zero_copy_mode=False)
        self.hicache.register_mem_pool_host(mock_host)
        self.hicache.register_mem_host_pool_v2(MockHybridPool(), PoolName.MAMBA)
        self.hicache._batch_xfer = lambda keys, key_strs, host_buffers, direction: (
            [True] * len(key_strs)
        )
        keys = ["metrics_k0", "metrics_k1"]
        host_indices = torch.arange(len(keys) * mock_host.page_size, dtype=torch.int64)
        transfers = [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=keys,
                host_indices=torch.arange(len(keys), dtype=torch.int64),
            )
        ]
        ops = [
            ("prefetch", lambda: self.hicache.batch_get_v1(keys, host_indices)),
            ("prefetch", lambda: self.hicache.batch_get_v2(transfers)),
            ("backup", lambda: self.hicache.batch_set_v1(keys, host_indices)),
            ("backup", lambda: self.hicache.batch_set_v2(transfers)),
        ]
        for expected_tier, run in ops:
            with self.subTest(tier=expected_tier):
                run()
                stats = self.hicache.get_stats()

                if expected_tier == "prefetch":
                    self.assertEqual(len(stats.prefetch_bandwidth), 1)
                    self.assertEqual(stats.backup_bandwidth, [])
                else:
                    self.assertEqual(len(stats.backup_bandwidth), 1)
                    self.assertEqual(stats.prefetch_bandwidth, [])

    def test_hybrid_transfers_count_pages_not_component_keys(self):
        """A mamba page moves as one key per component (temporal + conv), so the
        page counters must record logical pages, not the expanded NIXL keys."""
        pool = MockHybridPool()
        self.hicache.register_mem_host_pool_v2(pool, PoolName.MAMBA)
        self.hicache._batch_xfer = lambda keys, key_strs, host_buffers, direction: (
            [True] * len(key_strs)
        )
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["metrics_p0", "metrics_p1", "metrics_p2"],
            host_indices=torch.arange(3, dtype=torch.int64),
        )
        self.assertGreater(
            self.hicache._get_hybrid_key_multiplier(PoolName.MAMBA, pool), 1
        )

        self.hicache.batch_set_v2([transfer])
        self.hicache.batch_get_v2([transfer])
        stats = self.hicache.get_stats()

        self.assertEqual(stats.backup_pgs, [3])
        self.assertEqual(stats.prefetch_pgs, [3])

    def test_get_stats_never_splits_a_sample_being_recorded(self):
        """A worker records a page count and a bandwidth as two appends; a drain on
        the scheduler thread between them must not report one without the other."""
        paused, resume = threading.Event(), threading.Event()

        class PausingList(list):
            def append(self, item):
                super().append(item)
                paused.set()
                resume.wait(timeout=10)

        self.hicache._prefetch_pgs = PausingList()
        worker = threading.Thread(
            target=self.hicache._log_xfer_stats,
            args=("batch_get_v1", 2, torch.arange(2), [4096], 1.0, [True]),
            kwargs={"is_read": True},
        )
        worker.start()
        self.assertTrue(paused.wait(timeout=10))
        drained = []
        drain = threading.Thread(
            target=lambda: drained.append(self.hicache.get_stats())
        )
        drain.start()
        # Without the lock the drain finishes inside the worker's pause.
        drain.join(timeout=1)
        resume.set()
        worker.join(timeout=10)
        drain.join(timeout=10)

        stats = [drained[0], self.hicache.get_stats()]
        self.assertEqual(sum(len(s.prefetch_pgs) for s in stats), 1)
        for s in stats:
            self.assertEqual(len(s.prefetch_pgs), len(s.prefetch_bandwidth))

    def test_get_stats_drains_so_samples_are_not_double_counted(self):
        """The collector observes every returned sample, so a drain that only copies
        would replay the whole history into the histogram on each scrape.
        """
        self._run_batch_set_v1(self.hicache, [True, True])

        self.assertEqual(len(self.hicache.get_stats().backup_bandwidth), 1)
        self.assertEqual(self.hicache.get_stats().backup_bandwidth, [])

    def test_disabled_storage_metrics_accumulates_nothing(self):
        """Nothing calls get_stats() when metrics are off, so recording anyway grows
        the sample lists for the lifetime of the process.
        """
        hicache = self._make_hicache(enable_storage_metrics=False)

        self._run_batch_set_v1(hicache, [True, True])

        self.assertEqual(hicache._backup_bandwidth, [])
        self.assertEqual(hicache._backup_pgs, [])


if __name__ == "__main__":
    unittest.main()
