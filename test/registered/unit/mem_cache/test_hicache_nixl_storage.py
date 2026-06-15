"""Unit tests for the NIXL HiCache storage backend -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from typing import List

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.test.test_utils import CustomTestCase

# Stress tests are opt-in: CI never sets this; set locally to exercise them.
STRESS_ENABLED = bool(os.environ.get("SGLANG_RUN_NIXL_STRESS"))


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
    ):
        self.layout = "page_first" if is_zero_copy_mode else "layer_first"
        self.page_size = page_size
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
        # Test tensors are too small to satisfy 4 KiB stride alignment; the
        # O_DIRECT path correctly falls back to copy mode in this case.
        return False


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
        self.test_dir = "/tmp/test_nixl_unified"
        os.makedirs(self.test_dir, exist_ok=True)

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
            },
        )

        try:
            self.hicache = HiCacheNixl(
                storage_config=self.storage_config,
                file_path=self.test_dir,
            )
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

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

            set_results = hicache.batch_set_v1(keys, host_indices)
            self.assertTrue(
                all(set_results),
                f"batch_set_v1 failed (zero_copy={is_zero_copy_mode}): {set_results}",
            )

            get_results = hicache.batch_get_v1(keys, host_indices)
            self.assertTrue(
                all(get_results),
                f"batch_get_v1 failed (zero_copy={is_zero_copy_mode}): {get_results}",
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
            return HiCacheNixl(storage_config=obj_config, file_path="")
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

    def _run_concurrent_stress(
        self, is_zero_copy_mode: bool, hicache: HiCacheNixl = None
    ):
        """One getter thread + one setter thread share the same HiCacheNixl
        for ``is_zero_copy_mode``. Defaults to ``self.hicache`` (FILE backend);
        pass ``hicache`` to exercise a different backend (e.g. OBJ).

        Phase 1 pre-seeds N preset pages and stores them under fixed keys.
        Phase 2 runs the getter (reads the presets back and verifies content)
        concurrently with the setter (writes a stream of fresh distinct keys
        from a disjoint source region). The kv_buffer regions touched by the
        two threads are disjoint so any data corruption observed is from the
        backend's shared state (bounce buffers, devId maps, fd pool).
        """
        if hicache is None:
            hicache = self.hicache

        # 8 preset pages, 8 getter dst pages, 8 setter src pages -> 24 in use.
        mock_host = MockMemPoolHost(is_zero_copy_mode=is_zero_copy_mode, num_pages=32)
        hicache.register_mem_pool_host(mock_host)
        hicache.is_zero_copy = is_zero_copy_mode

        page_size = mock_host.page_size
        dtype = mock_host.dtype
        num_pages = 8

        # Disjoint per-thread regions in kv_buffer (indexed by token index).
        preset_src = (0, num_pages)
        getter_dst = (num_pages, 2 * num_pages)
        setter_src = (2 * num_pages, 3 * num_pages)

        # zero_copy=page_first uses dim 1 for the token axis; non-zero-copy=
        # layer_first uses dim 2. All buffer accesses below go through this so
        # the rest of the harness stays layout-agnostic.
        def token_index(start_token: int, n_tokens: int):
            s = slice(start_token, start_token + n_tokens)
            if is_zero_copy_mode:
                return (slice(None), s, slice(None), slice(None), slice(None))
            return (slice(None), slice(None), s, slice(None), slice(None))

        def page_index(start_page: int, n_pages: int):
            return token_index(start_page * page_size, n_pages * page_size)

        def fill_pages(start_page: int, n_pages: int, value_fn):
            """value_fn(i) -> scalar value for page i."""
            for i in range(n_pages):
                idx = page_index(start_page + i, 1)
                shape = mock_host.kv_buffer[idx].shape
                mock_host.kv_buffer[idx] = torch.full(
                    shape, float(value_fn(i)), dtype=dtype
                )

        # Phase 1: distinct value per preset page so a wrong-page result is
        # detectable; setter source is constant (value irrelevant to the
        # test, just needs to be valid).
        fill_pages(preset_src[0], num_pages, lambda i: i + 1)
        fill_pages(setter_src[0], num_pages, lambda i: -1.0)

        preset_keys = [f"preset_{int(is_zero_copy_mode)}_{i}" for i in range(num_pages)]
        preset_indices = torch.arange(
            preset_src[0] * page_size,
            preset_src[1] * page_size,
            dtype=torch.int64,
        )
        self.assertTrue(
            all(hicache.batch_set_v1(preset_keys, preset_indices)),
            "phase 1: presetting keys failed",
        )

        # Expected per-page-i payload after a successful get into getter_dst.
        expected_pages = [
            mock_host.kv_buffer[page_index(preset_src[0] + i, 1)].clone()
            for i in range(num_pages)
        ]

        # Phase 2.
        stop = threading.Event()
        errors: List[str] = []
        errors_lock = threading.Lock()

        def record_error(msg: str):
            with errors_lock:
                errors.append(msg)

        def getter_loop():
            dst_indices = torch.arange(
                getter_dst[0] * page_size,
                getter_dst[1] * page_size,
                dtype=torch.int64,
            )
            loops = 0
            while not stop.is_set():
                # Zero the dst pages so a no-op get is observable.
                mock_host.kv_buffer[page_index(getter_dst[0], num_pages)] = 0.0
                ok = hicache.batch_get_v1(preset_keys, dst_indices)
                if not all(ok):
                    record_error(f"getter loop {loops}: batch_get_v1 returned {ok}")
                    return
                for i in range(num_pages):
                    got = mock_host.kv_buffer[page_index(getter_dst[0] + i, 1)]
                    if not torch.equal(got, expected_pages[i]):
                        record_error(f"getter loop {loops}: preset page {i} corrupted")
                        return
                loops += 1

        def setter_loop():
            src_indices = torch.arange(
                setter_src[0] * page_size,
                setter_src[1] * page_size,
                dtype=torch.int64,
            )
            loops = 0
            while not stop.is_set():
                keys = [
                    f"setter_{int(is_zero_copy_mode)}_{loops}_{i}"
                    for i in range(num_pages)
                ]
                ok = hicache.batch_set_v1(keys, src_indices)
                if not all(ok):
                    record_error(f"setter loop {loops}: batch_set_v1 returned {ok}")
                    return
                loops += 1

        t_get = threading.Thread(target=getter_loop, daemon=True)
        t_set = threading.Thread(target=setter_loop, daemon=True)
        t_get.start()
        t_set.start()

        # Bounded run: long enough to interleave many ops under NIXL I/O
        # GIL release, short enough for a unit test.
        time.sleep(3.0)
        stop.set()
        t_get.join(timeout=10)
        t_set.join(timeout=10)

        self.assertFalse(
            t_get.is_alive() or t_set.is_alive(),
            "stress threads failed to stop",
        )
        self.assertEqual(errors, [], f"concurrency errors: {errors}")

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    def test_concurrent_getter_setter_file_zero_copy(self):
        """Stress: concurrent getter+setter, FILE backend, zero-copy."""
        self._run_concurrent_stress(is_zero_copy_mode=True)

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    def test_concurrent_getter_setter_file_non_zero_copy(self):
        """Stress: concurrent getter+setter, FILE backend, non-zero-copy."""
        self._run_concurrent_stress(is_zero_copy_mode=False)

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_concurrent_getter_setter_obj_zero_copy(self):
        """Stress: concurrent getter+setter, OBJ backend (MinIO), zero-copy."""
        self._run_concurrent_stress(
            is_zero_copy_mode=True, hicache=self._make_obj_hicache()
        )

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_concurrent_getter_setter_obj_non_zero_copy(self):
        """Stress: concurrent getter+setter, OBJ backend (MinIO), non-zero-copy."""
        self._run_concurrent_stress(
            is_zero_copy_mode=False, hicache=self._make_obj_hicache()
        )


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available on this platform")
class TestNixlDirectIO(CustomTestCase):
    """Tests for the O_DIRECT file I/O path in NixlFileManager and HiCacheNixl."""

    def setUp(self):
        self.test_dir = "/tmp/test_nixl_direct_io"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_open_file_sets_o_direct(self):
        """open_file sets O_DIRECT on the file descriptor when use_direct_io=True."""
        import fcntl

        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

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

    def _make_direct_io_hicache(self) -> HiCacheNixl:
        """Return a HiCacheNixl configured for O_DIRECT (default) with the POSIX backend."""
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
            extra_config={"plugin": {"posix": {"active": True}}},
            # use_direct_io defaults to True (env var)
        )
        try:
            return HiCacheNixl(storage_config=storage_config, file_path=self.test_dir)
        except ImportError:
            self.skipTest("NIXL not available")

    def test_needs_page_alignment_true_for_file_backend(self):
        """File-based backend + use_direct_io=True must set needs_page_alignment."""
        hicache = self._make_direct_io_hicache()
        self.assertTrue(hicache.needs_page_alignment)

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


class TestNixlFileLayout(CustomTestCase):
    """Tests for deterministic NIXL FILE storage path layout."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_layout_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_route_key_is_stable_and_bucketed(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_routing import (
            _BUCKET_HEX_CHARS,
            route_key,
        )

        self.assertEqual(route_key("page-123", 4), route_key("page-123", 4))
        disk_idx, bucket = route_key("page-123", 4)
        self.assertGreaterEqual(disk_idx, 0)
        self.assertLess(disk_idx, 4)
        self.assertEqual(len(bucket), _BUCKET_HEX_CHARS)
        self.assertRegex(bucket, rf"^[0-9a-f]{{{_BUCKET_HEX_CHARS}}}$")

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


if __name__ == "__main__":
    unittest.main()
