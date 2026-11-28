"""NIXL tier-3 storage on Intel XPU pinned host pools: O_DIRECT fallback and transfers.

torch-xpu pinned host memory can fail get_user_pages(), so an O_DIRECT write sourced
from it, or a read landing in it, returns EFAULT; which direction fails depends on the
kernel. Before the fallback, every tier-3 write then left a zero-length file that
batch_exists counted as cached. The CUDA-registered NIXL unit
tests stub the probe and use unpinned CPU tensors, so they cannot cover this.

Each test builds a real XPU host pool (pinned, as in serving), registers it with
HiCacheNixl over the POSIX plugin with O_DIRECT requested, and checks that:

* the O_DIRECT decision matches a raw O_DIRECT pwrite from every buffer the backend
  writes tier-3 data from and a raw O_DIRECT preadv into every buffer it reads it
  into (kv_buffer / component buffers both ways when zero-copy, bounce_set and
  bounce_get otherwise), so it is not agreeing with its own probe;
* a set/get round trip through NIXL returns every byte, and no tier-3 file is
  left empty.

The class skips when NIXL storage cannot run here: no importable NIXL backend
module, no POSIX plugin, or a POSIX backend that cannot start (e.g. io_uring
blocked by the container's seccomp profile).
"""

import ctypes
import errno
import mmap
import os
import shutil
import tempfile
import unittest
from typing import Optional

import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="extra-test-1-gpu-xpu")

XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False

DEVICE = "xpu"
PAGE_SIZE = 16
NUM_PAGES = 8
POOL_TOKENS = PAGE_SIZE * NUM_PAGES
LAYER_NUM = 4
DTYPE = torch.bfloat16
# Mirrors _OS_PAGE_BYTES in hicache_nixl; kept local so the expectation does not
# move with the code under test.
_PAGE_BYTES = 4096
# The errnos that make O_DIRECT unusable rather than transiently failing.
_UNUSABLE_ERRNOS = (errno.EFAULT, errno.EINVAL)


# Memory-backed: tmpfs accepts O_DIRECT since Linux 6.6 but never pins the user
# buffer, so pinned XPU memory that EFAULTs on a disk filesystem passes here.
_MEMORY_FILESYSTEMS = frozenset({"tmpfs", "ramfs"})


def _fs_type(path: str) -> Optional[str]:
    """Filesystem type of the mount holding ``path``, from /proc/self/mountinfo."""
    path = os.path.realpath(path)
    best, fs_type = "", None
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                fields, _, rest = line.partition(" - ")
                mount_point = fields.split()[4].replace("\\040", " ")
                prefix = mount_point.rstrip("/") + "/"
                if (path == mount_point or path.startswith(prefix)) and len(
                    mount_point
                ) >= len(best):
                    best, fs_type = mount_point, rest.split()[0]
    except OSError:
        return None
    return fs_type


def _o_direct_dir() -> Optional[str]:
    """A disk-backed directory whose filesystem accepts O_DIRECT opens, or None.

    Try the temp dir first and then the on-disk checkout this test lives in,
    skipping memory-backed filesystems, where the O_DIRECT decision is vacuous.
    """
    for parent in (tempfile.gettempdir(), os.path.dirname(os.path.abspath(__file__))):
        if _fs_type(parent) in _MEMORY_FILESYSTEMS:
            continue
        path = os.path.join(parent, f".nixl_odirect_check.{os.getpid()}")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
        except OSError:
            continue
        os.close(fd)
        os.unlink(path)
        return parent
    return None


def _nixl_posix_unavailable_reason() -> Optional[str]:
    """Why NIXL POSIX storage cannot run in this process, or None if it can.

    Creates the POSIX backend with the same default params the test config uses.
    """
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as e:
        # Includes the meta package picking a missing nixl_cuXX backend.
        return f"NIXL not importable: {e}"
    agent = nixl_agent(f"nixl_xpu_probe_{os.getpid()}", nixl_agent_config(backends=[]))
    if "POSIX" not in agent.get_plugin_list():
        return "NIXL POSIX plugin not installed"
    try:
        agent.create_backend("POSIX")
    except Exception as e:
        return f"NIXL POSIX backend cannot start: {e}"
    return None


def _aligned_block(buf: torch.Tensor) -> Optional[ctypes.Array]:
    """The first page-aligned 4 KiB block inside ``buf``, or None if none fits."""
    addr = buf.data_ptr()
    aligned = addr + (-addr) % _PAGE_BYTES
    if aligned - addr + _PAGE_BYTES > buf.numel() * buf.element_size():
        return None
    return (ctypes.c_char * _PAGE_BYTES).from_address(aligned)


def _raw_direct_write_errno(directory: str, buf: torch.Tensor) -> Optional[int]:
    """errno of a page-aligned O_DIRECT pwrite sourced from ``buf``, None if it works.

    Raw syscalls only, so the expectation is independent of NixlFileManager.
    """
    block = _aligned_block(buf)
    if block is None:
        # The backend falls back for a buffer that cannot hold an aligned page.
        return errno.EINVAL
    path = os.path.join(directory, f".raw_probe.{os.getpid()}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        os.pwrite(fd, block, 0)
    except OSError as e:
        return e.errno
    finally:
        os.close(fd)
        os.unlink(path)
    return None


def _raw_direct_read_errno(directory: str, buf: torch.Tensor) -> Optional[int]:
    """errno of a page-aligned O_DIRECT preadv into ``buf``, None if it works.

    Overwrites one block of ``buf``. A read pins its destination for writing, so a
    buffer can source an O_DIRECT write and still refuse this.
    """
    block = _aligned_block(buf)
    if block is None:
        return errno.EINVAL
    path = os.path.join(directory, f".raw_probe.{os.getpid()}")
    # Seeded from page-aligned anonymous memory, so only the read tests ``buf``.
    seed = mmap.mmap(-1, _PAGE_BYTES)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        os.pwrite(fd, seed, 0)
        os.preadv(fd, [block], 0)
    except OSError as e:
        return e.errno
    finally:
        os.close(fd)
        os.unlink(path)
        seed.close()
    return None


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available")
class TestHiCacheNixlStorageXpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        reason = _nixl_posix_unavailable_reason()
        if reason is not None:
            raise unittest.SkipTest(reason)

    def setUp(self):
        parent = _o_direct_dir()
        if parent is None:
            # The decision check then only agrees with the raw syscalls there; the
            # round trip and empty-file checks still hold.
            parent = tempfile.gettempdir()
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_xpu_", dir=parent)
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.hicache = self._make_hicache()

    def _make_hicache(self):
        from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl

        config = HiCacheStorageConfig(
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
                "use_direct_io": True,
                "l3_cleaner_enabled": False,
            },
        )
        try:
            hicache = HiCacheNixl(storage_config=config, file_path=self.test_dir)
        except ImportError:
            self.skipTest("NIXL not available")
        self.addCleanup(hicache.close)
        self.assertTrue(hicache.needs_page_alignment)
        return hicache

    def _assert_direct_io_decision(self, sources, destinations):
        """O_DIRECT stays on iff a raw O_DIRECT write works from every source and a
        raw O_DIRECT read works into every destination."""
        raw = {
            f"write from {i}": _raw_direct_write_errno(self.test_dir, buf)
            for i, buf in enumerate(sources)
        }
        raw.update(
            {
                f"read into {i}": _raw_direct_read_errno(self.test_dir, buf)
                for i, buf in enumerate(destinations)
            }
        )
        expect_direct = not any(e in _UNUSABLE_ERRNOS for e in raw.values())
        detail = {k: (errno.errorcode.get(e, e) if e else "ok") for k, e in raw.items()}
        self.assertEqual(
            self.hicache.file_manager.use_direct_io,
            expect_direct,
            f"raw O_DIRECT syscall per transfer buffer: {detail}",
        )
        self.assertEqual(self.hicache.needs_page_alignment, expect_direct)
        # The registry opens files through this manager; a copy would keep O_DIRECT.
        self.assertIs(self.hicache.registry.file_manager, self.hicache.file_manager)

    def _assert_no_empty_tier3_files(self):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(self.test_dir)
            for name in names
        ]
        self.assertTrue(files, "batch_set wrote no tier-3 files")
        empty = [f for f in files if os.path.getsize(f) == 0]
        self.assertEqual(empty, [], "tier-3 writes left zero-length files")

    def _mha_host(self, layout: str) -> MHATokenToKVPoolHost:
        device_pool = MHATokenToKVPool(
            size=POOL_TOKENS,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            head_num=2,
            head_dim=128,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        host = MHATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout=layout,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            write_back_staging=device_pool.hicache_write_back_staging,
        )
        # The contract under test is the pinned allocation serving uses.
        self.assertTrue(host.pin_memory)
        return host

    def _mha_round_trip(self, layout: str, expect_zero_copy: Optional[bool]):
        host = self._mha_host(layout)
        self.hicache.register_mem_pool_host(host)
        if expect_zero_copy is not None:
            self.assertEqual(self.hicache.is_zero_copy, expect_zero_copy)
        if self.hicache.is_zero_copy:
            # Zero-copy writes from and reads into kv_buffer itself.
            self._assert_direct_io_decision([host.kv_buffer], [host.kv_buffer])
        else:
            self._assert_direct_io_decision(
                [self.hicache._bounce_set], [self.hicache._bounce_get]
            )

        # Pages 1..N-1, so a transfer that ignores host_indices round-trips wrong.
        pages = range(1, NUM_PAGES)
        host_indices = torch.arange(PAGE_SIZE, PAGE_SIZE * NUM_PAGES, dtype=torch.int64)
        keys = [f"xpu_{layout}_{p}" for p in pages]
        host.kv_buffer.copy_(torch.randn(host.kv_buffer.shape, dtype=torch.float32))
        expected = [host.get_data_page(p * PAGE_SIZE).clone() for p in pages]

        self.assertEqual(
            self.hicache.batch_set_v1(keys, host_indices), [True] * len(keys)
        )
        self._assert_no_empty_tier3_files()
        host.kv_buffer.zero_()
        self.assertEqual(
            self.hicache.batch_get_v1(keys, host_indices), [True] * len(keys)
        )
        for p, want in zip(pages, expected):
            self.assertTrue(
                torch.equal(host.get_data_page(p * PAGE_SIZE), want),
                f"page {p} changed across the NIXL round trip ({layout})",
            )

    def test_mha_page_first_pinned_pool(self):
        # Zero-copy when the pinned base is page-aligned: transfers read kv_buffer
        # itself, the allocation torch-xpu pins.
        self._mha_round_trip("page_first", expect_zero_copy=None)

    def test_mha_layer_first_pinned_pool(self):
        # Copy mode: transfers read the mmap bounce buffer, not the pinned pool.
        self._mha_round_trip("layer_first", expect_zero_copy=False)

    def test_mamba_page_first_pinned_pool(self):
        def state(layers):
            # 4 KiB per slot across layers in each component, so a page-aligned
            # pinned base gives zero-copy.
            return MambaStateBufferInfo(
                temporal=torch.empty(
                    (layers, POOL_TOKENS, 8, 32), dtype=torch.float32, device=DEVICE
                ),
                conv=(
                    torch.empty(
                        (layers, POOL_TOKENS, 4, 128), dtype=DTYPE, device=DEVICE
                    ),
                ),
            )

        host = MambaPoolHost(
            state(LAYER_NUM),
            host_to_device_ratio=2.0,
            host_size=0,
            device_capacity=POOL_TOKENS,
            layout="page_first",
        )
        self.assertTrue(host.pin_memory)
        self.hicache.register_mem_host_pool_v2(host, PoolName.MAMBA)
        ctx = self.hicache._hybrid_pool_ctx[PoolName.MAMBA]
        if ctx.is_zero_copy:
            buffers = [b for b in host.get_hybrid_pool_buffer() if b.numel() > 0]
            self._assert_direct_io_decision(buffers, buffers)
        else:
            self._assert_direct_io_decision([ctx.bounce_set], [ctx.bounce_get])

        slots = [1, 3, 5]
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=[f"xpu_mamba_{s}" for s in slots],
            host_indices=torch.tensor(slots, dtype=torch.int64),
        )
        for buf in host.get_hybrid_pool_buffer():
            buf.copy_(torch.randn(buf.shape, dtype=torch.float32))
        expected = [host.get_data_page(s).clone() for s in slots]

        self.assertEqual(
            self.hicache.batch_set_v2([transfer]), {PoolName.MAMBA: [True] * len(slots)}
        )
        self._assert_no_empty_tier3_files()
        for buf in host.get_hybrid_pool_buffer():
            buf.zero_()
        self.assertEqual(
            self.hicache.batch_get_v2([transfer]), {PoolName.MAMBA: [True] * len(slots)}
        )
        for s, want in zip(slots, expected):
            self.assertTrue(
                torch.equal(host.get_data_page(s), want),
                f"mamba slot {s} changed across the NIXL round trip",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
