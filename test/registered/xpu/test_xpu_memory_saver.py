"""Unit tests for Intel XPU memory saver (release/resume memory occupation).

These exercise the upstream ``torch_memory_saver`` package's Level Zero
pause/resume path on XPU -- the same backend SGLang's ``release_memory_occupation``
/ ``resume_memory_occupation`` use -- without booting a full inference server.

The tests are skipped unless:
  * torch XPU is available, and
  * ``torch_memory_saver`` is installed with its XPU backend, built from source
    against the local oneAPI (see docs_new/docs/hardware-platforms/xpu.mdx).
    Pinned to the commit that merged XPU support; the PyPI wheel predates it:
      TMS_PLATFORM=xpu pip install --no-build-isolation \\
          git+https://github.com/fzyzcjy/torch_memory_saver.git@990ec133b7b178ca4e30fe7aaac921d3701bcb3a

Physical-memory release is verified via the saver's driver-independent
``tms_xpu_committed_bytes`` (physical bytes it holds ACTIVE on a device, which
pause() releases and resume() re-commits). Neither sysman free-bytes
(``tms_xpu_device_free_bytes`` / ``torch.xpu.mem_get_info``) nor
``torch.xpu.memory_allocated()`` is used: the former is frozen on newer Intel
drivers and the latter is allocator bookkeeping that ignores ``zeVirtualMemUnmap``.
"""

import unittest

import torch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")


def _xpu_saver_available():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        import torch_memory_saver  # noqa: F401
    except ImportError:
        return False
    return True


_XPU_OK = _xpu_saver_available()

if _XPU_OK:
    import torch_memory_saver as _tms

    # The package singleton. On XPU it requires hook_mode="torch"; importing the
    # adapter sets that exactly once (it must be set before the singleton is
    # initialized). We import the adapter here so the mode is configured before
    # any region()/pause() in these tests touches the singleton.
    import sglang.srt.utils.torch_memory_saver_adapter  # noqa: F401

    xpu_memory_saver = _tms.torch_memory_saver

_GIB = 1024**3
# 1 GiB tensor: large enough that a real physical release is unmistakable.
_N_FP32 = 256 * 1024 * 1024
# How much committed memory a 1 GiB region should drop on pause, with alignment slack.
_RELEASE_THRESHOLD_GIB = 0.8


@unittest.skipUnless(
    _XPU_OK,
    "Requires torch XPU and torch_memory_saver with its XPU backend "
    "(TMS_PLATFORM=xpu pip install --no-build-isolation git+https://"
    "github.com/fzyzcjy/torch_memory_saver.git@990ec133b7b178ca4e30fe7aaac921d3701bcb3a).",
)
class TestXpuMemorySaver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pin a single device for the whole class. Probing every device with
        # set_device() churns SYCL/L0 contexts and can destabilize the runtime
        # when many GPU-heavy tests share one process; pick once, by free memory
        # (mem_get_info(i) reads a specific device without changing the current one).
        best, best_free = 0, -1
        for i in range(torch.xpu.device_count()):
            free, _ = torch.xpu.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        torch.xpu.set_device(best)
        cls.device_index = best
        cls.device = f"xpu:{best}"
        xpu_memory_saver._ensure_initialized()
        cls._cdll = xpu_memory_saver._impl._binary_wrapper.cdll

    def tearDown(self):
        # Ensure no region is left paused across tests (paused state on the
        # process-global singleton would leak into the next test).
        try:
            xpu_memory_saver.resume(None)
        except Exception:
            pass
        torch.xpu.synchronize()

    def _committed_gib(self):
        # Physical bytes the saver holds ACTIVE on this device. Driver-independent
        # (unlike frozen sysman free-bytes); drops on pause(), restored on resume().
        torch.xpu.synchronize()
        return self._cdll.tms_xpu_committed_bytes(self.device_index) / _GIB

    # ------------------------------------------------------------------ basics
    def test_pause_releases_and_resume_restores(self):
        """A region tensor's physical memory is freed on pause and re-committed
        on resume at the same virtual address; the tensor stays usable."""
        from sglang.srt.utils.torch_memory_saver_adapter import (
            TorchMemorySaverAdapter,
        )

        adapter = TorchMemorySaverAdapter.create(enable=True)
        self.assertTrue(adapter.enabled)

        with adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            t = torch.ones(_N_FP32, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()
        self.assertEqual(float(t[0]), 1.0)
        committed_after_alloc = self._committed_gib()

        adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
        committed_after_pause = self._committed_gib()
        self.assertGreater(
            committed_after_alloc - committed_after_pause,
            _RELEASE_THRESHOLD_GIB,
            "pause should release ~1 GiB of physical device memory",
        )

        adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
        committed_after_resume = self._committed_gib()
        self.assertGreater(
            committed_after_resume - committed_after_pause,
            _RELEASE_THRESHOLD_GIB,
            "resume should re-commit ~1 GiB of physical device memory",
        )

        # Same VA must be writable again after resume.
        t.fill_(5.0)
        torch.xpu.synchronize()
        self.assertEqual(float(t[0]), 5.0)

    def test_region_scoping_isolates_unmanaged_memory(self):
        """pause(tag) must not touch allocations made outside a region()."""
        outside = torch.ones(_N_FP32 // 4, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()

        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            inside = torch.ones(_N_FP32, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()

        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        # The unmanaged tensor must remain valid through a pause of another tag.
        self.assertEqual(float(outside[0]), 1.0)
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)
        del inside, outside

    def test_cpu_backup_preserves_contents(self):
        """With enable_cpu_backup, contents survive a pause/resume cycle."""
        with xpu_memory_saver.region(
            tag=GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=True
        ):
            w = torch.full((_N_FP32,), 2.0, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()

        xpu_memory_saver.pause(GPU_MEMORY_TYPE_WEIGHTS)
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_WEIGHTS)
        torch.xpu.synchronize()

        # CPU backup should restore the exact prior contents.
        self.assertEqual(float(w[0]), 2.0)
        self.assertEqual(float(w[-1]), 2.0)
        del w

    def test_tag_selectivity(self):
        """pause(weights) must leave a kv_cache region mapped and usable.

        Asserted via tensor readability rather than global free memory: the
        saver is a process-global singleton and torch's MemPool caches freed
        blocks, so other-tagged allocations from earlier tests may still be live
        and would perturb a free-memory delta. Readability of *this* tensor is
        the precise selectivity property.
        """
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            kv = torch.full((_N_FP32,), 4.0, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()

        # Pausing a DIFFERENT tag must not unmap the kv_cache region.
        xpu_memory_saver.pause(GPU_MEMORY_TYPE_WEIGHTS)
        torch.xpu.synchronize()
        self.assertEqual(
            float(kv[0]),
            4.0,
            "pausing an unrelated tag must leave the kv_cache region mapped",
        )
        self.assertEqual(float(kv[-1]), 4.0)

        # Now pause the matching tag and confirm it frees real memory.
        committed_before_kv = self._committed_gib()
        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        committed_after_kv = self._committed_gib()
        self.assertGreater(
            committed_before_kv - committed_after_kv,
            _RELEASE_THRESHOLD_GIB,
            "pausing the matching tag should release ~1 GiB",
        )

        # Restore everything we paused so we don't leak paused state.
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_WEIGHTS)
        del kv

    # ----------------------------------------------------------- adapter wiring
    def test_adapter_selected_on_xpu(self):
        """TorchMemorySaverAdapter.create() returns the real (enabled) backend on
        XPU, and a disabled adapter is a no-op that reports not-enabled."""
        from sglang.srt.utils.torch_memory_saver_adapter import (
            TorchMemorySaverAdapter,
            _TorchMemorySaverAdapterNoop,
            _TorchMemorySaverAdapterReal,
        )

        adapter = TorchMemorySaverAdapter.create(enable=True)
        self.assertIsInstance(adapter, _TorchMemorySaverAdapterReal)
        self.assertTrue(adapter.enabled)

        noop = TorchMemorySaverAdapter.create(enable=False)
        self.assertIsInstance(noop, _TorchMemorySaverAdapterNoop)
        self.assertFalse(noop.enabled)

    def test_adapter_cuda_graph_and_configure_subprocess_are_noops(self):
        """On XPU, configure_subprocess (no LD_PRELOAD) and cuda_graph (no
        pauseable graph capture) must be harmless no-op context managers that
        do not raise."""
        from sglang.srt.utils.torch_memory_saver_adapter import (
            TorchMemorySaverAdapter,
        )

        adapter = TorchMemorySaverAdapter.create(enable=True)
        with adapter.configure_subprocess():
            pass
        with adapter.cuda_graph(tag=GPU_MEMORY_TYPE_CUDA_GRAPH):
            pass

    # --------------------------------------------------------------- all tags
    def test_release_all_tags_with_none(self):
        """pause(None)/resume(None) act on every region regardless of tag,
        matching release_memory_occupation() with no tags (all types)."""
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            kv = torch.ones(_N_FP32, dtype=torch.float32, device=self.device)
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_WEIGHTS):
            wt = torch.ones(_N_FP32, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()
        committed_alloc = self._committed_gib()

        # None == all tags: should free both regions (~2 GiB).
        xpu_memory_saver.pause(None)
        committed_pause = self._committed_gib()
        self.assertGreater(
            committed_alloc - committed_pause,
            2 * _RELEASE_THRESHOLD_GIB,
            "pause(None) should release every region (~2 GiB)",
        )

        xpu_memory_saver.resume(None)
        kv.fill_(1.0)
        wt.fill_(1.0)
        torch.xpu.synchronize()
        self.assertEqual(float(kv[-1]), 1.0)
        self.assertEqual(float(wt[-1]), 1.0)
        del kv, wt

    # ------------------------------------------------------------- robustness
    def test_double_pause_and_double_resume_are_idempotent(self):
        """Pausing an already-paused tag (or resuming an active one) must be a
        safe no-op, not a crash or double-free."""
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            t = torch.full((_N_FP32,), 3.0, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()

        # resume while already active -> no-op
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)
        committed_active = self._committed_gib()

        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        # pause again while already paused -> no-op (no extra free, no crash)
        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        committed_paused = self._committed_gib()
        self.assertGreater(committed_active - committed_paused, _RELEASE_THRESHOLD_GIB)

        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)  # second resume -> no-op
        t.fill_(9.0)
        torch.xpu.synchronize()
        self.assertEqual(float(t[0]), 9.0)
        del t

    def test_multiple_regions_same_tag_all_released(self):
        """Several allocations under one tag are all released together and all
        usable again after resume."""
        tensors = []
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            for v in (1.0, 2.0, 3.0):
                tensors.append(
                    torch.full(
                        (_N_FP32 // 2,), v, dtype=torch.float32, device=self.device
                    )
                )
        torch.xpu.synchronize()
        committed_alloc = self._committed_gib()

        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        committed_pause = self._committed_gib()
        # 3 x 0.5 GiB = ~1.5 GiB.
        self.assertGreater(committed_alloc - committed_pause, _RELEASE_THRESHOLD_GIB)

        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)
        for v, t in zip((1.0, 2.0, 3.0), tensors):
            # contents are not preserved without cpu_backup; just confirm the VA
            # is mapped and writable again.
            t.fill_(v)
        torch.xpu.synchronize()
        for v, t in zip((1.0, 2.0, 3.0), tensors):
            self.assertEqual(float(t[0]), v)
        del tensors

    def test_resume_without_backup_gives_writable_fresh_memory(self):
        """Without cpu_backup, resumed memory need not preserve contents (KV
        cache is flushed anyway) but must be freshly writable at the same VA."""
        with xpu_memory_saver.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            t = torch.full((_N_FP32,), 7.0, dtype=torch.float32, device=self.device)
        torch.xpu.synchronize()
        ptr_before = t.data_ptr()

        xpu_memory_saver.pause(GPU_MEMORY_TYPE_KV_CACHE)
        xpu_memory_saver.resume(GPU_MEMORY_TYPE_KV_CACHE)

        # Same virtual address preserved across the cycle.
        self.assertEqual(t.data_ptr(), ptr_before)
        # Writable again.
        t.fill_(42.0)
        torch.xpu.synchronize()
        self.assertEqual(float(t[0]), 42.0)
        self.assertEqual(float(t[-1]), 42.0)
        del t


if __name__ == "__main__":
    unittest.main()
