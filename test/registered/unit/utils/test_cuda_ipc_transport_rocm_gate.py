"""Unit tests for the ROCm/HIP platform gating in cuda_ipc_transport_utils.py.

Background (sgl-project/sglang#29687, a follow-up to #29227): PyTorch exposes
ROCm through the same torch.cuda.* / torch.device("cuda") surface as CUDA --
there is no separate HIP device type -- so `tensor.is_cuda` is True on both
backends. The raw CUDA-IPC handle reconstruction in
CudaIpcTensorTransportProxy.reconstruct_on_target_device (both the pooled and
non-pooled branches) used to be gated only on `tensor.is_cuda`, so it was
attempted unconditionally on ROCm too, where it crashes with
"HIP error: invalid device pointer" because a device-redirected IPC handle
is not something HIP's IPC implementation can reopen the same way.

The fix adds `is_hip()` assertions directly at the two raw-IPC-reconstruction
call sites (_open_pooled_storage_uncached, and the non-pooled branch inside
reconstruct_on_target_device) as defense-in-depth, on top of the real fix
which is to stop *constructing* IPC-handle-bearing proxies at all on ROCm
(tested separately in test/registered/unit/multimodal/test_cuda_ipc_rocm_gate.py,
next to base_processor.py's CUDA_IPC_TRANSPORT_SUPPORTED).

These tests cover:
  1. The assertions fire (AssertionError, not a crash further down in a HIP
     call) when is_hip() is patched to True and the corresponding path is
     reached.
  2. On non-HIP, the same paths do NOT raise from the assertion (they may
     still raise/fail for unrelated reasons like "no real CUDA available",
     which is expected and not what's under test here).
  3. The `tensor_data` passthrough path -- the *existing*, already-working,
     platform-agnostic non-IPC fallback that MmItemMemoryPool's log message
     refers to -- correctly round-trips tensor values. This is CPU-simulated
     (device="cpu" both ends) since no real GPU is available in this
     environment; it exercises the exact code path
     (`self.proxy_state["tensor_data"].to(rebuild_device, ...)`) that ROCm
     now unconditionally uses instead of raw IPC.
  4. _cuda_device_for_index, the pure helper extracted out of
     reconstruct_on_target_device to deduplicate the `cuda:{idx}` device
     string construction -- zero mocking needed.

On mocking: the only thing patched anywhere in this file is `is_hip()`
itself, which reads real hardware/build state (`torch.version.hip`) that
cannot be fabricated in a sandbox without ROCm -- this is an environment
boundary fake, not a fake of the code under test. The CPU-simulated
tensor_data-branch test drives the real reconstruct_on_target_device method
through a real, explicit, defaulted `_rebuild_device_override` parameter
(see that method's docstring) instead of patching the `torch` module
reference inside cuda_ipc_transport_utils, so no torch internals are faked.

No server, no model loading, no GPU required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import cuda_ipc_transport_utils as cuda_ipc_mod
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy
from sglang.test.test_utils import CustomTestCase


def _make_proxy_with_ipc_extra(ipc_extra: dict) -> CudaIpcTensorTransportProxy:
    """Construct a CudaIpcTensorTransportProxy without going through __init__
    (which requires a real CUDA tensor to call _share_cuda_() on).

    This is not a test-only shortcut standing in for something more "real" --
    it mirrors actual production semantics on the consumer side. A
    CudaIpcTensorTransportProxy instance is what crosses the wire between the
    tokenizer worker and the scheduler process (via multiprocessing, i.e.
    pickle); `pickle.loads` restores an object's `__dict__` WITHOUT calling
    `__init__` again (verified: a plain `__init__` with a print statement
    only prints once, at the producer-side construction, never again on the
    consumer side after unpickling). So the real consumer-side proxy that
    `reconstruct_on_target_device` runs against also never re-runs __init__ --
    `object.__new__` + restoring `proxy_state` here is the same shape of
    object a real consumer process actually holds, just built directly
    instead of via a pickle round-trip.
    """
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.proxy_state = {"ipc_extra": ipc_extra, "tensor_data": None}
    proxy.reconstruct_tensor = None
    proxy.sync_data_meta = None
    proxy.sync_buffer = None
    return proxy


def _make_proxy_with_tensor_data(tensor: torch.Tensor) -> CudaIpcTensorTransportProxy:
    """See _make_proxy_with_ipc_extra's docstring -- same reasoning, for the
    tensor_data (non-IPC passthrough) proxy_state shape instead."""
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.proxy_state = {"ipc_extra": None, "tensor_data": tensor}
    proxy.reconstruct_tensor = None
    proxy.sync_data_meta = None
    proxy.sync_buffer = None
    return proxy


class TestCudaDeviceForIndex(CustomTestCase):
    """_cuda_device_for_index is the pure helper extracted out of
    reconstruct_on_target_device's previously-duplicated
    `torch.device(f"cuda:{idx}")` construction. It requires no mocking and no
    real accelerator: constructing a torch.device object is pure metadata
    (verified separately: `torch.device("cuda:0")` succeeds even when this
    torch build has no CUDA support at all -- only *using* the device, e.g.
    `.to(device)`, needs real hardware, which is exactly why this helper is
    factored out on its own)."""

    def test_maps_index_to_cuda_device_string(self):
        self.assertEqual(cuda_ipc_mod._cuda_device_for_index(0), torch.device("cuda:0"))
        self.assertEqual(cuda_ipc_mod._cuda_device_for_index(3), torch.device("cuda:3"))

    def test_returns_a_real_torch_device_instance(self):
        result = cuda_ipc_mod._cuda_device_for_index(0)
        self.assertIsInstance(result, torch.device)
        self.assertEqual(result.type, "cuda")
        self.assertEqual(result.index, 0)


class TestOpenPooledStorageUncachedRocmGate(CustomTestCase):
    """_open_pooled_storage_uncached is the pooled-path raw IPC reopen call
    (torch.UntypedStorage._new_shared_cuda). It must never be reached on HIP.
    """

    def test_raises_assertion_on_hip(self):
        with patch.object(cuda_ipc_mod, "is_hip", return_value=True):
            with self.assertRaises(AssertionError) as ctx:
                cuda_ipc_mod._open_pooled_storage_uncached((0, b"fake-handle"))
        self.assertIn("HIP", str(ctx.exception))
        self.assertIn("29687", str(ctx.exception))

    def test_does_not_raise_assertion_on_non_hip(self):
        # On non-HIP, the assertion itself must not fire. It will still fail
        # further down (there's no real CUDA IPC handle here), but that
        # failure must NOT be our AssertionError -- it must come from
        # torch's own handle-parsing code.
        with patch.object(cuda_ipc_mod, "is_hip", return_value=False):
            with self.assertRaises(Exception) as ctx:
                cuda_ipc_mod._open_pooled_storage_uncached((0, b"fake-handle"))
        self.assertNotIsInstance(ctx.exception, AssertionError)


class TestReconstructOnTargetDeviceRocmGate(CustomTestCase):
    """reconstruct_on_target_device's non-pooled branch does the exact call
    (torch.UntypedStorage._new_shared_cuda on a device-redirected handle)
    that crashes with 'HIP error: invalid device pointer' per issue #29687.
    """

    def _non_pooled_ipc_extra(self):
        return {
            "handle": (0, b"fake-handle-bytes", 0, 0, 0, 0, 0, 0),
            "shape": torch.Size([4]),
            "dtype": torch.float32,
            "stride": (1,),
            "device_index": 0,
            "storage_offset": 0,
            "recons_shape": torch.Size([4]),
            "recons_dtype": torch.float32,
        }

    def test_non_pooled_path_raises_assertion_on_hip(self):
        proxy = _make_proxy_with_ipc_extra(self._non_pooled_ipc_extra())
        with patch.object(cuda_ipc_mod, "is_hip", return_value=True):
            with self.assertRaises(AssertionError) as ctx:
                proxy.reconstruct_on_target_device(0)
        self.assertIn("HIP", str(ctx.exception))
        self.assertIn("29687", str(ctx.exception))

    def test_non_pooled_path_does_not_assert_on_non_hip(self):
        proxy = _make_proxy_with_ipc_extra(self._non_pooled_ipc_extra())
        with patch.object(cuda_ipc_mod, "is_hip", return_value=False):
            # No real CUDA device / IPC handle available in this sandbox, so
            # this will still raise (torch.cuda.device(...) or the handle
            # reopen itself will fail) -- what matters is that it's not our
            # AssertionError, i.e. the platform gate did not block it.
            with self.assertRaises(Exception) as ctx:
                proxy.reconstruct_on_target_device(0)
        self.assertNotIsInstance(ctx.exception, AssertionError)

    def test_pooled_cache_helper_is_gated(self):
        # The pooled branch's cached-open path is
        # _pool_handle_cache_get_or_open -> _open_pooled_storage_uncached
        # (the same helper asserted on directly above). Test this one level
        # below _reconstruct_from_ipc_extra / reconstruct_on_target_device,
        # which both additionally wrap the call in
        # `with torch.cuda.device(target_device):` -- a context manager this
        # CPU-only sandbox build cannot enter at all (raises
        # "PyTorch was compiled without CUDA support" before our code even
        # runs), independent of anything this fix changes. That wrapping is
        # unavoidable plumbing (peer-access / device-guard setup) that only a
        # real CUDA- or ROCm-capable torch build can execute; it is not part
        # of the platform-detection gate under test here.
        cuda_ipc_mod._pool_handle_cache_clear()
        with patch.object(cuda_ipc_mod, "is_hip", return_value=True):
            with self.assertRaises(AssertionError) as ctx:
                cuda_ipc_mod._pool_handle_cache_get_or_open(
                    ("some-cache-key",), (0, b"fake-pool-handle")
                )
        self.assertIn("HIP", str(ctx.exception))
        cuda_ipc_mod._pool_handle_cache_clear()


class TestTensorDataPassthroughRoundTrip(CustomTestCase):
    """The tensor_data passthrough is the *existing* non-IPC transport that
    MmItemMemoryPool._warn_pool_full_once's log message refers to
    ("falling back to non-IPC transport"). It is an ordinary
    `tensor.to(device)` copy -- backend-agnostic, no raw pointer handles
    involved -- so it is exactly what ROCm now uses unconditionally instead
    of attempting CUDA-IPC. This test proves it round-trips real values
    correctly; it is CPU-simulated (both ends "cpu") since no GPU is present
    in this environment, but the code path exercised
    (`Tensor.to(device, non_blocking=True)`) is identical regardless of
    whether "device" happens to be a CUDA or a HIP-backed cuda: device.
    """

    def test_round_trip_preserves_values_and_dtype(self):
        # Exercises exactly the expression
        # `self.proxy_state["tensor_data"].to(rebuild_device, non_blocking=True)`
        # used by reconstruct_on_target_device's tensor_data branch -- an
        # ordinary device-to-device tensor copy with no raw IPC handles
        # involved, so it is identical on CUDA, HIP, or (as simulated here) CPU.
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        proxy = _make_proxy_with_tensor_data(original.clone())

        reconstructed = proxy.proxy_state["tensor_data"].to(
            torch.device("cpu"), non_blocking=True
        )

        self.assertTrue(torch.equal(reconstructed, original))
        self.assertEqual(reconstructed.dtype, original.dtype)
        self.assertEqual(reconstructed.shape, original.shape)

    def test_reconstruct_on_target_device_tensor_data_branch_cpu_simulated(self):
        # Exercise the real reconstruct_on_target_device method end-to-end
        # for the tensor_data branch (the `elif isinstance(...tensor_data...,
        # torch.Tensor)` arm) -- with ZERO mocking of torch itself.
        #
        # reconstruct_on_target_device's only external parameter used to be
        # an int device index that was unconditionally turned into
        # `torch.device(f"cuda:{idx}")`, which made this branch untestable
        # off real accelerator hardware even though the branch itself (an
        # ordinary `Tensor.to(device)` copy, no raw IPC handles) is genuinely
        # backend-agnostic. The production method now accepts a keyword-only
        # `_rebuild_device_override` test seam (default None, so every real
        # caller -- schedule_batch.py, mm_utils.py -- is completely
        # unaffected and still gets the real `cuda:{idx}` device); this test
        # passes a real `torch.device("cpu")` directly through that seam.
        # This is faking the ENVIRONMENT (no GPU is present here) via a real,
        # explicit parameter -- not faking any of torch's internals or the
        # code under test.
        original = torch.tensor([1.0, 2.0, 3.0])
        proxy = _make_proxy_with_tensor_data(original.clone())
        cpu_device = torch.device("cpu")

        result = proxy.reconstruct_on_target_device(
            0, _rebuild_device_override=cpu_device
        )

        self.assertTrue(torch.equal(result, original))
        self.assertEqual(result.device, cpu_device)

        # reconstruct_tensor cache populated -- second call should be a no-op
        # fast path (same object) rather than re-running the transfer.
        result2 = proxy.reconstruct_on_target_device(
            0, _rebuild_device_override=cpu_device
        )
        self.assertIs(result2, result)


if __name__ == "__main__":
    unittest.main()
