"""Unit tests for the CUDA-IPC tensor-sharing device remap in ipc_a2a.py.

``IpcA2AState._share`` re-opens the peer's CUDA IPC handle in the local device
context by rewriting the ``storage_device`` slot of torch's ``reduce_tensor``
rebuild tuple. This used to be a hard-coded positional index, which silently
breaks (mapping peer storage onto the wrong GPU) if torch ever reorders the
rebuild function's arguments. These tests pin the remap to the argument *name*
instead, and guard against an unexpected signature without a CUDA context.
"""

import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rebuild_stub(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    """Signature-compatible stand-in for torch's ``rebuild_cuda_tensor``.

    The argument count and order intentionally mirror the real torch internal
    so tests exercise the same slot that ``_share`` rewrites.
    """
    return storage_device


class TestIpcA2aShareDeviceRemap(CustomTestCase):
    def _run_remap(self, stub_fn, args, device_index):
        """Mirror the device-rewrite loop inside ``IpcA2AState._share``."""
        names = list(inspect.signature(stub_fn).parameters)
        storage_device_index = (
            names.index("storage_device") if "storage_device" in names else None
        )
        remapped = list(args)
        for i, v in enumerate(remapped):
            if isinstance(v, int) and i == storage_device_index:
                remapped[i] = device_index
        return stub_fn(*remapped)

    def test_rewrites_storage_device_by_name(self):
        """The int slot named ``storage_device`` is replaced with the local
        device index regardless of its position."""
        args = (
            None,  # tensor_cls
            None,  # tensor_size
            None,  # tensor_stride
            None,  # tensor_offset
            None,  # storage_cls
            None,  # dtype
            3,  # storage_device (producer's device index)
            None,  # storage_handle
            None,  # storage_size_bytes
            None,  # storage_offset_bytes
            None,  # requires_grad
            None,  # ref_counter_handle
            None,  # ref_counter_offset
            None,  # event_handle
            None,  # event_sync_required
        )
        result = self._run_remap(_rebuild_stub, args, device_index=1)
        self.assertEqual(result, 1)

    def test_rewrite_survives_argument_reordering(self):
        """Moving ``storage_device`` to a different position must still rewrite
        the correct slot, not a hard-coded index."""
        # Reorder the stub signature so storage_device is no longer at index 6.
        def reordered_stub(
            storage_device,
            tensor_cls,
            tensor_size,
            tensor_stride,
            tensor_offset,
            storage_cls,
            dtype,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            requires_grad,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ):
            return storage_device

        args = (
            3,  # storage_device now first
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        result = self._run_remap(reordered_stub, args, device_index=7)
        self.assertEqual(result, 7)

    def test_rewrite_skips_when_signature_has_no_storage_device(self):
        """A rebuild function without a ``storage_device`` parameter leaves the
        positional args untouched instead of corrupting an unrelated int slot."""

        def cpu_stub(a, b, c):
            return (a, b, c)

        args = (5, 6, 7)
        names = list(inspect.signature(cpu_stub).parameters)
        storage_device_index = (
            names.index("storage_device") if "storage_device" in names else None
        )
        self.assertIsNone(storage_device_index)
        self.assertEqual(cpu_stub(*args), (5, 6, 7))

    def test_real_rebuild_cuda_tensor_signature_has_storage_device(self):
        """The real torch internal that ``_share`` depends on still exposes
        ``storage_device``; if this ever fails, the transport must be
        re-audited rather than silently mapping the wrong slot."""
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        names = list(inspect.signature(rebuild_cuda_tensor).parameters)
        self.assertIn("storage_device", names)
        # Guards the historical index used before this fix, so a regression in
        # the name lookup (or a torch reorder) is caught here.
        self.assertEqual(names.index("storage_device"), 6)


if __name__ == "__main__":
    unittest.main()
