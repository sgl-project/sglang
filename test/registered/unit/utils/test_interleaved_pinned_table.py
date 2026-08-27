"""The PLE offload table lives in host RAM and is read by a GPU kernel.

Two properties have to hold for that to work, and both broke in production:

  1. The pages must not all land on the NUMA node local to the GPU. A
     multi-GiB node-local pinned allocation can exhaust a small node, which
     shows up as an OOM kill, or as the NVIDIA driver failing to allocate the
     GPU page tables that map the pinning. A partially mapped pinning then
     faults asynchronously (Xid 31, MMU FAULT_PDE at a host address) and
     reaches Python as "CUDA error: an illegal memory access was encountered"
     at whatever unrelated call happens to synchronize next.
  2. The raw host pointer must stay readable from a device kernel, so the
     mapping has to outlive every view of it.

Neither needs a model, a checkpoint, or tensor parallelism to exercise.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.numa_utils import (
    InterleavedPinnedBuffer,
    allocate_interleaved_pinned_table,
    numa_memory_nodes,
    numa_page_counts,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

# Large enough that the interleave is visible over page-granularity noise,
# small enough to allocate on a busy host.
_TABLE_ROWS = 1 << 16
_TABLE_DIM = 160


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestInterleavedPinnedTable(CustomTestCase):
    def test_ple_call_site_interleaves_only_exact_sm120(self):
        from sglang.srt.models import qwen4_exp

        cases = ((False, False, False), (True, False, True), (True, True, False))
        for sm120_supported, sm121, expected in cases:
            with (
                self.subTest(
                    sm120_supported=sm120_supported,
                    sm121=sm121,
                ),
                patch.object(
                    qwen4_exp,
                    "is_sm120_supported",
                    return_value=sm120_supported,
                ),
                patch.object(qwen4_exp, "is_sm121", return_value=sm121),
            ):
                self.assertIs(qwen4_exp._should_interleave_ple_table(), expected)

    def test_pages_span_multiple_numa_nodes(self):
        nodes = numa_memory_nodes()
        if len(nodes) < 2:
            self.skipTest(f"single NUMA node host: {nodes}")
        tensor, buffer = allocate_interleaved_pinned_table(
            (_TABLE_ROWS, _TABLE_DIM), torch.float8_e4m3fn
        )
        if buffer is None:
            del tensor
            self.skipTest("NUMA interleaving is not active on this runner")
        try:
            counts = numa_page_counts(buffer.ptr, buffer.nbytes)
            populated = set(counts).intersection(nodes)
            self.assertGreaterEqual(
                len(populated),
                2,
                f"pinned pages were not interleaved: {counts}",
            )
        finally:
            del tensor
            if buffer is not None:
                buffer.release()

    def test_device_kernel_reads_the_host_table(self):
        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        tensor, buffer = allocate_interleaved_pinned_table(
            (_TABLE_ROWS, _TABLE_DIM), torch.float8_e4m3fn
        )
        try:
            self.assertTrue(tensor.is_pinned(), "table is not page-locked")
            rows = [0, _TABLE_ROWS // 2, _TABLE_ROWS - 1]
            reference = torch.arange(_TABLE_DIM, dtype=torch.bfloat16)
            for offset, row in enumerate(rows):
                tensor[row] = (reference + offset).to(torch.float8_e4m3fn)

            ids = torch.tensor(rows, dtype=torch.long, device="cuda")
            out = torch.empty(
                (len(rows), _TABLE_DIM), dtype=torch.bfloat16, device="cuda"
            )
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                tensor.data_ptr(),
                ids,
                out,
                embedding_dim=_TABLE_DIM,
                tp_vocab_start=0,
                tp_vocab_end=_TABLE_ROWS,
                is_fp8=True,
                BLOCK_D=256,
            )
            torch.cuda.synchronize()
            for offset in range(len(rows)):
                expected = (
                    (reference + offset).to(torch.float8_e4m3fn).to(torch.bfloat16)
                )
                self.assertTrue(torch.equal(out[offset].cpu(), expected))
        finally:
            del tensor
            if buffer is not None:
                buffer.release()

    def test_disabled_flag_uses_node_local_pinning(self):
        with envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(False):
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn
            )
        try:
            self.assertIsNone(buffer)
            self.assertTrue(tensor.is_pinned())
        finally:
            del tensor

    def test_interleave_request_uses_all_available_nodes(self):
        sentinel = object()
        self.assertNotIn(
            "is_sm120_supported", allocate_interleaved_pinned_table.__code__.co_names
        )
        with (
            envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(True),
            patch("sglang.srt.utils.numa_utils.numa_memory_nodes", return_value=[0, 1]),
            patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True),
            patch("sglang.srt.utils.numa_utils.InterleavedPinnedBuffer") as interleaved,
            patch(
                "sglang.srt.utils.numa_utils.numa_page_counts",
                return_value={0: 1, 1: 1},
            ),
        ):
            interleaved.return_value = MagicMock(
                ptr=1234, nbytes=4096, as_tensor=MagicMock(return_value=sentinel)
            )
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn, interleave=True
            )

        self.assertIs(tensor, sentinel)
        self.assertIs(buffer, interleaved.return_value)
        interleaved.assert_called_once()

    def test_interleaved_failure_falls_back_to_node_local_pinning(self):
        sentinel = object()
        with (
            envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(True),
            patch("sglang.srt.utils.numa_utils.numa_memory_nodes", return_value=[0, 1]),
            patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True),
            patch(
                "sglang.srt.utils.numa_utils.InterleavedPinnedBuffer",
                side_effect=RuntimeError("cudaHostRegister failed"),
            ),
            patch(
                "sglang.srt.utils.numa_utils.torch.empty", return_value=sentinel
            ) as plain,
        ):
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn, interleave=True
            )

        self.assertIs(tensor, sentinel)
        self.assertIsNone(buffer)
        plain.assert_called_once()

    def test_permission_denied_uses_node_local_pinning(self):
        sentinel = object()
        with (
            envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(True),
            patch("sglang.srt.utils.numa_utils.numa_memory_nodes", return_value=[0, 1]),
            patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=False),
            patch("sglang.srt.utils.numa_utils.InterleavedPinnedBuffer") as interleaved,
            patch("sglang.srt.utils.numa_utils.torch.empty", return_value=sentinel),
        ):
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn, interleave=True
            )

        self.assertIs(tensor, sentinel)
        self.assertIsNone(buffer)
        interleaved.assert_not_called()

    def test_non_sm120_policy_uses_node_local_pinning(self):
        sentinel = object()
        with (
            envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(True),
            patch("sglang.srt.utils.numa_utils.InterleavedPinnedBuffer") as interleaved,
            patch("sglang.srt.utils.numa_utils.torch.empty", return_value=sentinel),
        ):
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn, interleave=False
            )

        self.assertIs(tensor, sentinel)
        self.assertIsNone(buffer)
        interleaved.assert_not_called()

    def test_single_node_placement_is_not_logged_as_interleaved(self):
        sentinel = object()
        buffer = MagicMock(ptr=1234, nbytes=4096)
        buffer.as_tensor.return_value = sentinel
        with (
            envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(True),
            patch("sglang.srt.utils.numa_utils.numa_memory_nodes", return_value=[0, 1]),
            patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True),
            patch(
                "sglang.srt.utils.numa_utils.InterleavedPinnedBuffer",
                return_value=buffer,
            ),
            patch("sglang.srt.utils.numa_utils.numa_page_counts", return_value={0: 1}),
            patch("sglang.srt.utils.numa_utils._handle_numa_bind_failure") as failure,
            patch("sglang.srt.utils.numa_utils.logger.info") as info,
        ):
            tensor, owner = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn
            )

        self.assertIs(tensor, sentinel)
        self.assertIs(owner, buffer)
        failure.assert_called_once()
        info.assert_not_called()

    def test_destructor_uses_the_synchronized_release_during_runtime(self):
        buffer = InterleavedPinnedBuffer.__new__(InterleavedPinnedBuffer)
        buffer.ptr = 1234
        buffer.nbytes = 4096
        buffer._registered = True
        buffer._libc = MagicMock()
        cudart = MagicMock()
        cudart.cudaHostUnregister.return_value = 0
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "cudart", return_value=cudart),
            patch("sglang.srt.utils.numa_utils.sys.is_finalizing", return_value=False),
        ):
            buffer.__del__()

        synchronize.assert_called_once_with()
        cudart.cudaHostUnregister.assert_called_once_with(1234)

    def test_destructor_leaves_mapping_to_os_during_interpreter_teardown(self):
        buffer = InterleavedPinnedBuffer.__new__(InterleavedPinnedBuffer)
        buffer.ptr = 1234
        buffer.nbytes = 4096
        buffer._registered = True
        buffer._libc = MagicMock()
        with (
            patch("sglang.srt.utils.numa_utils.sys.is_finalizing", return_value=True),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "cudart") as cudart,
        ):
            buffer.__del__()

        synchronize.assert_not_called()
        cudart.assert_not_called()
        buffer._libc.munmap.assert_not_called()
        # The pointer is synthetic. Disarm the later Python test-object cleanup
        # after verifying that the production destructor left it untouched.
        buffer.ptr = None

    def test_explicit_release_synchronizes_before_unregister(self):
        buffer = InterleavedPinnedBuffer.__new__(InterleavedPinnedBuffer)
        buffer.ptr = 1234
        buffer.nbytes = 4096
        buffer._registered = True
        buffer._libc = MagicMock()
        cudart = MagicMock()
        cudart.cudaHostUnregister.return_value = 0
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            buffer.release()

        synchronize.assert_called_once_with()
        cudart.cudaHostUnregister.assert_called_once_with(1234)


if __name__ == "__main__":
    unittest.main()
