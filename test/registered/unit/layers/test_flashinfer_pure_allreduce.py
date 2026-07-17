import unittest

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-b200")


class _FakeWorkspace:
    def __init__(self, world_size):
        self.world_size = world_size

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kAllReduce = object()
        kARResidualRMSNorm = object()

    def __init__(self, world_size):
        self._world_size = world_size

    def allreduce_fusion(self, *, input, workspace, pattern, **_kwargs):
        if pattern is self.AllReduceFusionPattern.kAllReduce:
            return input * workspace.world_size
        raise ValueError(f"Unexpected pattern: {pattern}")


class TestFlashInferPureAllReduce(unittest.TestCase):
    def _make_manager(self, world_size):
        manager = fusion.FlashInferWorkspaceManager()
        manager.workspace = _FakeWorkspace(world_size)
        manager.initialized = True
        manager.max_token_num = 2048
        manager.hidden_dim = 4096
        return manager

    def test_allreduce_output_equals_input_times_world_size(self):
        world_size = 4
        fake_comm = _FakeFlashInferComm(world_size)
        manager = self._make_manager(world_size)

        original_comm = fusion._flashinfer_comm
        original_manager = fusion._attn_tp_workspace_manager
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._attn_tp_workspace_manager = manager
            fusion._flashinfer_allreduce_unavailable = False

            if not torch.cuda.is_available():
                self.skipTest("CUDA required for flashinfer custom op")
            device = torch.device("cuda")
            input_ = torch.randn(8, 16, dtype=torch.bfloat16, device=device)
            expected = input_ * world_size

            with get_parallel().override(attn_tp_size=world_size):
                result = fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)

            self.assertIsNotNone(result)
            torch.testing.assert_close(result, expected)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._attn_tp_workspace_manager = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_shape_guard_returns_none_for_non_2d(self):
        world_size = 4
        fake_comm = _FakeFlashInferComm(world_size)
        manager = self._make_manager(world_size)

        original_comm = fusion._flashinfer_comm
        original_manager = fusion._attn_tp_workspace_manager
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._attn_tp_workspace_manager = manager
            fusion._flashinfer_allreduce_unavailable = False

            device = torch.device("cpu")
            input_1d = torch.randn(16, device=device)
            input_3d = torch.randn(2, 8, 16, device=device)

            self.assertIsNone(
                fusion.flashinfer_allreduce(input_1d, use_attn_tp_group=True)
            )
            self.assertIsNone(
                fusion.flashinfer_allreduce(input_3d, use_attn_tp_group=True)
            )
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._attn_tp_workspace_manager = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_shape_guard_returns_none_for_non_contiguous(self):
        world_size = 4
        fake_comm = _FakeFlashInferComm(world_size)
        manager = self._make_manager(world_size)

        original_comm = fusion._flashinfer_comm
        original_manager = fusion._attn_tp_workspace_manager
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._attn_tp_workspace_manager = manager
            fusion._flashinfer_allreduce_unavailable = False

            # A transposed tensor is non-contiguous
            base = torch.randn(16, 8)
            non_contiguous = base.t()
            self.assertFalse(non_contiguous.is_contiguous())

            self.assertIsNone(
                fusion.flashinfer_allreduce(non_contiguous, use_attn_tp_group=True)
            )
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._attn_tp_workspace_manager = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_returns_none_when_unavailable(self):
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = True
            input_ = torch.randn(8, 16)
            self.assertIsNone(
                fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)
            )
        finally:
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_returns_none_when_workspace_uninitialized(self):
        world_size = 4
        fake_comm = _FakeFlashInferComm(world_size)
        manager = fusion.FlashInferWorkspaceManager()  # not initialized

        original_comm = fusion._flashinfer_comm
        original_manager = fusion._attn_tp_workspace_manager
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._attn_tp_workspace_manager = manager
            fusion._flashinfer_allreduce_unavailable = False

            input_ = torch.randn(8, 16)
            with get_parallel().override(attn_tp_size=world_size):
                result = fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)
            self.assertIsNone(result)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._attn_tp_workspace_manager = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable


if __name__ == "__main__":
    unittest.main()
