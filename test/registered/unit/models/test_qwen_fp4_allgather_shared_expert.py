"""The FP4 all-gather path must compute each DP-local shared expert in full."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    post_experts_output_is_complete,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.models import qwen2_moe
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RoutedExperts(torch.nn.Module):
    """Only the shared expert is exercised; no FP4 kernels or process group."""

    def __init__(self, **kwargs):
        super().__init__()
        self.runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        self.routing_method_type = kwargs["routing_method_type"]


class TestQwenFp4AllgatherSharedExpert(CustomTestCase):
    def setUp(self):
        super().setUp()
        context = get_context().override_server_args()
        context.install()
        self.addCleanup(context.restore)
        generator = torch.Generator().manual_seed(42)
        self.gate = torch.randn(8, 4, generator=generator)
        self.up = torch.randn(8, 4, generator=generator)
        self.down = torch.randn(4, 8, generator=generator)
        self.shared_gate = torch.randn(1, 4, generator=generator)

    @contextmanager
    def _block(self, rank, *, disable_fp4_allgather=False, quantization="modelopt_fp4"):
        config = SimpleNamespace(
            model_type="qwen4_exp_text",
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            hidden_size=4,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            hidden_act="silu",
        )
        with (
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.NONE,
                runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
                quantization=quantization,
                disable_fp4_allgather=disable_fp4_allgather,
            ),
            get_flags().dp.override(enabled=True),
            get_parallel().override(
                tp_group=None,
                tp_size=2,
                tp_rank=rank,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=2,
                attn_dp_rank=rank,
                attn_cp_size=1,
                moe_ep_size=2,
                moe_ep_rank=rank,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_dp_size=1,
                dwdp_size=1,
            ),
            patch.object(qwen2_moe, "get_moe_impl_class", return_value=_RoutedExperts),
        ):
            # Shared experts excluded from NVFP4 quantization use these ordinary
            # linears. Load full checkpoint tensors through the real shard loaders.
            block = qwen2_moe.Qwen2MoeSparseMoeBlock(0, config)
            shared = block.shared_expert
            shared.gate_up_proj.weight.weight_loader(
                shared.gate_up_proj.weight, self.gate, 0
            )
            shared.gate_up_proj.weight.weight_loader(
                shared.gate_up_proj.weight, self.up, 1
            )
            shared.down_proj.weight.weight_loader(shared.down_proj.weight, self.down)
            with torch.no_grad():
                block.shared_expert_gate.weight.copy_(self.shared_gate)
            yield block

    def _reference(self, x):
        return F.linear(
            F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down
        ) * torch.sigmoid(F.linear(x, self.shared_gate))

    def test_dp_local_outputs_match_the_complete_shared_expert(self):
        # Different tokens/counts on each DP rank, including the idle-rank case.
        for rank, rows in ((0, 1), (1, 3), (1, 0)):
            with self.subTest(rank=rank, rows=rows), self._block(rank) as block:
                self.assertTrue(should_use_flashinfer_cutlass_moe_fp4_allgather())
                self.assertTrue(post_experts_output_is_complete(is_tp_path=True))
                x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 10
                actual = block._forward_shared_experts(x)
                torch.testing.assert_close(actual, self._reference(x))
                self.assertEqual(block.shared_expert.gate_up_proj.weight.shape, (16, 4))
                self.assertEqual(block.shared_expert.down_proj.weight.shape, (4, 8))

    def test_disabled_allgather_keeps_tp_shards(self):
        self._check_tp_shards(disable_fp4_allgather=True)

    def test_other_quantization_keeps_tp_shards(self):
        self._check_tp_shards(quantization="modelopt_mixed")

    def _check_tp_shards(self, **kwargs):
        # Without the special route the shared MLP runs on gathered rows, and
        # its two partial contributions must still sum to the full reference.
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        outputs = []
        for rank in range(2):
            with self._block(rank, **kwargs) as block:
                self.assertFalse(should_use_flashinfer_cutlass_moe_fp4_allgather())
                self.assertEqual(block.shared_expert.gate_up_proj.weight.shape, (8, 4))
                self.assertEqual(block.shared_expert.down_proj.weight.shape, (4, 4))
                outputs.append(block._forward_shared_experts(x))
        torch.testing.assert_close(outputs[0] + outputs[1], self._reference(x))


if __name__ == "__main__":
    unittest.main()
