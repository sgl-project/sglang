import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import fused_moe_dispatch_index
from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    PermuteMethodPool,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    _max_tokens_per_scattered_source,
    _scattered_source_token_counts,
    _workspace_size_for_namespace,
)
from sglang.srt.layers.quantization import fp8  # noqa: F401
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestFlashinferA2AWideEPPlumbing(CustomTestCase):
    def test_runner_paths_are_registered(self):
        self.assertIn(("flashinfer", "flashinfer_trtllm"), FusedOpPool._fused_funcs)
        self.assertIn(
            ("flashinfer", "flashinfer_trtllm_routed"), FusedOpPool._fused_funcs
        )
        self.assertIn(
            ("flashinfer", "deep_gemm"), PermuteMethodPool._pre_permute_methods
        )
        self.assertIn(
            ("deep_gemm", "flashinfer"), PermuteMethodPool._post_permute_methods
        )

    def test_dp4_tp4_uses_physical_source_rank_geometry(self):
        self.assertEqual(_max_tokens_per_scattered_source([2048] * 4, 4), 512)
        self.assertEqual(_max_tokens_per_scattered_source([1, 0, 0, 0], 4), 1)
        self.assertEqual(_max_tokens_per_scattered_source([7, 3, 2, 1], 4), 2)
        self.assertEqual(_max_tokens_per_scattered_source([512] * 16, 1), 512)

    def test_target_and_draft_decode_use_distinct_workspaces(self):
        sizes = {
            _workspace_size_for_namespace(4096, speculative=speculative)
            for speculative in (False, True)
        }
        self.assertEqual(sizes, {4096, 4224})

    def test_prefill_ag_expands_dp_counts_to_physical_source_ranks(self):
        self.assertEqual(
            _scattered_source_token_counts([7, 3], 4),
            [2, 2, 2, 1, 1, 1, 1, 0],
        )
        self.assertEqual(_scattered_source_token_counts([4] * 16, 1), [4] * 16)

    def test_deepgemm_dispatch_marks_empty_expert_lanes_invalid(self):
        topk_ids = torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device="cuda")
        masked_m, src2dst = fused_moe_dispatch_index(
            topk_ids, num_local_experts=2, m_max=4
        )

        torch.testing.assert_close(
            masked_m, torch.tensor([1, 1], dtype=torch.int32, device="cuda")
        )
        torch.testing.assert_close(
            src2dst,
            torch.tensor([-1, 0, -1, 4], dtype=torch.int32, device="cuda"),
        )

    def test_global_expert_mapping_is_fused_into_dispatch_index(self):
        global_ids = torch.tensor(
            [-1, 15, 16, 17, 31, 32], dtype=torch.int32, device="cuda"
        )
        masked_m, src2dst = fused_moe_dispatch_index(
            global_ids, num_local_experts=2, m_max=4, expert_start=16
        )

        torch.testing.assert_close(
            masked_m, torch.tensor([1, 1], dtype=torch.int32, device="cuda")
        )
        torch.testing.assert_close(
            src2dst,
            torch.tensor([-1, -1, 0, 4, -1, -1], dtype=torch.int32, device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
