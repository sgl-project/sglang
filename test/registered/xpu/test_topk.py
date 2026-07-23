import unittest

import torch

from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_gpu,
)
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl as native_biased_grouped_topk,
)
from sglang.srt.layers.moe.topk import grouped_topk_gpu as native_grouped_topk
from sglang.srt.layers.moe.topk import (
    grouped_topk_xpu,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=5, suite="stage-b-test-1-gpu-xpu")


def _scatter_by_expert(
    weights: torch.Tensor, indices: torch.Tensor, num_columns: int
) -> torch.Tensor:
    """Scatter (weight, id) pairs into a dense ``[M, num_columns]`` tensor.

    Makes the comparison independent of the per-row slot order, so the test does
    not depend on how ties between equal scores are broken.
    """
    dense = torch.zeros(
        (weights.shape[0], num_columns), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, indices.long(), weights.float())
    return dense

def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"

# Nemotron-3 uses biased_grouped_topk
class TestBiasedGroupedTopK(CustomTestCase):
    def _run_single_test(
        self,
        M,
        E,
        G,
        topk,
        topk_group,
        renormalize,
        gating_dtype,
        bias_dtype,
        routed_scaling_factor,
    ):
        torch.manual_seed(1024)
        device = torch.device("xpu")

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=torch.bfloat16, device=device)
        gating_output = torch.randn(M, E, dtype=gating_dtype, device=device)
        correction_bias = torch.randn(E, dtype=bias_dtype, device=device)

        ref_topk_weights, ref_topk_ids = native_biased_grouped_topk(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        # fused version
        topk_weights, topk_ids = biased_grouped_topk_gpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            routed_scaling_factor,
            None,
        )

        res = torch.zeros(M, E, dtype=torch.float, device=device)
        ref = torch.zeros(M, E, dtype=torch.float, device=device)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    # Nemotron-3-Nano-30B-A3B uses fast biased_grouped_topk with num_expert_group = 1 and topk_group = 1
    def test_fast_biased_grouped_topk(self):
        # The test config is also from this nemotron model.
        E_num = 128
        num_expert_group = 1
        topk_value = 6
        topk_group = 1
        gating_dtype = torch.bfloat16
        bias_dtype = torch.float32
        renormalize = True
        routed_scaling_factor = 2.5

        bs = [1, 2, 4, 8]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]

        for M in num_tokens:
            self._run_single_test(
                M,
                E_num,
                num_expert_group,
                topk_value,
                topk_group,
                renormalize,
                gating_dtype,
                bias_dtype,
                routed_scaling_factor,
            )

    def test_biased_grouped_topk(self):
        # DeepSeek-V3 style grouped routing shape
        E_num = 256
        num_expert_group = 8
        topk_value = 8
        topk_group = 4
        gating_dtype = torch.bfloat16
        bias_dtype = torch.float32
        renormalize = True
        routed_scaling_factor = 2.5

        torch.manual_seed(1024)
        device = torch.device("xpu")

        bs = [1, 2, 4, 8]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]
        num_fused_shared_experts_list = [0, 1]

        for M in num_tokens:
            for num_fused_shared_experts in num_fused_shared_experts_list:

                topk_routed = topk_value - num_fused_shared_experts
                hidden_states = torch.randn(M, 100, dtype=torch.bfloat16, device=device)
                gating_output = torch.randn(M, E_num, dtype=gating_dtype, device=device)
                correction_bias = torch.randn(E_num, dtype=bias_dtype, device=device)

                ref_topk_weights, ref_topk_ids = native_biased_grouped_topk(
                    hidden_states.float(),
                    gating_output.float(),
                    correction_bias,
                    topk_value,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    num_fused_shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                )

                # fused version
                topk_weights, topk_ids = biased_grouped_topk_gpu(
                    hidden_states,
                    gating_output,
                    correction_bias,
                    topk_value,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    num_fused_shared_experts,
                    routed_scaling_factor,
                )

                torch.testing.assert_close(
                    _scatter_by_expert(
                        topk_weights[:, :topk_routed], topk_ids[:, :topk_routed], E_num
                    ),
                    _scatter_by_expert(
                        ref_topk_weights[:, :topk_routed],
                        ref_topk_ids[:, :topk_routed],
                        E_num,
                    ),
                )

    def test_grouped_topk(self):
        # DeepSeek-V3 style grouped routing shape
        E_num = 256
        num_expert_group = 8
        topk_value = 8
        topk_group = 4
        gating_dtype = torch.bfloat16
        renormalize = True
        routed_scaling_factor = 2.5

        torch.manual_seed(1024)
        device = torch.device("xpu")

        bs = [1]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]
        num_fused_shared_experts_list = [0, 1]

        for M in num_tokens:
            for num_fused_shared_experts in num_fused_shared_experts_list:

                topk_routed = topk_value - num_fused_shared_experts
                hidden_states = torch.randn(M, 100, dtype=torch.bfloat16, device=device)
                gating_output = torch.randn(M, E_num, dtype=gating_dtype, device=device)

                ref_topk_weights, ref_topk_ids = native_grouped_topk(
                    hidden_states.float(),
                    gating_output.float(),
                    topk_value,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    num_fused_shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                )

                # fused version
                topk_weights, topk_ids = grouped_topk_xpu(
                    hidden_states,
                    gating_output,
                    topk_value,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    num_fused_shared_experts,
                    routed_scaling_factor,
                )

                assert_equal(
                    gating_output,
                    ref_topk_ids[:, :topk_routed],
                    topk_ids[:, :topk_routed],
                    bs=len(bs),
                    k=topk_value,
                    seq_len=seq_len,
                )


if __name__ == "__main__":
    unittest.main()
