import unittest

import torch

from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_gpu,
)
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl as native_biased_grouped_topk,
)
from sglang.srt.layers.moe.topk import biased_topk_impl as native_biased_topk
from sglang.srt.layers.moe.topk import (
    biased_topk_xpu,
)
from sglang.srt.layers.moe.topk import grouped_topk_gpu as native_grouped_topk
from sglang.srt.layers.moe.topk import (
    grouped_topk_xpu,
)
from sglang.srt.runtime_context import get_context
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

    def test_biased_topk(self):
        # DeepSeek-V4 style routing shape
        E_num_list = [256, 384]
        topk_value = 6
        gating_dtype = torch.float32
        bias_dtype = torch.float32
        renormalize = True
        scoring_func = "sqrtsoftplus"
        routed_scaling_factor = 2.5

        torch.manual_seed(1024)
        device = torch.device("xpu")

        bs = [1]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]
        num_fused_shared_experts_list = [0, 1]

        for E_num in E_num_list:
            for M in num_tokens:
                for num_fused_shared_experts in num_fused_shared_experts_list:

                    topk_routed = topk_value - num_fused_shared_experts
                    hidden_states = torch.randn(
                        M, 100, dtype=gating_dtype, device=device
                    )
                    gating_output = torch.randn(
                        M, E_num, dtype=gating_dtype, device=device
                    )
                    correction_bias = torch.randn(
                        E_num, dtype=bias_dtype, device=device
                    )

                    ref_topk_weights, ref_topk_ids = native_biased_topk(
                        hidden_states,
                        gating_output,
                        correction_bias,
                        topk_value,
                        renormalize,
                        scoring_func,
                        num_fused_shared_experts,
                        routed_scaling_factor,
                        apply_routed_scaling_factor_on_output=True,
                    )

                    # fused version
                    topk_weights, topk_ids = biased_topk_xpu(
                        hidden_states,
                        gating_output,
                        correction_bias,
                        topk_value,
                        renormalize,
                        scoring_func,
                        num_fused_shared_experts,
                        routed_scaling_factor,
                        apply_routed_scaling_factor_on_output=True,
                    )

                    torch.testing.assert_close(
                        _scatter_by_expert(
                            topk_weights[:, :topk_routed],
                            topk_ids[:, :topk_routed],
                            E_num,
                        ),
                        _scatter_by_expert(
                            ref_topk_weights[:, :topk_routed],
                            ref_topk_ids[:, :topk_routed],
                            E_num,
                        ),
                    )

    def test_hash_topk(self):
        """Guard the XPU fused hash-topk path against math/ID drift from torch."""
        torch.manual_seed(1024)
        device = torch.device("xpu")

        E_num_list = [256, 384]
        topk = 6
        vocab_size = 128
        dtype = torch.float32

        bs = [1]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]
        num_fused_shared_experts_list = [0, 1]

        with get_context().override_server_args(enable_waterfill=False):
            for E_num in E_num_list:
                for M in num_tokens:
                    for num_fused_shared_experts in num_fused_shared_experts_list:
                        hidden_states = torch.randn(
                            M, 1, dtype=torch.float32, device=device
                        )
                        router_logits = torch.randn(
                            M, E_num, dtype=dtype, device=device
                        )
                        input_ids = torch.randint(
                            low=0,
                            high=vocab_size,
                            size=(M,),
                            dtype=torch.int64,
                            device=device,
                        )

                        hash_topk = HashTopK(
                            topk=topk,
                            num_experts=E_num,
                            num_fused_shared_experts=num_fused_shared_experts,
                            vocab_size=vocab_size,
                            scoring_func="sqrtsoftplus",
                            routed_scaling_factor=2.5,
                        ).to(device)
                        topk_routed = hash_topk.tid2eid.shape[1]
                        with torch.no_grad():
                            hash_topk.tid2eid.copy_(
                                torch.randint(
                                    low=0,
                                    high=E_num,
                                    size=(vocab_size, topk_routed),
                                    dtype=torch.int32,
                                    device=device,
                                )
                            )

                            ref_topk_weights, ref_topk_ids = hash_topk._forward_torch(
                                router_logits, input_ids
                            )

                            output = hash_topk(
                                hidden_states=hidden_states,
                                router_logits=router_logits,
                                input_ids=input_ids,
                            )

                        torch.testing.assert_close(output.topk_ids, ref_topk_ids)
                        torch.testing.assert_close(
                            output.topk_weights, ref_topk_weights
                        )


if __name__ == "__main__":
    unittest.main()
