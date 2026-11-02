import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.moe_front import NUM_EXPERTS, TOPK, fused_front
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


class TestMoeFront(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_moe_front(self):
        torch.manual_seed(4)
        num_tokens, latent_dim = 1, 128
        hidden = (
            torch.randn(
                num_tokens,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        weight = (
            torch.randn(
                NUM_EXPERTS + latent_dim,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        bias = torch.randn(NUM_EXPERTS, device="cuda")

        weights, ids, routed = fused_front(
            hidden,
            weight,
            bias,
            latent_dim,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        merged = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        ref_weights, ref_ids = moe_fused_gate(
            merged[:, :NUM_EXPERTS],
            bias,
            topk=TOPK,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        order = ids.argsort(dim=-1)
        ref_order = ref_ids.argsort(dim=-1)
        self.assertTrue(
            torch.equal(
                ids.gather(1, order),
                ref_ids.to(torch.int32).gather(1, ref_order),
            )
        )
        torch.testing.assert_close(
            weights.gather(1, order),
            ref_weights.gather(1, ref_order),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(
            torch.equal(
                routed,
                merged[:, NUM_EXPERTS:].to(torch.bfloat16),
            )
        )

    def test_kimi_front_applies_placement_and_batch_padding(self):
        from sglang.srt.eplb.expert_location import ExpertLocationMetadata
        from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.kimi_k3 import KimiK3MoE
        from sglang.srt.runtime_context import get_context, get_exec, get_resources

        torch.manual_seed(42)
        tokens, latent = 4, 128
        hidden = (
            torch.randn(tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / 32
        )
        weight = (
            torch.randn(
                NUM_EXPERTS + latent, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            )
            / 32
        )
        bias = torch.randn(NUM_EXPERTS, device="cuda")
        physical_to_logical = torch.randperm(NUM_EXPERTS, device="cuda").unsqueeze(0)
        logical_to_physical = physical_to_logical.argsort(dim=-1)
        with get_context().override_server_args(
            ep_dispatch_algorithm="static",
            init_expert_location="permuted.json",
            moe_a2a_backend="megamoe",
        ):
            metadata = ExpertLocationMetadata._init_raw(
                ep_size=1,
                physical_to_logical_map=physical_to_logical,
                logical_to_all_physical_map=logical_to_physical.unsqueeze(-1),
                moe_ep_rank=0,
            )
            previous = get_resources().expert_location_metadata
            get_resources().expert_location_metadata = metadata
            try:
                router = TopK(
                    top_k=TOPK,
                    layer_id=0,
                    renormalize=True,
                    use_grouped_topk=True,
                    num_expert_group=1,
                    topk_group=1,
                    correction_bias=bias,
                    scoring_func="sigmoid",
                    output_format=TopKOutputFormat.STANDARD,
                )
                # Exercise the model method and actual merged GEMM/router with
                # only the state that method reads; no checkpoint is needed.
                layer = SimpleNamespace(
                    _ep_front_eligible=True,
                    _eligible_for_fused_front=False,
                    topk=router,
                    gate=SimpleNamespace(e_score_correction_bias=bias),
                    _front_w=weight,
                    moe_hidden_size=latent,
                    layer_idx=0,
                )
                self.assertTrue(KimiK3MoE._routing_contract_ok.func(layer))
                with get_exec().moe.override(ep_dispatch_algorithm="fake"):
                    self.assertFalse(KimiK3MoE._routing_contract_ok.func(layer))
                empty = torch.empty(0, dtype=torch.int32, device="cuda")
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.arange(tokens, device="cuda"),
                    req_pool_indices=empty,
                    seq_lens=empty,
                    out_cache_loc=empty,
                    seq_lens_sum=0,
                )
                batch.num_token_non_padded = torch.tensor(
                    3, device="cuda", dtype=torch.int32
                )
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=batch
                ):
                    actual, routed = KimiK3MoE._ep_front(layer, hidden)
                weights, logical_ids, ref_routed = fused_front(
                    hidden, weight, bias, latent
                )
                expected = logical_to_physical[0, logical_ids.long()].to(
                    logical_ids.dtype
                )
                expected[3:] = -1
                torch.testing.assert_close(actual.topk_ids, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual.topk_weights, weights, rtol=0, atol=0)
                torch.testing.assert_close(routed, ref_routed, rtol=0, atol=0)
            finally:
                get_resources().expert_location_metadata = previous


if __name__ == "__main__":
    unittest.main()
