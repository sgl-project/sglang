import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# isort: off
# mxfp4 first: importing flashinfer_trtllm first hits a circular import.
from sglang.srt.layers.quantization import mxfp4
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

# isort: on
from sglang.kernels.ops.moe import kimi_k3_fused_routing
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 3584
NUM_EXPERTS = 896
TOPK = 16
SCALE = 2.827
# EP2 rank 1, so the routing metadata must offset the local expert ids.
FIRST_EXPERT = 448
NUM_LOCAL = 448
INTERMEDIATE = 768
NUM_PADDED_ROWS = 3


def _make_experts(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape: int, lo: int = 0, hi: int = 256) -> torch.Tensor:
        return torch.randint(
            lo, hi, shape, generator=g, device="cuda", dtype=torch.uint8
        )

    # Random MXFP4 bytes: both paths read the same weights, so parity needs no
    # meaningful values. UE8M0 scales of 2^-6 .. 2^-3 keep the GEMMs finite.
    w13 = u8(NUM_LOCAL, 2 * INTERMEDIATE, HIDDEN // 2)
    w2 = u8(NUM_LOCAL, HIDDEN, INTERMEDIATE // 2)
    w13_scale = u8(NUM_LOCAL, 2 * INTERMEDIATE, HIDDEN // 32, lo=121, hi=125)
    w2_scale = u8(NUM_LOCAL, HIDDEN, INTERMEDIATE // 32, lo=121, hi=125)
    w13_scale = w13_scale.view(torch.float8_e4m3fn)
    w2_scale = w2_scale.view(torch.float8_e4m3fn)
    alpha = torch.full((NUM_LOCAL,), 1.702, dtype=torch.float32, device="cuda")
    beta = torch.ones(NUM_LOCAL, dtype=torch.float32, device="cuda")
    limit = torch.full((NUM_LOCAL,), 7.0, dtype=torch.float32, device="cuda")

    method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
    method.use_flashinfer = True
    method.use_marlin = False
    method._fi_kernel = "trtllm_sm100"
    method.flashinfer_mxfp4_moe_precision = "default"
    method.moe_runner_config = MoeRunnerConfig(activation="situ")
    method.hidden_size = HIDDEN
    method.intermediate_size_per_partition = INTERMEDIATE
    layer = SimpleNamespace(
        quant_method=method,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL,
        moe_ep_rank=FIRST_EXPERT // NUM_LOCAL,
        use_cuda_mxfp4_decode_moe=False,
    )
    quant_info = flashinfer_trtllm.FlashInferTrtllmGenMxfp4MoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_weight_bias=None,
        w2_weight_bias=None,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=FIRST_EXPERT,
        local_num_experts=NUM_LOCAL,
        intermediate_size_per_partition=INTERMEDIATE,
        hidden_size=HIDDEN,
        flashinfer_mxfp4_moe_precision="default",
    )
    return layer, quant_info


def _route(scores, bias, x, *, layer, num_valid):
    # Drives the handoff as select_experts does; layer=None runs the unfused chain.
    route_quant_handoff.stage(x, layer=layer)
    if layer is not None:
        route_quant_handoff.bind_select_experts(num_valid, fused_routing_ok=True)
    out = route_quant_handoff.try_route_quant_fused(
        scores,
        bias,
        TOPK,
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
    )
    if out is None:
        out = route_radix(scores, bias, TOPK, True, SCALE, True)
    return out


def _run_moe(x, scores, bias, quant_info, *, layer, num_valid, defer):
    weights, ids = _route(scores, bias, x, layer=layer, num_valid=num_valid)
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(weights, ids, scores),
    )
    latent = torch.empty(x.shape[0], HIDDEN, dtype=torch.bfloat16, device="cuda")
    try:
        with (
            zero_copy_context.set_moe_output(latent),
            flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(defer),
        ):
            out = flashinfer_trtllm._fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
                dispatch, quant_info, MoeRunnerConfig(activation="situ")
            ).hidden_states
    finally:
        route_quant_handoff.clear()
    torch.cuda.synchronize()
    return out, ids


class TestKimiK3PreparedRouting(CustomTestCase):
    """The fused routing plus FlashInfer's prepared-routing body must stay bitwise
    equal to the unfused routed call, so a FlashInfer bump that changes the
    runner kwargs, tuning config or default tile fails here."""

    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("trtllm-gen MXFP4 MoE needs SM10x")
        if not kimi_k3_fused_routing.available():
            self.skipTest("Kimi-K3 fused routing kernel unavailable")

    def _assert_same(self, out, ref, valid: int, defer: bool):
        if not defer:
            self.assertTrue(bool(torch.isfinite(ref[:valid]).all()))
            self.assertTrue(torch.equal(out[:valid], ref[:valid]))
            return
        self.assertTrue(
            torch.equal(
                out.expert_weights[:valid].float(), ref.expert_weights[:valid].float()
            )
        )
        out_perm = out.expanded_idx_to_permuted_idx.view(-1, TOPK)[:valid].long()
        ref_perm = ref.expanded_idx_to_permuted_idx.view(-1, TOPK)[:valid].long()
        self.assertTrue(torch.equal(out_perm >= 0, ref_perm >= 0))
        routed = ref_perm >= 0
        self.assertTrue(
            torch.equal(
                out.gemm2_out[out_perm[routed]], ref.gemm2_out[ref_perm[routed]]
            )
        )

    def test_prepared_body_matches_routed_call(self):
        from flashinfer.autotuner import autotune

        layer, quant_info = _make_experts(seed=0)
        num_prepared = 0
        run_prepared = flashinfer_trtllm._run_trtllm_prepared_routing_moe

        def counting_run_prepared(*args, **kwargs):
            nonlocal num_prepared
            num_prepared += 1
            return run_prepared(*args, **kwargs)

        with patch.object(
            flashinfer_trtllm,
            "_run_trtllm_prepared_routing_moe",
            counting_run_prepared,
        ):
            # Default tactic first: autotuning caches a tactic per M bucket.
            for tuned in (False, True):
                # 65 crosses the 64-row switch to FP32 expert weights.
                for m in (6, 65, 192):
                    g = torch.Generator(device="cuda").manual_seed(m)
                    scores = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda")
                    bias = 0.1 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
                    x = torch.randn(m, HIDDEN, generator=g, device="cuda")
                    x = x.to(torch.bfloat16)
                    if tuned:
                        with autotune(True):
                            _run_moe(
                                x,
                                scores,
                                bias,
                                quant_info,
                                layer=None,
                                num_valid=None,
                                defer=False,
                            )
                    valid = m - NUM_PADDED_ROWS
                    num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
                    for defer in (False, True):
                        with self.subTest(tuned=tuned, m=m, defer=defer):
                            ref, _ = _run_moe(
                                x,
                                scores,
                                bias,
                                quant_info,
                                layer=None,
                                num_valid=None,
                                defer=defer,
                            )
                            before = num_prepared
                            out, ids = _run_moe(
                                x,
                                scores,
                                bias,
                                quant_info,
                                layer=layer,
                                num_valid=num_valid,
                                defer=defer,
                            )
                            self.assertEqual(num_prepared, before + 1)
                            self.assertTrue(bool((ids[valid:] == -1).all()))
                            self._assert_same(out, ref, valid, defer)


if __name__ == "__main__":
    unittest.main()
