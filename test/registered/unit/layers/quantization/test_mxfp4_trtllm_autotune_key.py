"""A deferred-finalize MXFP4 trtllm-gen MoE call above 32 rows must pass the same
output shape as a finalized one: FlashInfer keys its tactic cache on that shape, so a
mismatch silently runs the untuned default tactic. At <= 32 rows deferred calls keep
the (M, 0) placeholder key on purpose (the default tactic measured faster there)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

# The quantization package must load before the MoE runner (circular import otherwise).
import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.srt.environ import envs
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase

HIDDEN, INTER, EXPERTS, TOP_K = 64, 32, 4, 2


def _layer():
    layer = SimpleNamespace(
        w13_weight=torch.zeros(EXPERTS, 2 * INTER, HIDDEN),
        w2_weight=torch.zeros(EXPERTS, HIDDEN, INTER),
        w13_weight_scale_inv=torch.zeros(EXPERTS, 2 * INTER, 1),
        w2_weight_scale_inv=torch.zeros(EXPERTS, HIDDEN, 1),
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        moe_ep_rank=0,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        use_cuda_mxfp4_decode_moe=False,
    )
    method = object.__new__(mxfp4.Mxfp4FlashinferTrtllmMoEMethod)
    method.flashinfer_mxfp4_moe_precision = "bf16"
    method._gemm1_clamp_limit_tensor = None
    method._bypassed_routing_method = None
    return method, layer


class TestMxfp4AutotuneKey(CustomTestCase):
    def test_deferred_calls_pass_the_finalized_output_shape_above_32_rows(self):
        method, layer = _layer()
        outputs = []

        def moe_call(**kwargs):
            # FlashInfer keys a missing output as its (M, 0) deferred placeholder.
            out = kwargs["output"]
            rows = kwargs["hidden_states"].shape[0]
            outputs.append((rows, 0) if out is None else tuple(out.shape))
            return (out,)

        with (
            envs.SGLANG_OPT_MXFP4_DEFERRED_FINALIZED_SHAPE.override(True),
            get_parallel().override(tp_group=None),
            patch.object(
                mxfp4, "trtllm_fp4_block_scale_routed_moe", moe_call, create=True
            ),
            patch.object(mxfp4, "use_symmetric_memory", return_value=nullcontext()),
            patch.object(mxfp4, "is_allocation_symmetric", return_value=False),
            patch.object(flashinfer_trtllm, "_make_deferred_finalize_output"),
            patch.object(
                flashinfer_trtllm, "trtllm_moe_enable_pdl", return_value=False
            ),
        ):
            # Pins the 32 / 33 floor and both sides of the old 97-384 range.
            for rows in (32, 33, 96, 385):
                x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
                ids = torch.zeros(rows, TOP_K, dtype=torch.int32)
                topk = StandardTopKOutput(torch.ones(rows, TOP_K), ids, x.new_empty(0))
                dispatch = StandardDispatchOutput(x, None, topk)
                for deferred in (False, True):
                    with self.subTest(rows=rows, deferred=deferred):
                        with flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(
                            deferred
                        ):
                            method.apply(layer, dispatch)
                        width = 0 if deferred and rows <= 32 else HIDDEN
                        self.assertEqual(outputs[-1], (rows, width))


class TestNvfp4BypassedAutotuneKey(CustomTestCase):
    """The NVFP4 logits-bypassed call: at TP1 a deferred finalize passes a full-width
    output, so FlashInfer's AutoTuner looks up the (N, hidden) keys tuned at startup
    instead of a (N, 0) placeholder that misses them (flashinfer-ai/flashinfer#6077)."""

    def _call(self, rows, deferred, tp_size):
        seen = []

        def moe_call(**kwargs):
            seen.append(kwargs)
            return (torch.zeros(rows, HIDDEN, dtype=torch.bfloat16),)

        fused = types.ModuleType("flashinfer.fused_moe")
        fused.trtllm_fp4_block_scale_moe = moe_call
        fused.trtllm_fp4_block_scale_routed_moe = moe_call
        x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        topk = SimpleNamespace(
            router_logits=torch.zeros(rows, EXPERTS),
            topk_config=SimpleNamespace(
                correction_bias=None, top_k=TOP_K, num_expert_group=1, topk_group=1
            ),
        )
        dispatch = SimpleNamespace(
            hidden_states=x, hidden_states_scale=None, topk_output=topk
        )
        info = SimpleNamespace(
            padded_hidden_size=None,
            use_per_token_activation=False,
            w13_input_scale_quant=torch.ones(1),
            w13_weight=torch.zeros(1),
            w2_weight=torch.zeros(1),
            w13_weight_scale=torch.zeros(1, dtype=torch.uint8),
            w2_weight_scale=torch.zeros(1, dtype=torch.uint8),
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            g1_scale_c=torch.ones(1),
            g1_alphas=torch.ones(1),
            g2_alphas=torch.ones(1),
            global_num_experts=EXPERTS,
            intermediate_size_per_partition=INTER,
            local_expert_offset=0,
            local_num_experts=EXPERTS,
            routing_method_type=None,
        )
        config = SimpleNamespace(
            routed_scaling_factor=None, activation="silu", is_gated=True
        )
        quantized = (
            torch.zeros(rows, HIDDEN // 2, dtype=torch.uint8),
            torch.zeros(rows, HIDDEN // 16, dtype=torch.uint8),
        )
        with (
            patch.dict(sys.modules, {"flashinfer.fused_moe": fused}),
            patch.object(
                flashinfer_trtllm, "quantize_hidden_states_fp4", return_value=quantized
            ),
            patch.object(
                flashinfer_trtllm,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=tp_size),
            ),
            patch.object(flashinfer_trtllm, "get_activation_type", return_value=0),
            patch.object(
                flashinfer_trtllm, "trtllm_moe_enable_pdl", return_value=False
            ),
            patch.object(
                flashinfer_trtllm, "_make_deferred_finalize_output", lambda r, **k: r
            ),
            patch.object(
                flashinfer_trtllm,
                "_get_fp4_moe_output_buffer",
                lambda **kw: torch.zeros(kw["num_tokens"], kw["hidden_size"]),
            ),
            patch.object(
                topk_module.TopKOutputChecker,
                "format_is_standard",
                return_value=False,
            ),
            patch.object(
                topk_module.TopKOutputChecker,
                "format_is_bypassed",
                return_value=True,
            ),
            flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(deferred),
        ):
            flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch, info, config
            )
        return seen[-1]

    def test_deferred_tp1_call_passes_the_full_width_output(self):
        for rows in (8, 64):
            for deferred in (False, True):
                for tp_size in (1, 4):
                    with self.subTest(rows=rows, deferred=deferred, tp_size=tp_size):
                        kw = self._call(rows, deferred, tp_size)
                        self.assertIs(kw["do_finalize"], not deferred)
                        if deferred and tp_size != 1:
                            self.assertNotIn("output", kw)
                        elif deferred:
                            self.assertEqual(tuple(kw["output"].shape), (rows, HIDDEN))
                            self.assertEqual(kw["output"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
