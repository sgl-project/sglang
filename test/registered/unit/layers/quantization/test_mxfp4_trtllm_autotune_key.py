"""FlashInfer keys its trtllm-gen MoE tactic cache on the output shape, so a
deferred-finalize call that passes a different shape from the finalized call silently
runs the untuned default tactic. MXFP4 deferred calls pass the finalized shape above 32
rows; at <= 32 rows they keep the (M, 0) placeholder key on purpose (the default tactic
measured faster there). NVFP4 logits-bypassed deferred calls pass it at every row count,
as the routed path does (flashinfer-ai/flashinfer#6077)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

# The quantization package must load before the MoE runner (circular import otherwise).
import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
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
    """A deferred NVFP4 logits-bypassed call passes the finalized output shape."""

    def test_deferred_calls_pass_the_finalized_output_shape(self):
        outputs = []

        def moe_call(**kwargs):
            # FlashInfer keys a missing output as its (M, 0) deferred placeholder.
            out = kwargs.get("output")
            rows = kwargs["hidden_states"].shape[0]
            outputs.append((rows, 0) if out is None else tuple(out.shape))
            return (out,)

        fused = ModuleType("flashinfer.fused_moe")
        fused.trtllm_fp4_block_scale_moe = moe_call
        fused.trtllm_fp4_block_scale_routed_moe = moe_call
        byte, one = torch.zeros(1, dtype=torch.uint8), torch.ones(1)
        # Weights and weight scales, the four scalar scales, then expert metadata.
        info = flashinfer_trtllm.FlashInferTrtllmFp4MoeQuantInfo(
            *[byte] * 4, *[one] * 4, EXPERTS, 0, EXPERTS, INTER, None
        )
        with (
            patch.dict(sys.modules, {"flashinfer.fused_moe": fused}),
            patch.object(flashinfer_trtllm, "get_activation_type", return_value=0),
            patch.object(
                flashinfer_trtllm, "trtllm_moe_enable_pdl", return_value=False
            ),
            patch.object(flashinfer_trtllm, "_make_deferred_finalize_output"),
            flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(True),
        ):
            for rows in (8, 64):
                # FP4 activations as an NVFP4 dispatcher delivers them.
                x = torch.zeros(rows, HIDDEN // 2, dtype=torch.uint8)
                scale = torch.zeros(rows, HIDDEN // 16, dtype=torch.uint8)
                logits = torch.zeros(rows, EXPERTS)
                topk = BypassedTopKOutput(x, logits, TopKConfig(top_k=TOP_K))
                with self.subTest(rows=rows):
                    flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
                        StandardDispatchOutput(x, scale, topk), info, MoeRunnerConfig()
                    )
                    self.assertEqual(outputs[-1], (rows, HIDDEN))


if __name__ == "__main__":
    unittest.main()
