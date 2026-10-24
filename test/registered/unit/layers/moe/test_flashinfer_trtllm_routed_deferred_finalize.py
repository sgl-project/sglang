"""CPU unit tests for deferred finalize on the routed FlashInfer TRT-LLM FP8 runner.

MXFP8 experts on ``flashinfer_trtllm_routed`` hand the unfinalized GEMM2 rows to a
fused finalize consumer instead of writing the MoE output. FlashInfer is replaced
by a module stub, so these tests pin the runner's half of that contract on CPU;
the kernels are covered on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import enum
import sys
import unittest
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

# Import the FusedMoE layer before flashinfer_trtllm (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer  # noqa: F401
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
    FlashInferTrtllmFp8MoeQuantInfo,
    flashinfer_trtllm_deferred_finalize_context,
    fused_experts_none_to_flashinfer_trtllm_fp8,
)

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_forward
from sglang.test.test_utils import CustomTestCase

NUM_TOKENS = 3
HIDDEN = 64
INTERMEDIATE = 32
NUM_EXPERTS = 4
TOP_K = 2
ROUTED_SCALING_FACTOR = 1.5


class _Fp8QuantizationType(enum.IntEnum):
    DeepSeekFp8 = 0
    MxFp8 = 1


class _ActivationType(enum.IntEnum):
    Swiglu = 3


@contextmanager
def _flashinfer_stub(routed_moe):
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.__path__ = []
    fused_moe.Fp8QuantizationType = _Fp8QuantizationType
    fused_moe.trtllm_fp8_block_scale_routed_moe = routed_moe
    core = ModuleType("flashinfer.fused_moe.core")
    core.ActivationType = _ActivationType
    with mock.patch.dict(
        sys.modules,
        {
            "flashinfer": flashinfer,
            "flashinfer.fused_moe": fused_moe,
            "flashinfer.fused_moe.core": core,
        },
    ):
        yield


@contextmanager
def _cpu_runner_env(eager_wrapper=None):
    """Stub the GPU-only collaborators of the FP8 runner."""
    with (
        mock.patch.object(flashinfer_trtllm, "trtllm_moe_enable_pdl", lambda n: False),
        mock.patch.object(
            flashinfer_trtllm, "use_symmetric_memory", lambda *a, **k: nullcontext()
        ),
        mock.patch.object(
            flashinfer_trtllm,
            "get_parallel",
            lambda: SimpleNamespace(tp_group=None),
        ),
        mock.patch.object(flashinfer_trtllm, "is_allocation_symmetric", lambda: False),
        mock.patch.object(
            flashinfer_trtllm,
            "trtllm_fp8_block_scale_routed_moe_out_wrapper",
            eager_wrapper or mock.Mock(side_effect=AssertionError("eager call")),
        ),
    ):
        yield


def _quant_info(*, use_mxfp8: bool, block_quant: bool = True):
    block_k = 32 if use_mxfp8 else 128
    return FlashInferTrtllmFp8MoeQuantInfo(
        w13_weight=torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        w2_weight=torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        intermediate_size=INTERMEDIATE,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
        block_quant=block_quant,
        use_mxfp8=use_mxfp8,
        weight_block_k=block_k if block_quant else None,
        w13_weight_scale_inv=torch.zeros(1) if block_quant else None,
        w2_weight_scale_inv=torch.zeros(1) if block_quant else None,
        activation_type=int(_ActivationType.Swiglu),
    )


def _routed_dispatch_output() -> StandardDispatchOutput:
    # MXFP8-prequantized activations skip the in-runner quantize kernel.
    # FP8+routed folds the routed scaling factor into the top-k weights; fp32 is
    # what the STANDARD top-k produces.
    topk_weights = (
        torch.tensor([[0.75, 0.25], [0.5, 0.5], [0.875, 0.125]], dtype=torch.float32)
        * ROUTED_SCALING_FACTOR
    )
    return StandardDispatchOutput(
        hidden_states=torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.zeros(NUM_TOKENS, HIDDEN // 32, dtype=torch.uint8),
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int32),
            router_logits=torch.zeros(NUM_TOKENS, NUM_EXPERTS),
        ),
    )


def _runner_config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=TOP_K,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )


class _DeferredRoutedMoe:
    """FlashInfer's ``do_finalize=False`` routed op: it returns the unpacked
    caller weights as-is, next to padded GEMM2 rows and the permutation."""

    def __init__(self):
        self.calls = []
        self.gemm2_out = torch.ones(16, HIDDEN, dtype=torch.bfloat16)
        self.permuted_idx = torch.arange(NUM_TOKENS * TOP_K, dtype=torch.int32)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        _, topk_weights = kwargs["topk_ids"]
        return [self.gemm2_out, topk_weights, self.permuted_idx]


class TestFp8RoutedDeferredFinalize(CustomTestCase):
    def test_mxfp8_routed_defers_with_caller_weights_and_a_tuned_buffer(self):
        dispatch_output = _routed_dispatch_output()
        topk_weights = dispatch_output.topk_output.topk_weights.clone()
        routed_moe = _DeferredRoutedMoe()

        with (
            _flashinfer_stub(routed_moe),
            _cpu_runner_env(),
            flashinfer_trtllm_deferred_finalize_context(),
        ):
            combine_input = fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                _quant_info(use_mxfp8=True),
                _runner_config(),
                use_routed_topk=True,
            )

        deferred = combine_input.hidden_states
        self.assertIsInstance(deferred, FlashInferTrtllmDeferredFinalizeOutput)
        self.assertIs(deferred.gemm2_out, routed_moe.gemm2_out)
        self.assertIs(deferred.expanded_idx_to_permuted_idx, routed_moe.permuted_idx)
        self.assertEqual(deferred.top_k, TOP_K)
        # No deferred consumer rescales, so the weights must reach the finalize
        # exactly as the top-k scaled them: once by the routed scaling factor.
        torch.testing.assert_close(
            deferred.expert_weights, topk_weights, rtol=0, atol=0
        )

        (kwargs,) = routed_moe.calls
        self.assertIs(kwargs["do_finalize"], False)
        self.assertIs(kwargs["fp8_quantization_type"], _Fp8QuantizationType.MxFp8)
        self.assertIs(kwargs["activation_type"], _ActivationType.Swiglu)
        # flashinfer-ai/flashinfer#6077: a zero-width placeholder output misses the
        # autotune entry, so the ignored buffer must keep the finalized shape.
        self.assertEqual(kwargs["output"].shape, (NUM_TOKENS, HIDDEN))
        self.assertEqual(kwargs["output"].dtype, torch.bfloat16)

    def test_mxfp8_routed_without_defer_writes_the_final_output(self):
        dispatch_output = _routed_dispatch_output()
        written = []

        routed_moe = mock.Mock(side_effect=AssertionError("deferred call"))
        with (
            _flashinfer_stub(routed_moe),
            _cpu_runner_env(lambda **kwargs: written.append(kwargs)),
        ):
            combine_input = fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                _quant_info(use_mxfp8=True),
                _runner_config(),
                use_routed_topk=True,
            )

        (kwargs,) = written
        # The MoE output is the buffer the eager kernel finalized into.
        self.assertIs(combine_input.hidden_states, kwargs["output"])
        self.assertEqual(combine_input.hidden_states.shape, (NUM_TOKENS, HIDDEN))
        self.assertIs(kwargs["topk_weights"], dispatch_output.topk_output.topk_weights)

    def test_non_mxfp8_routed_defer_is_rejected(self):
        for name, quant_info in (
            ("block_fp8", _quant_info(use_mxfp8=False)),
            ("per_tensor", _quant_info(use_mxfp8=False, block_quant=False)),
        ):
            with self.subTest(quant=name):
                routed_moe = mock.Mock(side_effect=AssertionError("deferred call"))
                with (
                    _flashinfer_stub(routed_moe),
                    _cpu_runner_env(),
                    flashinfer_trtllm_deferred_finalize_context(),
                    self.assertRaisesRegex(RuntimeError, "deferred finalize"),
                ):
                    fused_experts_none_to_flashinfer_trtllm_fp8(
                        _routed_dispatch_output(),
                        quant_info,
                        _runner_config(),
                        use_routed_topk=True,
                    )


class _Experts:
    """FusedMoE stand-in that records which forward the MoE block picked."""

    quant_method = None
    moe_runner_config = SimpleNamespace(inplace=False)

    def __init__(self, *, logits_flag: bool, routed_flag: bool):
        self.supports_deferred_finalize = logits_flag
        self.supports_routed_deferred_finalize = routed_flag
        self.paths = []

    def forward_deferred_finalize(self, hidden_states, topk_output, **kwargs):
        self.paths.append(("deferred", topk_output.format))
        return "deferred-handle"

    def forward_impl(self, hidden_states, topk_output):
        self.paths.append(("eager", topk_output.format))
        return hidden_states.clone()

    def __call__(self, hidden_states, topk_output, **kwargs):
        return self.forward_impl(hidden_states, topk_output)


class _StandardTopK:
    topk_config = None

    def __call__(self, hidden_states, router_logits, **kwargs):
        num_tokens = hidden_states.shape[0]
        return StandardTopKOutput(
            topk_weights=torch.ones(num_tokens, TOP_K),
            topk_ids=torch.zeros(num_tokens, TOP_K, dtype=torch.int32),
            router_logits=router_logits,
        )


def _deepseek_moe(experts: _Experts) -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    vars(moe).update(
        reduce_results=False,
        _shared_expert_tp1=False,
        layer_id=0,
        is_nextn=False,
        is_hash=False,
        num_fused_shared_experts=0,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        experts=experts,
        gate=lambda x, *args: x,
        topk=_StandardTopK(),
        alt_stream=mock.MagicMock(),
        routed_quant_stream=None,
        _deferred_finalize_max_tokens=192,
        _maybe_quant_moe_input_once=lambda x: None,
        _should_quant_routed_input_mxfp8=lambda x: False,
        _forward_shared_experts=lambda x, *args, **kwargs: x * 0.5,
    )
    return moe


class TestDeepseekMoeDeferredFlagByTopKFormat(CustomTestCase):
    def _run(self, *, bypass: bool, logits_flag: bool, routed_flag: bool):
        experts = _Experts(logits_flag=logits_flag, routed_flag=routed_flag)
        with (
            get_forward().scoped(flashinfer_trtllm_bypass=bypass),
            mock.patch.object(deepseek_v2, "_is_cuda", True),
            mock.patch.object(
                deepseek_v2,
                "get_exec",
                lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
            ),
            mock.patch.object(
                deepseek_v2,
                "maybe_fuse_routed_scale_and_shared_add",
                lambda experts, routed, shared, scale: routed,
            ),
            mock.patch.object(
                flashinfer_trtllm,
                "finalize_flashinfer_trtllm_deferred_output",
                lambda deferred, shared: torch.zeros_like(shared),
            ),
            mock.patch("torch.cuda.current_stream", mock.MagicMock),
            mock.patch("torch.cuda.stream", lambda stream: nullcontext()),
        ):
            _deepseek_moe(experts).forward_normal_dual_stream(
                torch.ones(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
            )
        ((path, _),) = experts.paths
        return path

    def test_routed_top_k_reads_the_routed_flag(self):
        self.assertEqual(
            self._run(bypass=False, logits_flag=False, routed_flag=True), "deferred"
        )
        self.assertEqual(
            self._run(bypass=False, logits_flag=True, routed_flag=False), "eager"
        )

    def test_bypassed_top_k_reads_the_logits_flag(self):
        self.assertEqual(
            self._run(bypass=True, logits_flag=True, routed_flag=False), "deferred"
        )
        self.assertEqual(
            self._run(bypass=True, logits_flag=False, routed_flag=True), "eager"
        )


if __name__ == "__main__":
    unittest.main()
