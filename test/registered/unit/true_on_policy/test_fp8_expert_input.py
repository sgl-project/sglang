"""TOP FP8 expert input: what the quantizer and the expert forward refuse instead of running another program."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.quantization.per_token_group_quant import _jit_module
from sglang.srt.true_on_policy import fp8
from sglang.srt.true_on_policy.fp8.fp8_expert_forward import (
    _gemm_input_scales,
    expert_forward,
    quantize_rows,
)
from sglang.srt.true_on_policy.slot_order_deepep import (
    SlotOrderDeepEPError,
    _check_expert_input,
    _row_words,
    enable_low_latency_fp8,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_HOPPER = patch("torch.cuda.get_device_capability", return_value=(9, 0))
_BLACKWELL = patch("torch.cuda.get_device_capability", return_value=(10, 0))


class TestQuantizeRowsSwigluClamp(CustomTestCase):
    def test_refuses_a_limit_without_the_fused_swiglu(self):
        """The clamp lives in the fused SwiGLU entry points; a plain row quant would drop it silently."""
        x = torch.zeros(4, 256, dtype=torch.bfloat16)
        with _HOPPER, self.assertRaisesRegex(ValueError, "fused SwiGLU"):
            quantize_rows(x, swiglu_limit=10.0)


class TestExpertInputAmaxFloor(CustomTestCase):
    def test_swiglu_output_keeps_the_default_floor(self):
        """Only the expert input crosses DeepEP; the down projection's input keeps 1e-10 like serving's other
        quantizers, so the floor must not reach the fused SwiGLU entry points."""
        gate = torch.zeros(4, 512, dtype=torch.bfloat16)
        with _HOPPER, self.assertRaisesRegex(ValueError, "expert input only"):
            quantize_rows(gate, fused_swiglu=True, deepep_amax_floor=True)
        for fused, ue8m0 in ((True, True), (False, False)):
            with self.assertRaises(AssertionError):
                _jit_module(
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    128,
                    ue8m0,
                    False,
                    True,
                    fused,
                    False,
                    False,
                    deepep_amax_floor=True,
                )


def _weights():
    w = torch.zeros(2, 256, 256, dtype=torch.float8_e4m3fn)
    s = torch.zeros(2, 2, 2, dtype=torch.float32)
    return (w, s), (w, s)


class TestPrequantizedDispatch(CustomTestCase):
    def test_only_top_fp8_experts_send_fp8(self):
        """BF16 experts (GLM-5.3-Flash) and non-TOP serving keep the BF16 dispatch in both modes."""
        cases = [
            (fp8.BLOCK, True, True),
            (fp8.BLOCK32, True, True),
            (fp8.BLOCK, False, False),
            (None, True, False),
        ]
        for block, quantized, expected in cases:
            with patch.object(fp8, "served_block", return_value=block):
                self.assertIs(fp8.fp8_dispatch(quantized_experts=quantized), expected)

    def test_expert_forward_refuses_scales_with_bf16_rows(self):
        w13, w2 = _weights()
        scale = torch.zeros(4, 2, dtype=torch.float32)
        x = torch.zeros(4, 256, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "E4M3 rows"):
            expert_forward(x, w13, w2, x_scale=scale, m_indices=torch.zeros(4))

    def test_runner_refuses_bf16_input_to_the_masked_gemm(self):
        """Decode's expert input is DeepEP's fused FP8 cast; BF16 rows there mean the dispatcher skipped it."""
        runner_input = SimpleNamespace(
            hidden_states=torch.zeros(2, 4, 256, dtype=torch.bfloat16),
            hidden_states_scale=None,
            use_masked_gemm=True,
        )
        quant_info = SimpleNamespace(
            w13_weight=None, w13_scale=None, w2_weight=None, w2_scale=None
        )
        with patch.object(fp8, "served_block", return_value=fp8.BLOCK):
            with self.assertRaisesRegex(NotImplementedError, "low-latency receive"):
                fp8.maybe_expert_forward(
                    runner_input,
                    quant_info,
                    weight_dtype=torch.float8_e4m3fn,
                    activation="silu",
                    swiglu_limit=None,
                )


class TestLowLatencyFp8(CustomTestCase):
    @staticmethod
    def _impl(*, ue8m0):
        return SimpleNamespace(
            use_fp8=False, fp8_round_scale=ue8m0, fp8_use_ue8m0=ue8m0
        )

    def test_power_of_two_scales_on_both_architectures(self):
        """Stock asks DeepEP for power-of-two scales on Blackwell only; on Hopper it would send FP32 amax/448
        scales, which no TOP quantizer writes."""
        for arch, ue8m0 in ((_HOPPER, False), (_BLACKWELL, True)):
            impl = self._impl(ue8m0=ue8m0)
            with arch:
                enable_low_latency_fp8(impl)
            self.assertEqual(
                (impl.use_fp8, impl.fp8_round_scale, impl.fp8_use_ue8m0),
                (True, True, ue8m0),
            )

    def test_refuses_a_scale_storage_the_expert_gemm_does_not_read(self):
        for arch, ue8m0 in ((_HOPPER, True), (_BLACKWELL, False)):
            with arch, self.assertRaisesRegex(NotImplementedError, "scale format"):
                enable_low_latency_fp8(self._impl(ue8m0=ue8m0))

    def test_masked_scales_reach_the_gemm_in_its_layout(self):
        """DeepEP's low-latency scales are [experts, rows, groups] MN-major; another layout would be read
        as different scales by the masked GEMM."""
        mn_major = torch.zeros(2, 2, 8).transpose(1, 2)
        with _HOPPER:
            self.assertIs(_gemm_input_scales(mn_major), mn_major)
            with self.assertRaisesRegex(ValueError, "MN-major"):
                _gemm_input_scales(torch.zeros(2, 8, 2))
        with _BLACKWELL, self.assertRaisesRegex(ValueError, "packed MN-major int32"):
            _gemm_input_scales(mn_major)


class TestSlotReceiveDtype(CustomTestCase):
    def _out(self, dtype, scale):
        return SimpleNamespace(
            hidden_states=torch.zeros(3, 256, dtype=dtype), hidden_states_scale=scale
        )

    def test_receive_must_carry_what_the_dispatcher_sent(self):
        scale = torch.zeros(3, 2)
        cases = [
            (self._out(torch.bfloat16, None), True),
            (self._out(torch.float8_e4m3fn, None), True),
            (self._out(torch.float8_e4m3fn, scale), False),
            (self._out(torch.bfloat16, scale), False),
        ]
        for out, fp8_sent in cases:
            with self.assertRaises(SlotOrderDeepEPError):
                _check_expert_input(out, fp8=fp8_sent)

    def test_fingerprints_refuse_rows_that_are_not_16_bit_words(self):
        with self.assertRaisesRegex(SlotOrderDeepEPError, "BF16 or FP8"):
            _row_words(torch.zeros(3, 256, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
