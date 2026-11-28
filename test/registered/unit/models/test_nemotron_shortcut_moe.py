"""Nemotron Shortcut's soft-clamped squared-ReLU experts: runner selection and
routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (  # noqa: E402
    _cutedsl_wrapper_activation_type,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # noqa: E402
from sglang.srt.layers.quantization.mxfp8_cutedsl_moe import (  # noqa: E402
    Mxfp8CuteDslMoEMethod,
)
from sglang.srt.models.nemotron_shortcut import (  # noqa: E402
    _check_clamped_relu2_backend,
    route_logit_space,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ACTIVATIONS = SimpleNamespace(
    Swiglu="Swiglu", Relu2="Relu2", ClampedRelu2="ClampedRelu2"
)


def _check_backend(backend: MoeRunnerBackend, cutedsl_supported: bool = True):
    with (
        patch(
            "sglang.srt.models.nemotron_shortcut.get_moe_runner_backend",
            return_value=backend,
        ),
        patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl."
            "cutedsl_supports_clamped_relu2",
            return_value=cutedsl_supported,
        ),
    ):
        _check_clamped_relu2_backend()


class TestExpertActivation(CustomTestCase):
    def test_clamp_limit_selects_clamped_relu2(self):
        pick = _cutedsl_wrapper_activation_type
        self.assertEqual(pick("relu2", _ACTIVATIONS, 16.0), "ClampedRelu2")
        # Without a limit the plain squared ReLU stays plain.
        self.assertEqual(pick("relu2", _ACTIVATIONS, None), "Relu2")
        self.assertEqual(pick("silu", _ACTIVATIONS, None), "Swiglu")
        with self.assertRaises(ValueError):
            pick("gelu", _ACTIVATIONS, None)

    def test_only_runners_with_the_soft_clamp_are_accepted(self):
        _check_backend(MoeRunnerBackend.MARLIN)
        _check_backend(MoeRunnerBackend.FLASHINFER_CUTEDSL)
        with self.assertRaises(ValueError):
            _check_backend(MoeRunnerBackend.FLASHINFER_CUTEDSL, cutedsl_supported=False)
        with self.assertRaises(ValueError):
            _check_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)

    def test_triton_serves_the_bf16_mtp_experts(self):
        _check_backend(MoeRunnerBackend.TRITON)


class TestMxfp8CuteDsl(CustomTestCase):
    def test_gated_experts_are_rejected(self):
        # Gated experts would need w13 interleaved for the kernel; serving them
        # as stored would silently pair the wrong rows.
        method = Mxfp8CuteDslMoEMethod.__new__(Mxfp8CuteDslMoEMethod)
        config = MoeRunnerConfig(activation="silu", is_gated=True)
        with self.assertRaises(NotImplementedError):
            method.create_moe_runner(SimpleNamespace(), config)


class TestLogitSpaceRouting(CustomTestCase):
    def test_selects_on_biased_logits_and_weights_unbiased_scores(self):
        # topk(logits + bias) picks experts {0, 2}; topk(sigmoid(logits) + bias)
        # would pick {0, 1}, so this case tells the two bias spaces apart.
        logits = torch.tensor([[4.0, 0.0, 3.5]])
        bias = torch.tensor([0.0, 0.6, 0.0])
        weights, ids = route_logit_space(
            hidden_states=torch.zeros(1, 8),
            gating_output=logits,
            topk=2,
            renormalize=True,
            bias=bias,
        )
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(sorted(ids[0].tolist()), [0, 2])
        expected = logits[0, ids[0].long()].sigmoid()
        torch.testing.assert_close(weights[0], expected / expected.sum())


if __name__ == "__main__":
    unittest.main()
