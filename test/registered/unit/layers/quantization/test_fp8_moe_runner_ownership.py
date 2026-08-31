"""Unit tests for srt/layers/quantization/fp8.py MoE runner ownership.

MoE backend flags are global, so `Fp8MoEMethod.process_weights_after_loading`
must require that this instance owns a MoeRunner before materializing
backend-specific parameters or layouts. MxFP4 wrappers borrow an `Fp8MoEMethod`
for weight loading without giving it a runner (issue #36264).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ACTIVATION_PARAMS = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")


class TestFp8MoERunnerOwnership(CustomTestCase):
    def setUp(self):
        moe = get_flags().moe
        self._saved_runner_backend = moe.runner_backend
        moe.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM
        # _use_hip_int4 would divert this to the ROCm int4 branch, past the guard.
        hip_int4 = patch("sglang.srt.layers.quantization.fp8._use_hip_int4", False)
        hip_int4.start()
        self.addCleanup(hip_int4.stop)

    def tearDown(self):
        get_flags().moe.runner_backend = self._saved_runner_backend

    @staticmethod
    def _make_block_fp8_method() -> Fp8MoEMethod:
        # The real constructor's _owns_moe_runner default is what a delegate relies on.
        return Fp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        )

    @staticmethod
    def _make_layer(num_local_experts: int = 2) -> SimpleNamespace:
        return SimpleNamespace(
            num_local_experts=num_local_experts,
            w13_weight=torch.empty(num_local_experts, 4),
        )

    def _run_post_load(self, method: Fp8MoEMethod, layer: SimpleNamespace) -> None:
        with patch.object(method, "process_weights_after_loading_block_quant") as work:
            method.process_weights_after_loading(layer)
        work.assert_called_once_with(layer)

    def _assert_activation_params_absent(self, layer: SimpleNamespace) -> None:
        for name in _ACTIVATION_PARAMS:
            self.assertFalse(hasattr(layer, f"_flashinfer_trtllm_{name}"))

    def test_borrowed_delegate_skips_trtllm_activation_params(self):
        """A method with no MoeRunner must not read moe_runner_config; doing so
        aborts weight loading whenever a TRT-LLM runner backend is selected."""
        method = self._make_block_fp8_method()
        layer = self._make_layer()

        self._run_post_load(method=method, layer=layer)

        self._assert_activation_params_absent(layer)

    def test_owning_method_prepares_trtllm_activation_params(self):
        """The owning method must still materialize the params it consumes;
        apply() dereferences layer._flashinfer_trtllm_* on the TRT-LLM branch."""
        method = self._make_block_fp8_method()
        layer = self._make_layer()
        method.create_moe_runner(
            layer=layer,
            moe_runner_config=MoeRunnerConfig(
                gemm1_alpha=1.5, gemm1_beta=0.25, gemm1_clamp_limit=None
            ),
        )

        self._run_post_load(method=method, layer=layer)

        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_alpha,
                torch.full((2,), 1.5, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_beta,
                torch.full((2,), 0.25, dtype=torch.float32),
            )
        )
        # None stays None: a zero-filled tensor would not mean "no clamp".
        self.assertIsNone(layer._flashinfer_trtllm_gemm1_clamp_limit)

    def test_owning_method_skips_params_on_non_trtllm_backend(self):
        """Ownership alone must not materialize params no kernel consumes."""
        get_flags().moe.runner_backend = MoeRunnerBackend.TRITON
        method = self._make_block_fp8_method()
        layer = self._make_layer()
        method.create_moe_runner(
            layer=layer, moe_runner_config=MoeRunnerConfig(gemm1_alpha=1.5)
        )

        self._run_post_load(method=method, layer=layer)

        self._assert_activation_params_absent(layer)

    def test_global_aiter_does_not_shuffle_triton_runner_weights(self):
        """SGLANG_USE_AITER also enables non-MoE kernels, so the selected MoE
        runner must own the preshuffled layout before weights are rewritten."""
        method = self._make_block_fp8_method()
        method.convert_mxfp8_to_block = True
        method.use_mxfp8 = True
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        method._owns_moe_runner = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.arange(16.0).reshape(1, 4, 4)),
            w2_weight=torch.nn.Parameter(torch.arange(16.0).reshape(1, 4, 4)),
        )

        with (
            patch("sglang.srt.layers.quantization.fp8._use_aiter", True),
            patch("sglang.srt.layers.quantization.fp8._is_fp8_fnuz", False),
            patch.object(method, "_convert_mxfp8_moe_to_block_fp8"),
            patch(
                "sglang.srt.layers.quantization.fp8.shuffle_weight", create=True
            ) as shuffle,
        ):
            method.process_weights_after_loading_block_quant(layer)

        shuffle.assert_not_called()

    def test_borrowed_delegate_skips_aiter_weight_shuffle(self):
        """MxFP4 delegates do not own a runner, so global AITER enablement must
        not make weight loading dereference or prepare an absent runner."""
        method = self._make_block_fp8_method()
        method.convert_mxfp8_to_block = True
        method.use_mxfp8 = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.arange(16.0).reshape(1, 4, 4)),
            w2_weight=torch.nn.Parameter(torch.arange(16.0).reshape(1, 4, 4)),
        )

        with (
            patch("sglang.srt.layers.quantization.fp8._use_aiter", True),
            patch("sglang.srt.layers.quantization.fp8._is_fp8_fnuz", False),
            patch.object(method, "_convert_mxfp8_moe_to_block_fp8"),
            patch(
                "sglang.srt.layers.quantization.fp8.shuffle_weight", create=True
            ) as shuffle,
        ):
            method.process_weights_after_loading_block_quant(layer)

        shuffle.assert_not_called()

    def test_aiter_runner_keeps_preshuffled_weight_layout(self):
        method = self._make_block_fp8_method()
        method.convert_mxfp8_to_block = True
        method.use_mxfp8 = True
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)
        method._owns_moe_runner = True
        original_w13 = torch.arange(16.0).reshape(1, 4, 4)
        original_w2 = torch.arange(16.0, 32.0).reshape(1, 4, 4)
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(original_w13.clone()),
            w2_weight=torch.nn.Parameter(original_w2.clone()),
        )

        def add_one(weight, _tile):
            return weight + 1

        with (
            patch("sglang.srt.layers.quantization.fp8._use_aiter", True),
            patch("sglang.srt.layers.quantization.fp8._is_fp8_fnuz", False),
            patch.object(method, "_convert_mxfp8_moe_to_block_fp8"),
            patch(
                "sglang.srt.layers.quantization.fp8.shuffle_weight",
                side_effect=add_one,
                create=True,
            ) as shuffle,
        ):
            method.process_weights_after_loading_block_quant(layer)

        self.assertEqual(shuffle.call_count, 2)
        torch.testing.assert_close(layer.w13_weight, original_w13 + 1)
        torch.testing.assert_close(layer.w2_weight, original_w2 + 1)


if __name__ == "__main__":
    unittest.main()
