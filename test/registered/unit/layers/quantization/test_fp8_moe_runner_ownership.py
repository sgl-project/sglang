"""Unit tests for srt/layers/quantization/fp8.py MoE runner ownership.

`--moe-runner-backend flashinfer_trtllm[_routed]` is a global setting, so
`Fp8MoEMethod.process_weights_after_loading` must also require that this
instance owns a MoeRunner before materializing the TRT-LLM SwiGLU params:
the MxFP4 wrapper methods borrow an `Fp8MoEMethod` for weight loading only
and never give it a `moe_runner_config` (issue #36264).
"""

import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


class TestFp8MoEAiterQuantInfo(CustomTestCase):
    """maybe_get_hip_aiter_quant_info assembles what the AITER runner consumes.

    The gfx950 e2e builds AiterMoeQuantInfo by hand, so dropping the gate/up
    layout or the clamp here would leave it passing while served experts read
    the gate and up halves swapped.
    """

    def test_block_fp8_forwards_separated_layout_and_clamp(self):
        method = Fp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        )
        # create_moe_runner is not called: it resolves a global backend and
        # builds a MoeRunner, none of which this assembly reads.
        method.moe_runner_config = MoeRunnerConfig(swiglu_limit=10.0)
        layer = SimpleNamespace(
            w13_weight=torch.zeros((1, 4, 4), dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros((1, 4, 2), dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            w2_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            hidden_pad=0,
            intermediate_pad=0,
            _aiter_gate_up_interleaved=False,
            dispatcher=SimpleNamespace(expert_mask_gpu=torch.tensor([True, False])),
        )
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )

        with (
            patch.dict(sys.modules, {"aiter.ops.flydsl.moe_common": fake_moe_common}),
            patch.object(fp8_module, "_use_aiter", True),
        ):
            quant_info = method.maybe_get_hip_aiter_quant_info(layer)

        self.assertIsNotNone(quant_info)
        self.assertEqual(quant_info.quant_type, AiterQuantType.PER_128X128)
        self.assertEqual(quant_info.swiglu_limit, 10.0)
        self.assertEqual(quant_info.fused_moe_kwargs, {"gate_mode": "separated"})
        self.assertIs(quant_info.expert_mask, layer.dispatcher.expert_mask_gpu)


class TestFp8MoEAiterWeightLayout(CustomTestCase):
    @staticmethod
    def _make_layer():
        layer = torch.nn.Module()
        for name, shape in (("w13_weight", (2, 32, 16)), ("w2_weight", (2, 16, 16))):
            weight = (torch.arange(math.prod(shape)).reshape(shape) % 13).to(
                torch.float8_e4m3fn
            )
            weight.view(torch.int8).reshape(-1)[0] = -128
            layer.register_parameter(
                name, torch.nn.Parameter(weight, requires_grad=False)
            )
            layer.register_parameter(
                f"{name}_scale_inv",
                torch.nn.Parameter(torch.ones((2, 1, 1)), requires_grad=False),
            )
        return layer

    @staticmethod
    def _shuffle(weight, block):
        return weight.view(torch.uint8).flip(-1).contiguous().view(weight.dtype)

    def _check_layout(self, fnuz):
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
            for backend in (MoeRunnerBackend.TRITON, None, MoeRunnerBackend.AITER):
                with self.subTest(backend=backend, fnuz=fnuz):
                    method = Fp8MoEMethod(
                        Fp8Config(
                            is_checkpoint_fp8_serialized=True,
                            weight_block_size=[128, 128],
                        )
                    )
                    if backend is not None:
                        method.runner = SimpleNamespace(runner_backend=backend)
                    layer = self._make_layer()
                    expected = {}
                    for name in ("w13_weight", "w2_weight"):
                        weight = getattr(layer, name).detach().clone()
                        if fnuz:
                            weight, _, _ = fp8_module.normalize_e4m3fn_to_e4m3fnuz(
                                weight, getattr(layer, f"{name}_scale_inv"), None
                            )
                        if backend == MoeRunnerBackend.AITER:
                            weight = self._shuffle(weight, (16, 16))
                        expected[name] = weight
                    with (
                        patch.object(fp8_module, "_is_fp8_fnuz", fnuz),
                        patch.object(fp8_module, "_use_aiter", True),
                        patch.object(
                            fp8_module, "shuffle_weight", self._shuffle, create=True
                        ),
                    ):
                        method.process_weights_after_loading_block_quant(layer)
                    for name, weight in expected.items():
                        actual = getattr(layer, name)
                        self.assertEqual(actual.dtype, weight.dtype)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), weight.view(torch.uint8)
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                getattr(layer, f"{name}_scale_inv"),
                                torch.full((2, 1, 1), 2.0 if fnuz else 1.0),
                            )
                        )

    def test_fnuz_shuffles_only_for_aiter_runner(self):
        self._check_layout(fnuz=True)

    def test_non_fnuz_shuffles_only_for_aiter_runner(self):
        self._check_layout(fnuz=False)


if __name__ == "__main__":
    unittest.main()
