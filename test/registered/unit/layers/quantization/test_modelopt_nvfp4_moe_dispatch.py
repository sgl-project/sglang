"""CPU unit tests for NVFP4 fused-MoE backend dispatch in ModelOptNvFp4FusedMoEMethod.apply.

apply() picks the kernel path from the backend the method cached in
create_moe_runner, not from the process-wide MoE runner backend, which
speculative decoding changes after the weights were prepared. These tests pin
that for the FlashInfer TRT-LLM path, which serves the regular and the routed
TRT-LLM backend from one weight prep.

The platform check and the runner are stubbed and the layer is a bag of small
tensors, so the tests stay on CPU; the kernels are covered on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant before flashinfer_trtllm (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp4MoeQuantInfo,
)

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 32


class _Runner:
    """Records the quant_info apply() hands to the runner."""

    def __init__(self):
        self.calls = []

    def run(self, dispatch_output, quant_info):
        self.calls.append((dispatch_output, quant_info))
        return "combine-input"


def _trtllm_prepared_layer() -> SimpleNamespace:
    """A layer as align_fp4_moe_weights_for_flashinfer_trtllm leaves it: packed FP4
    weights, FP8 block scales and the TRT-LLM output scalars, incl. g1_scale_c."""

    def p(t: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(t, requires_grad=False)

    return SimpleNamespace(
        w13_weight=p(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8)
        ),
        w2_weight=p(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
        ),
        w13_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.float8_e4m3fn
            )
        ),
        w2_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16, dtype=torch.float8_e4m3fn
            )
        ),
        g1_scale_c=p(torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32)),
        g1_alphas=p(torch.full((NUM_EXPERTS,), 0.5, dtype=torch.float32)),
        g2_alphas=p(torch.full((NUM_EXPERTS,), 0.75, dtype=torch.float32)),
        w13_input_scale_quant=torch.tensor(2.0, dtype=torch.float32),
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        moe_ep_rank=0,
        intermediate_size_per_partition=INTERMEDIATE,
        # The weight alignment sets this; HIDDEN needs no padding.
        trtllm_padded_hidden_size=HIDDEN,
        # FusedMoE.__init__ sets this; apply() reads it to reject the fused
        # fallback for MegaMoE experts.
        _mega_moe_nvfp4=False,
    )


@contextmanager
def _live_backend(backend: MoeRunnerBackend):
    """The process-wide MoE runner backend, as the method sees it."""
    with mock.patch.object(
        modelopt_quant, "get_moe_runner_backend", return_value=backend
    ):
        yield


def _method_set_up_for(backend: MoeRunnerBackend) -> ModelOptNvFp4FusedMoEMethod:
    """A method constructed and given its runner while ``backend`` was the live one."""
    with (
        _live_backend(backend),
        mock.patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=True),
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(use_per_token_activation=False)
        )
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                activation="silu",
                is_gated=True,
            ),
        )
    method.runner = _Runner()
    return method


class TestNvFp4LoraAdmission(CustomTestCase):
    def _method(self, backend, capability=(10, 0)):
        with (
            _live_backend(backend),
            mock.patch.object(modelopt_quant, "is_cuda", return_value=True),
            mock.patch.object(
                modelopt_quant, "get_device_capability", return_value=capability
            ),
            mock.patch.object(
                modelopt_quant,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=capability >= (10, 0)),
            ),
        ):
            return ModelOptNvFp4FusedMoEMethod(SimpleNamespace())

    def test_actual_quant_method_rejects_non_marlin_lora(self):
        for backend in (MoeRunnerBackend.LORA_TRITON, MoeRunnerBackend.LORA_CUTEDSL):
            with self.subTest(backend=backend), _live_backend(backend):
                with mock.patch.object(modelopt_quant, "get_platform") as platform:
                    with self.assertRaisesRegex(ValueError, "lora_marlin"):
                        ModelOptNvFp4FusedMoEMethod(SimpleNamespace())
                platform.assert_not_called()
        for backend in (MoeRunnerBackend.LORA_MARLIN, MoeRunnerBackend.MARLIN):
            for capability in ((9, 0), (10, 0)):
                with self.subTest(backend=backend, capability=capability):
                    self._method(backend, capability)

    def test_checkpoint_detected_nvfp4_uses_the_same_guard(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        config = ModelOptFp4Config.from_config(
            {"quant_algo": "NVFP4", "group_size": 16, "ignore": []}
        )
        layer = FusedMoE.__new__(FusedMoE)
        for backend in (MoeRunnerBackend.LORA_TRITON, MoeRunnerBackend.LORA_CUTEDSL):
            with self.subTest(backend=backend), _live_backend(backend):
                with self.assertRaisesRegex(ValueError, "lora_marlin"):
                    config.get_quant_method(layer, "experts")

    def test_lora_rejects_nongated_before_weight_creation(self):
        layer = SimpleNamespace(moe_runner_config=SimpleNamespace(is_gated=False))
        for backend in (MoeRunnerBackend.LORA_MARLIN, MoeRunnerBackend.MARLIN):
            method = self._method(backend)
            with mock.patch.object(
                method,
                "prepare_weight_loader",
                side_effect=RuntimeError("loader reached"),
            ) as loader:
                if backend.is_lora_marlin():
                    with self.assertRaisesRegex(ValueError, "two W13 slices"):
                        method.create_weights(layer, 2, 64, 32, torch.bfloat16)
                    loader.assert_not_called()
                    self.assertFalse(hasattr(layer, "params_dtype"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "loader reached"):
                        method.create_weights(layer, 2, 64, 32, torch.bfloat16)
                    loader.assert_called_once()

    def test_lora_uses_the_existing_marlin_scale_preparation(self):
        cases = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([[1.0, 1.000001], [2.0, 2.0]]),
            torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        )
        for backend in (MoeRunnerBackend.LORA_MARLIN, MoeRunnerBackend.MARLIN):
            for w13 in cases:
                method = self._method(backend)
                w2 = torch.ones(2)
                expected = w13 if w13.ndim == 1 else w13[:, 0]
                prepared = []
                layer = SimpleNamespace(
                    num_local_experts=2,
                    moe_runner_config=SimpleNamespace(is_gated=True),
                    w13_weight_scale_2=w13,
                    w2_weight_scale_2=w2,
                )
                with (
                    self.subTest(backend=backend, w13=w13),
                    _live_backend(backend),
                    mock.patch.object(
                        modelopt_quant,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_megamoe=lambda: False),
                    ),
                    mock.patch.object(
                        modelopt_quant, "_use_nvfp4_dispatch", return_value=False
                    ),
                    mock.patch.object(
                        modelopt_quant,
                        "copy_or_rebind_param",
                        side_effect=lambda layer, name, value: setattr(
                            layer, name, value
                        ),
                    ),
                    mock.patch.object(
                        modelopt_quant,
                        "prepare_moe_nvfp4_layer_for_marlin",
                        side_effect=lambda layer: prepared.append(
                            (
                                layer.w13_weight_scale_2.clone(),
                                layer.w13_weight_scale_2.is_contiguous(),
                            )
                        ),
                    ) as prepare,
                ):
                    method.process_weights_after_loading(layer)
                    prepare.assert_called_once_with(layer)
                    scales, contiguous = prepared[0]
                    torch.testing.assert_close(scales, expected, rtol=0, atol=0)
                    self.assertTrue(contiguous)
                    self.assertIs(layer.w2_weight_scale_2, w2)


class TestNvFp4MoeDispatch(CustomTestCase):
    def test_routed_trtllm_takes_the_trtllm_path(self):
        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with self.subTest(backend=backend.value):
                method = _method_set_up_for(backend)
                layer = _trtllm_prepared_layer()
                dispatch_output = object()

                with _live_backend(backend):
                    out = method.apply(layer, dispatch_output)

                self.assertEqual(out, "combine-input")
                ((seen_dispatch, quant_info),) = method.runner.calls
                self.assertIs(seen_dispatch, dispatch_output)
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)
                self.assertEqual(
                    quant_info.g1_scale_c.data_ptr(), layer.g1_scale_c.data_ptr()
                )
                self.assertEqual(
                    quant_info.g1_alphas.data_ptr(), layer.g1_alphas.data_ptr()
                )
                self.assertEqual(quant_info.local_num_experts, NUM_EXPERTS)
                self.assertEqual(
                    quant_info.intermediate_size_per_partition, INTERMEDIATE
                )

    def test_dispatch_follows_the_backend_the_runner_was_created_for(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        for live in (
            MoeRunnerBackend.AUTO,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with self.subTest(live=live.value):
                method.runner = _Runner()

                with _live_backend(live):
                    method.apply(_trtllm_prepared_layer(), object())

                ((_, quant_info),) = method.runner.calls
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)

    def test_missing_trtllm_prep_names_the_missing_field(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        layer = _trtllm_prepared_layer()
        del layer.g1_scale_c

        with _live_backend(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED):
            with self.assertRaises(AttributeError) as ctx:
                method.apply(layer, object())

        self.assertIn("g1_scale_c", str(ctx.exception))
        self.assertEqual(method.runner.calls, [])


if __name__ == "__main__":
    unittest.main()
