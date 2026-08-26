"""MoE weight gating plus ROCm AITER padding/backend regressions."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 640
BLOCK_N = 128
BLOCK_K = 128


class _RecordingLayer:
    """Collects the parameters ``create_fp8_moe_weight_`` registers."""

    def __init__(self, is_gated: bool):
        self.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        self.params = {}

    def register_parameter(self, name, param):
        self.params[name] = param


def _create_weights(
    is_gated: bool,
    block_quant: bool,
    *,
    intermediate: int = INTERMEDIATE,
    backend=None,
    use_aiter: bool = False,
    is_hip: bool = False,
    tp_size: int = 1,
    moe_tp_size: int = 1,
):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    layer = _RecordingLayer(is_gated)
    quant_config = MagicMock(
        weight_block_size=[BLOCK_N, BLOCK_K],
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=False,
    )

    if backend is None:
        backend = MoeRunnerBackend.AUTO

    with (
        patch.object(fp8_quant, "get_parallel") as parallel,
        patch.object(
            fp8_quant,
            "will_use_aiter_moe",
            return_value=is_hip
            and (backend.is_aiter() or (backend.is_auto() and use_aiter)),
        ),
    ):
        parallel.return_value.tp_size = tp_size
        parallel.return_value.moe_tp_size = moe_tp_size
        fp8_quant.Fp8MoEMethod.create_fp8_moe_weight_(
            layer=layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=intermediate,
            block_quant=block_quant,
            quant_config=quant_config,
            use_mxfp8=False,
            is_checkpoint_fp8_serialized=False,
            is_fp4_expert=False,
            params_dtype=torch.bfloat16,
        )
    return layer.params


class TestFp8MoEWeightGating(CustomTestCase):
    def test_gated_fuses_gate_and_up(self):
        params = _create_weights(is_gated=True, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], 2 * INTERMEDIATE)
        self.assertEqual(
            params["w13_weight_scale_inv"].shape[1], 2 * (INTERMEDIATE // BLOCK_N)
        )

    def test_non_gated_w13_holds_up_only(self):
        # Regression: w13 was always sized 2*intermediate, so the upper half
        # stayed uninitialised for NemotronH.
        params = _create_weights(is_gated=False, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], INTERMEDIATE)

    def test_non_gated_block_scale_matches_weight(self):
        params = _create_weights(is_gated=False, block_quant=True)
        weight_rows = params["w13_weight"].shape[1]
        scale_rows = params["w13_weight_scale_inv"].shape[1]
        self.assertEqual(scale_rows * BLOCK_N, weight_rows)

    def test_non_gated_per_tensor_scale_is_single(self):
        # One shard means one scale per expert; nothing to fuse afterwards.
        params = _create_weights(is_gated=False, block_quant=False)
        self.assertEqual(params["w13_weight_scale"].shape, (NUM_EXPERTS, 1))

    def test_explicit_aiter_backend_pads_block_fp8_without_global_aiter(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        params = _create_weights(
            is_gated=True,
            block_quant=True,
            intermediate=80,
            backend=MoeRunnerBackend.AITER,
            use_aiter=False,
            is_hip=True,
        )

        self.assertEqual(params["w13_weight"].shape[1], 2 * BLOCK_N)
        self.assertEqual(params["w2_weight"].shape[2], BLOCK_K)
        self.assertTrue(params["w13_weight"].weight_padded)
        self.assertTrue(params["w2_weight"].weight_padded)

    def test_explicit_aiter_backend_rejects_unaligned_moe_tp_block_split(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        with self.assertRaisesRegex(ValueError, "not divisible"):
            _create_weights(
                is_gated=True,
                block_quant=True,
                intermediate=80,
                backend=MoeRunnerBackend.AITER,
                is_hip=True,
                tp_size=2,
                moe_tp_size=2,
            )

    def test_explicit_aiter_backend_allows_ep_with_unaligned_expert(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        params = _create_weights(
            is_gated=True,
            block_quant=True,
            intermediate=80,
            backend=MoeRunnerBackend.AITER,
            is_hip=True,
            tp_size=2,
            moe_tp_size=1,
        )

        self.assertEqual(params["w13_weight"].shape[1], 2 * BLOCK_N)
        self.assertEqual(params["w2_weight"].shape[2], BLOCK_K)

    def test_explicit_aiter_backend_uses_padded_checkpoint_loading(self):
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

        layer = object.__new__(fused_moe_layer.FusedMoE)
        layer.w2_weight = SimpleNamespace(weight_padded=True)
        layer.use_flashinfer_trtllm_moe = False

        with (
            patch.object(fused_moe_layer, "_is_cpu", False),
            patch.object(fused_moe_layer, "will_use_aiter_moe", return_value=True),
        ):
            self.assertTrue(fused_moe_layer.FusedMoE.use_padded_loading.func(layer))

        with (
            patch.object(fused_moe_layer, "_is_cpu", False),
            patch.object(fused_moe_layer, "will_use_aiter_moe", return_value=False),
        ):
            self.assertFalse(fused_moe_layer.FusedMoE.use_padded_loading.func(layer))

    def test_explicit_aiter_backend_loads_and_zeroes_padded_fp8_experts(self):
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

        layer = object.__new__(fused_moe_layer.FusedMoE)
        layer.w2_weight = SimpleNamespace(weight_padded=True)
        layer.use_flashinfer_trtllm_moe = False
        layer.use_presharded_weights = False
        layer.use_triton_kernels = False
        layer.moe_tp_size = 1
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        layer.quant_method = SimpleNamespace(load_up_proj_weight_first=False)
        layer.quant_config = None

        w13 = torch.full((2 * BLOCK_N, 4), -1.0)
        w1 = torch.arange(80 * 4, dtype=torch.float32).reshape(80, 4)
        w3 = w1 + 1000
        w2 = torch.full((4, BLOCK_K), -1.0)
        loaded_w2 = torch.arange(4 * 80, dtype=torch.float32).reshape(4, 80)

        with (
            patch.object(fused_moe_layer, "_is_cpu", False),
            patch.object(fused_moe_layer, "will_use_aiter_moe", return_value=True),
        ):
            layer._load_w13(w13, 0, "w1", w1, tp_rank=0)
            layer._load_w13(w13, 0, "w3", w3, tp_rank=0)
            layer._load_w2(w2, 1, "w2", loaded_w2, tp_rank=0)

        torch.testing.assert_close(w13[:80], w1)
        torch.testing.assert_close(w13[80:BLOCK_N], torch.zeros(BLOCK_N - 80, 4))
        torch.testing.assert_close(w13[BLOCK_N : BLOCK_N + 80], w3)
        torch.testing.assert_close(w13[BLOCK_N + 80 :], torch.zeros(BLOCK_N - 80, 4))
        torch.testing.assert_close(w2[:, :80], loaded_w2)
        torch.testing.assert_close(w2[:, 80:], torch.zeros(4, BLOCK_K - 80))

    def test_explicit_triton_backend_does_not_inherit_global_aiter_layout(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        with self.assertRaisesRegex(ValueError, "not divisible"):
            _create_weights(
                is_gated=True,
                block_quant=True,
                intermediate=80,
                backend=MoeRunnerBackend.TRITON,
                use_aiter=True,
                is_hip=True,
            )

    def test_effective_aiter_moe_honors_explicit_runner(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

        with (
            patch.object(moe_utils, "is_hip", return_value=True),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AITER,
            ),
            patch.object(
                moe_utils,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            envs.SGLANG_USE_AITER.override(False),
            envs.SGLANG_INT4_WEIGHT.override(False),
        ):
            self.assertTrue(moe_utils.will_use_aiter_moe())

        with (
            patch.object(moe_utils, "is_hip", return_value=True),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            envs.SGLANG_USE_AITER.override(True),
        ):
            self.assertFalse(moe_utils.will_use_aiter_moe())

    def test_effective_aiter_moe_is_disabled_off_hip(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import utils as moe_utils

        with (
            patch.object(moe_utils, "is_hip", return_value=False),
            patch.object(moe_utils, "get_moe_runner_backend") as get_backend,
            envs.SGLANG_USE_AITER.override(True),
            envs.SGLANG_INT4_WEIGHT.override(True),
        ):
            self.assertFalse(moe_utils.will_use_aiter_moe())

        get_backend.assert_not_called()

    def test_explicit_aiter_rejects_incompatible_a2a_backend(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

        with (
            patch.object(moe_utils, "is_hip", return_value=True),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AITER,
            ),
            patch.object(
                moe_utils,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.FLASHINFER,
            ),
            envs.SGLANG_USE_AITER.override(False),
            self.assertRaisesRegex(ValueError, "incompatible.*flashinfer"),
        ):
            moe_utils.will_use_aiter_moe()

    def test_explicit_aiter_requires_global_switch_for_distributed_dispatch(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

        with (
            patch.object(moe_utils, "is_hip", return_value=True),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AITER,
            ),
            patch.object(
                moe_utils,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            envs.SGLANG_USE_AITER.override(False),
            self.assertRaisesRegex(ValueError, "currently supported only"),
        ):
            moe_utils.will_use_aiter_moe()

    def test_quantized_moe_compatibility_accepts_aiter_padding_with_ep(self):
        from sglang.srt.model_executor.model_runner_components import moe_ep_setup

        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config={"weight_block_size": [BLOCK_N, BLOCK_K]}
            ),
            hf_text_config=SimpleNamespace(moe_intermediate_size=640),
        )

        with patch.object(moe_ep_setup, "will_use_aiter_moe", return_value=True):
            moe_ep_setup.check_quantized_moe_compatibility(
                model_config=model_config,
                tp_size=8,
                moe_ep_size=8,
                moe_dp_size=1,
            )

        with (
            patch.object(moe_ep_setup, "will_use_aiter_moe", return_value=True),
            self.assertRaisesRegex(ValueError, "split checkpoint quantization blocks"),
        ):
            moe_ep_setup.check_quantized_moe_compatibility(
                model_config=model_config,
                tp_size=8,
                moe_ep_size=1,
                moe_dp_size=1,
            )

        with (
            patch.object(moe_ep_setup, "will_use_aiter_moe", return_value=False),
            self.assertRaisesRegex(ValueError, "weight_block_size_n=128"),
        ):
            moe_ep_setup.check_quantized_moe_compatibility(
                model_config=model_config,
                tp_size=8,
                moe_ep_size=1,
                moe_dp_size=1,
            )


if __name__ == "__main__":
    unittest.main()
