"""CPU contracts for the FlashInfer MegaMOE MoeRunner integration."""

import importlib.machinery
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.flashinfer_megamoe import (
    FlashInferMegaMoeQuantInfo,
    _validate_nvfp4_fc1_alpha,
    ensure_bf16_moe_layer_for_flashinfer_megamoe,
    ensure_fp4_moe_layer_for_flashinfer_megamoe,
    ensure_mxfp8_bf16_moe_layer_for_flashinfer_megamoe,
    prepare_bf16_moe_weights_for_flashinfer_megamoe,
    prepare_fp4_moe_weights_for_flashinfer_megamoe,
    prepare_mxfp8_bf16_moe_weights_for_flashinfer_megamoe,
    run_flashinfer_megamoe,
)
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    _input_scale_to_local_experts,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferMegaMoeRunner(CustomTestCase):
    def test_registers_fused_runner_path(self):
        fused_func = FusedOpPool.get_fused_func(
            MoeA2ABackend.FLASHINFER_MEGAMOE.value,
            MoeRunnerBackend.FLASHINFER_MEGAMOE.value,
        )
        self.assertIs(fused_func, run_flashinfer_megamoe)

    @patch(
        "sglang.srt.layers.moe.moe_runner.runner.get_moe_a2a_backend",
        return_value=MoeA2ABackend.FLASHINFER_MEGAMOE,
    )
    def test_runner_does_not_own_weight_preparation(self, _):
        runner = MoeRunner(
            MoeRunnerBackend.FLASHINFER_MEGAMOE,
            MoeRunnerConfig(),
        )
        self.assertFalse(hasattr(runner, "build_flashinfer_megamoe"))
        self.assertFalse(hasattr(runner, "quant_info"))

    @patch(
        "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_MEGAMOE,
    )
    @patch("sglang.srt.layers.quantization.unquant.MoeRunner")
    def test_unquantized_method_selects_megamoe_runner(self, mock_runner, _):
        method = UnquantizedFusedMoEMethod()
        config = MoeRunnerConfig()
        method.create_moe_runner(SimpleNamespace(), config)
        mock_runner.assert_called_once_with(MoeRunnerBackend.FLASHINFER_MEGAMOE, config)

    def test_prepares_weights_then_lazily_builds_layer(self):
        preprocess_args = {}
        transformed_weights = (
            (torch.ones(2), torch.ones(3)),
            (torch.ones(4), torch.ones(5)),
        )

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_preprocess(weights, **kwargs):
            preprocess_args.update(weights=weights, **kwargs)
            return transformed_weights

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader=None)
        moe_ep = types.ModuleType("flashinfer.moe_ep")
        for name in (
            "BootstrapConfig",
            "FleetParams",
            "MegaConfig",
            "MoEEpMegaLayer",
            "MoEWeightPack",
            "Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig",
        ):
            setattr(moe_ep, name, FakeConfig)
        moe_ep.preprocess_mega_weights = fake_preprocess
        flashinfer.moe_ep = moe_ep

        layer = SimpleNamespace(
            layer_id=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=2,
            w13_weight=torch.nn.Parameter(torch.empty(1), requires_grad=False),
            w2_weight=torch.nn.Parameter(torch.empty(1), requires_grad=False),
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(torch.empty(1), requires_grad=False),
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            should_fuse_routed_scaling_factor_in_topk=False,
        )

        with (
            patch.dict(
                sys.modules,
                {"flashinfer": flashinfer, "flashinfer.moe_ep": moe_ep},
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._resolve_max_tokens_per_rank",
                return_value=1024,
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._get_moe_ep_process_group",
                return_value=object(),
            ),
        ):
            prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)
            self.assertFalse(hasattr(layer, "_flashinfer_megamoe_layer"))
            mega = ensure_fp4_moe_layer_for_flashinfer_megamoe(layer)
            self.assertIs(ensure_fp4_moe_layer_for_flashinfer_megamoe(layer), mega)

        self.assertFalse(hasattr(layer, "flashinfer_megamoe_prepared_state"))
        self.assertEqual(preprocess_args["intermediate_size"], 128)
        self.assertEqual(preprocess_args["hidden_size"], 128)
        torch.testing.assert_close(layer.w13_weight, transformed_weights[0][0])
        torch.testing.assert_close(
            layer.w13_weight_scale_inv, transformed_weights[0][1]
        )
        torch.testing.assert_close(layer.w2_weight, transformed_weights[1][0])
        torch.testing.assert_close(layer.w2_weight_scale_inv, transformed_weights[1][1])
        backend = mega.kwargs["backend"]
        self.assertFalse(backend.kwargs["preprocess_weights"])
        transformed = backend.kwargs["transformed_weights"]
        self.assertEqual(transformed[0][0].data_ptr(), layer.w13_weight.data.data_ptr())
        self.assertEqual(
            transformed[0][1].data_ptr(),
            layer.w13_weight_scale_inv.data.data_ptr(),
        )
        self.assertEqual(transformed[1][0].data_ptr(), layer.w2_weight.data.data_ptr())
        self.assertEqual(
            transformed[1][1].data_ptr(),
            layer.w2_weight_scale_inv.data.data_ptr(),
        )

    def test_prepares_bf16_weights_then_lazily_builds_layer(self):
        preprocess_args = {}
        process_group = object()
        transformed_weights = ((torch.ones(2), None), (torch.ones(4), None))

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_preprocess(weights, **kwargs):
            preprocess_args.update(weights=weights, **kwargs)
            return transformed_weights

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader=None)
        moe_ep = types.ModuleType("flashinfer.moe_ep")
        for name in (
            "BootstrapConfig",
            "FleetParams",
            "MegaConfig",
            "MoEEpMegaLayer",
            "MoEWeightPack",
            "Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig",
        ):
            setattr(moe_ep, name, FakeConfig)
        moe_ep.preprocess_bf16_cutedsl_mega_weights = fake_preprocess
        flashinfer.moe_ep = moe_ep
        layer = SimpleNamespace(
            layer_id=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=2,
            w13_weight=torch.nn.Parameter(
                torch.empty(1, dtype=torch.bfloat16), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.empty(1, dtype=torch.bfloat16), requires_grad=False
            ),
            moe_runner_config=SimpleNamespace(
                activation="silu", is_gated=True, swiglu_limit=None
            ),
        )

        with (
            patch.dict(
                sys.modules,
                {"flashinfer": flashinfer, "flashinfer.moe_ep": moe_ep},
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._resolve_max_tokens_per_rank",
                return_value=1024,
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._get_moe_ep_process_group",
                return_value=process_group,
            ),
        ):
            prepare_bf16_moe_weights_for_flashinfer_megamoe(layer)
            mega = ensure_bf16_moe_layer_for_flashinfer_megamoe(layer)
            self.assertIs(ensure_bf16_moe_layer_for_flashinfer_megamoe(layer), mega)

        self.assertEqual(preprocess_args["intermediate_size"], 128)
        self.assertEqual(preprocess_args["hidden_size"], 128)
        torch.testing.assert_close(layer.w13_weight, transformed_weights[0][0])
        torch.testing.assert_close(layer.w2_weight, transformed_weights[1][0])
        bootstrap = mega.kwargs["bootstrap"]
        self.assertIs(bootstrap.kwargs["process_group"], process_group)
        transformed = mega.kwargs["backend"].kwargs["transformed_weights"]
        self.assertIsNone(transformed[0][1])
        self.assertIsNone(transformed[1][1])

    def test_prepares_mxfp8_bf16_weights_then_lazily_builds_layer(self):
        preprocess_args = {}
        transformed_weights = (
            (
                torch.empty(2, dtype=torch.float8_e4m3fn),
                torch.empty(3, dtype=torch.uint8),
            ),
            (
                torch.empty(4, dtype=torch.float8_e4m3fn),
                torch.empty(5, dtype=torch.uint8),
            ),
        )

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_preprocess(weights, **kwargs):
            preprocess_args.update(weights=weights, **kwargs)
            return transformed_weights

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader=None)
        moe_ep = types.ModuleType("flashinfer.moe_ep")
        mixed_backend = types.ModuleType(
            "flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl"
        )
        mixed_backend.preprocess_mega_weights = fake_preprocess
        for name in (
            "BootstrapConfig",
            "FleetParams",
            "MegaConfig",
            "MoEEpMegaLayer",
            "MoEWeightPack",
            "Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
        ):
            setattr(moe_ep, name, FakeConfig)
        flashinfer.moe_ep = moe_ep
        layer = SimpleNamespace(
            layer_id=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=2,
            w13_weight=torch.nn.Parameter(
                torch.empty(1, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.empty(1, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, dtype=torch.uint8), requires_grad=False
            ),
            moe_runner_config=SimpleNamespace(
                activation="silu", is_gated=True, swiglu_limit=None
            ),
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer": flashinfer,
                    "flashinfer.moe_ep": moe_ep,
                    "flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl": mixed_backend,
                },
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._resolve_max_tokens_per_rank",
                return_value=1024,
            ),
            patch(
                "sglang.srt.layers.moe.flashinfer_megamoe._get_moe_ep_process_group",
                return_value=object(),
            ),
        ):
            prepare_mxfp8_bf16_moe_weights_for_flashinfer_megamoe(layer)
            mega = ensure_mxfp8_bf16_moe_layer_for_flashinfer_megamoe(layer)
            self.assertIs(
                ensure_mxfp8_bf16_moe_layer_for_flashinfer_megamoe(layer), mega
            )

        self.assertEqual(preprocess_args["intermediate_size"], 128)
        self.assertEqual(preprocess_args["hidden_size"], 128)
        self.assertEqual(preprocess_args["kind"], "bf16_mxfp8_e4m3")
        weights = preprocess_args["weights"].kwargs
        self.assertEqual(weights["w13_scale"].dtype, torch.uint8)
        self.assertEqual(weights["w2_scale"].dtype, torch.uint8)
        backend = mega.kwargs["backend"]
        self.assertFalse(backend.kwargs["preprocess_weights"])
        config = backend.kwargs["megakernel"]
        self.assertEqual(config.kwargs["kind"], "bf16_mxfp8_e4m3")
        transformed = backend.kwargs["transformed_weights"]
        self.assertEqual(transformed[0][0].data_ptr(), layer.w13_weight.data.data_ptr())
        self.assertEqual(transformed[1][0].data_ptr(), layer.w2_weight.data.data_ptr())
        self.assertEqual(
            transformed[0][1].data_ptr(), layer.w13_weight_scale_inv.data.data_ptr()
        )
        self.assertEqual(
            transformed[1][1].data_ptr(), layer.w2_weight_scale_inv.data.data_ptr()
        )

    def test_bf16_weight_preparation_rejects_unsupported_experts(self):
        layer = SimpleNamespace(
            hidden_size=128,
            intermediate_size_per_partition=128,
            num_experts=8,
            moe_ep_size=1,
            w13_weight=torch.nn.Parameter(torch.empty(1), requires_grad=False),
            w2_weight=torch.nn.Parameter(torch.empty(1), requires_grad=False),
            moe_runner_config=SimpleNamespace(activation="silu", is_gated=True),
        )
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            prepare_bf16_moe_weights_for_flashinfer_megamoe(layer)

    def test_fused_forward_converts_inputs_and_applies_scaling(self):
        tensor_args = {}

        class FakeTensors:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                tensor_args.update(kwargs)

        class FakeMega:
            def forward(self, _):
                return torch.ones((2, 4), dtype=torch.bfloat16)

        class FakeCombineInput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader=None)
        moe_ep = types.ModuleType("flashinfer.moe_ep")
        moe_ep.MoEEpTensors = FakeTensors
        flashinfer.moe_ep = moe_ep
        token_dispatcher = types.ModuleType("sglang.srt.layers.moe.token_dispatcher")
        token_dispatcher.StandardCombineInput = FakeCombineInput
        dispatch_output = SimpleNamespace(
            hidden_states=torch.ones((2, 4), dtype=torch.float32),
            topk_output=SimpleNamespace(
                topk_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                topk_weights=torch.ones((2, 2), dtype=torch.bfloat16),
            ),
        )
        quant_info = FlashInferMegaMoeQuantInfo(
            mega=FakeMega(),
            apply_routed_scaling_factor=True,
        )

        with patch.dict(
            sys.modules,
            {
                "flashinfer": flashinfer,
                "flashinfer.moe_ep": moe_ep,
                "sglang.srt.layers.moe.token_dispatcher": token_dispatcher,
            },
        ):
            result = run_flashinfer_megamoe(
                dispatch_output,
                quant_info,
                MoeRunnerConfig(routed_scaling_factor=0.5),
            )

        self.assertEqual(tensor_args["hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(tensor_args["topk_ids"].dtype, torch.int32)
        self.assertEqual(tensor_args["topk_weights"].dtype, torch.float32)
        torch.testing.assert_close(
            result.hidden_states,
            torch.full((2, 4), 0.5, dtype=torch.bfloat16),
        )

    def test_fused_forward_passes_zero_token_rank(self):
        tensor_args = {}

        class FakeTensors:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                tensor_args.update(kwargs)

        class FakeMega:
            def forward(self, tensors):
                return torch.ones(
                    (tensors.kwargs["hidden_states"].shape[0], 4),
                    dtype=torch.bfloat16,
                )

        class FakeCombineInput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader=None)
        moe_ep = types.ModuleType("flashinfer.moe_ep")
        moe_ep.MoEEpTensors = FakeTensors
        flashinfer.moe_ep = moe_ep
        token_dispatcher = types.ModuleType("sglang.srt.layers.moe.token_dispatcher")
        token_dispatcher.StandardCombineInput = FakeCombineInput
        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 4), dtype=torch.bfloat16),
            topk_output=SimpleNamespace(
                topk_ids=torch.empty((0, 2), dtype=torch.int32),
                topk_weights=torch.empty((0, 2), dtype=torch.bfloat16),
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "flashinfer": flashinfer,
                "flashinfer.moe_ep": moe_ep,
                "sglang.srt.layers.moe.token_dispatcher": token_dispatcher,
            },
        ):
            result = run_flashinfer_megamoe(
                dispatch_output,
                FlashInferMegaMoeQuantInfo(mega=FakeMega()),
                MoeRunnerConfig(),
            )

        self.assertEqual(tensor_args["hidden_states"].shape, (0, 4))
        self.assertEqual(tensor_args["topk_ids"].shape, (0, 2))
        self.assertEqual(tensor_args["topk_weights"].shape, (0, 2))
        self.assertEqual(result.hidden_states.shape, (0, 4))


class TestNvfp4MegaMoeScaleReuse(CustomTestCase):
    """Guard for reusing per-expert ModelOpt scales in the NVFP4 MegaMOE path.

    fc1_alpha / fc2_alpha / fc1_norm_const are read straight off the layer as
    g1_alphas / g2_alphas / w2_input_scale_quant. ModelOpt derives the w2 scales
    per-expert for megamoe (via _input_scale_to_local_experts), so these tests pin
    the canonical params against a direct per-expert derivation.
    """

    @staticmethod
    def _make_layer(w2_input_scale, *, num_experts, num_local, moe_ep_rank):
        # Per-local-expert gate/up FC1 alphas (gate == up, as the kernel requires).
        g1 = torch.rand(num_local, dtype=torch.float32) + 0.5
        w2_weight_scale_2 = torch.rand(num_local, dtype=torch.float32) + 0.5
        layer = SimpleNamespace(
            num_experts=num_experts,
            num_local_experts=num_local,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=num_experts // num_local,
            moe_runner_config=SimpleNamespace(is_gated=True),
            g1_alphas=g1,
            g1_alphas_up=g1.clone(),
            w2_input_scale=w2_input_scale.to(torch.float32),
            w2_weight_scale_2=w2_weight_scale_2,
        )
        # Mirror ModelOpt's megamoe branch: per-expert (local) w2 input scale.
        w2_local = _input_scale_to_local_experts(
            layer.w2_input_scale, num_local, num_experts, moe_ep_rank
        )
        layer.g2_alphas = (w2_local * w2_weight_scale_2).to(torch.float32)
        layer.w2_input_scale_quant = (1 / w2_local).to(torch.float32)
        return layer

    @staticmethod
    def _reference_scales(layer):
        """Direct per-expert FC2 dequant alpha and FC2-input renorm."""
        start = layer.moe_ep_rank * layer.num_local_experts
        end = start + layer.num_local_experts
        w2 = layer.w2_input_scale
        if w2.shape == (layer.num_experts,):
            w2 = w2[start:end]
        fc2_alpha = (w2 * layer.w2_weight_scale_2).to(torch.float32)
        fc1_norm_const = (1 / w2).to(torch.float32)
        return fc2_alpha, fc1_norm_const

    def _cases(self):
        return {
            "non_ep_nonuniform": self._make_layer(
                torch.tensor([0.10, 0.25, 0.40, 0.55]),
                num_experts=4,
                num_local=4,
                moe_ep_rank=0,
            ),
            "non_ep_uniform": self._make_layer(
                torch.full((4,), 0.30),
                num_experts=4,
                num_local=4,
                moe_ep_rank=0,
            ),
            "ep_global_nonuniform": self._make_layer(
                torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                num_experts=8,
                num_local=4,
                moe_ep_rank=1,
            ),
        }

    def test_canonical_scales_match_per_expert(self):
        # g2_alphas / w2_input_scale_quant must equal the per-expert FC2 scales the
        # mega kernel needs (fc2_alpha / fc1_norm_const) across non-EP and EP cases.
        for name, layer in self._cases().items():
            with self.subTest(case=name):
                _validate_nvfp4_fc1_alpha(layer)  # gate == up, must not raise
                ref_fc2, ref_norm = self._reference_scales(layer)
                torch.testing.assert_close(layer.g2_alphas, ref_fc2)
                torch.testing.assert_close(layer.w2_input_scale_quant, ref_norm)

    def test_input_scale_to_local_experts(self):
        # scalar -> expanded; local vector -> unchanged; global vector -> EP slice.
        torch.testing.assert_close(
            _input_scale_to_local_experts(torch.tensor(0.3), 4, 8, 1),
            torch.full((4,), 0.3),
        )
        local = torch.tensor([0.1, 0.2, 0.3, 0.4])
        torch.testing.assert_close(_input_scale_to_local_experts(local, 4, 4, 0), local)
        glob = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        torch.testing.assert_close(
            _input_scale_to_local_experts(glob, 4, 8, 1),
            torch.tensor([0.5, 0.6, 0.7, 0.8]),
        )

    def test_validate_fc1_alpha_rejects_gate_up_mismatch(self):
        layer = self._cases()["non_ep_uniform"]
        layer.g1_alphas_up = layer.g1_alphas + 1.0
        with self.assertRaises(ValueError):
            _validate_nvfp4_fc1_alpha(layer)


if __name__ == "__main__":
    unittest.main()
