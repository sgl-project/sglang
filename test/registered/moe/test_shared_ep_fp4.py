import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.shared_ep import (
    SharedEpQuantCapability,
    SharedEpQuantInfo,
    SharedEpQuantization,
    SharedEpScaleLayout,
    SharedEpWeightLayout,
)
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatcher,
    _profile_quantization,
)
from sglang.srt.layers.moe.shared_ep.fp4 import (
    publish_bf16_owner_input,
    run_shared_ep_mxfp4,
)
from sglang.srt.layers.moe.shared_ep.kernels import RoutePreparation
from sglang.srt.layers.moe.shared_ep.layout import SharedEpInputViews
from sglang.srt.layers.moe.shared_ep.profiles import select_profile
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")


_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}


def _small_profile():
    return SimpleNamespace(
        name="test_mxfp4",
        quantization=SharedEpQuantization.MXFP4,
        hidden_size=128,
        intermediate_size=128,
        top_k=2,
        num_local_experts=2,
        block_size_m=64,
        w13_kernel_config=lambda _tokens: _CONFIG,
        w2_kernel_config=lambda _tokens: _CONFIG,
    )


def _small_quant_info() -> SharedEpQuantInfo:
    return SharedEpQuantInfo(
        w13_weight=torch.zeros((2, 256, 64), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 128, 64), dtype=torch.uint8),
        w13_scale=torch.zeros((2, 256, 4), dtype=torch.uint8),
        w2_scale=torch.zeros((2, 128, 4), dtype=torch.uint8),
        block_shape=(1, 32),
        fallback_quant_info=Mock(),
        fallback_backend=MoeRunnerBackend.AITER,
        quantization=SharedEpQuantization.MXFP4,
        weight_layout=SharedEpWeightLayout.CANONICAL,
        scale_layout=SharedEpScaleLayout.CANONICAL,
        weight_group_size=32,
        scale_format="e8m0",
        capabilities=frozenset({SharedEpQuantCapability.CANONICAL_MXFP4}),
        fallback_weight_layout=SharedEpWeightLayout.AITER_SHUFFLED,
        fallback_scale_layout=SharedEpScaleLayout.AITER_SHUFFLED,
        fallback_uses_duplicate_tensors=True,
    )


class TestSharedEpMXFP4Contracts(unittest.TestCase):
    def test_bf16_owner_publish_is_direct_and_clears_unused_routes(self):
        backing = torch.zeros((4, 256), dtype=torch.bfloat16)
        target = SharedEpInputViews(
            activations=backing[:, :128],
            scales=None,
            topk_ids=torch.full((4, 2), 77, dtype=torch.int32),
            topk_weights=torch.full((4, 2), 9.0, dtype=torch.float32),
        )
        source = torch.arange(2 * 128, dtype=torch.bfloat16).view(2, 128)
        ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.int64)
        weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)

        publish_bf16_owner_input(
            target,
            source=source,
            source_ids=ids,
            source_weights=weights,
        )

        self.assertEqual(target.activations.stride(), (256, 1))
        torch.testing.assert_close(target.activations[:2], source)
        self.assertTrue(torch.equal(target.topk_ids[:2], ids.to(torch.int32)))
        self.assertTrue(torch.equal(target.topk_weights[:2], weights))
        self.assertTrue(torch.all(target.topk_ids[2:] == -1))
        self.assertTrue(torch.all(target.topk_weights[2:] == 0))

    def test_mxfp4_metadata_fails_closed_on_shuffled_scales(self):
        info = _small_quant_info()
        info.require_decode_capability(SharedEpQuantCapability.CANONICAL_MXFP4)
        self.assertTrue(info.fallback_uses_duplicate_tensors)
        self.assertIs(
            info.fallback_weight_layout,
            SharedEpWeightLayout.AITER_SHUFFLED,
        )

        info.w13_scale.is_shuffled = True
        with self.assertRaisesRegex(ValueError, "pre-shuffled"):
            info.require_decode_capability(SharedEpQuantCapability.CANONICAL_MXFP4)

    def test_quark_exposes_canonical_decode_and_explicit_prefill_duplicate(self):
        import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe as quark_module

        scheme = quark_module.QuarkW4A4MXFp4MoE.__new__(quark_module.QuarkW4A4MXFp4MoE)
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.arange(8, dtype=torch.uint8).view(1, 2, 4),
                requires_grad=False,
            ),
            w2_weight=torch.nn.Parameter(
                torch.arange(8, dtype=torch.uint8).view(1, 2, 4),
                requires_grad=False,
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.arange(4, dtype=torch.uint8).view(1, 2, 2),
                requires_grad=False,
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.arange(4, dtype=torch.uint8).view(1, 2, 2),
                requires_grad=False,
            ),
        )
        canonical = tuple(
            tensor.detach().clone()
            for tensor in (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            )
        )

        with (
            patch.object(quark_module, "_use_aiter", True),
            patch.object(quark_module, "_is_shuffle_moe_mxfp4", True),
            patch.object(
                quark_module,
                "shuffle_weight",
                side_effect=lambda value, _layout: value.detach().clone() + 10,
                create=True,
            ),
            patch.object(
                quark_module,
                "e8m0_shuffle",
                side_effect=lambda value: value.detach().clone() + 20,
                create=True,
            ),
        ):
            scheme._prepare_shared_ep_aiter_fallback(layer)

        for actual, expected in zip(
            (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            ),
            canonical,
        ):
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(actual.is_shuffled)
        self.assertTrue(layer._shared_ep_aiter_w13_weight.is_shuffled)
        self.assertTrue(layer._shared_ep_aiter_w13_weight_scale.is_shuffled)

        scheme.runner = SimpleNamespace(
            runner_backend=MoeRunnerBackend.AITER,
            run=lambda _dispatch, quant_info: quant_info,
        )
        layer.dispatcher = SimpleNamespace(expert_mask_gpu=None)
        with patch.object(
            quark_module,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.SHARED_EP,
        ):
            info = scheme.apply_weights(layer, object())
        self.assertIsInstance(info, SharedEpQuantInfo)
        self.assertIn(
            SharedEpQuantCapability.CANONICAL_MXFP4,
            info.capabilities,
        )
        self.assertTrue(info.fallback_uses_duplicate_tensors)
        self.assertIs(info.w13_weight, layer.w13_weight)
        self.assertIs(info.w13_scale, layer.w13_weight_scale)

    def test_fp8_fp4_experts_publish_canonical_mxfp4_metadata(self):
        import sglang.srt.layers.quantization.fp8 as fp8_module

        method = fp8_module.Fp8MoEMethod.__new__(fp8_module.Fp8MoEMethod)
        method.is_fp4_expert = True
        method.weight_block_size = [128, 128]
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)
        layer = SimpleNamespace(
            w13_weight=torch.zeros((2, 256, 64), dtype=torch.uint8),
            w2_weight=torch.zeros((2, 128, 64), dtype=torch.uint8),
            w13_weight_scale_inv=torch.zeros((2, 256, 4), dtype=torch.uint8),
            w2_weight_scale_inv=torch.zeros((2, 128, 4), dtype=torch.uint8),
            _shared_ep_aiter_w13_weight=torch.ones(
                (2, 256, 64),
                dtype=torch.uint8,
            ),
            _shared_ep_aiter_w2_weight=torch.ones(
                (2, 128, 64),
                dtype=torch.uint8,
            ),
        )
        with patch.object(
            fp8_module,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.SHARED_EP,
        ):
            info = method._wrap_shared_ep_quant_info(layer, Mock())

        self.assertIs(info.quantization, SharedEpQuantization.MXFP4)
        self.assertEqual(info.block_shape, (1, 32))
        self.assertEqual(info.weight_group_size, 32)
        self.assertIn(
            SharedEpQuantCapability.CANONICAL_MXFP4,
            info.capabilities,
        )
        self.assertTrue(info.fallback_uses_duplicate_tensors)
        info.require_decode_capability(SharedEpQuantCapability.CANONICAL_MXFP4)

    def test_cpu_adapter_preserves_canonical_route_ids_and_direct_output(self):
        profile = _small_profile()
        global_ids = torch.tensor(
            [[[0, 1], [1, 0], [0, 1]]],
            dtype=torch.int32,
        )
        global_weights = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
            dtype=torch.float32,
        )
        activations = torch.ones((1, 3, 128), dtype=torch.bfloat16)
        global_output = torch.zeros((1, 3, 2, 128), dtype=torch.bfloat16)
        output_epoch = Mock()
        state = SimpleNamespace(
            global_input=SimpleNamespace(
                topk_ids=global_ids,
                topk_weights=global_weights,
            ),
            global_output=global_output,
            local_output=global_output[0],
            input_epoch=SimpleNamespace(
                allocation=SimpleNamespace(local_storage=object()),
                epoch=object(),
            ),
            output_epoch=output_epoch,
        )
        dispatch_output = SimpleNamespace(
            profile=profile,
            state=state,
            hidden_states=activations,
            num_tokens=3,
            local_expert_start=0,
        )
        sorted_routes = torch.full((128,), 6, dtype=torch.int32)
        sorted_routes[:6] = torch.tensor([5, 0, 3, 2, 4, 1], dtype=torch.int32)
        routes = RoutePreparation(
            local_ids=global_ids.clone(),
            local_weights=global_weights.clone(),
            sorted_token_ids=sorted_routes,
            expert_ids=torch.tensor([0, 1], dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([128], dtype=torch.int32),
        )
        seen = {}

        def fake_w13(*args, **kwargs):
            seen["w13_routes"] = args[3]
            seen["swiglu_limit"] = kwargs["swiglu_limit"]
            kwargs["out"].fill_(1)
            return kwargs["out"]

        def fake_w2(*args, **kwargs):
            seen["w2_routes"] = args[4]
            seen["direct_out"] = kwargs["out"]
            out = kwargs["out"]
            for route in range(6):
                out[route].fill_(route + 1)
            return out

        def fake_reduce(source, *, num_tokens):
            self.assertIs(source, state.local_output)
            return source[:num_tokens].float().sum(dim=1).to(torch.bfloat16)

        with (
            patch(
                "sglang.srt.layers.moe.shared_ep.fp4.prepare_routes",
                return_value=routes,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.fp4._load_aiter_mxfp4_ops",
                return_value=(fake_w13, fake_w2),
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.fp4.reduce_owner_output",
                side_effect=fake_reduce,
            ),
        ):
            result = run_shared_ep_mxfp4(
                dispatch_output,
                _small_quant_info(),
                MoeRunnerConfig(
                    activation="silu",
                    is_gated=True,
                    swiglu_limit=10.0,
                ),
            )

        self.assertIs(seen["w13_routes"], sorted_routes)
        self.assertEqual(seen["swiglu_limit"], 10.0)
        self.assertIs(seen["w2_routes"], sorted_routes)
        self.assertEqual(
            seen["direct_out"].data_ptr(),
            global_output.data_ptr(),
        )
        expected = torch.tensor([3, 7, 11], dtype=torch.bfloat16)[:, None].expand(
            3, 128
        )
        torch.testing.assert_close(result.hidden_states, expected)
        output_epoch.publish.assert_called_once_with()
        output_epoch.wait_all.assert_called_once_with()

    def test_dsv4_pro_profile_metadata_matches_packed_shapes(self):
        config = MoeRunnerConfig(
            hidden_size=7168,
            intermediate_size_per_partition=3072,
            top_k=6,
            num_experts=384,
            num_local_experts=48,
            params_dtype=torch.bfloat16,
        )
        profile = select_profile(
            config,
            platform="rocm",
            capability=(9, 5),
            ep_size=8,
            quantization=SharedEpQuantization.MXFP4,
            block_shape=(1, 32),
            max_tokens_per_rank=32,
        )
        self.assertEqual(
            (
                profile.num_local_experts,
                profile.intermediate_size * 2,
                profile.hidden_size // 2,
            ),
            (48, 6144, 3584),
        )
        self.assertEqual(
            (
                profile.num_local_experts,
                profile.hidden_size,
                profile.intermediate_size // 32,
            ),
            (48, 7168, 96),
        )
        self.assertEqual(
            _profile_quantization(config),
            (SharedEpQuantization.MXFP4, (1, 32)),
        )

        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = profile
        descriptor = dispatcher._validate_mxfp4_quant_config(
            {
                "shared_ep_quantization": "mxfp4",
                "shared_ep_weight_layout": "canonical",
                "shared_ep_scale_layout": "canonical",
                "block_shape": (1, 32),
                "weight_group_size": 32,
                "scale_format": "e8m0",
                "weight_dtype": getattr(
                    torch,
                    "float4_e2m1fn_x2",
                    torch.uint8,
                ),
                "weight_scale_dtype": getattr(
                    torch,
                    "float8_e8m0fnu",
                    torch.uint8,
                ),
                "w13_shape": (48, 6144, 3584),
                "w2_shape": (48, 7168, 1536),
                "w13_scale_shape": (48, 6144, 224),
                "w2_scale_shape": (48, 7168, 96),
                "fallback_uses_duplicate_tensors": True,
            }
        )
        self.assertEqual(descriptor[0], "mxfp4")
        self.assertTrue(descriptor[-1])

    def test_gfx950_aiter_route_indexing_and_direct_output(self):
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.skipTest("requires a ROCm GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")
        try:
            from aiter.ops.triton.moe.shared_ep_mxfp4 import (
                shared_ep_mxfp4_route_rows,
                shared_ep_mxfp4_w2,
                shared_ep_mxfp4_w13,
            )
        except ImportError as error:
            self.skipTest(f"requires the SharedEP AITER API: {error}")

        device = torch.device("cuda")
        route_ids = torch.tensor([5, 0, 3, 2], dtype=torch.int32, device=device)
        w13_rows, w13_out_rows = shared_ep_mxfp4_route_rows(
            route_ids,
            stage="w13",
            top_k=2,
        )
        self.assertTrue(
            torch.equal(
                w13_rows.cpu(),
                torch.tensor([2, 0, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(w13_out_rows, route_ids))

        sorted_routes = torch.full((128,), 6, dtype=torch.int32, device=device)
        sorted_routes[:3] = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
        sorted_routes[64:67] = torch.tensor(
            [1, 2, 5],
            dtype=torch.int32,
            device=device,
        )
        experts = torch.tensor([0, 1], dtype=torch.int32, device=device)
        num_padded = torch.tensor([128], dtype=torch.int32, device=device)
        activations = torch.full(
            (3, 128),
            1 / 64,
            dtype=torch.bfloat16,
            device=device,
        )
        w13 = torch.full((2, 256, 64), 0x22, dtype=torch.uint8, device=device)
        s13 = torch.full((2, 256, 4), 127, dtype=torch.uint8, device=device)
        intermediate = shared_ep_mxfp4_w13(
            activations,
            w13,
            s13,
            sorted_routes,
            experts,
            num_padded,
            top_k=2,
            route_block_size=64,
            config=_CONFIG,
        )
        w2 = torch.full((2, 128, 64), 0x22, dtype=torch.uint8, device=device)
        s2 = torch.full((2, 128, 4), 127, dtype=torch.uint8, device=device)
        route_weights = torch.ones((3, 2), dtype=torch.float32, device=device)
        owner_slots = torch.empty(
            (1, 3, 2, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        direct = shared_ep_mxfp4_w2(
            intermediate,
            w2,
            s2,
            route_weights,
            sorted_routes,
            experts,
            num_padded,
            top_k=2,
            route_block_size=64,
            out=owner_slots.view(6, 128),
            config=_CONFIG,
        )
        self.assertEqual(direct.data_ptr(), owner_slots.data_ptr())
        self.assertTrue(torch.isfinite(owner_slots).all().item())


if __name__ == "__main__":
    unittest.main()
