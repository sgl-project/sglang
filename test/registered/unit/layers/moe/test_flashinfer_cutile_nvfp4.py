"""CPU tests for the FlashInfer cuTile NVFP4 W4A4 MoE runner integration.

Eligibility, weight-layout conversion and the forward contract are host-side, so
they run on CPU with the device mocked. FlashInfer's own weight preparation is
pure torch; the kernel launch is replaced by a recording stand-in.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import importlib.metadata
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups import overrides as arg_overrides
from sglang.srt.arg_groups.choices import MOE_RUNNER_BACKEND_CHOICES
from sglang.srt.arg_groups.model_overrides import minimax_m3 as minimax_m3_overrides
from sglang.srt.layers.moe.moe_runner import flashinfer_cutile as cutile
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import PackedTopKOutput, StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import minimax_m3_nvfp4
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

try:
    import flashinfer.cutile
    from flashinfer.fused_moe import SwiGLU

    _HAS_CUTILE_API = True
except ImportError:
    _HAS_CUTILE_API = False


def _runner_config(**overrides) -> MoeRunnerConfig:
    fields = dict(
        num_experts=32,
        num_local_experts=32,
        hidden_size=6144,
        intermediate_size_per_partition=3072,
        top_k=4,
        num_fused_shared_experts=0,
        activation="silu",
        is_gated=True,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    fields.update(overrides)
    return MoeRunnerConfig(**fields)


def _checkpoint_layer(num_experts=2, hidden=128, intermediate=128, seed=0):
    """Random experts in the compressed-tensors layout: GEMM1 rows are [gate; up]."""
    g = torch.Generator().manual_seed(seed)

    def packed(rows, cols):
        return torch.randint(
            0, 256, (num_experts, rows, cols // 2), generator=g, dtype=torch.uint8
        )

    def block_scale(rows, cols):
        return (torch.rand(num_experts, rows, cols // 16, generator=g) + 0.5).to(
            torch.float8_e4m3fn
        )

    return dict(
        w13_packed=packed(2 * intermediate, hidden),
        w13_block_scale=block_scale(2 * intermediate, hidden),
        w13_weight_global_scale=torch.rand(num_experts, 2, generator=g) * 900 + 100,
        w13_input_global_scale=torch.ones(num_experts, 2),
        w2_packed=packed(hidden, intermediate),
        w2_block_scale=block_scale(hidden, intermediate),
        w2_weight_global_scale=torch.rand(num_experts, generator=g) * 900 + 100,
        w2_input_global_scale=torch.ones(num_experts),
    )


class TestPlatformEligibility(CustomTestCase):
    def _reason(self, device_sm, *, version="0.7.3", tile_available=True):
        with (
            override_platform(is_cuda=True, device_sm=device_sm),
            mock.patch.object(importlib.metadata, "version", return_value=version),
            mock.patch.object(
                flashinfer.cutile, "is_cuda_tile_available", return_value=tile_available
            ),
        ):
            return cutile.cutile_nvfp4_platform_unsupported_reason()

    @unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
    def test_only_sm120_and_sm121_are_eligible(self):
        self.assertIsNone(self._reason(120))
        self.assertIsNone(self._reason(121))
        # SM100/SM110 are Blackwell too but have no cuTile NVFP4 W4A4 kernels.
        for device_sm in (80, 89, 90, 100, 103, 110):
            self.assertIn(f"sm_{device_sm}", self._reason(device_sm))

    @unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
    def test_old_flashinfer_and_missing_toolchain_are_reported(self):
        self.assertIn("0.7.1", self._reason(120, version="0.7.1"))
        # A pre-release of the first supporting version is still too old.
        self.assertIn("0.7.2rc6", self._reason(120, version="0.7.2rc6"))
        self.assertIsNone(self._reason(120, version="0.7.2.post1"))
        self.assertIn("tileiras", self._reason(120, tile_available=False))

    def test_non_cuda_and_missing_flashinfer_do_not_raise(self):
        with override_platform(is_cuda=False, device_sm=0):
            self.assertIn("CUDA", cutile.cutile_nvfp4_platform_unsupported_reason())
        with (
            override_platform(is_cuda=True, device_sm=120),
            mock.patch.object(
                importlib.metadata,
                "version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ),
        ):
            self.assertIn(
                "not installed", cutile.cutile_nvfp4_platform_unsupported_reason()
            )


class TestLayerEligibility(CustomTestCase):
    def test_minimax_m31_shapes_are_eligible(self):
        # TP1 and the TP2 shard of the M3.1 experts (intermediate 3072 / 1536).
        for intermediate in (3072, 1536):
            config = _runner_config(intermediate_size_per_partition=intermediate)
            self.assertIsNone(cutile.cutile_nvfp4_config_unsupported_reason(config))

    def test_unsupported_layers_are_rejected(self):
        cases = {
            "expert parallelism": dict(num_local_experts=16),
            "fused shared": dict(num_fused_shared_experts=1),
            "gated SiLU": dict(activation="relu2", is_gated=False),
            "clamp limit": dict(gemm1_clamp_limit=None),
            "alpha": dict(gemm1_alpha=None),
            "multiples of 64": dict(intermediate_size_per_partition=1000),
            "no_combine": dict(no_combine=True),
        }
        for expected, overrides in cases.items():
            reason = cutile.cutile_nvfp4_config_unsupported_reason(
                _runner_config(**overrides)
            )
            self.assertIsNotNone(reason, overrides)
            self.assertIn(expected, reason)


class TestWeightLayout(CustomTestCase):
    def test_gemm1_is_reordered_to_up_gate(self):
        layer = _checkpoint_layer()
        intermediate = layer["w13_packed"].shape[1] // 2
        w13, scale, global_scale = cutile.to_flashinfer_canonical_gemm1(
            layer["w13_packed"],
            layer["w13_block_scale"],
            layer["w13_weight_global_scale"],
        )
        gate, up = layer["w13_packed"].split(intermediate, dim=1)
        gate_scale, up_scale = layer["w13_block_scale"].split(intermediate, dim=1)
        self.assertTrue(torch.equal(w13, torch.cat((up, gate), dim=1)))
        self.assertTrue(
            torch.equal(
                scale.view(torch.uint8),
                torch.cat((up_scale, gate_scale), dim=1).view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(global_scale, layer["w13_weight_global_scale"][:, [1, 0]])
        )

    @unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
    def test_prepared_view_keeps_gate_and_up_with_their_own_scales(self):
        """A gate/up mix-up in either library's layout would pair one branch's
        weights with the other's block or global scales."""
        layer = _checkpoint_layer()
        intermediate = layer["w13_packed"].shape[1] // 2
        # Distinct constant block scales per branch survive FlashInfer's swizzle,
        # which only permutes inside 128-row tiles, without modelling its layout.
        gate_code, up_code = 0.5, 2.0
        layer["w13_block_scale"][:, :intermediate] = gate_code
        layer["w13_block_scale"][:, intermediate:] = up_code
        view = cutile.prepare_cutile_nvfp4_weight_view(
            **layer, activation=SwiGLU(alpha=1.702, beta=1.0, limit=7.0)
        )
        # FlashInfer's native GEMM1 order is [gate, up]: the checkpoint order.
        self.assertTrue(torch.equal(view["w1"], layer["w13_packed"]))
        tiles = intermediate // 128
        self.assertTrue(bool((view["w1_scale"][:, :tiles].float() == gate_code).all()))
        self.assertTrue(bool((view["w1_scale"][:, tiles:].float() == up_code).all()))
        # Checkpoint scales are encode scales; the kernel multiplies by the inverse.
        torch.testing.assert_close(
            view["w1_global_scale"], 1.0 / layer["w13_weight_global_scale"]
        )
        torch.testing.assert_close(
            view["w2_global_scale"], 1.0 / layer["w2_weight_global_scale"]
        )
        self.assertTrue(torch.equal(view["w2"], layer["w2_packed"]))

    @unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
    def test_non_unit_activation_global_scale_is_rejected(self):
        activation = SwiGLU(alpha=1.702, beta=1.0, limit=7.0)
        for name in ("w13_input_global_scale", "w2_input_global_scale"):
            layer = _checkpoint_layer()
            layer[name] = layer[name] * 0.25
            with self.assertRaisesRegex(ValueError, name):
                cutile.prepare_cutile_nvfp4_weight_view(**layer, activation=activation)


@unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
class TestSharedFlashInferLayer(CustomTestCase):
    def test_same_shape_layers_share_one_workspace_owner(self):
        """FlashInfer keeps its workspaces per MoELayer; one per model layer would
        hold a gigabyte-scale workspace once for every MoE layer."""
        built = []

        def fake_moe_layer(config, device=None):
            built.append(config)
            return object()

        def build(seed, top_k):
            return cutile.build_cutile_nvfp4_quant_info(
                **_checkpoint_layer(seed=seed),
                top_k=top_k,
                swiglu_alpha=1.702,
                swiglu_beta=1.0,
                swiglu_limit=7.0,
            )

        with (
            mock.patch("flashinfer.fused_moe.MoELayer", fake_moe_layer),
            mock.patch.dict(cutile._SHARED_MOE_LAYERS, clear=True),
        ):
            first, second, other = build(0, 4), build(1, 4), build(2, 2)
        self.assertIs(first.moe_layer, second.moe_layer)
        self.assertIsNot(first.weight_pack, second.weight_pack)
        self.assertIsNot(first.moe_layer, other.moe_layer)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0].execution.tune_max_num_tokens, 8192)


class _RecordingLayer:
    """Stands in for flashinfer.fused_moe.MoELayer: a per-token function of the
    inputs, so a misrouted or reordered chunk changes the output."""

    def __init__(self):
        self.calls = []

    def __call__(self, act, weight_pack):
        self.calls.append((act.num_tokens, weight_pack))
        assert act.hidden_states_q.is_contiguous() and act.topk_ids.is_contiguous()
        assert act.topk_ids.dtype == torch.int32
        assert act.topk_weights.dtype == torch.float32
        scale = (act.topk_ids.float() * act.topk_weights).sum(dim=1, keepdim=True)
        return (act.hidden_states_q.float() * scale).to(torch.bfloat16)


@unittest.skipUnless(_HAS_CUTILE_API, "FlashInfer cuTile MoE API not installed")
class TestForwardContract(CustomTestCase):
    def setUp(self):
        self.layer = _RecordingLayer()
        self.quant_info = cutile.FlashInferCutileNvfp4MoeQuantInfo(
            moe_layer=self.layer, weight_pack="weights", max_tokens_per_call=4
        )

    @staticmethod
    def _dispatch(num_tokens, hidden=16, top_k=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        return StandardDispatchOutput(
            hidden_states=torch.randn(num_tokens, hidden, generator=g).to(
                torch.bfloat16
            ),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.rand(num_tokens, top_k, generator=g),
                topk_ids=torch.randint(0, 8, (num_tokens, top_k), generator=g).to(
                    torch.int32
                ),
                router_logits=None,
            ),
        )

    def _expected(self, dispatch):
        topk = dispatch.topk_output
        scale = (topk.topk_ids.float() * topk.topk_weights).sum(dim=1, keepdim=True)
        return (dispatch.hidden_states.float() * scale).to(torch.bfloat16)

    def _run(self, dispatch, **config):
        return cutile.fused_experts_none_to_flashinfer_cutile(
            dispatch, self.quant_info, _runner_config(**config)
        ).hidden_states

    def test_token_chunks_cover_every_row_in_order(self):
        # One past a multiple of the per-call bound: the 8193-token case.
        dispatch = self._dispatch(num_tokens=9)
        output = self._run(dispatch)
        self.assertEqual([n for n, _ in self.layer.calls], [4, 4, 1])
        self.assertTrue(torch.equal(output, self._expected(dispatch)))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_batch_at_the_bound_is_one_call(self):
        dispatch = self._dispatch(num_tokens=4)
        self._run(dispatch)
        self.assertEqual([n for n, _ in self.layer.calls], [4])

    def test_routed_scaling_factor_is_applied_once(self):
        dispatch = self._dispatch(num_tokens=3)
        expected = self._expected(dispatch)
        self.assertTrue(torch.equal(self._run(dispatch), expected))
        scaled = self._run(dispatch, routed_scaling_factor=2.0)
        self.assertTrue(torch.equal(scaled, (expected.float() * 2.0).bfloat16()))

    def test_padded_slots_reach_the_kernel_as_no_ops(self):
        """CUDA-graph padding rows arrive with expert id -1; the kernel must see an
        in-range id with zero weight, never the -1."""
        dispatch = self._dispatch(num_tokens=4)
        ids = dispatch.topk_output.topk_ids.clone()
        ids[2:] = -1  # padded rows
        ids[0, 1] = -1  # one unrouted slot in a real row
        dispatch = dispatch._replace(
            topk_output=dispatch.topk_output._replace(topk_ids=ids)
        )
        topk = dispatch.topk_output
        valid = topk.topk_ids >= 0
        scale = torch.where(valid, topk.topk_ids.float() * topk.topk_weights, 0.0)
        expected = (dispatch.hidden_states.float() * scale.sum(1, keepdim=True)).to(
            torch.bfloat16
        )
        self.assertTrue(torch.equal(self._run(dispatch), expected))

    def test_zero_tokens_skip_the_kernel(self):
        output = self._run(self._dispatch(num_tokens=0))
        self.assertEqual(tuple(output.shape), (0, 16))
        self.assertEqual(self.layer.calls, [])

    def test_unsupported_inputs_are_errors(self):
        dispatch = self._dispatch(num_tokens=2)
        packed = dispatch._replace(
            topk_output=PackedTopKOutput(
                packed_topk_ids=torch.zeros(2, 2, dtype=torch.int32),
                router_logits=None,
            )
        )
        with self.assertRaisesRegex(TypeError, "standard top-k"):
            self._run(packed)
        fp16 = dispatch._replace(hidden_states=dispatch.hidden_states.half())
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            self._run(fp16)
        self.assertEqual(self.layer.calls, [])


class TestBackendRegistration(CustomTestCase):
    def test_backend_is_registered_end_to_end(self):
        self.assertIn("flashinfer_cutile", MOE_RUNNER_BACKEND_CHOICES)
        backend = MoeRunnerBackend("flashinfer_cutile")
        self.assertTrue(backend.is_flashinfer_cutile())
        self.assertFalse(MoeRunnerBackend.FLASHINFER_CUTEDSL.is_flashinfer_cutile())
        self.assertIs(
            FusedOpPool.get_fused_func("none", "flashinfer_cutile"),
            cutile.fused_experts_none_to_flashinfer_cutile,
        )


class TestArgumentResolution(CustomTestCase):
    def _m31_overrides(self, **cfg):
        view = SimpleNamespace(
            quantization="minimax_m3_nvfp4",
            speculative_algorithm=None,
            page_size=None,
            moe_runner_backend="flashinfer_cutile",
            moe_a2a_backend="none",
            ep_size=1,
            kv_cache_dtype="auto",
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
        )
        view.__dict__.update(cfg)
        hf_config = SimpleNamespace(quantization_config={"quant_method": "mxfp8"})
        with (
            override_platform(is_hip=False, is_blackwell=True),
            mock.patch.object(minimax_m3_overrides, "resolving_view", lambda _: view),
        ):
            return minimax_m3_overrides._minimax_m3_overrides(
                SimpleNamespace(_quantization_explicitly_unset=False), hf_config
            )

    def test_cutile_request_is_not_rerouted_to_megamoe(self):
        # The Blackwell default forces MegaMoE, which bypasses the expert runner.
        overrides = self._m31_overrides()
        self.assertNotIn("moe_a2a_backend", overrides)
        self.assertNotIn("moe_runner_backend", overrides)
        default = self._m31_overrides(moe_runner_backend="auto")
        self.assertEqual(default["moe_a2a_backend"], "megamoe")
        self.assertEqual(default["moe_runner_backend"], "deep_gemm")

    def test_cutile_request_rejects_megamoe_and_expert_parallelism(self):
        with self.assertRaisesRegex(ValueError, "moe-a2a-backend none"):
            self._m31_overrides(moe_a2a_backend="megamoe")
        with self.assertRaisesRegex(ValueError, "ep-size 1"):
            self._m31_overrides(ep_size=2)

    def test_cutile_request_needs_the_m31_nvfp4_quantization(self):
        # Every other quant method would ignore the flag and run its own backend.
        constraints = arg_overrides._moe_runner_backend_quant_constraints
        view = SimpleNamespace(
            moe_runner_backend="flashinfer_cutile", quantization="fp8"
        )
        with self.assertRaisesRegex(ValueError, "minimax_m3_nvfp4"):
            constraints(view)
        view.quantization = "minimax_m3_nvfp4"
        self.assertEqual(constraints(view), {})


class TestMiniMaxSchemeSelection(CustomTestCase):
    def _scheme(self, *, runner, a2a="none", is_blackwell, device_sm, reason=None):
        with (
            override_platform(
                is_cuda=True,
                is_hip=False,
                is_blackwell=is_blackwell,
                device_sm=device_sm,
            ),
            mock.patch.object(
                minimax_m3_nvfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend(runner),
            ),
            mock.patch.object(
                minimax_m3_nvfp4,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend(a2a),
            ),
            mock.patch.object(
                cutile, "cutile_nvfp4_platform_unsupported_reason", return_value=reason
            ),
        ):
            return minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()

    def test_cutile_is_selected_only_on_request(self):
        scheme = self._scheme(
            runner="flashinfer_cutile", is_blackwell=True, device_sm=120
        )
        self.assertTrue(scheme.use_flashinfer_cutile)
        self.assertFalse(scheme.use_marlin or scheme.use_flashinfer_trtllm)

    def test_existing_selection_is_unchanged(self):
        marlin = self._scheme(runner="marlin", is_blackwell=False, device_sm=90)
        self.assertTrue(marlin.use_marlin)
        self.assertFalse(marlin.use_flashinfer_cutile)
        for device_sm in (100, 120):
            mega = self._scheme(
                runner="deep_gemm",
                a2a="megamoe",
                is_blackwell=True,
                device_sm=device_sm,
            )
            self.assertFalse(
                mega.use_marlin
                or mega.use_flashinfer_cutile
                or mega.use_flashinfer_trtllm
            )
            routed = self._scheme(
                runner="flashinfer_trtllm_routed",
                is_blackwell=True,
                device_sm=device_sm,
            )
            self.assertTrue(routed.use_flashinfer_trtllm)

    def test_explicit_cutile_request_never_falls_back(self):
        # Off SM12x the scheme would otherwise pick Marlin and ignore the flag.
        with self.assertRaisesRegex(ValueError, "sm_90"):
            self._scheme(
                runner="flashinfer_cutile",
                is_blackwell=False,
                device_sm=90,
                reason="it supports compute capability (120, 121) only, got sm_90",
            )
        with self.assertRaisesRegex(ValueError, "moe-a2a-backend none"):
            self._scheme(
                runner="flashinfer_cutile",
                a2a="megamoe",
                is_blackwell=True,
                device_sm=120,
            )

    def test_cutile_runner_rejects_expert_parallel_layers(self):
        scheme = self._scheme(
            runner="flashinfer_cutile", is_blackwell=True, device_sm=120
        )
        config = _runner_config(num_local_experts=16)
        with self.assertRaisesRegex(ValueError, "expert parallelism"):
            scheme.create_moe_runner(SimpleNamespace(), config)


if __name__ == "__main__":
    unittest.main()
