"""Precision and checkpoint-layout regressions for native Beam."""

import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from compressed_tensors.quantization import preset_name_to_scheme

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.configs.model_config import (
    _get_and_verify_dtype,
    get_hybrid_layer_ids,
    is_hybrid_swa_model,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.topk import TopKOutputFormat, biased_grouped_topk_impl
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.models.beam import (
    BeamAttention,
    BeamDecoderLayer,
    BeamForCausalLM,
    BeamModel,
    BeamMoE,
    BeamUnquantizedFusedMoEMethod,
    apply_attention_gate,
    apply_ssmax,
    depth_scaled_residual,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils.hf_transformers import get_context_length
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tiny_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=8,
        max_position_embeddings=1024,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        moe_intermediate_size=32,
        num_experts=16,
    )
    kwargs.update(overrides)
    return BeamConfig(**kwargs)


def loader_model(params):
    model = BeamForCausalLM.__new__(BeamForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = tiny_config()
    model.start_layer, model.end_layer = 0, 4
    model.model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True)
    )
    model.named_parameters = lambda: iter(params.items())
    return model


def fp8_config(preset="FP8_DYNAMIC", ignore=(), group_name=None):
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "config_groups": {
                group_name or preset: preset_name_to_scheme(
                    preset, targets=["Linear", "lm_head"]
                ).model_dump()
            },
            "ignore": list(ignore),
            "packed_modules_mapping": BeamForCausalLM.packed_modules_mapping,
        }
    )


class _LoadedExpert:
    def __init__(self):
        self.shards = {}

    def weight_loader(self, param, weight, name, *, shard_id, expert_id):
        self.shards[expert_id, shard_id] = weight.clone()


class TestBeamConfig(CustomTestCase):
    def test_bf16_and_fp8_reject_unvalidated_moe_runners(self):
        from sglang.srt.runtime_context import get_flags

        for quant in (None, fp8_config()):
            for backend in (
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                MoeRunnerBackend.FLASHINFER_CUTLASS,
                MoeRunnerBackend.DEEP_GEMM,
            ):
                with (
                    self.subTest(quant=quant, backend=backend),
                    get_context().override_server_args(),
                    get_flags().moe.override(
                        runner_backend=backend, a2a_backend=MoeA2ABackend.NONE
                    ),
                    get_parallel().override(moe_tp_size=1, moe_ep_size=1),
                    self.assertRaisesRegex(ValueError, "auto or triton"),
                ):
                    BeamMoE(tiny_config(), 1, quant_config=quant)

    def test_unsupported_quantization_fails_before_allocation(self):
        # Other FP8 layouts must not pass merely because they use the same exporter.
        mixed = fp8_config().config
        mixed["config_groups"]["other"] = preset_name_to_scheme(
            "FP8", targets=["model.layers.1.mlp.experts"]
        ).model_dump()
        for quant in (
            Fp8Config(),
            Fp8Config(is_checkpoint_fp8_serialized=True),
            W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
            ModelOptFp8Config(is_checkpoint_fp8_serialized=True),
            fp8_config("FP8"),
            fp8_config("W8A8"),
            CompressedTensorsConfig.from_config(mixed),
            CompressedTensorsConfig.from_config(
                fp8_config().config
                | {"linear_fp8_config": {"activation_scheme": "static"}}
            ),
            CompressedTensorsConfig.from_config(
                {"format": "float-quantized", "config_groups": {}, "ignore": []}
            ),
        ):
            with (
                self.subTest(quant=quant),
                patch("sglang.srt.models.beam.BeamModel") as model,
            ):
                with self.assertRaisesRegex(NotImplementedError, "FP8_DYNAMIC"):
                    BeamForCausalLM(tiny_config(), quant_config=quant)
                model.assert_not_called()

    def test_unsupported_attention_and_head_parallelism_fail_before_allocation(self):
        for field in ("enable_dp_attention", "enable_dp_lm_head"):
            with (
                self.subTest(field=field),
                get_context().override_server_args(**{field: True}),
                patch("sglang.srt.models.beam.BeamModel") as model,
            ):
                with self.assertRaisesRegex(
                    NotImplementedError, "attention data parallelism"
                ):
                    BeamForCausalLM(tiny_config())
                model.assert_not_called()

    def test_expert_relocation_fails_before_allocation(self):
        for options in (
            {"enable_eplb": True},
            {"init_expert_location": "random"},
            {"ep_num_redundant_experts": 1},
        ):
            with (
                self.subTest(options=options),
                get_context().override_server_args(**options),
                patch("sglang.srt.models.beam.BeamModel") as model,
            ):
                with self.assertRaisesRegex(
                    NotImplementedError, "expert relocation or redundant experts"
                ):
                    BeamForCausalLM(tiny_config())
                model.assert_not_called()

    def test_release_and_explicit_layer_schedules_choose_native_pools(self):
        release = BeamConfig()
        full_ids = [5, 11, 17, 23, 29, 35, 41, 47, 51]
        sliding_ids = [layer for layer in range(52) if layer not in full_ids]
        self.assertEqual(
            get_hybrid_layer_ids(release.architectures, release),
            (sliding_ids, full_ids),
        )
        self.assertEqual(
            release.layer_types,
            [
                "full_attention" if layer in full_ids else "sliding_attention"
                for layer in range(52)
            ],
        )
        self.assertEqual(release.mlp_layer_types, ["dense"] * 3 + ["sparse"] * 49)
        self.assertEqual(release.max_position_embeddings, 1_048_576)
        self.assertTrue(is_hybrid_swa_model(release.architectures, release))

        explicit = BeamConfig(
            num_hidden_layers=3,
            layer_types=["full_attention", "sliding_attention", "full_attention"],
        )
        self.assertEqual(
            get_hybrid_layer_ids(explicit.architectures, explicit), ([1], [0, 2])
        )
        self.assertEqual(
            BeamForCausalLM.get_attention_sliding_window_size(
                SimpleNamespace(config=explicit)
            ),
            511,
        )

        full_only = BeamConfig(num_hidden_layers=3, layer_types=["full_attention"] * 3)
        self.assertFalse(is_hybrid_swa_model(full_only.architectures, full_only))
        self.assertEqual(
            get_hybrid_layer_ids(full_only.architectures, full_only), ([], [0, 1, 2])
        )
        self.assertIsNone(
            BeamForCausalLM.get_attention_sliding_window_size(
                SimpleNamespace(config=full_only)
            )
        )

        sliding_only = BeamConfig(
            num_hidden_layers=3,
            layer_types=["sliding_attention"] * 3,
            sliding_window=2,
        )
        self.assertTrue(is_hybrid_swa_model(sliding_only.architectures, sliding_only))
        self.assertEqual(
            get_hybrid_layer_ids(sliding_only.architectures, sliding_only),
            ([0, 1, 2], []),
        )
        self.assertEqual(
            BeamForCausalLM.get_attention_sliding_window_size(
                SimpleNamespace(config=sliding_only)
            ),
            1,
        )

    def test_rope_parameters_are_nested_per_layer_type_and_survive_roundtrip(self):
        parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 20000.0},
            "full_attention": None,
        }
        config = BeamConfig(rope_parameters=parameters)
        # save_pretrained runs the Transformers validators; see BeamConfig.validate.
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            saved = BeamConfig.from_pretrained(directory)
        for candidate in (config, saved):
            self.assertEqual(candidate.rope_parameters, parameters)
            self.assertFalse(hasattr(candidate, "rope_theta"))
            self.assertEqual(
                get_context_length(candidate), candidate.max_position_embeddings
            )

    def test_unsupported_rope_and_single_token_window_fail_early(self):
        with self.assertRaisesRegex(ValueError, "per-layer-type"):
            BeamConfig(rope_parameters={"rope_type": "default", "rope_theta": 500000.0})
        with self.assertRaisesRegex(ValueError, "only default RoPE"):
            BeamConfig(
                rope_parameters={
                    "sliding_attention": {"rope_type": "linear", "factor": 2},
                    "full_attention": None,
                }
            )
        with self.assertRaisesRegex(ValueError, "without rotary embeddings"):
            BeamConfig(
                rope_parameters={
                    "sliding_attention": {"rope_type": "default"},
                    "full_attention": {"rope_type": "default"},
                }
            )

        # Checked before the Transformers constructor, which raises a non-ValueError.
        with self.assertRaisesRegex(ValueError, "mlp_layer_types length"):
            BeamConfig(num_hidden_layers=2, mlp_layer_types=["dense"])

        for window in (None, 0, 1):
            with (
                self.subTest(window=window),
                self.assertRaisesRegex(ValueError, "sliding_window >= 2"),
            ):
                BeamConfig(sliding_window=window)

    def test_missing_dtype_defaults_to_bf16_without_overriding_metadata(self):
        for metadata in ({}, {"dtype": None}, {"torch_dtype": None}):
            with self.subTest(metadata=metadata):
                config = BeamConfig(**metadata)
                self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.bfloat16)
                restored = BeamConfig.from_dict(config.to_dict())
                self.assertEqual(
                    _get_and_verify_dtype(restored, "auto"), torch.bfloat16
                )

        for field in ("dtype", "torch_dtype"):
            with self.subTest(field=field):
                config = BeamConfig(**{field: "float16"})
                self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.float16)


class TestBeamNumerics(CustomTestCase):
    def test_norms_round_after_weight_multiplication(self):
        config = tiny_config()
        pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
        )
        with (
            get_context().override_server_args(),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                pp_group=pp_group,
            ),
            patch(
                "sglang.srt.models.beam.BeamRotaryEmbedding",
                side_effect=lambda config: torch.nn.Identity(),
            ) as rotary,
            patch("sglang.srt.models.beam.BeamMoE", return_value=torch.nn.Identity()),
        ):
            model = BeamModel(config)
        rotary.assert_called_once_with(config)
        self.assertIs(
            model.layers[0].self_attn.rotary_emb,
            model.layers[2].self_attn.rotary_emb,
        )
        for layer_id in (1, 3):
            self.assertIsNone(model.layers[layer_id].self_attn.rotary_emb)
        generator = torch.Generator().manual_seed(46)
        for name, norm in model.named_modules():
            if not isinstance(norm, RMSNorm):
                continue
            with self.subTest(norm=name):
                norm = norm.bfloat16()
                value = torch.randn(5, norm.hidden_size, generator=generator).bfloat16()
                weight = (
                    0.5 + torch.rand(norm.hidden_size, generator=generator)
                ).bfloat16()
                with torch.no_grad():
                    norm.weight.copy_(weight)
                normalized = value.float() * torch.rsqrt(
                    value.float().square().mean(dim=-1, keepdim=True)
                    + config.rms_norm_eps
                )
                expected = (normalized * weight.float()).bfloat16()
                rounded_before_weight = normalized.bfloat16() * weight

                actual = norm.forward_native(value)

                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertFalse(torch.equal(expected, rounded_before_weight))

    def test_ssmax_absolute_positions_and_fp32_product(self):
        positions = torch.tensor([0, 511, 512, 2047, 8192, 262143, 1048575])
        generator = torch.Generator().manual_seed(42)
        query = torch.randn(positions.numel(), 32, generator=generator).bfloat16()
        scale = torch.tensor(
            [math.exp(0.1 * math.log1p((p + 1) / 2048)) for p in positions],
            dtype=torch.float32,
        ).unsqueeze(-1)
        expected = (query.float() * scale).bfloat16()

        actual = apply_ssmax(query, positions)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(expected, query * scale.bfloat16()))
        # Chunking and cached decode must retain the supplied absolute positions.
        for start, end in ((0, 3), (3, 6), (6, 7)):
            torch.testing.assert_close(
                apply_ssmax(query[start:end], positions[start:end]),
                actual[start:end],
                atol=0,
                rtol=0,
            )

    def test_gate_bias_broadcast_and_bf16_serving_arithmetic(self):
        generator = torch.Generator().manual_seed(43)
        output = torch.randn(3, 4, 8, generator=generator).bfloat16()
        gate = torch.randn(3, 4, 8, generator=generator).bfloat16()
        bias = torch.tensor([-1.03125, 0.00390625, 0.53125, 2.125]).bfloat16()
        expected = output * torch.sigmoid(gate + bias[None, :, None])
        fp32_gate = (
            output.float() * torch.sigmoid(gate.float() + bias.float()[None, :, None])
        ).bfloat16()

        actual = apply_attention_gate(output, gate, bias)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertFalse(torch.equal(expected, fp32_gate))

    def test_depth_scaled_residual_rounds_scaled_operands_and_addition(self):
        generator = torch.Generator().manual_seed(44)
        branch = torch.randn(5, 32, generator=generator).bfloat16()
        residual = torch.randn(5, 32, generator=generator).bfloat16()
        # Layer 51 is the release depth; the default schedules cover 52 layers.
        deep_config = tiny_config(
            num_hidden_layers=52, layer_types=None, mlp_layer_types=None
        )
        for layer_id in (0, 1, 51):
            config = deep_config if layer_id == 51 else tiny_config()
            with (
                patch(
                    "sglang.srt.models.beam.BeamAttention",
                    return_value=torch.nn.Identity(),
                ),
                patch(
                    "sglang.srt.models.beam.BeamMLP", return_value=torch.nn.Identity()
                ),
                patch(
                    "sglang.srt.models.beam.BeamMoE", return_value=torch.nn.Identity()
                ),
            ):
                layer = BeamDecoderLayer(config, layer_id)
            for is_ffn in (False, True):
                depth = 2 * (layer_id + 1) + int(is_ffn)
                with self.subTest(layer_id=layer_id, is_ffn=is_ffn):
                    expected = branch * (1 / math.sqrt(depth)) + residual * (
                        math.sqrt(depth - 1) / math.sqrt(depth)
                    )
                    fp32_temporaries = (
                        branch.float() * (1 / math.sqrt(depth))
                        + residual.float() * (math.sqrt(depth - 1) / math.sqrt(depth))
                    ).bfloat16()
                    scales = (
                        layer.mlp_residual_scales
                        if is_ffn
                        else layer.attn_residual_scales
                    )
                    actual = depth_scaled_residual(branch, residual, scales)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertFalse(torch.equal(expected, fp32_temporaries))

    def test_moe_uses_combined_runner_output_without_reweighting(self):
        config = tiny_config()
        for num_tokens in (0, 1, 3):
            with self.subTest(num_tokens=num_tokens):
                hidden = torch.zeros(num_tokens, config.hidden_size).bfloat16()
                combined = torch.full_like(hidden, 0.75)
                expected = combined.clone()
                gate = torch.nn.Module()
                gate.forward = Mock(
                    return_value=(torch.zeros(num_tokens, config.num_experts), None)
                )
                topk_output = object()
                topk = torch.nn.Module()
                topk.forward = Mock(return_value=topk_output)
                topk.empty_topk_output = Mock(return_value=topk_output)
                experts = torch.nn.Module()
                experts.should_fuse_routed_scaling_factor_in_topk = False
                experts.forward = Mock(return_value=combined)
                factory = Mock(return_value=experts)
                with (
                    get_context().override_server_args(),
                    get_parallel().override(moe_tp_size=1, moe_ep_size=1),
                    patch("sglang.srt.models.beam.ReplicatedLinear", return_value=gate),
                    patch("sglang.srt.models.beam.TopK", return_value=topk),
                    patch(
                        "sglang.srt.models.beam.get_moe_impl_class",
                        return_value=factory,
                    ),
                ):
                    moe = BeamMoE(config, layer_id=1)
                    self.assertFalse(factory.call_args.kwargs.get("no_combine", False))
                    self.assertEqual(
                        factory.call_args.kwargs["routed_scaling_factor"],
                        config.routed_scaling_factor,
                    )
                    actual = moe(hidden)

                experts.forward.assert_called_once_with(hidden, topk_output)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.shape, hidden.shape)
                if num_tokens:
                    topk.forward.assert_called_once()
                    topk.empty_topk_output.assert_not_called()
                else:
                    topk.forward.assert_not_called()
                    topk.empty_topk_output.assert_called_once()

    def _select_beam_experts(self, config, bias, logits):
        # Exercise the TopK constructed by Beam, replacing only the hardware
        # selector with its shared eager implementation for CPU CI.
        hidden = torch.ones(logits.shape[0], config.hidden_size, dtype=torch.bfloat16)
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0, moe_tp_size=1, moe_ep_size=1),
            patch.object(
                torch.nn.Identity,
                "should_fuse_routed_scaling_factor_in_topk",
                False,
                create=True,
            ),
            patch(
                "sglang.srt.models.beam.get_moe_impl_class",
                return_value=Mock(return_value=torch.nn.Identity()),
            ),
            patch(
                "sglang.srt.layers.moe.topk.biased_grouped_topk",
                wraps=biased_grouped_topk_impl._torchdynamo_orig_callable,
            ) as selector,
        ):
            moe = BeamMoE(config, layer_id=1)
            with torch.no_grad():
                moe.gate.e_score_correction_bias.copy_(bias)
            routing = moe.topk.topk_config
            self.assertTrue(routing.use_grouped_topk)
            self.assertEqual(routing.num_expert_group, config.n_group)
            self.assertEqual(routing.topk_group, config.topk_group)
            self.assertEqual(routing.top_k, config.num_experts_per_tok)
            self.assertTrue(routing.renormalize)
            self.assertEqual(routing.scoring_func, "sigmoid")
            self.assertEqual(routing.output_format, TopKOutputFormat.STANDARD)
            self.assertIsNone(routing.custom_routing_function)
            self.assertIsNone(routing.routed_scaling_factor)
            self.assertFalse(routing.apply_routed_scaling_factor_on_output)
            self.assertIs(routing.correction_bias, moe.gate.e_score_correction_bias)
            self.assertEqual(moe.gate.weight.dtype, torch.float32)
            self.assertEqual(routing.correction_bias.dtype, torch.float32)

            output = moe.topk.forward_native(hidden, logits)

            selector.assert_called_once()
        self.assertEqual(output.format, TopKOutputFormat.STANDARD)
        self.assertEqual(output.topk_weights.dtype, torch.float32)
        self.assertEqual(output.topk_ids.dtype, torch.int32)
        return output.topk_weights, output.topk_ids

    def test_router_bias_selects_experts_with_unscaled_fp32_combine_scores(self):
        config = tiny_config()
        config.num_experts = 32
        bias = torch.linspace(0, 1, 32)
        logits = torch.stack(
            (
                torch.linspace(1, -1, 32),
                # The epsilon is significant, and the old 1e-9 clamp would
                # reduce the coefficient sum by many orders of magnitude.
                torch.linspace(-49, -47, 32),
                torch.full((32,), -100.0),
            )
        )
        weights, ids = self._select_beam_experts(config, bias, logits)
        _, unbiased_ids = self._select_beam_experts(
            config, torch.zeros_like(bias), logits[:1]
        )
        self.assertNotEqual(set(ids[0].tolist()), set(unbiased_ids[0].tolist()))

        for row in range(3):
            scores = logits[row].sigmoid()
            corrected = scores + bias
            group_scores = [
                sum(sorted(corrected[group * 4 : (group + 1) * 4].tolist())[-2:])
                for group in range(8)
            ]
            groups = sorted(range(8), key=lambda g: group_scores[g], reverse=True)[:4]
            selected = sorted(
                [
                    expert
                    for group in groups
                    for expert in range(group * 4, group * 4 + 4)
                ],
                key=lambda expert: corrected[expert].item(),
                reverse=True,
            )[:8]
            self.assertEqual(set(ids[row].tolist()), set(selected))
            # Align coefficients with actual IDs; slot order is not a contract.
            unbiased = scores[ids[row].long()]
            expected = unbiased / (unbiased.sum() + 1e-20)
            torch.testing.assert_close(weights[row], expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(weights[0], weights[0].bfloat16().float()))
        self.assertAlmostEqual(weights[0].sum().item(), 1.0, places=6)
        self.assertGreater(weights[1].sum().item(), 0.1)
        self.assertLess(weights[1].sum().item(), 0.95)
        self.assertEqual(weights[2].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(weights).all())

    def test_router_logits_change_selected_groups_and_experts_with_fixed_bias(self):
        config = tiny_config()
        config.num_experts = 32
        bias = torch.linspace(-0.015, 0.015, 32)
        # Rows 0/1 choose different groups; rows 0/2 keep the same groups but
        # change both winning experts in every group. Fixed, small correction
        # biases cannot select these memberships without the sigmoid scores.
        selected = torch.tensor(
            [
                [0, 1, 4, 5, 8, 9, 12, 13],
                [18, 19, 22, 23, 26, 27, 30, 31],
                [2, 3, 6, 7, 10, 11, 14, 15],
            ]
        )
        scores = torch.linspace(0.03, 0.12, 32).repeat(3, 1)
        scores.scatter_(1, selected, torch.linspace(0.85, 0.64, 8).repeat(3, 1))
        # Each rejected group contains a single expert stronger than any
        # winner. Its top-two sum is still weaker: a group-max or global-topk
        # implementation must fail these explicit membership expectations.
        scores[torch.arange(3), torch.tensor([28, 0, 24])] = 0.95
        weights, ids = self._select_beam_experts(config, bias, scores.logit())

        for row in range(3):
            self.assertEqual(set(ids[row].tolist()), set(selected[row].tolist()))
        self.assertTrue(torch.isfinite(weights).all())


class TestBeamLoading(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_fp8_model_uses_shared_methods_and_loads_scales(self):
        config = tiny_config(
            vocab_size=64,
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            mlp_layer_types=["dense", "sparse"],
        )
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        # Exporters may use arbitrary group names for the FP8_DYNAMIC recipe.
        for quant in (None, fp8_config(group_name="group_0")):
            with (
                self.subTest(quant=quant),
                get_parallel().override(
                    tp_rank=0,
                    tp_size=1,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    moe_tp_rank=0,
                    moe_tp_size=1,
                    moe_ep_rank=0,
                    moe_ep_size=1,
                    pp_group=SimpleNamespace(
                        is_first_rank=True,
                        is_last_rank=True,
                        rank_in_group=0,
                        world_size=1,
                    ),
                ),
                # Exercise real allocation and loaders without executing CUDA kernels.
                patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            ):
                model = BeamForCausalLM(config, quant_config=quant)
                dense, sparse = model.model.layers
                experts = sparse.mlp.experts
                dtype = torch.float8_e4m3fn if quant is not None else torch.bfloat16
                for layer in (dense, sparse):
                    for proj in ("qkv_proj", "gate_proj", "o_proj"):
                        self.assertEqual(
                            getattr(layer.self_attn, proj).weight.dtype, dtype
                        )
                for param in (
                    dense.mlp.gate_up_proj.weight,
                    dense.mlp.down_proj.weight,
                ):
                    self.assertEqual(param.dtype, dtype)
                self.assertEqual(experts.w13_weight.dtype, dtype)
                self.assertEqual(experts.w2_weight.dtype, dtype)
                self.assertEqual(model.model.embed_tokens.weight.dtype, torch.bfloat16)
                self.assertEqual(sparse.mlp.gate.weight.dtype, torch.float32)
                self.assertEqual(
                    sparse.mlp.gate.e_score_correction_bias.dtype, torch.float32
                )
                self.assertEqual(
                    experts.moe_runner_config.routed_scaling_factor,
                    config.routed_scaling_factor,
                )
                self.assertFalse(
                    sparse.mlp.topk.topk_config.apply_routed_scaling_factor_on_output
                )
                with (
                    torch.no_grad(),
                    patch(
                        "sglang.srt.layers.moe.topk.biased_grouped_topk",
                        wraps=biased_grouped_topk_impl._torchdynamo_orig_callable,
                    ),
                ):
                    sparse.mlp.gate.e_score_correction_bias.zero_()
                    routing = sparse.mlp.topk.forward_native(
                        torch.ones(1, config.hidden_size),
                        torch.linspace(1, -1, config.num_experts)[None],
                    )
                # FP8/BF16 runners scale after expert combination, not in routing.
                torch.testing.assert_close(
                    routing.topk_weights.sum(-1), torch.ones(1, dtype=torch.float32)
                )
                if quant is None:
                    self.assertIsInstance(
                        experts.quant_method, BeamUnquantizedFusedMoEMethod
                    )
                    self.assertIsNone(experts.quant_method._torch_compile_forward(1))
                    continue
                self.assertNotIsInstance(
                    experts.quant_method, BeamUnquantizedFusedMoEMethod
                )
                weights = []
                for name, width, value in (
                    ("0.self_attn.q_proj", 384, 1),
                    ("0.self_attn.k_proj", 64, 2),
                    ("0.self_attn.v_proj", 64, 3),
                    ("0.mlp.gate_proj", 32, 4),
                    ("0.mlp.up_proj", 32, 5),
                    ("1.mlp.experts.0.gate_proj", 32, 6),
                    ("1.mlp.experts.0.up_proj", 32, 7),
                ):
                    prefix = f"model.layers.{name}"
                    weights.append(
                        (
                            f"{prefix}.weight",
                            torch.full((width, 24), value, dtype=dtype),
                        )
                    )
                    weights.append(
                        (
                            f"{prefix}.weight_scale",
                            torch.full(
                                (width, 1),
                                value / 16,
                                dtype=torch.float32,
                            ),
                        )
                    )
                model.load_weights(weights)
                for layer, expected in (
                    (dense.self_attn.qkv_proj, [1] * 384 + [2] * 64 + [3] * 64),
                    (dense.mlp.gate_up_proj, [4] * 32 + [5] * 32),
                ):
                    torch.testing.assert_close(
                        layer.weight.float()[:, 0], torch.tensor(expected).float()
                    )
                    torch.testing.assert_close(
                        layer.weight_scale.flatten(),
                        torch.tensor(expected).float() / 16,
                    )
                torch.testing.assert_close(
                    experts.w13_weight[0].float()[:, 0],
                    torch.tensor([6] * 32 + [7] * 32).float(),
                )
                torch.testing.assert_close(
                    experts.w13_weight_scale[0].flatten(),
                    torch.tensor([6] * 32 + [7] * 32).float() / 16,
                )
                self.assertEqual(model.lm_head.weight.dtype, dtype)
                scales = torch.full_like(model.lm_head.weight_scale, 0.25)
                model.load_weights(
                    [
                        ("lm_head.weight", torch.full((64, 24), 9, dtype=dtype)),
                        ("lm_head.weight_scale", scales),
                    ]
                )
                torch.testing.assert_close(
                    model.lm_head.weight.float(),
                    torch.full((64, 24), 9.0, dtype=torch.float32),
                )
                torch.testing.assert_close(model.lm_head.weight_scale, scales)

    def test_fp8_exclusions_use_unpacked_projection_names(self):
        ignored = [
            f"model.layers.0.{name}"
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            )
        ]
        with (
            get_parallel().override(
                tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1
            ),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            layer = BeamDecoderLayer(
                tiny_config(layer_types=["full_attention"] * 4),
                0,
                quant_config=fp8_config(ignore=ignored),
                prefix="model.layers.0",
            )
        self.assertEqual(
            layer.self_attn.qkv_proj.weight.dtype, torch.get_default_dtype()
        )
        self.assertEqual(layer.mlp.gate_up_proj.weight.dtype, torch.get_default_dtype())
        self.assertEqual(layer.self_attn.gate_proj.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.mlp.down_proj.weight.dtype, torch.float8_e4m3fn)

    def test_non_last_pipeline_stage_skips_lm_head_scales(self):
        model = loader_model({})
        model.model.pp_group.is_last_rank = False
        self.assertEqual(
            model.load_weights(
                [
                    (f"lm_head.{suffix}", torch.ones(1))
                    for suffix in ("weight", "weight_scale", "input_scale")
                ]
            ),
            set(),
        )

    def test_attention_tp_shards_and_packed_projections(self):
        config = tiny_config()
        for tp_size in (1, 2, 8, 16):
            rank = tp_size - 1
            with (
                self.subTest(tp_size=tp_size),
                get_parallel().override(
                    tp_size=tp_size,
                    attn_tp_rank=rank,
                    attn_tp_size=tp_size,
                    moe_tp_size=tp_size,
                ),
                patch(
                    # This is a loading test; constructing the unused CPU rotary
                    # kernel would require vLLM. Native rotation is covered in E2E.
                    "sglang.srt.models.beam.BeamRotaryEmbedding",
                    return_value=torch.nn.Identity(),
                ),
            ):
                attention = BeamAttention(config, layer_id=0)
                dense = MergedColumnParallelLinear(
                    24, [32, 32], bias=False, tp_rank=rank, tp_size=tp_size
                )
                params = {
                    "model.layers.0.self_attn.qkv_proj.weight": attention.qkv_proj.weight,
                    "model.layers.0.self_attn.gate_proj.weight": attention.gate_proj.weight,
                    "model.layers.0.self_attn.gate_bias": attention.gate_bias,
                    "model.layers.0.mlp.gate_up_proj.weight": dense.weight,
                }
                weights = {
                    f"model.layers.0.self_attn.{name}.weight": torch.arange(width * 24)
                    .reshape(width, 24)
                    .float()
                    for name, width in (
                        ("q_proj", 384),
                        ("k_proj", 64),
                        ("v_proj", 64),
                        ("gate_proj", 384),
                    )
                }
                weights["model.layers.0.self_attn.v_proj.weight"] += 20000
                weights["model.layers.0.self_attn.gate_bias"] = torch.arange(48).float()
                weights["model.layers.0.mlp.gate_proj.weight"] = (
                    torch.arange(32 * 24).reshape(32, 24).float()
                )
                weights["model.layers.0.mlp.up_proj.weight"] = (
                    weights["model.layers.0.mlp.gate_proj.weight"] + 1000
                )
                loader_model(params).load_weights(weights.items())
                q_start = rank * attention.q_size
                kv_rank = rank if tp_size <= 8 else rank // (tp_size // 8)
                kv_start = kv_rank * attention.kv_size
                expected_qkv = torch.cat(
                    [
                        weights["model.layers.0.self_attn.q_proj.weight"][
                            q_start : q_start + attention.q_size
                        ],
                        weights["model.layers.0.self_attn.k_proj.weight"][
                            kv_start : kv_start + attention.kv_size
                        ],
                        weights["model.layers.0.self_attn.v_proj.weight"][
                            kv_start : kv_start + attention.kv_size
                        ],
                    ]
                )
                torch.testing.assert_close(attention.qkv_proj.weight, expected_qkv)
                torch.testing.assert_close(
                    attention.gate_proj.weight,
                    weights["model.layers.0.self_attn.gate_proj.weight"][
                        q_start : q_start + attention.q_size
                    ],
                )
                torch.testing.assert_close(
                    attention.gate_bias, torch.arange(48).float().chunk(tp_size)[rank]
                )
                torch.testing.assert_close(
                    dense.weight,
                    torch.cat(
                        [
                            weights["model.layers.0.mlp.gate_proj.weight"].chunk(
                                tp_size
                            )[rank],
                            weights["model.layers.0.mlp.up_proj.weight"].chunk(tp_size)[
                                rank
                            ],
                        ]
                    ),
                )
                self.assertEqual(attention.attn.layer_id, 0)
                self.assertEqual(attention.attn.sliding_window_size, 511)
                self.assertEqual(attention.attn.tp_q_head_num, 48 // tp_size)
                self.assertEqual(attention.attn.tp_k_head_num, max(1, 8 // tp_size))
                full = BeamAttention(config, layer_id=3)
                self.assertEqual(full.attn.layer_id, 3)
                self.assertEqual(full.attn.sliding_window_size, -1)
                self.assertIsNone(full.rotary_emb)

    def test_split_and_packed_experts_keep_gate_up_order(self):
        tensors = {
            (expert, shard): torch.full(
                (4, 6) if shard != "w2" else (6, 4), expert * 10 + offset
            )
            for expert in (0, 1, 15)
            for offset, shard in enumerate(("w1", "w3", "w2"))
        }
        for packed in (False, True):
            with self.subTest(packed=packed):
                w13, w2 = _LoadedExpert(), _LoadedExpert()
                model = loader_model(
                    {
                        "model.layers.1.mlp.experts.w13_weight": w13,
                        "model.layers.1.mlp.experts.w2_weight": w2,
                    }
                )
                if packed:
                    packed_gate_up = torch.zeros(16, 8, 6)
                    packed_down = torch.zeros(16, 6, 4)
                    for expert in (0, 1, 15):
                        packed_gate_up[expert] = torch.cat(
                            [tensors[expert, "w1"], tensors[expert, "w3"]]
                        )
                        packed_down[expert] = tensors[expert, "w2"]
                    weights = [
                        ("model.layers.1.mlp.experts.gate_up_proj", packed_gate_up),
                        ("model.layers.1.mlp.experts.down_proj", packed_down),
                    ]
                else:
                    weights = [
                        (
                            f"model.layers.1.mlp.experts.{expert}.{name}.weight",
                            tensors[expert, shard],
                        )
                        for expert in (0, 1, 15)
                        for name, shard in (
                            ("gate_proj", "w1"),
                            ("up_proj", "w3"),
                            ("down_proj", "w2"),
                        )
                    ]
                model.load_weights(weights)
                for (expert, shard), tensor in tensors.items():
                    actual = (w2 if shard == "w2" else w13).shards[expert, shard]
                    torch.testing.assert_close(actual, tensor.to(actual.dtype))

    def test_nvfp4_expert_weights_and_scales_use_same_shard_mapping(self):
        suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                w13, w2 = _LoadedExpert(), _LoadedExpert()
                params = {
                    f"model.layers.3.mlp.experts.w13_{suffix}": w13,
                    f"model.layers.3.mlp.experts.w2_{suffix}": w2,
                }
                weights = [
                    (
                        f"model.layers.3.mlp.experts.7.{projection}.{suffix}",
                        torch.tensor([value]),
                    )
                    for value, projection in enumerate(
                        ("gate_proj", "up_proj", "down_proj"), start=1
                    )
                ]
                loaded = loader_model(params).load_weights(weights)
                self.assertEqual(loaded, set(params))
                for param, shard, expected in (
                    (w13, "w1", 1),
                    (w13, "w3", 2),
                    (w2, "w2", 3),
                ):
                    torch.testing.assert_close(
                        param.shards[7, shard], torch.tensor([expected])
                    )

    def test_router_parameters_preserve_fp32_checkpoint_values(self):
        config = tiny_config()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with (
                get_parallel().override(
                    moe_tp_size=1,
                    moe_ep_size=1,
                ),
                patch("sglang.srt.models.beam.TopK", return_value=torch.nn.Identity()),
                patch.object(
                    torch.nn.Identity,
                    "should_fuse_routed_scaling_factor_in_topk",
                    False,
                    create=True,
                ),
                patch(
                    "sglang.srt.models.beam.get_moe_impl_class",
                    return_value=lambda **kwargs: torch.nn.Identity(),
                ),
            ):
                moe = BeamMoE(config, layer_id=1)
        finally:
            torch.set_default_dtype(previous_dtype)
        params = {
            "model.layers.1.mlp.gate.weight": moe.gate.weight,
            "model.layers.1.mlp.gate.e_score_correction_bias": moe.gate.e_score_correction_bias,
        }
        weights = {
            name: torch.linspace(0.000123, 0.998765, param.numel()).view(param.shape)
            for name, param in params.items()
        }
        loader_model(params).load_weights(weights.items())
        for name, param in params.items():
            self.assertEqual(param.dtype, torch.float32)
            torch.testing.assert_close(param, weights[name], atol=0, rtol=0)
            self.assertFalse(torch.equal(param, weights[name].bfloat16().float()))

    def test_load_weights_returns_loaded_parameter_names(self):
        params = {
            "model.norm.weight": torch.nn.Parameter(torch.zeros(24)),
            "model.layers.0.self_attn.gate_bias": torch.nn.Parameter(torch.zeros(48)),
        }
        complete = {name: torch.ones_like(param) for name, param in params.items()}
        loaded = loader_model(params).load_weights(complete.items())
        self.assertEqual(loaded, set(params))
        torch.testing.assert_close(params["model.norm.weight"], torch.ones(24))

        # A missing tensor is reported through the returned set, not an error.
        del complete["model.layers.0.self_attn.gate_bias"]
        loaded = loader_model(params).load_weights(complete.items())
        self.assertEqual(loaded, {"model.norm.weight"})

    def test_head_load_keeps_bf16_storage_and_respects_projection_precision(self):
        config = tiny_config()
        stub_model = torch.nn.Module()
        stub_model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        stub_model.start_layer, stub_model.end_layer = 0, 4
        generator = torch.Generator().manual_seed(45)
        stored = torch.randn(32, 24, generator=generator)
        loaded = stored.bfloat16()
        hidden = torch.randn(3, 24, generator=generator).bfloat16()
        bf16_projection = hidden @ loaded.T
        fp32_projection = hidden.float() @ loaded.float().T
        self.assertFalse(torch.equal(bf16_projection.float(), fp32_projection))
        for use_fp32_lm_head in (False, True):
            with self.subTest(use_fp32_lm_head=use_fp32_lm_head):
                previous_dtype = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(torch.bfloat16)
                    with (
                        get_context().override_server_args(
                            enable_fp32_lm_head=use_fp32_lm_head,
                            enable_dp_lm_head=False,
                        ),
                        get_parallel().override(
                            tp_size=1,
                            tp_rank=0,
                            attn_tp_size=1,
                            attn_tp_rank=0,
                            attn_dp_size=1,
                        ),
                        patch(
                            "sglang.srt.models.beam.BeamModel",
                            return_value=stub_model,
                        ),
                    ):
                        model = BeamForCausalLM(config)
                finally:
                    torch.set_default_dtype(previous_dtype)
                model.load_weights([("lm_head.weight", stored)])
                self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)
                self.assertEqual(
                    model.logits_processor.use_fp32_lm_head, use_fp32_lm_head
                )
                torch.testing.assert_close(
                    model.lm_head.weight[:32], loaded, atol=0, rtol=0
                )
                actual = LogitsProcessor._compute_lm_head(
                    model.logits_processor, hidden, model.lm_head
                )[:, :32]
                expected = fp32_projection if use_fp32_lm_head else bf16_projection
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.dtype, expected.dtype)


if __name__ == "__main__":
    unittest.main()
