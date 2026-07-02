import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.mimo_v2 import (
    MiMoV2ForCausalLM,
    MiMoV2MoE,
    _block_quantize_fp8_weight,
    _get_cp_v2_local_pad_size,
    _get_cp_v2_tp_pad_size,
    load_mimo_v2_qkv_proj_weight,
)
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMiMoV2WeightLoading(CustomTestCase):
    def test_cp_v2_local_padding_aligns_uneven_prefill_shards(self):
        self.assertEqual(_get_cp_v2_local_pad_size(135, cp_v2_active=True), 1)
        self.assertEqual(_get_cp_v2_local_pad_size(134, cp_v2_active=True), 2)
        self.assertEqual(_get_cp_v2_local_pad_size(136, cp_v2_active=True), 0)
        self.assertEqual(_get_cp_v2_local_pad_size(135, cp_v2_active=False), 0)
        self.assertEqual(_get_cp_v2_local_pad_size(127, cp_v2_active=True), 0)

    def test_cp_v2_tp_padding_uses_cp_group_max_tokens(self):
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                per_rank_actual_token=[1800, 1792, 1792, 1792]
            )
        )

        with patch("sglang.srt.models.mimo_v2.is_cp_v2_active", return_value=True):
            self.assertEqual(_get_cp_v2_tp_pad_size(1792, forward_batch), 8)
            self.assertEqual(_get_cp_v2_tp_pad_size(1800, forward_batch), 0)

    def test_cp_v2_tp_padding_aligns_moe_full_gathered_tokens(self):
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[213, 213, 212, 212])
        )

        with (
            patch("sglang.srt.models.mimo_v2.is_cp_v2_active", return_value=True),
            patch("sglang.srt.models.mimo_v2.get_moe_cp_size", return_value=4),
        ):
            self.assertEqual(
                _get_cp_v2_tp_pad_size(852, forward_batch, is_moe_full=True),
                4,
            )

    def test_fused_qkv_loader_accepts_runtime_tp_divisor(self):
        loaded_weight = torch.arange(16, dtype=torch.float32).view(8, 2)

        with get_parallel().override(attn_tp_size=4, attn_tp_rank=2):
            param = torch.nn.Parameter(torch.empty(2, 2))
            load_mimo_v2_qkv_proj_weight(
                "model.layers.0.self_attn.qkv_proj.weight",
                param,
                loaded_weight,
                expected_fused_tp_size=8,
            )

        self.assertTrue(torch.equal(param.data, loaded_weight.chunk(4, dim=0)[2]))

    def test_fused_qkv_loader_accepts_replicated_attention_tp(self):
        loaded_weight = torch.arange(16, dtype=torch.float32).view(8, 2)

        with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
            param = torch.nn.Parameter(torch.empty(8, 2))
            load_mimo_v2_qkv_proj_weight(
                "model.layers.0.self_attn.qkv_proj.weight",
                param,
                loaded_weight,
                expected_fused_tp_size=8,
            )

        self.assertTrue(torch.equal(param.data, loaded_weight))

    def test_fused_qkv_loader_delegates_padded_fp8_scale_tensor(self):
        loaded_weight = torch.arange(108 * 32, dtype=torch.float32).view(108, 32)
        param = torch.nn.Parameter(torch.empty(106, 32))
        loader_calls = []

        def fake_qkv_weight_loader(param, loaded_weight):
            loader_calls.append(tuple(loaded_weight.shape))
            param.data.copy_(loaded_weight[: param.shape[0]])

        param.weight_loader = fake_qkv_weight_loader

        with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
            load_mimo_v2_qkv_proj_weight(
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv",
                param,
                loaded_weight,
                expected_fused_tp_size=8,
            )

        self.assertEqual(loader_calls, [(108, 32)])
        self.assertTrue(torch.equal(param.data, loaded_weight[:106]))

    def test_block_quantize_fp8_weight_returns_runtime_scale_shape(self):
        weight = torch.randn(11, 5, dtype=torch.bfloat16)

        qweight, scale = _block_quantize_fp8_weight(
            weight,
            block_size=[4, 2],
            fp8_dtype=torch.float8_e4m3fn,
        )

        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scale.shape, (3, 3))
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)

    def test_fused_qkv_loader_rejects_incompatible_runtime_tp(self):
        loaded_weight = torch.arange(12, dtype=torch.float32).view(6, 2)

        with get_parallel().override(attn_tp_size=3, attn_tp_rank=0):
            param = torch.nn.Parameter(torch.empty(2, 2))
            with self.assertRaisesRegex(ValueError, "checkpoint TP size"):
                load_mimo_v2_qkv_proj_weight(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    param,
                    loaded_weight,
                    expected_fused_tp_size=8,
                )

    def test_language_only_skips_multimodal_weights(self):
        model = object.__new__(MiMoV2ForCausalLM)
        torch.nn.Module.__init__(model)
        model._is_multimodal = False
        model.config = SimpleNamespace(
            encoder_only=False,
            tie_word_embeddings=False,
            n_routed_experts=0,
            attention_projection_layout=None,
        )
        model.model = SimpleNamespace(start_layer=0, end_layer=1)
        model.pp_group = SimpleNamespace(world_size=1, is_last_rank=True)
        language_param = torch.nn.Parameter(torch.zeros(2, 2))

        with patch.object(
            model,
            "named_parameters",
            return_value=[("model.foo.weight", language_param)],
        ):
            model.load_weights(
                [
                    ("visual.patch_embed.proj.weight", torch.ones(2, 2)),
                    ("audio_encoder.foo.weight", torch.ones(2, 2)),
                    ("model.foo.weight", torch.full((2, 2), 3.0)),
                ]
            )

        self.assertTrue(torch.equal(language_param.data, torch.full((2, 2), 3.0)))

    def test_cp_v2_enabled_disables_mimo_legacy_cp_fallback(self):
        model = object.__new__(MiMoV2ForCausalLM)
        torch.nn.Module.__init__(model)
        model._is_multimodal = False
        model._MIN_LEGACY_CP_TOKENS = 64
        model.config = SimpleNamespace(encoder_only=False)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.attn_cp_rank = 0
        model.attn_cp_size = 4

        class LanguageModel:
            start_layer = 0
            end_layer = 1

            def __call__(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=None,
                pp_proxy_tensors=None,
            ):
                return torch.zeros(input_ids.shape[0], 2), None

        model.model = LanguageModel()
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=64,
            attn_cp_metadata=None,
            seq_lens_cpu=[64],
            extend_seq_lens_cpu=[64],
        )

        with (
            patch(
                "sglang.srt.models.mimo_v2.is_prefill_context_parallel_enabled",
                return_value=True,
            ),
            patch("sglang.srt.models.mimo_v2.enable_cp_v2", return_value=True),
            patch("sglang.srt.models.mimo_v2.is_cp_v2_active", return_value=False),
            patch("sglang.srt.models.mimo_v2.can_cp_split") as can_cp_split,
            patch(
                "sglang.srt.models.mimo_v2.prepare_context_parallel_metadata"
            ) as prepare_metadata,
        ):
            output = model.forward(
                torch.arange(64),
                torch.arange(64),
                forward_batch,
            )

        can_cp_split.assert_not_called()
        prepare_metadata.assert_not_called()
        self.assertIsNone(forward_batch.attn_cp_metadata)
        self.assertEqual(tuple(output.shape), (64, 2))

    def test_moe_forward_uses_moe_tp_all_reduce(self):
        moe = object.__new__(MiMoV2MoE)
        moe.tp_size = 2

        class Gate:
            def __call__(self, hidden_states):
                return torch.ones(hidden_states.shape[0], 4)

        class TopK:
            def __call__(self, hidden_states, router_logits):
                return router_logits

        class Experts:
            def __call__(self, hidden_states, topk_output):
                return hidden_states + 1

        moe.gate = Gate()
        moe.topk = TopK()
        moe.experts = Experts()
        hidden_states = torch.ones(3, 2)
        all_reduce_inputs = []

        def fake_moe_all_reduce(tensor):
            all_reduce_inputs.append(tensor.clone())
            return tensor + 10

        with (
            patch(
                "sglang.srt.models.mimo_v2.should_skip_post_experts_all_reduce",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.mimo_v2.moe_tensor_model_parallel_all_reduce",
                side_effect=fake_moe_all_reduce,
            ),
        ):
            output = moe.forward_normal(hidden_states)

        self.assertEqual(len(all_reduce_inputs), 1)
        self.assertTrue(torch.equal(all_reduce_inputs[0], hidden_states + 1))
        self.assertTrue(torch.equal(output, hidden_states + 11))

    def test_moe_forward_pads_cp_v2_local_tokens_before_experts(self):
        moe = object.__new__(MiMoV2MoE)
        moe._enable_a2a_moe = False
        moe.tp_size = 1
        seen_shapes = []

        class Gate:
            def __call__(self, hidden_states):
                seen_shapes.append(("gate", tuple(hidden_states.shape)))
                return torch.ones(hidden_states.shape[0], 4)

        class TopK:
            def __call__(self, hidden_states, router_logits):
                seen_shapes.append(("topk", tuple(hidden_states.shape)))
                return router_logits

        class Experts:
            def __call__(self, hidden_states, topk_output):
                seen_shapes.append(("experts", tuple(hidden_states.shape)))
                return hidden_states + 1

        moe.gate = Gate()
        moe.topk = TopK()
        moe.experts = Experts()
        hidden_states = torch.ones(852, 2)

        with patch("sglang.srt.models.mimo_v2.is_cp_v2_active", return_value=True):
            output = moe.forward(hidden_states, forward_batch=object())

        self.assertEqual(
            seen_shapes,
            [("gate", (856, 2)), ("topk", (856, 2)), ("experts", (856, 2))],
        )
        self.assertEqual(tuple(output.shape), (852, 2))
        self.assertTrue(torch.equal(output, hidden_states + 1))

    def test_moe_forward_pads_cp_v2_local_tokens_to_cp_group_max(self):
        moe = object.__new__(MiMoV2MoE)
        moe._enable_a2a_moe = False
        moe.tp_size = 1
        seen_shapes = []

        class Gate:
            def __call__(self, hidden_states):
                seen_shapes.append(("gate", tuple(hidden_states.shape)))
                return torch.ones(hidden_states.shape[0], 4)

        class TopK:
            def __call__(self, hidden_states, router_logits):
                seen_shapes.append(("topk", tuple(hidden_states.shape)))
                return router_logits

        class Experts:
            def __call__(self, hidden_states, topk_output):
                seen_shapes.append(("experts", tuple(hidden_states.shape)))
                return hidden_states + 1

        moe.gate = Gate()
        moe.topk = TopK()
        moe.experts = Experts()
        hidden_states = torch.ones(1792, 2)
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                per_rank_actual_token=[1800, 1792, 1792, 1792]
            )
        )

        with patch("sglang.srt.models.mimo_v2.is_cp_v2_active", return_value=True):
            output = moe.forward(hidden_states, forward_batch=forward_batch)

        self.assertEqual(
            seen_shapes,
            [("gate", (1800, 2)), ("topk", (1800, 2)), ("experts", (1800, 2))],
        )
        self.assertEqual(tuple(output.shape), (1792, 2))
        self.assertTrue(torch.equal(output, hidden_states + 1))


if __name__ == "__main__":
    unittest.main()
