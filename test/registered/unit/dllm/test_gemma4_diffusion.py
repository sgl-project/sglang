# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import gemma4_diffusion, gemma4_mm
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaForBlockDiffusion
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeTritonBackend:
    def __init__(self):
        self.forward_metadata = SimpleNamespace(mask_indptr=None, custom_mask=None)


class _LanguageModelStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 4)
        self.input_embeds = []

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        self.input_embeds.append(input_embeds)
        if input_embeds is not None:
            return input_embeds
        return self.embed_tokens(input_ids)

    def self_conditioning(self, input_embeds, signal):
        return input_embeds + signal


class _LogitsProcessorStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = object()

    def forward(self, input_ids, hidden_states, lm_head, forward_batch):
        return self.output


def _diffusion_model_stub():
    model = DiffusionGemmaForBlockDiffusion.__new__(DiffusionGemmaForBlockDiffusion)
    torch.nn.Module.__init__(model)
    model.model = _LanguageModelStub()
    model.vision_tower = None
    model.embed_vision = None
    model.lm_head = model.model.embed_tokens
    model.logits_processor = _LogitsProcessorStub()
    return model


def _dispatch_batch(*, encoder):
    return SimpleNamespace(
        forward_mode=(ForwardMode.EXTEND if encoder else ForwardMode.DLLM_EXTEND),
        dllm_input_preparation_state=None,
        contains_image_inputs=lambda: True,
    )


class TestGemma4DiffusionImageMasks(unittest.TestCase):
    def test_mask_is_bidirectional_only_within_each_image_span(self):
        backend = _FakeTritonBackend()
        image_items = [
            SimpleNamespace(is_image=lambda: True, offsets=[(3, 5)]),
            SimpleNamespace(is_image=lambda: True, offsets=[(6, 7)]),
        ]
        forward_batch = SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=[6],
            extend_prefix_lens=[2],
            mm_inputs=[SimpleNamespace(mm_items=image_items)],
        )
        input_ids = torch.arange(6)
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(use_bidirectional_attention="vision")
            )
        )

        with (
            patch.object(gemma4_mm, "TritonAttnBackend", _FakeTritonBackend),
            patch.object(gemma4_mm, "get_attn_backend", return_value=backend),
        ):
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                model, forward_batch, input_ids, torch.bool
            )

        expected = torch.ones((6, 8), dtype=torch.bool).tril(diagonal=2)
        expected[1:4, 3:6] = True
        expected[4:6, 6:8] = True
        actual = backend.forward_metadata.custom_mask.view(6, 8)

        torch.testing.assert_close(
            backend.forward_metadata.mask_indptr,
            torch.tensor([0, 48], dtype=torch.int64),
        )
        torch.testing.assert_close(actual, expected)
        self.assertTrue(actual[1, 5])
        self.assertFalse(actual[1, 6])
        self.assertFalse(actual[0, 5])
        self.assertTrue(actual[4, 7])

    def test_encoder_prepares_image_mask_with_and_without_input_embeds(self):
        model = _diffusion_model_stub()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        forward_batch = _dispatch_batch(encoder=True)
        input_embeds = torch.randn(2, 4)

        with patch.object(
            Gemma4ForConditionalGeneration,
            "prepare_attn_masks",
            autospec=True,
        ) as prepare_attn_masks:
            self.assertIsNone(model(input_ids, positions, forward_batch))
            self.assertIsNone(
                model(
                    input_ids,
                    positions,
                    forward_batch,
                    input_embeds=input_embeds,
                )
            )

        self.assertEqual(
            prepare_attn_masks.call_args_list,
            [
                call(
                    model,
                    forward_batch,
                    input_ids,
                    torch.bool,
                    sliding_layers_only=False,
                ),
                call(
                    model,
                    forward_batch,
                    input_ids,
                    torch.bool,
                    sliding_layers_only=False,
                ),
            ],
        )
        self.assertIsNone(model.model.input_embeds[0])
        self.assertIs(model.model.input_embeds[1], input_embeds)

    def test_decoder_does_not_prepare_image_mask(self):
        model = _diffusion_model_stub()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        forward_batch = _dispatch_batch(encoder=False)

        with patch.object(
            Gemma4ForConditionalGeneration,
            "prepare_attn_masks",
            autospec=True,
        ) as prepare_attn_masks:
            output = model(input_ids, positions, forward_batch)

        prepare_attn_masks.assert_not_called()
        self.assertIs(output, model.logits_processor.output)

    def test_denoise_inputs_use_generic_input_embeds(self):
        model = _diffusion_model_stub()
        input_ids = torch.tensor([1, 2])
        signal = torch.full((2, 4), 0.25)

        actual = model.prepare_dllm_input_embeds(input_ids, signal)
        expected = model.model.embed_tokens(input_ids) + signal
        torch.testing.assert_close(actual, expected)

        without_signal = model.prepare_dllm_input_embeds(input_ids, None)
        torch.testing.assert_close(without_signal, model.model.embed_tokens(input_ids))


def _image_item(*offsets):
    return SimpleNamespace(is_image=lambda: True, offsets=list(offsets))


def _fa_batch(extend_lens, prefix_lens, mm_items_per_req):
    return SimpleNamespace(
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        mm_inputs=[SimpleNamespace(mm_items=items) for items in mm_items_per_req],
    )


def _prepare_on(backend, forward_batch):
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(use_bidirectional_attention="vision")
        )
    )
    with patch.object(gemma4_mm, "get_attn_backend", return_value=backend):
        Gemma4ForConditionalGeneration.prepare_attn_masks(
            model, forward_batch, torch.zeros(1, dtype=torch.int64), torch.bfloat16
        )


class TestGemma4Fa4ImageTokenRanges(unittest.TestCase):
    """The FA4 metadata path: fa4, and fa3 upgraded to FA4 for the 512-wide
    global layers, both emit image_token_ranges; other non-triton backends
    fall back to causal attention and leave the metadata alone."""

    @staticmethod
    def _backend(**flags):
        return SimpleNamespace(forward_metadata=SimpleNamespace(), **flags)

    def test_fa3_upgraded_to_fa4_emits_image_token_ranges(self):
        # A fresh request with an image at [2, 5] and a chunked one whose
        # image at absolute [5, 7] sits inside the extend window.
        backend = self._backend(fa3_upgrades_to_fa4=True)
        _prepare_on(
            backend,
            _fa_batch((8, 5), (0, 4), ([_image_item((2, 5))], [_image_item((5, 7))])),
        )

        ranges = backend.forward_metadata.image_token_ranges
        self.assertEqual(ranges.dtype, torch.int32)
        expected = torch.full((13, 2), -1, dtype=torch.int32)
        expected[2:6] = torch.tensor([2, 5], dtype=torch.int32)
        expected[8 + 1 : 8 + 4] = torch.tensor([5, 7], dtype=torch.int32)
        self.assertTrue(torch.equal(ranges, expected), (ranges, expected))

    def test_plain_fa3_without_upgrade_leaves_metadata_alone(self):
        backend = self._backend(fa_impl_ver=3, fa3_upgrades_to_fa4=False)
        _prepare_on(backend, _fa_batch((4,), (0,), ([_image_item((1, 2))],)))
        self.assertFalse(hasattr(backend.forward_metadata, "image_token_ranges"))


class TestGemma4DiffusionExpertActivation(unittest.TestCase):
    def test_decoder_passes_checkpoint_activation_to_experts(self):
        for activation, expected in (
            ("gelu_pytorch_tanh", "gelu_tanh"),
            ("gelu", "gelu"),
        ):
            with self.subTest(activation=activation), ExitStack() as stack:
                config = SimpleNamespace(
                    layer_types=["full_attention"],
                    head_dim=2,
                    num_key_value_heads=1,
                    hidden_size=4,
                    intermediate_size=8,
                    hidden_activation=activation,
                    rms_norm_eps=1e-6,
                )
                for name in (
                    "DiffusionGemmaAttention",
                    "Gemma3MLP",
                    "Gemma4Router",
                    "RMSNorm",
                ):
                    stack.enter_context(
                        patch.object(
                            gemma4_diffusion, name, return_value=torch.nn.Identity()
                        )
                    )
                experts = stack.enter_context(
                    patch.object(
                        gemma4_diffusion, "Gemma4MoE", return_value=torch.nn.Identity()
                    )
                )
                gemma4_diffusion.DiffusionGemmaDecoderLayer(0, config)
                self.assertEqual(experts.call_args.kwargs["activation"], expected)


class TestGemma4DiffusionWeightLoading(unittest.TestCase):
    def test_missing_weight_warning_ignores_derived_buffers(self):
        model = DiffusionGemmaForBlockDiffusion.__new__(DiffusionGemmaForBlockDiffusion)
        torch.nn.Module.__init__(model)
        model.text_config = SimpleNamespace(
            layer_types=[], num_experts=0, tie_word_embeddings=True
        )
        model.vision_tower = None
        model.register_parameter("learned_weight", torch.nn.Parameter(torch.ones(1)))
        skipped_param = torch.nn.Parameter(torch.ones(1))
        skipped_param._skip_weight_check = True
        model.register_parameter("k_scale", skipped_param)
        model.register_parameter(
            "w13_blockscale_swizzled", torch.nn.Parameter(torch.ones(1))
        )
        model.register_buffer("derived_cache", torch.ones(1), persistent=False)

        with patch("sglang.srt.models.gemma4_diffusion.logger.warning") as warning:
            model.load_weights([])

        self.assertEqual(warning.call_args.args[1], ["learned_weight"])

    def test_per_expert_quantized_and_full_attn_k_to_v_loading(self):
        model = DiffusionGemmaForBlockDiffusion.__new__(DiffusionGemmaForBlockDiffusion)
        torch.nn.Module.__init__(model)
        model.text_config = SimpleNamespace(
            layer_types=["full_attention"], num_experts=2, tie_word_embeddings=True
        )
        model.vision_tower = None

        recorded_calls = []

        def _make_tracked_param(tag):
            p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            def _loader(param, loaded_weight, *args, **kwargs):
                recorded_calls.append((tag, float(loaded_weight.item()), args, kwargs))

            p.weight_loader = _loader
            return p

        params = {
            "model.layers.0.moe.experts.w13_weight": _make_tracked_param("w13_weight"),
            "model.layers.0.moe.experts.w13_weight_scale": _make_tracked_param(
                "w13_weight_scale"
            ),
            "model.layers.0.moe.experts.w13_weight_scale_2": _make_tracked_param(
                "w13_weight_scale_2"
            ),
            "model.layers.0.moe.experts.w13_input_scale": _make_tracked_param(
                "w13_input_scale"
            ),
            "model.layers.0.moe.experts.w2_weight": _make_tracked_param("w2_weight"),
            "model.layers.0.moe.experts.w2_weight_scale": _make_tracked_param(
                "w2_weight_scale"
            ),
            "model.layers.0.self_attn.qkv_proj.weight": _make_tracked_param(
                "qkv_weight"
            ),
            "model.layers.0.self_attn.qkv_proj.weight_scale": _make_tracked_param(
                "qkv_weight_scale"
            ),
            "model.layers.0.self_attn.attn.k_scale": _make_tracked_param("attn_k_scale"),
            "model.layers.0.self_attn.attn.v_scale": _make_tracked_param("attn_v_scale"),
            "model.layers.0.self_attn.attn_bidir.k_scale": _make_tracked_param(
                "bidir_k_scale"
            ),
            "model.layers.0.self_attn.attn_bidir.v_scale": _make_tracked_param(
                "bidir_v_scale"
            ),
        }

        with (
            patch.object(
                model, "named_parameters", return_value=list(params.items())
            ),
            patch.object(model, "named_buffers", return_value=[]),
            patch("sglang.srt.models.gemma4_diffusion.logger.warning") as warning,
        ):
            loaded = model.load_weights(
                [
                    (
                        "model.decoder.layers.0.experts.0.gate_proj.weight",
                        torch.tensor([1.0]),
                    ),
                    (
                        "model.decoder.layers.0.experts.0.gate_proj.weight_scale",
                        torch.tensor([2.0]),
                    ),
                    (
                        "model.decoder.layers.0.experts.0.up_proj.weight_scale_2",
                        torch.tensor([3.0]),
                    ),
                    (
                        "model.decoder.layers.0.experts.1.down_proj.weight",
                        torch.tensor([4.0]),
                    ),
                    (
                        "model.decoder.layers.0.experts.1.down_proj.weight_scale",
                        torch.tensor([5.0]),
                    ),
                    (
                        "model.decoder.layers.0.experts.1.gate_proj.input_scale",
                        torch.tensor([6.0]),
                    ),
                    (
                        "model.decoder.layers.0.self_attn.q_proj.weight",
                        torch.tensor([7.0]),
                    ),
                    (
                        "model.decoder.layers.0.self_attn.k_proj.weight",
                        torch.tensor([8.0]),
                    ),
                    (
                        "model.decoder.layers.0.self_attn.k_proj.weight_scale",
                        torch.tensor([9.0]),
                    ),
                    (
                        "model.decoder.layers.0.self_attn.k_proj.k_scale",
                        torch.tensor([10.0]),
                    ),
                ]
            )

        warning.assert_not_called()
        self.assertEqual(loaded, set(params.keys()))
        self.assertIn(
            (
                "w13_weight",
                1.0,
                ("model.layers.0.moe.experts.w13_weight",),
                {"shard_id": "w1", "expert_id": 0},
            ),
            recorded_calls,
        )
        self.assertIn(
            (
                "w13_weight_scale",
                2.0,
                ("model.layers.0.moe.experts.w13_weight_scale",),
                {"shard_id": "w1", "expert_id": 0},
            ),
            recorded_calls,
        )
        self.assertIn(
            (
                "w13_weight_scale_2",
                3.0,
                ("model.layers.0.moe.experts.w13_weight_scale_2",),
                {"shard_id": "w3", "expert_id": 0},
            ),
            recorded_calls,
        )
        self.assertIn(
            (
                "w2_weight",
                4.0,
                ("model.layers.0.moe.experts.w2_weight",),
                {"shard_id": "w2", "expert_id": 1},
            ),
            recorded_calls,
        )
        self.assertIn(
            (
                "w2_weight_scale",
                5.0,
                ("model.layers.0.moe.experts.w2_weight_scale",),
                {"shard_id": "w2", "expert_id": 1},
            ),
            recorded_calls,
        )
        self.assertIn(
            (
                "w13_input_scale",
                6.0,
                ("model.layers.0.moe.experts.w13_input_scale",),
                {"shard_id": "w1", "expert_id": 1},
            ),
            recorded_calls,
        )
        # k_proj.weight and k_proj.weight_scale must duplicate into both "k" and "v" shards
        self.assertIn(("qkv_weight", 8.0, ("k",), {}), recorded_calls)
        self.assertIn(("qkv_weight", 8.0, ("v",), {}), recorded_calls)
        self.assertIn(("qkv_weight_scale", 9.0, ("k",), {}), recorded_calls)
        self.assertIn(("qkv_weight_scale", 9.0, ("v",), {}), recorded_calls)
        # k_proj.k_scale on a full-attention layer populates k_scale & v_scale on attn & attn_bidir
        for scale_tag in (
            "attn_k_scale",
            "attn_v_scale",
            "bidir_k_scale",
            "bidir_v_scale",
        ):
            self.assertIn((scale_tag, 10.0, (), {}), recorded_calls)


class TestGemma4DiffusionQuantConfigMapping(unittest.TestCase):
    def test_compressed_tensors_fp8_ignore_mapping(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from sglang.srt.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        raw_quant_cfg = {
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "input_activations": {
                        "dynamic": True,
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "float",
                    },
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": False,
                        "num_bits": 8,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "float",
                    },
                }
            },
            "format": "float-quantized",
            "ignore": [
                "lm_head",
                "model.decoder.layers.0.router.proj",
                "model.decoder.self_conditioning.down_proj",
                "model.decoder.self_conditioning.gate_proj",
                "model.decoder.self_conditioning.up_proj",
                "model.encoder.embed_vision.embedding_projection",
                "model.encoder.language_model.layers.0.router.proj",
                "model.encoder.vision_tower.encoder.layers.0.mlp.down_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.mlp.up_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.self_attn.k_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.self_attn.o_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                "model.encoder.vision_tower.encoder.layers.0.self_attn.v_proj.linear",
                "model.encoder.vision_tower.patch_embedder.input_proj",
            ],
            "packed_modules_mapping": DiffusionGemmaForBlockDiffusion.packed_modules_mapping,
        }
        qcfg = CompressedTensorsConfig.from_config(raw_quant_cfg)
        qcfg.apply_weight_name_mapper(
            DiffusionGemmaForBlockDiffusion.hf_to_sglang_mapper
        )

        ignored_sglang_prefixes = [
            "lm_head",
            "model.layers.0.router.proj",
            "model.self_conditioning.gate_up_proj",
            "model.self_conditioning.down_proj",
            "embed_vision.embedding_projection",
            "vision_tower.patch_embedder.input_proj",
            "vision_tower.encoder.layers.0.self_attn.qkv_proj",
            "vision_tower.encoder.layers.0.self_attn.o_proj.linear",
            "vision_tower.encoder.layers.0.mlp.gate_up_proj",
            "vision_tower.encoder.layers.0.mlp.down_proj.linear",
        ]
        for pfx in ignored_sglang_prefixes:
            self.assertTrue(
                should_ignore_layer(
                    pfx, ignore=qcfg.ignore, fused_mapping=qcfg.packed_modules_mapping
                ),
                f"Expected {pfx} to be ignored in CompressedTensorsConfig",
            )

        quantized_sglang_prefixes = [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.moe.experts.0.gate_proj",
            "model.layers.0.moe.experts.0.up_proj",
            "model.layers.0.moe.experts.0.down_proj",
        ]
        for pfx in quantized_sglang_prefixes:
            self.assertFalse(
                should_ignore_layer(
                    pfx, ignore=qcfg.ignore, fused_mapping=qcfg.packed_modules_mapping
                ),
                f"Expected {pfx} to remain quantized in CompressedTensorsConfig",
            )

    def test_modelopt_nvfp4_exclusion_mapping(self):
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        qcfg = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": [
                        "lm_head",
                        "*embed_vision*",
                        "*mlp*",
                        "*router*",
                        "*self_attn*",
                        "*self_conditioning*",
                        "*vision_tower*",
                        "model.decoder.layers.0.experts.99",
                    ],
                },
                "packed_modules_mapping": DiffusionGemmaForBlockDiffusion.packed_modules_mapping,
            }
        )
        qcfg.apply_weight_name_mapper(
            DiffusionGemmaForBlockDiffusion.hf_to_sglang_mapper
        )

        for excluded in (
            "lm_head",
            "embed_vision.embedding_projection",
            "vision_tower.patch_embedder.input_proj",
            "vision_tower.encoder.layers.0.self_attn.qkv_proj",
            "model.self_conditioning.gate_up_proj",
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.router.proj",
            "model.layers.0.moe.experts.99",
        ):
            self.assertTrue(
                qcfg.is_layer_excluded(excluded),
                f"Expected {excluded} to be excluded in ModelOptQuantConfig",
            )

        self.assertFalse(qcfg.is_layer_excluded("model.layers.0.moe.experts"))


if __name__ == "__main__":
    unittest.main()
