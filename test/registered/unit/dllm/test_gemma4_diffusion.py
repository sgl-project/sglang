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
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import gemma4_mm
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaForBlockDiffusion
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        dllm_is_encoder=encoder,
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
            forward_mode=ForwardMode.DLLM_EXTEND,
            extend_seq_lens=[6],
            extend_prefix_lens=[2],
            mm_inputs=[SimpleNamespace(mm_items=image_items)],
        )
        input_ids = torch.arange(6)

        with patch.object(
            gemma4_mm, "TritonAttnBackend", _FakeTritonBackend
        ), patch.object(gemma4_mm, "get_attn_backend", return_value=backend):
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                None, forward_batch, input_ids, torch.bool
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
                call(model, forward_batch, input_ids, torch.bool),
                call(model, forward_batch, input_ids, torch.bool),
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


class TestGemma4DiffusionWeightLoading(unittest.TestCase):
    def test_missing_weight_warning_ignores_derived_buffers(self):
        model = DiffusionGemmaForBlockDiffusion.__new__(DiffusionGemmaForBlockDiffusion)
        torch.nn.Module.__init__(model)
        model.text_config = SimpleNamespace(
            layer_types=[], num_experts=0, tie_word_embeddings=True
        )
        model.vision_tower = None
        model.register_parameter("learned_weight", torch.nn.Parameter(torch.ones(1)))
        model.register_buffer("derived_cache", torch.ones(1), persistent=False)

        with patch("sglang.srt.models.gemma4_diffusion.logger.warning") as warning:
            model.load_weights([])

        self.assertEqual(warning.call_args.args[1], ["learned_weight"])


if __name__ == "__main__":
    unittest.main()
