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

import copy
import unittest

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from sglang.multimodal_gen.runtime.models.vlas.activevln import (
    ActiveVLNModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def tiny_qwen(attention_backend="sdpa"):
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 97,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
        image_token_id=94,
        video_token_id=95,
        vision_start_token_id=93,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = attention_backend
    return Qwen2_5_VLForConditionalGeneration(config).eval()


class TestActiveVLN(CustomTestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
        self.positions = torch.arange(6)[None, None].expand(3, 1, -1)

    def test_logits_and_gradients_match_transformers(self):
        reference = tiny_qwen()
        model = ActiveVLNModel(copy.deepcopy(reference))
        actual = model(self.ids, self.positions)
        actual = model.qwen.lm_head(actual)
        expected = reference(
            input_ids=self.ids, position_ids=self.positions, use_cache=False
        ).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        reference_parameters = dict(reference.named_parameters())
        checked = 0
        for name, parameter in model.qwen.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(
                    parameter.grad,
                    reference_parameters[name].grad,
                    atol=1e-7,
                    rtol=2e-4,
                )
                checked += 1
        self.assertGreater(checked, 10)
        self.assertTrue(all(p.grad is None for p in model.visual.parameters()))

    def test_image_embeddings_and_multiaxis_positions_match_transformers(self):
        reference = tiny_qwen()
        model = ActiveVLNModel(copy.deepcopy(reference))
        ids = torch.tensor([[5, 94, 94, 8, 9, 10]])
        positions = torch.tensor(
            [
                [[0, 1, 1, 2, 3, 4]],
                [[0, 1, 2, 3, 4, 5]],
                [[0, 2, 1, 3, 4, 5]],
            ]
        )
        features = torch.randn(2, 32)
        embedded = reference.get_input_embeddings()(ids)
        embedded = embedded.masked_scatter(
            (ids == 94).unsqueeze(-1).expand_as(embedded), features
        )
        expected = reference(
            inputs_embeds=embedded, position_ids=positions, use_cache=False
        ).logits
        actual = model.project(model(ids, positions, image_features=features))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_checkpointed_backward_matches_eager(self):
        for layers in (-1, 1):
            with self.subTest(train_last_n_layers=layers):
                eager = ActiveVLNModel(tiny_qwen(), train_last_n_layers=layers)
                checkpointed = copy.deepcopy(eager)
                checkpointed.gradient_checkpointing = True
                for model in (eager, checkpointed):
                    model(self.ids, self.positions).square().mean().backward()
                for a, b in zip(eager.parameters(), checkpointed.parameters()):
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_response_scores_include_eos_and_exclude_prompt(self):
        model = ActiveVLNModel(tiny_qwen(), train_last_n_layers=1)
        self.ids[0, -1] = 2
        probs, entropy = model.response_logprobs(
            self.ids, self.positions, 3, temperature=0.7, compute_entropy=True
        )
        logits = model.project(model(self.ids, self.positions))[:, 2:-1].float() / 0.7
        expected = logits.log_softmax(-1).gather(-1, self.ids[:, 3:, None]).squeeze(-1)
        torch.testing.assert_close(probs, expected)
        self.assertTrue(torch.isfinite(entropy).all())
        (-probs.sum()).backward()
        self.assertIsNone(model.text.layers[0].self_attn.q_proj.weight.grad)
        self.assertGreater(
            model.text.layers[1].self_attn.q_proj.weight.grad.abs().max(), 0
        )
        self.assertIsNone(model.qwen.lm_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.visual.parameters()))

    def test_export_preserves_checkpoint_tensors_and_trainable_selection(self):
        model = ActiveVLNModel(tiny_qwen(), train_last_n_layers=1)
        exported = dict(model.named_srt_weights())
        trainable = dict(model.named_srt_weights(trainable_only=True))
        self.assertIn("model.layers.0.self_attn.q_proj.weight", exported)
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", trainable)
        self.assertIn("model.layers.1.self_attn.q_proj.weight", trainable)
        self.assertIn("model.norm.weight", trainable)
        self.assertIn("visual.patch_embed.proj.weight", exported)
        self.assertFalse(any("visual" in name for name in trainable))
        torch.testing.assert_close(
            exported["model.layers.1.self_attn.q_proj.weight"],
            model.text.layers[1].self_attn.q_proj.weight,
            atol=0,
            rtol=0,
        )

    def test_invalid_response_and_image_layout(self):
        model = ActiveVLNModel(tiny_qwen())
        with self.assertRaisesRegex(ValueError, "nonempty response"):
            model.response_logprobs(self.ids, self.positions, self.ids.shape[1])
        with self.assertRaisesRegex(ValueError, "positive"):
            model.response_logprobs(self.ids, self.positions, 3, temperature=0)
        ids = self.ids.clone()
        ids[0, 1] = 94
        with self.assertRaisesRegex(ValueError, "Image tokens"):
            model(ids, self.positions)


if __name__ == "__main__":
    unittest.main()
