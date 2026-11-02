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


"""Image graph regression for Bolt E4B BF16.

Checkpoint: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16.
CPU fixtures reproduce the image-token lookup without downloading weights.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.mm_utils import embed_mm_inputs  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models import gemma4_mm  # noqa: E402
from sglang.srt.models.gemma4_causal import Gemma4TextModel  # noqa: E402
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ZeroProjection(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros_like(inputs), None


class TestGemma4MultimodalGraphInputs(CustomTestCase):
    def test_embedding_clamp_preserves_graph_per_layer_lookup(self):
        language_model = Gemma4TextModel.__new__(Gemma4TextModel)
        torch.nn.Module.__init__(language_model)
        language_model.config = SimpleNamespace(num_hidden_layers=1)
        language_model.vocab_size_per_layer_input = 16
        language_model.hidden_size_per_layer_input = 2
        language_model.embed_tokens_per_layer = torch.nn.Embedding.from_pretrained(
            torch.arange(32, dtype=torch.float32).reshape(16, 2)
        )
        language_model.pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=False
        )
        language_model.start_layer = language_model.end_layer = 0
        language_model.layers_to_capture = set()
        language_model.per_layer_model_projection = _ZeroProjection()
        language_model.per_layer_projection_norm = torch.nn.Identity()
        language_model.per_layer_projection_scale = 1.0
        language_model.per_layer_input_scale = 1.0
        model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.language_model = language_model
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        model.config = SimpleNamespace(
            text_config=SimpleNamespace(pad_token_id=0),
            image_token_id=16,
            video_token_id=17,
            audio_token_id=18,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            contains_image_inputs=lambda: False,
            contains_mm_inputs=lambda: True,
        )
        # Image placeholders are hashes outside the embedding vocabulary.
        static_ids = torch.tensor([2, 1000, 3])
        expected = language_model.get_per_layer_inputs(static_ids)

        def embed_then_replay(input_ids, per_layer_inputs, **kwargs):
            embeddings, _ = embed_mm_inputs(
                mm_inputs_list=[],
                extend_prefix_lens=[],
                extend_seq_lens=[],
                input_ids=input_ids,
                input_embedding=torch.nn.Embedding(16, 2),
            )
            torch.testing.assert_close(per_layer_inputs, expected)
            replay = language_model(
                static_ids,
                torch.arange(3),
                batch,
                input_embeds=embeddings,
            )
            torch.testing.assert_close(replay.tensors["hidden_states"], embeddings)
            return replay.tensors["per_layer_inputs"]

        with patch.object(
            gemma4_mm, "general_mm_embed_routine", side_effect=embed_then_replay
        ):
            actual = model(static_ids, torch.arange(3), batch)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(static_ids.tolist(), [2, 1000, 3])


if __name__ == "__main__":
    unittest.main()
