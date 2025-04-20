# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

import multiprocessing as mp
import unittest

import torch
from transformers import AutoProcessor

from sglang.srt.utils import load_image
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import get_similarities

TEXTS = "two Subway Series sandwiches with meats, cheese, lettuce, tomatoes, and onions on a black background, accompanied by the Subway Series logo, highlighting a new sandwich series."
IMAGES = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"
MODELS = [
    ("openai/clip-vit-large-patch14-336", 1e-5),
]
TORCH_DTYPES = [torch.float16]


class TestClipModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_embeddings(self, model, prefill_tolerance, torch_dtype):

        with HFRunner(
            model,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            hf_text_embeds = hf_runner.forward(prompts=TEXTS)
            hf_image_embeds = hf_runner.forward(image_data=IMAGES)

        with SRTRunner(
            model,
            tp_size=1,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as srt_runner:
            text_embeds = srt_runner.forward(prompts=TEXTS)
            image_embeds = srt_runner.forward(prompts="padding", image_data=IMAGES)

        text_similarity = get_similarities(
            text_embeds.embed_logits[0], hf_text_embeds.embed_logits[0]
        )
        image_similarity = get_similarities(
            image_embeds.embed_logits[0], hf_image_embeds.embed_logits[0]
        )
        print("text similarity diff", abs(text_similarity - 1))
        print("image similarity diff", abs(image_similarity - 1))
        assert torch.all(
            abs(text_similarity - 1) < prefill_tolerance
        ), "embeddings are not all close"
        assert torch.all(
            abs(image_similarity - 1) < prefill_tolerance
        ), "embeddings are not all close"

    def test_accuracy(self):
        for model, prefill_tolerance in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_embeddings(model, prefill_tolerance, torch_dtype)


if __name__ == "__main__":
    unittest.main()
