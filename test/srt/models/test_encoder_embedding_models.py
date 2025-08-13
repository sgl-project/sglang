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

# python -m unittest test_encoder_embedding_models.TestEncoderEmbeddingModels.test_prefill_logits

import multiprocessing as mp
import random
import time
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities, is_in_ci

MODELS = [("BAAI/bge-small-en", 1, 1e-5), ("BAAI/bge-m3", 1, 1e-5)]

ATTENTION_BACKEND = ["torch_native", "triton"]
BATCH_SIZE = [1, 2]
TORCH_DTYPES = [torch.float32]
sgl_to_st_ratio = []


class TestEncoderEmbeddingModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", 512) - 20

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated_text = tokenizer.decode(
                    tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                )
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)

        return truncated_prompts

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
        attention_backend,
        batch_size,
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)
        truncated_prompts = truncated_prompts * batch_size

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            # warm up
            hf_outputs = hf_runner.forward(truncated_prompts)

            st_start_time = time.perf_counter()
            hf_outputs = hf_runner.forward(truncated_prompts)
            st_end_time = time.perf_counter()

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend=attention_backend,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
        ) as srt_runner:
            # warm up
            srt_outputs = srt_runner.forward(truncated_prompts)

            sgl_start_time = time.perf_counter()
            srt_outputs = srt_runner.forward(truncated_prompts)
            sgl_end_time = time.perf_counter()

        transformer_time = st_end_time - st_start_time
        sgl_time = sgl_end_time - sgl_start_time
        sgl_to_st_ratio.append(sgl_time / transformer_time)

        for i in range(len(truncated_prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            # If something is wrong, uncomment this to observe similarity.
            # print("similarity diff", abs(similarity - 1))

            if len(truncated_prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model, tp_size, prefill_tolerance in models_to_test:
            for attention_backend in ATTENTION_BACKEND:
                for batch_size in BATCH_SIZE:
                    for torch_dtype in TORCH_DTYPES:
                        self.assert_close_prefill_logits(
                            DEFAULT_PROMPTS,
                            model,
                            tp_size,
                            torch_dtype,
                            prefill_tolerance,
                            attention_backend,
                            batch_size,
                        )

        for i in range(len(BATCH_SIZE)):
            print(
                "bacth size: ",
                BATCH_SIZE[i] * 5,
                "sgl_time/st_time",
                round(sgl_to_st_ratio[i], 3),
            )


if __name__ == "__main__":
    unittest.main()
