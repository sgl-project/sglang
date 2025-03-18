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
"""
Usage:

To test a specific model:
1. Add it to ALL_OTHER_MODELS
2. Run `ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others`
"""

import dataclasses
import multiprocessing as mp
import os
import unittest
from typing import List

import torch

from sglang.test.runners import HFRunner, SRTRunner, check_close_model_outputs


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False


MODELS = [
    ModelCase("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    ModelCase("lmms-lab/llava-onevision-qwen2-72b-ov-chat"),
    ModelCase("Qwen/Qwen2-VL-7B-Instruct"),
]

TORCH_DTYPES = [torch.float16]

PROMPTS = [
    "Describe this image in detail",
]

IMAGES = [
    "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
]


class TestVisionLangModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        image_data: List[str],
        model_case: ModelCase,
        torch_dtype: torch.dtype,
    ) -> None:
        model_path = model_case.model_path
        prefill_tolerance, decode_tolerance, rouge_l_tolerance = (
            model_case.prefill_tolerance,
            model_case.decode_tolerance,
            model_case.rouge_l_tolerance,
        )
        max_new_tokens = 32

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation_with_image",  # Handle image input to generation model
            trust_remote_code=model_case.trust_remote_code,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts, image_data=image_data, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            model_path,
            tp_size=model_case.tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts, image_data=image_data, max_new_tokens=max_new_tokens
            )

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=prefill_tolerance,
            decode_tolerance=decode_tolerance,
            rouge_l_tolerance=rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={prompts} image_data={image_data}",
        )

    def test_models(self):
        for model_case in MODELS:
            # Only run a specified model
            if (
                "ONLY_RUN" in os.environ
                and os.environ["ONLY_RUN"] != model_case.model_path
            ):
                continue

            # Skip long prompts for models that do not have a long context
            prompts = PROMPTS
            image_data = IMAGES
            if model_case.skip_long_prompt:
                prompts = []
                image_data = []
                for i, p in enumerate(PROMPTS):
                    if len(p) < 1000:
                        prompts.append(p)
                        image_data.append(IMAGES[i])

            # Assert the logits and output strs are close
            self.assert_close_logits_and_output_strs(
                prompts, image_data, model_case, torch.float16
            )


if __name__ == "__main__":
    unittest.main()
