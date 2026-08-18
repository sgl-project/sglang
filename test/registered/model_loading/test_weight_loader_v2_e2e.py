# Copyright 2023-2025 SGLang Team
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

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

import multiprocessing as mp

import torch

from sglang.srt.environ import envs
from sglang.test.runners import SRTRunner, check_close_model_outputs
from sglang.test.test_utils import CustomTestCase

MODEL = "Qwen/Qwen2-0.5B"
SHORT_PROMPT = "The capital of the United Kingdom is"


class TestWeightLoaderV2E2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _runner_kwargs(self):
        return dict(
            torch_dtype=torch.float16,
            model_type="generation",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            trust_remote_code=True,
            max_total_tokens=2048,
        )

    def test_qwen2_native_v1_v2_generation_match(self):
        prompts = [SHORT_PROMPT]
        max_new_tokens = 32
        kwargs = self._runner_kwargs()

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(False):
            with SRTRunner(MODEL, **kwargs) as runner_v1:
                out_v1 = runner_v1.forward(prompts, max_new_tokens=max_new_tokens)

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(True):
            with SRTRunner(MODEL, **kwargs) as runner_v2:
                out_v2 = runner_v2.forward(prompts, max_new_tokens=max_new_tokens)

        check_close_model_outputs(
            hf_outputs=out_v1,
            srt_outputs=out_v2,
            prefill_tolerance=1e-6,
            decode_tolerance=1e-6,
            rouge_l_tolerance=1.0,
            debug_text="qwen2 native v1 vs v2 weight loader",
        )

    def test_transformers_impl_loads_and_generates(self):
        prompts = [SHORT_PROMPT]
        max_new_tokens = 16

        with SRTRunner(
            MODEL,
            model_impl="transformers",
            **self._runner_kwargs(),
        ) as runner:
            outputs = runner.forward(prompts, max_new_tokens=max_new_tokens)

        self.assertEqual(len(outputs.output_strs), 1)
        self.assertGreater(len(outputs.output_strs[0]), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
