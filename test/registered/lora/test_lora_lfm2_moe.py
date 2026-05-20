# Copyright 2023-2026 SGLang Team
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
"""Smoke test for LFM2-MoE LoRA support.

Loads the published PEFT 0.18+ smoltalk adapter (which targets the batched
3D MoE expert weights via `target_parameters`) into SGLang and asserts the
LoRA-on output differs from the base output. Exercises the full
`Lfm2MoeForCausalLM` LoRA path end-to-end: attention, ShortConv, dense MLP,
MoE router, and the FusedMoEWithLoRA expert codepath.

Usage:
    python test/registered/lora/test_lora_lfm2_moe.py
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")

import sglang as sgl

BASE_MODEL = "LiquidAI/LFM2-8B-A1B"
LORA_REPO = "LiquidAI/LFM2-8B-A1B-smoltalk-LoRA"
LORA_NAME = "smoltalk"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
    "in_proj",
    "gate",
    "gate_up_proj",
    "down_proj",
]
PROMPT = (
    "What are some ideas for a good short story about a city not on a planet, "
    "but rather a generation ship, or on the moon of a gas giant, "
    "or somewhere else unusual?"
)
SAMPLING = {"max_new_tokens": 32, "temperature": 0.0}


class TestLFM2MoELoRA(CustomTestCase):
    def test_lora_changes_outputs(self):
        engine = sgl.Engine(
            model_path=BASE_MODEL,
            enable_lora=True,
            max_lora_rank=8,
            lora_paths=[f"{LORA_NAME}={LORA_REPO}"],
            lora_target_modules=LORA_TARGET_MODULES,
        )
        try:
            base = engine.generate(PROMPT, sampling_params=SAMPLING)["text"]
            lora = engine.generate(
                PROMPT, sampling_params=SAMPLING, lora_path=LORA_NAME
            )["text"]
        finally:
            engine.shutdown()
        self.assertNotEqual(
            base,
            lora,
            "LoRA-on and base outputs should differ; if they're identical, the adapter "
            "isn't reaching the forward path.",
        )


if __name__ == "__main__":
    unittest.main()
