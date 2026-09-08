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
"""MoE LoRA under the breakable prefill CUDA graph must match eager prefill."""

import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")

# Measured graph noise is <=0.25; a missing adapter changes logprobs by 7-17.
PROMPT_LOGPROB_THRESHOLD = 1.0
MAX_NEW_TOKENS = 8
# Keep padded buckets within the runner's 2x token-count limit.
PREFILL_GRAPH_BATCH_SIZES = [32, 64, 128, 256, 512, 1024]


class TestMoELoRAPrefillCudaGraph(CustomTestCase):
    def test_breakable_prefill_graph_matches_eager(self):
        prompts = MOE_LORA_TEST_PROMPTS
        lora_paths = ["moe_lora"] * len(prompts)
        results = {}
        for prefill_graph in (False, True):
            kwargs = dict(
                model_path=MOE_BASE_MODEL_PATH,
                enable_lora=True,
                lora_paths={"moe_lora": MOE_LORA_PATH},
                max_loras_per_batch=1,
                lora_backend="triton",
                trust_remote_code=True,
                enable_tokenizer_batch_encode=True,
                enable_metrics=prefill_graph,
                disable_radix_cache=True,
                mem_fraction_static=0.8,
                cuda_graph_max_bs_decode=4,
                cuda_graph_backend_prefill="breakable" if prefill_graph else "disabled",
            )
            if prefill_graph:
                kwargs["cuda_graph_bs_prefill"] = PREFILL_GRAPH_BATCH_SIZES

            engine = sgl.Engine(**kwargs)
            try:
                # Compare prefill outputs before decoding.
                prompt_out = engine.generate(
                    prompts,
                    sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                    return_logprob=True,
                    logprob_start_len=0,
                    lora_path=lora_paths,
                )
                prompt_logprobs = [
                    torch.tensor(
                        [lp for lp, _, _ in o["meta_info"]["input_token_logprobs"][1:]]
                    )
                    for o in prompt_out
                ]
                if prefill_graph:
                    from prometheus_client import CollectorRegistry, multiprocess

                    # Wait for scheduler metric reporting after the response.
                    engine.get_server_info()
                    registry = CollectorRegistry()
                    multiprocess.MultiProcessCollector(registry)
                    replays = sum(
                        sample.value
                        for metric in registry.collect()
                        for sample in metric.samples
                        if sample.name == "sglang:cuda_graph_passes_total"
                        and sample.labels.get("mode") == "prefill_cuda_graph"
                    )
                    self.assertGreaterEqual(
                        replays,
                        1,
                        "MoE LoRA fell back to eager prefill",
                    )
                gen_out = engine.generate(
                    prompts,
                    sampling_params={
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0.0,
                    },
                    lora_path=lora_paths,
                )
                results[prefill_graph] = {
                    "prompt_logprobs": prompt_logprobs,
                    "texts": [o["text"] for o in gen_out],
                }
            finally:
                engine.shutdown()
            if not prefill_graph:
                torch.cuda.empty_cache()

        eager, graph = results[False], results[True]
        for i, prompt in enumerate(prompts):
            e_lp, g_lp = eager["prompt_logprobs"][i], graph["prompt_logprobs"][i]
            self.assertEqual(e_lp.numel(), g_lp.numel(), f"prompt {i}: token count")
            max_diff = (e_lp - g_lp).abs().max().item()
            self.assertLess(
                max_diff,
                PROMPT_LOGPROB_THRESHOLD,
                f"prompt {i} ({prompt[:40]!r}): prefill graph logprobs drift "
                f"{max_diff:.2e} from eager",
            )
            self.assertEqual(
                eager["texts"][i],
                graph["texts"][i],
                f"prompt {i}: greedy continuation differs under the prefill graph",
            )


if __name__ == "__main__":
    unittest.main()
