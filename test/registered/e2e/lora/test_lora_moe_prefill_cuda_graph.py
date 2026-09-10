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
"""MoE LoRA prefill graphs must replay adapters and match eager prefill."""

import os
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

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")

# Missing adapters shift logprobs by 7-17.
LOGPROB_THRESHOLD = 1.0
MAX_NEW_TOKENS = 8
# Keep padded buckets within the runner's 2x token-count limit.
PREFILL_GRAPH_BATCH_SIZES = [512, 1024, 2048, 4096]


class TestMoELoRAPrefillCudaGraph(CustomTestCase):
    def test_prefill_graph_matches_eager(self):
        from prometheus_client import REGISTRY

        prompts = (MOE_LORA_TEST_PROMPTS * 4)[:64]
        lora_paths = [None if i % 3 == 1 else "moe_lora" for i in range(len(prompts))]
        results = {}
        for lora_backend, backend in (
            ("triton", "disabled"),
            ("triton", "breakable"),
            ("triton", "full"),
            ("csgmv", "breakable"),
            ("csgmv", "full"),
        ):
            label = f"{lora_backend}/{backend}"
            prefill_graph = backend != "disabled"
            if prefill_graph:
                # Isolate replay counts between engines.
                os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            kwargs = dict(
                model_path=MOE_BASE_MODEL_PATH,
                enable_lora=True,
                lora_paths={"moe_lora": MOE_LORA_PATH},
                max_loras_per_batch=2,
                lora_backend=lora_backend,
                attention_backend="flashinfer",
                trust_remote_code=True,
                enable_tokenizer_batch_encode=True,
                enable_metrics=prefill_graph,
                disable_radix_cache=True,
                mem_fraction_static=0.8,
                max_running_requests=len(prompts),
                chunked_prefill_size=4096,
                cuda_graph_max_bs_decode=4,
                cuda_graph_backend_prefill=backend,
            )
            if prefill_graph:
                kwargs["cuda_graph_bs_prefill"] = PREFILL_GRAPH_BATCH_SIZES
            if backend == "full":
                kwargs["cuda_graph_config"] = {
                    "prefill": {"full_prefill_max_req": len(prompts)}
                }

            collectors_before = set(REGISTRY._collector_to_names)
            engine = None
            try:
                engine = sgl.Engine(**kwargs)
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
                gen_out = engine.generate(
                    prompts,
                    sampling_params={
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0.0,
                    },
                    lora_path=lora_paths,
                    return_logprob=True,
                    logprob_start_len=-1,
                    top_logprobs_num=5,
                )
                if prefill_graph:
                    from prometheus_client import CollectorRegistry, multiprocess

                    # Wait for scheduler metric reporting after the response.
                    engine.get_server_info()
                    registry = CollectorRegistry()
                    multiprocess.MultiProcessCollector(registry)
                    samples = [
                        sample
                        for metric in registry.collect()
                        for sample in metric.samples
                        if sample.name == "sglang:cuda_graph_passes_total"
                    ]
                    passes = {
                        mode: sum(
                            sample.value
                            for sample in samples
                            if sample.labels.get("mode") == mode
                        )
                        for mode in ("prefill_cuda_graph", "prefill_none")
                    }
                    self.assertEqual(
                        passes,
                        {"prefill_cuda_graph": 2, "prefill_none": 0},
                        f"{label}: expected two 64-request graph prefills",
                    )
                results[label] = {
                    "prompt_logprobs": prompt_logprobs,
                    "next_token_scores": [
                        (
                            o["meta_info"]["output_token_logprobs"][0][1],
                            {
                                token: lp
                                for lp, token, _ in o["meta_info"][
                                    "output_top_logprobs"
                                ][0]
                            },
                        )
                        for o in gen_out
                    ],
                }
            finally:
                if engine is not None:
                    engine.shutdown()
                for collector in set(REGISTRY._collector_to_names) - collectors_before:
                    REGISTRY.unregister(collector)
                torch.cuda.empty_cache()

        eager = results.pop("triton/disabled")
        for backend, graph in results.items():
            for i, prompt in enumerate(prompts):
                e_lp, g_lp = eager["prompt_logprobs"][i], graph["prompt_logprobs"][i]
                self.assertEqual(e_lp.numel(), g_lp.numel(), f"prompt {i}: token count")
                max_diff = (e_lp - g_lp).abs().max().item()
                self.assertLess(
                    max_diff,
                    LOGPROB_THRESHOLD,
                    f"{backend}, prompt {i} ({prompt[:40]!r}): logprobs drift "
                    f"{max_diff:.2e} from eager",
                )
                # Compare first-token scores; argmax can flip near ties.
                e_token, e_scores = eager["next_token_scores"][i]
                g_token, g_scores = graph["next_token_scores"][i]
                for token in {e_token, g_token}:
                    self.assertIn(
                        token, e_scores, f"{backend}, prompt {i}: eager top-5"
                    )
                    self.assertIn(
                        token, g_scores, f"{backend}, prompt {i}: graph top-5"
                    )
                    self.assertLess(
                        abs(e_scores[token] - g_scores[token]),
                        LOGPROB_THRESHOLD,
                        f"{backend}, prompt {i}: first-token logprob drift",
                    )


if __name__ == "__main__":
    unittest.main()
