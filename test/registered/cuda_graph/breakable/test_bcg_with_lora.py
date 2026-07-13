"""LoRA under the breakable (BCG) prefill CUDA graph.

Launches the same LoRA-enabled engine twice — once with
--cuda-graph-backend-prefill=breakable and once with =disabled — and
compares per-token prompt logprobs for:
  * single LoRA requests across prompt lengths spanning several token buckets,
  * base-model (no-adapter) requests served by the same LoRA-enabled engine,
  * one multi-request batch mixing LoRA and base requests (bs > 1 replay).

Also asserts the adapter materially changes prompt logprobs while the
prefill graph is enabled, which catches a graph that silently captured
without applying LoRA.
"""

import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_ADAPTER = "nissenj/Qwen3-4B-lora-v2"
LORA_NAME = "lora0"

# Max absolute per-token prompt logprob difference tolerated between the
# graph-enabled and graph-disabled runs (bf16; graph replay pads the token
# axis up to the captured bucket, which perturbs GEMM tiling slightly).
GRAPH_VS_EAGER_TOLERANCE = 1e-1
# The adapter must move at least one prompt logprob by this much, otherwise
# LoRA was not applied under the graph at all.
LORA_EFFECT_THRESHOLD = 5e-2

# Prompt lengths chosen to land in different captured token buckets.
PROMPTS = [
    "What is the capital of France?",
    "List three benefits of regular exercise. " * 8,
    "The quick brown fox jumps over the lazy dog. " * 40,
]

MIXED_BATCH_PROMPTS = [
    "Summarize the plot of Romeo and Juliet.",
    "Explain how photosynthesis works. " * 6,
    "Write a haiku about the sea.",
    "Describe the water cycle in detail. " * 12,
]
MIXED_BATCH_LORA_PATHS = [LORA_NAME, None, LORA_NAME, None]


def _prompt_logprobs(engine, prompt, lora_path):
    out = engine.generate(
        prompt=prompt,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_path,
    )
    # First entry has no logprob (no preceding context).
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def _batch_prompt_logprobs(engine, prompts, lora_paths):
    outs = engine.generate(
        prompt=prompts,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_paths,
    )
    return [
        [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]
        for out in outs
    ]


def _collect(cuda_graph_backend_prefill):
    engine = sgl.Engine(
        model_path=BASE_MODEL,
        enable_lora=True,
        lora_paths={LORA_NAME: LORA_ADAPTER},
        max_loras_per_batch=2,
        cuda_graph_backend_prefill=cuda_graph_backend_prefill,
        cuda_graph_max_bs_prefill=1024,
        cuda_graph_max_bs_decode=8,
        disable_radix_cache=True,
        mem_fraction_static=0.8,
        random_seed=42,
    )
    try:
        results = {
            "lora": [_prompt_logprobs(engine, p, LORA_NAME) for p in PROMPTS],
            "base": [_prompt_logprobs(engine, p, None) for p in PROMPTS],
            "mixed": _batch_prompt_logprobs(
                engine, MIXED_BATCH_PROMPTS, MIXED_BATCH_LORA_PATHS
            ),
        }
    finally:
        engine.shutdown()
    return results


def _max_abs_diff(a, b):
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    assert ta.shape == tb.shape, f"token count mismatch: {ta.shape} vs {tb.shape}"
    return (ta - tb).abs().max().item()


class TestBreakableCudaGraphWithLoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.with_graph = _collect("breakable")
        cls.without_graph = _collect("disabled")

    def test_lora_prefill_logprobs_match_eager(self):
        for i, prompt in enumerate(PROMPTS):
            diff = _max_abs_diff(
                self.with_graph["lora"][i], self.without_graph["lora"][i]
            )
            print(f"[lora] prompt {i} ({len(prompt)} chars): max_abs_diff={diff:.5f}")
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"LoRA prompt logprobs diverge between breakable prefill graph "
                f"and eager prefill for prompt {i}",
            )

    def test_base_prefill_logprobs_match_eager(self):
        for i, prompt in enumerate(PROMPTS):
            diff = _max_abs_diff(
                self.with_graph["base"][i], self.without_graph["base"][i]
            )
            print(f"[base] prompt {i} ({len(prompt)} chars): max_abs_diff={diff:.5f}")
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"Base-model prompt logprobs diverge between breakable prefill "
                f"graph and eager prefill for prompt {i}",
            )

    def test_mixed_batch_matches_eager(self):
        for i in range(len(MIXED_BATCH_PROMPTS)):
            diff = _max_abs_diff(
                self.with_graph["mixed"][i], self.without_graph["mixed"][i]
            )
            print(
                f"[mixed] req {i} (lora={MIXED_BATCH_LORA_PATHS[i]}): "
                f"max_abs_diff={diff:.5f}"
            )
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"Mixed LoRA/base batch logprobs diverge between breakable "
                f"prefill graph and eager prefill for request {i}",
            )

    def test_lora_is_applied_under_graph(self):
        # If the captured graph dropped the LoRA kernels (or read stale
        # metadata), LoRA and base logprobs would coincide.
        for i in range(len(PROMPTS)):
            diff = _max_abs_diff(self.with_graph["lora"][i], self.with_graph["base"][i])
            print(f"[effect] prompt {i}: lora-vs-base max_abs_diff={diff:.5f}")
            self.assertGreater(
                diff,
                LORA_EFFECT_THRESHOLD,
                f"Adapter did not change prompt logprobs under the breakable "
                f"prefill graph for prompt {i}; LoRA was likely not applied.",
            )


if __name__ == "__main__":
    unittest.main()
