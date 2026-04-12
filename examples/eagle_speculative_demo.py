"""
EAGLE3 Speculative Decoding Benchmark (Llama-3.1-8B)
=====================================================
自动连续跑两遍（baseline → EAGLE3 speculative），最后输出速度对比。

常用 EAGLE 模型对（target → draft）：
  meta-llama/Llama-3.1-8B-Instruct  → jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
  Qwen/Qwen3-4B                      → AngelSlim/Qwen3-4B_eagle3

运行方式：
  python eagle_speculative_demo.py
"""

import os
import time

import sglang as sgl

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL  = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci numbers recursively.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "What is the capital of France and what is it famous for?",
    "How does garbage collection work in modern programming languages?",
    "Summarize the plot of Shakespeare's Hamlet.",
    "What are the key principles of object-oriented programming?",
]

SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 256}

BASE_ENGINE_KWARGS = dict(
    model_path=TARGET_MODEL,
    cuda_graph_max_bs=len(PROMPTS),
    dtype="float16",
    mem_fraction_static=0.7,
    chunked_prefill_size=128,
    max_running_requests=8,
)

EAGLE_KWARGS = dict(
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path=DRAFT_MODEL,
    speculative_num_steps=3,
    speculative_eagle_topk=4,
    speculative_num_draft_tokens=16,
)


def benchmark(use_spec: bool) -> float:
    label = "EAGLE3 spec" if use_spec else "Baseline"
    print(f"\n{'='*60}")
    print(f"  [{label}] 启动中...")
    print(f"{'='*60}")

    kwargs = {**BASE_ENGINE_KWARGS}
    if use_spec:
        kwargs.update(EAGLE_KWARGS)

    llm = sgl.Engine(**kwargs)
    llm.generate(PROMPTS[:1], SAMPLING_PARAMS)  # 预热

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - t0
    llm.shutdown()

    for prompt, out in zip(PROMPTS, outputs):
        print(f"\nPrompt : {prompt}")
        print(f"Output : {out['text'].strip()[:200]}...")

    print(f"\n[{label}] Wall time: {elapsed:.2f} s  ({len(PROMPTS)} requests)")
    return elapsed


if __name__ == "__main__":
    t_base = benchmark(use_spec=False)
    t_spec = benchmark(use_spec=True)

    print(f"\n{'='*60}")
    print(f"  Baseline      : {t_base:.2f} s")
    print(f"  EAGLE3 spec   : {t_spec:.2f} s")
    print(f"  Speedup       : {t_base / t_spec:.2f}x")
    print(f"{'='*60}")
