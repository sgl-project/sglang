"""
Standalone Speculative Decoding Demo (Qwen) — Benchmark
=========================================================
自动连续跑两遍（baseline → speculative），最后输出速度对比。

运行方式：
  python standalone_speculative_demo.py
"""

import time
import sglang as sgl

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"

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
    mem_fraction_static=0.5,
    tp_size=2,
)

SPEC_KWARGS = dict(
    speculative_algorithm="STANDALONE",
    speculative_draft_model_path=DRAFT_MODEL,
    speculative_num_steps=4,
    speculative_eagle_topk=2,
    speculative_num_draft_tokens=7,
)


def benchmark(use_spec: bool) -> float:
    label = "STANDALONE spec" if use_spec else "Baseline"
    print(f"\n{'='*60}")
    print(f"  [{label}] 启动中...")
    print(f"{'='*60}")

    kwargs = {**BASE_ENGINE_KWARGS}
    if use_spec:
        kwargs.update(SPEC_KWARGS)

    llm = sgl.Engine(**kwargs)
    llm.generate(PROMPTS[:1], SAMPLING_PARAMS)          # 预热

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - t0
    llm.shutdown()

    for prompt, out in zip(PROMPTS, outputs):
        print(f"\nPrompt : {prompt}")
        print(f"Output : {out['text'].strip()[:200]}...")   # 截断避免刷屏

    print(f"\n[{label}] Wall time: {elapsed:.2f} s  ({len(PROMPTS)} requests)")
    return elapsed


if __name__ == "__main__":
    t_base = benchmark(use_spec=False)
    t_spec = benchmark(use_spec=True)

    speedup = t_base / t_spec
    print(f"\n{'='*60}")
    print(f"  Baseline      : {t_base:.2f} s")
    print(f"  Speculative   : {t_spec:.2f} s")
    print(f"  Speedup       : {speedup:.2f}x")
    print(f"{'='*60}")
