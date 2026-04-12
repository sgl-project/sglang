"""
Standalone Speculative Decoding Benchmark (Qwen)
=================================================
对比 baseline vs speculative decoding 在以下两个维度的速度差异：
  1. 不同 batch size（1 / 4 / 8）
  2. 逐条串行推理（模拟在线 chat 场景）

运行方式：
  python standalone_speculative_demo.py
"""

import time
import sglang as sgl

TARGET_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DRAFT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

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
    mem_fraction_static=0.7,
    tp_size=2,
)

SPEC_KWARGS = dict(
    speculative_algorithm="STANDALONE",
    speculative_draft_model_path=DRAFT_MODEL,
    speculative_num_steps=4,
    speculative_eagle_topk=2,
    speculative_num_draft_tokens=7,
)


def run_engine(use_spec: bool):
    kwargs = {**BASE_ENGINE_KWARGS}
    if use_spec:
        kwargs.update(SPEC_KWARGS)
    llm = sgl.Engine(**kwargs)
    llm.generate(PROMPTS[:1], SAMPLING_PARAMS)  # 预热
    return llm


def bench_batch(llm, batch_size: int) -> float:
    prompts = PROMPTS[:batch_size]
    t0 = time.perf_counter()
    llm.generate(prompts, SAMPLING_PARAMS)
    return time.perf_counter() - t0


def bench_serial(llm) -> float:
    t0 = time.perf_counter()
    for p in PROMPTS:
        llm.generate([p], SAMPLING_PARAMS)
    return time.perf_counter() - t0


if __name__ == "__main__":
    BATCH_SIZES = [1, 4, 8]

    print("Loading baseline engine...")
    llm_base = run_engine(use_spec=False)
    base_batch  = {bs: bench_batch(llm_base, bs) for bs in BATCH_SIZES}
    base_serial = bench_serial(llm_base)
    llm_base.shutdown()

    print("Loading speculative engine...")
    llm_spec = run_engine(use_spec=True)
    spec_batch  = {bs: bench_batch(llm_spec, bs) for bs in BATCH_SIZES}
    spec_serial = bench_serial(llm_spec)
    llm_spec.shutdown()

    # ── 结果汇总 ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'场景':<20} {'Baseline':>10} {'Spec':>10} {'Speedup':>10}")
    print(f"  {'-'*50}")
    for bs in BATCH_SIZES:
        speedup = base_batch[bs] / spec_batch[bs]
        print(f"  {'Batch bs='+str(bs):<20} {base_batch[bs]:>9.2f}s {spec_batch[bs]:>9.2f}s {speedup:>9.2f}x")
    speedup_serial = base_serial / spec_serial
    print(f"  {'Serial (8 req)':<20} {base_serial:>9.2f}s {spec_serial:>9.2f}s {speedup_serial:>9.2f}x")
    print(f"{'='*55}")
