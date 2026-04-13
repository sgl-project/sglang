"""
Standalone Speculative Decoding Benchmark
==========================================
支持三种配置，通过 --config 参数切换：

  qwen-32b   : 7B draft for Qwen2.5-32B（2x H100，无需授权）
               target = Qwen/Qwen2.5-32B-Instruct
               draft  = Qwen/Qwen2.5-7B-Instruct

  llama-70b  : 8B draft for Llama-3.1-70B（2x H100，需要 Meta 授权）
               target = meta-llama/Llama-3.1-70B-Instruct
               draft  = meta-llama/Llama-3.1-8B-Instruct

  sdar-8b    : 1.7B draft for SDAR-8B（单卡）
               target = JetLM/SDAR-8B-Chat
               draft  = JetLM/SDAR-1.7B-Chat

运行方式：
  python standalone_speculative_demo.py --config qwen-32b
  python standalone_speculative_demo.py --config llama-70b
  python standalone_speculative_demo.py --config sdar-8b
"""

import argparse
import time

import sglang as sgl

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

CONFIGS = {
    "qwen-32b": dict(
        target="Qwen/Qwen2.5-32B-Instruct",
        draft="Qwen/Qwen2.5-7B-Instruct",
        base_engine=dict(mem_fraction_static=0.7, tp_size=2),
        spec_kwargs=dict(
            speculative_algorithm="STANDALONE",
            speculative_num_steps=4,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=7,
        ),
    ),
    "llama-70b": dict(
        target="meta-llama/Llama-3.1-70B-Instruct",
        draft="meta-llama/Llama-3.1-8B-Instruct",
        base_engine=dict(mem_fraction_static=0.7, tp_size=4),
        spec_kwargs=dict(
            speculative_algorithm="STANDALONE",
            speculative_num_steps=4,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=7,
        ),
    ),
    "sdar-8b": dict(
        target="JetLM/SDAR-8B-Chat",
        draft="JetLM/SDAR-1.7B-Chat",
        base_engine=dict(mem_fraction_static=0.85, tp_size=4, trust_remote_code=True),
        spec_kwargs=dict(
            speculative_algorithm="STANDALONE",
            speculative_num_steps=4,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
        ),
    ),
}

BATCH_SIZES = [1, 4, 8]


def run_engine(cfg: dict, use_spec: bool) -> sgl.Engine:
    kwargs = dict(
        model_path=cfg["target"],
        cuda_graph_max_bs=max(BATCH_SIZES),
        **cfg["base_engine"],
    )
    if use_spec:
        kwargs["speculative_draft_model_path"] = cfg["draft"]
        kwargs.update(cfg["spec_kwargs"])
    llm = sgl.Engine(**kwargs)
    llm.generate(PROMPTS[:1], SAMPLING_PARAMS)  # 预热
    return llm


def bench_batch(llm, batch_size: int) -> float:
    t0 = time.perf_counter()
    llm.generate(PROMPTS[:batch_size], SAMPLING_PARAMS)
    return time.perf_counter() - t0


def bench_serial(llm) -> float:
    t0 = time.perf_counter()
    for p in PROMPTS:
        llm.generate([p], SAMPLING_PARAMS)
    return time.perf_counter() - t0


if __name__ == "__main__":
    import gc
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=list(CONFIGS), default="qwen-32b",
        help="qwen-32b: 7B draft for 32B Qwen | llama-70b: 8B draft for 70B Llama"
    )
    args = parser.parse_args()
    cfg = CONFIGS[args.config]

    print(f"Config : {args.config}")
    print(f"Target : {cfg['target']}")
    print(f"Draft  : {cfg['draft']}")

    print("\nLoading baseline engine...")
    llm_base = run_engine(cfg, use_spec=False)
    base_batch  = {bs: bench_batch(llm_base, bs) for bs in BATCH_SIZES}
    base_serial = bench_serial(llm_base)
    llm_base.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading speculative engine...")
    llm_spec = run_engine(cfg, use_spec=True)
    spec_batch  = {bs: bench_batch(llm_spec, bs) for bs in BATCH_SIZES}
    spec_serial = bench_serial(llm_spec)
    llm_spec.shutdown()

    print(f"\n{'='*55}")
    print(f"  {'场景':<20} {'Baseline':>10} {'Spec':>10} {'Speedup':>10}")
    print(f"  {'-'*50}")
    for bs in BATCH_SIZES:
        speedup = base_batch[bs] / spec_batch[bs]
        print(f"  {'Batch bs='+str(bs):<20} {base_batch[bs]:>9.2f}s {spec_batch[bs]:>9.2f}s {speedup:>9.2f}x")
    speedup_serial = base_serial / spec_serial
    print(f"  {'Serial (8 req)':<20} {base_serial:>9.2f}s {spec_serial:>9.2f}s {speedup_serial:>9.2f}x")
    print(f"{'='*55}")
