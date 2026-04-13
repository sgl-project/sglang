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

  sdar-8b    : 1.7B draft for SDAR-8B（4x GPU）
               target = JetLM/SDAR-8B-Chat
               draft  = JetLM/SDAR-1.7B-Chat

运行方式：
  python standalone_speculative_demo.py --config qwen-32b
  python standalone_speculative_demo.py --config llama-70b
  python standalone_speculative_demo.py --config sdar-8b

内部机制：baseline 和 speculative 各自在独立子进程中运行，
避免 SGLang engine shutdown 后 CUDA 显存未完全释放导致 OOM。
"""

import argparse
import json
import multiprocessing
import sys
import time

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
        # mem_fraction_static 只控制目标模型的 KV cache，STANDALONE 还需给草稿模型
        # 分配单独的 KV cache，因此需要留出足够余量
        base_engine=dict(mem_fraction_static=0.4, tp_size=4, trust_remote_code=True),
        spec_kwargs=dict(
            speculative_algorithm="STANDALONE",
            speculative_num_steps=4,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
        ),
    ),
}

BATCH_SIZES = [1, 4, 8]


# ─────────────────────────────────────────────
# 在独立子进程里运行的 worker 函数
# ─────────────────────────────────────────────
def _worker(cfg_json: str, use_spec: bool, result_queue: multiprocessing.Queue):
    """在独立进程中启动 engine、跑 benchmark，结果放入 queue。"""
    import sglang as sgl

    cfg = json.loads(cfg_json)
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

    batch_times = {}
    for bs in BATCH_SIZES:
        t0 = time.perf_counter()
        llm.generate(PROMPTS[:bs], SAMPLING_PARAMS)
        batch_times[bs] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in PROMPTS:
        llm.generate([p], SAMPLING_PARAMS)
    serial_time = time.perf_counter() - t0

    llm.shutdown()
    result_queue.put({"batch": batch_times, "serial": serial_time})


def run_isolated(cfg: dict, use_spec: bool, label: str) -> dict:
    """在独立子进程中运行 benchmark，进程退出后 CUDA 显存完全释放。"""
    print(f"\nLoading {label} engine...")
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(json.dumps(cfg), use_spec, q))
    p.start()
    p.join()
    if p.exitcode != 0:
        print(f"[ERROR] {label} worker exited with code {p.exitcode}", file=sys.stderr)
        sys.exit(1)
    return q.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=list(CONFIGS), default="qwen-32b",
        help="qwen-32b | llama-70b | sdar-8b"
    )
    args = parser.parse_args()
    cfg = CONFIGS[args.config]

    print(f"Config : {args.config}")
    print(f"Target : {cfg['target']}")
    print(f"Draft  : {cfg['draft']}")

    base_stats = run_isolated(cfg, use_spec=False, label="baseline")
    spec_stats = run_isolated(cfg, use_spec=True,  label="speculative")

    print(f"\n{'='*55}")
    print(f"  {'场景':<20} {'Baseline':>10} {'Spec':>10} {'Speedup':>10}")
    print(f"  {'-'*50}")
    for bs in BATCH_SIZES:
        b = base_stats["batch"][bs]
        s = spec_stats["batch"][bs]
        print(f"  {'Batch bs='+str(bs):<20} {b:>9.2f}s {s:>9.2f}s {b/s:>9.2f}x")
    b = base_stats["serial"]
    s = spec_stats["serial"]
    print(f"  {'Serial (8 req)':<20} {b:>9.2f}s {s:>9.2f}s {b/s:>9.2f}x")
    print(f"{'='*55}")
