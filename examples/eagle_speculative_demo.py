"""
EAGLE Speculative Decoding Demo
================================
两种使用方式：
  1. 离线推理（sgl.Engine，最简单）
  2. 在线服务 + OpenAI 客户端

常用 EAGLE 模型对（target → draft）：
  meta-llama/Llama-3.1-8B-Instruct  → jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
  Qwen/Qwen3-4B                      → AngelSlim/Qwen3-4B_eagle3
  Qwen/Qwen3-1.7B                    → AngelSlim/Qwen3-1.7B_eagle3

运行方式：
  # 离线模式（默认）
  python eagle_speculative_demo.py

  # 在线服务模式
  python eagle_speculative_demo.py --mode server

  # 关闭 speculative decoding，用于对比基线速度
  python eagle_speculative_demo.py --no-spec
"""

import argparse
import time

# ── 修改为你本地已下载的模型路径 ──────────────────────────────────────────
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
# ─────────────────────────────────────────────────────────────────────────

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

SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 256,
}

# ── EAGLE 超参数 ──────────────────────────────────────────────────────────
EAGLE_CONFIG = dict(
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path=DRAFT_MODEL,
    speculative_num_steps=3,          # draft 树深度
    speculative_eagle_topk=4,         # 每步 top-k 分支数
    speculative_num_draft_tokens=16,  # 最终 draft token 总数（树节点数）
)
# ─────────────────────────────────────────────────────────────────────────


def run_offline(use_spec: bool):
    """使用 sgl.Engine 做离线批量推理。"""
    import sglang as sgl

    engine_kwargs = dict(
        model_path=TARGET_MODEL,
        cuda_graph_max_bs=len(PROMPTS),
    )
    if use_spec:
        engine_kwargs.update(EAGLE_CONFIG)
        print(f"[EAGLE] target={TARGET_MODEL}")
        print(f"        draft={DRAFT_MODEL}")
        print(f"        steps={EAGLE_CONFIG['speculative_num_steps']}  "
              f"topk={EAGLE_CONFIG['speculative_eagle_topk']}  "
              f"draft_tokens={EAGLE_CONFIG['speculative_num_draft_tokens']}\n")
    else:
        print(f"[Baseline] target={TARGET_MODEL} (no speculative decoding)\n")

    llm = sgl.Engine(**engine_kwargs)

    # 预热
    _ = llm.generate(PROMPTS[:1], SAMPLING_PARAMS)

    # 计时推理
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o["meta_info"]["completion_tokens"])
                       if "completion_tokens" in o.get("meta_info", {})
                       else o["meta_info"].get("completion_tokens", 0)
                       for o in outputs)

    # 打印结果
    for prompt, output in zip(PROMPTS, outputs):
        print("=" * 60)
        print(f"Prompt : {prompt}")
        print(f"Output : {output['text'].strip()}")

    print("\n" + "=" * 60)
    print(f"  Requests  : {len(PROMPTS)}")
    print(f"  Wall time : {elapsed:.2f} s")

    llm.shutdown()


def run_server_mode(use_spec: bool):
    """启动 HTTP 服务端，用 OpenAI 客户端调用。"""
    import subprocess
    import sys

    import openai

    port = 30000
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", TARGET_MODEL,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    if use_spec:
        cmd += [
            "--speculative-algorithm", EAGLE_CONFIG["speculative_algorithm"],
            "--speculative-draft-model-path", DRAFT_MODEL,
            "--speculative-num-steps", str(EAGLE_CONFIG["speculative_num_steps"]),
            "--speculative-eagle-topk", str(EAGLE_CONFIG["speculative_eagle_topk"]),
            "--speculative-num-draft-tokens", str(EAGLE_CONFIG["speculative_num_draft_tokens"]),
        ]

    print("Starting SGLang server...")
    print("Command:", " ".join(cmd))
    server = subprocess.Popen(cmd)

    # 等待服务就绪
    import socket
    for _ in range(120):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        server.terminate()
        raise RuntimeError("Server did not start in time.")

    print("Server is ready.\n")

    client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

    t0 = time.perf_counter()
    for prompt in PROMPTS:
        resp = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=SAMPLING_PARAMS["max_new_tokens"],
            temperature=SAMPLING_PARAMS["temperature"],
        )
        text = resp.choices[0].message.content
        print("=" * 60)
        print(f"Prompt : {prompt}")
        print(f"Output : {text.strip()}")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"  Requests  : {len(PROMPTS)}")
    print(f"  Wall time : {elapsed:.2f} s")

    server.terminate()
    server.wait()


def main():
    parser = argparse.ArgumentParser(description="EAGLE Speculative Decoding Demo")
    parser.add_argument("--mode", choices=["offline", "server"], default="offline",
                        help="offline: sgl.Engine;  server: launch HTTP server + OpenAI client")
    parser.add_argument("--no-spec", action="store_true",
                        help="Disable speculative decoding (baseline comparison)")
    args = parser.parse_args()

    use_spec = not args.no_spec

    if args.mode == "offline":
        run_offline(use_spec)
    else:
        run_server_mode(use_spec)


# sgl.Engine 使用 spawn 多进程，必须保护入口点
if __name__ == "__main__":
    main()
