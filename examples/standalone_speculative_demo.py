"""
Standalone Speculative Decoding Demo (Qwen)
============================================
使用 Qwen2.5-7B-Instruct 作为 target，Qwen2.5-1.5B-Instruct 作为 draft，
采用 STANDALONE 算法（draft 模型无需专门训练，直接使用小模型）。

两种使用方式：
  1. 离线推理（sgl.Engine）
  2. 在线服务 + OpenAI 客户端

运行方式：
  # 离线模式（默认）
  python standalone_speculative_demo.py

  # 在线服务模式
  python standalone_speculative_demo.py --mode server

  # 关闭 speculative decoding，用于对比基线速度
  python standalone_speculative_demo.py --no-spec
"""

import argparse
import time

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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

SPEC_CONFIG = dict(
    speculative_algorithm="STANDALONE",
    speculative_draft_model_path=DRAFT_MODEL,
    speculative_num_steps=4,
    speculative_eagle_topk=2,
    speculative_num_draft_tokens=7,
    mem_fraction_static=0.5,
    tp_size=2,
)


def run_offline(use_spec: bool):
    """使用 sgl.Engine 做离线批量推理。"""
    import sglang as sgl

    engine_kwargs = dict(
        model_path=TARGET_MODEL,
        cuda_graph_max_bs=len(PROMPTS),
    )
    if use_spec:
        engine_kwargs.update(SPEC_CONFIG)
        print(f"[STANDALONE] target={TARGET_MODEL}")
        print(f"             draft={DRAFT_MODEL}")
        print(f"             steps={SPEC_CONFIG['speculative_num_steps']}  "
              f"topk={SPEC_CONFIG['speculative_eagle_topk']}  "
              f"draft_tokens={SPEC_CONFIG['speculative_num_draft_tokens']}\n")
    else:
        print(f"[Baseline] target={TARGET_MODEL} (no speculative decoding)\n")

    llm = sgl.Engine(**engine_kwargs)

    # 预热
    _ = llm.generate(PROMPTS[:1], SAMPLING_PARAMS)

    # 计时推理
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - t0

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
        "--mem-fraction-static", str(SPEC_CONFIG["mem_fraction_static"]),
        "--tp-size", str(SPEC_CONFIG["tp_size"]),
        "--cuda-graph-max-bs", str(len(PROMPTS)),
        "--log-level", "warning",
    ]
    if use_spec:
        cmd += [
            "--speculative-algorithm", SPEC_CONFIG["speculative_algorithm"],
            "--speculative-draft-model-path", DRAFT_MODEL,
            "--speculative-num-steps", str(SPEC_CONFIG["speculative_num_steps"]),
            "--speculative-eagle-topk", str(SPEC_CONFIG["speculative_eagle_topk"]),
            "--speculative-num-draft-tokens", str(SPEC_CONFIG["speculative_num_draft_tokens"]),
        ]

    print("Starting SGLang server...")
    print("Command:", " ".join(cmd))
    server = subprocess.Popen(cmd)

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
    parser = argparse.ArgumentParser(description="Standalone Speculative Decoding Demo")
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


if __name__ == "__main__":
    main()
