"""Collect per-layer per-expert routing stats from a running SGLang server.

Uses SGLang's built-in ExpertDistributionRecorder (EPLB infrastructure).
The server must be launched with --expert-distribution-recorder-mode per_token.

Usage:
  # 1. Launch server (separate terminal):
  CUDA_VISIBLE_DEVICES=4,5 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  PYTHONPATH=python python -m sglang.launch_server \
      --model-path /data/heter-moe/models/qwen3-30b-a3b-bf16 \
      --port 30000 --tp 2 \
      --expert-distribution-recorder-mode per_token

  # 2. Run this client:
  python3 scripts/heter_moe_collect_routing.py \
      --server-url http://localhost:30000 \
      --sharegpt-path /data/heter-moe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-requests 64 --max-tokens 128

  # 3. Synthetic mode (no server needed):
  python3 scripts/heter_moe_collect_routing.py --synthetic
"""

import argparse
import concurrent.futures
import json
import os
import time

import numpy as np
import requests

NUM_EXPERTS = 128
NUM_LAYERS = 48
TOP_K = 8
OUT_DIR = "/data/heter-moe/routing_stats"
EPLB_OUT_DIR = "/data/heter-moe/routing_stats/eplb"


def generate_synthetic_routing(batch_size, phase, seed=42):
    rng = np.random.default_rng(seed + batch_size + (0 if phase == "prefill" else 1))
    result = {}
    for layer_idx in range(NUM_LAYERS):
        zipf_weights = 1.0 / np.arange(1, NUM_EXPERTS + 1) ** 1.1
        zipf_weights /= zipf_weights.sum()
        layer_perm = rng.permutation(NUM_EXPERTS)
        shuffled_weights = zipf_weights[layer_perm]
        total_tokens = batch_size * TOP_K
        counts = rng.multinomial(total_tokens, shuffled_weights)
        result[f"transformer_block_{layer_idx}"] = counts.tolist()
    return result


def load_sharegpt(path, max_samples=2048):
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        for turn in turns:
            if turn.get("from") == "human" and turn.get("value", "").strip():
                prompts.append(turn["value"].strip())
                if len(prompts) >= max_samples:
                    return prompts
    return prompts


def check_server(url):
    try:
        return requests.get(f"{url}/health", timeout=5).status_code == 200
    except Exception:
        return False


def start_recording(url):
    r = requests.post(f"{url}/start_expert_distribution_record", timeout=10)
    r.raise_for_status()
    print(f"  Recording started: {r.text.strip()}")


def stop_recording(url):
    r = requests.post(f"{url}/stop_expert_distribution_record", timeout=10)
    r.raise_for_status()
    print(f"  Recording stopped: {r.text.strip()}")


def dump_recording(url):
    r = requests.post(f"{url}/dump_expert_distribution_record", timeout=120)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}


def send_requests(url, prompts, max_tokens=128, num_concurrent=8):
    def send_one(prompt):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            return True
        except Exception:
            return False

    completed, failed = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as pool:
        futures = [pool.submit(send_one, p) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            if f.result():
                completed += 1
            else:
                failed += 1
            if (completed + failed) % 10 == 0:
                print(f"    {completed + failed}/{len(prompts)} done", flush=True)

    print(f"  Completed: {completed}/{len(prompts)}, failed: {failed}")
    return completed


def collect_eplb(server_url, sharegpt_path, batch_sizes, max_tokens, num_concurrent):
    if not check_server(server_url):
        print(f"ERROR: Server not reachable at {server_url}")
        return

    if not os.path.exists(sharegpt_path):
        print(f"Downloading ShareGPT...")
        os.makedirs(os.path.dirname(sharegpt_path), exist_ok=True)
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
            local_dir=os.path.dirname(sharegpt_path),
        )

    all_prompts = load_sharegpt(sharegpt_path, max_samples=max(batch_sizes) * 2)
    print(f"Loaded {len(all_prompts)} prompts from ShareGPT")
    os.makedirs(EPLB_OUT_DIR, exist_ok=True)

    for bs in batch_sizes:
        prompts = all_prompts[:bs]
        if len(prompts) < bs:
            prompts = (prompts * (bs // len(prompts) + 1))[:bs]

        print(f"\n{'=' * 60}")
        print(f"Batch size: {bs}, max_tokens: {max_tokens}")
        print(f"{'=' * 60}")

        start_recording(server_url)
        completed = send_requests(
            server_url, prompts, max_tokens, min(num_concurrent, bs)
        )
        stop_recording(server_url)

        dump_result = dump_recording(server_url)
        dump_path = os.path.join(EPLB_OUT_DIR, f"batch{bs}.json")
        with open(dump_path, "w") as f:
            json.dump(dump_result, f)
        print(f"  Saved: {dump_path}")
        if isinstance(dump_result, dict):
            print(f"  Keys: {list(dump_result.keys())}")

    print(f"\nAll dumps saved to {EPLB_OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--server-url", type=str, default="http://localhost:30000")
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default="/data/heter-moe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64, 128],
        help="Batch sizes to collect (one round per size)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrent", type=int, default=8)
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic routing stats...")
        for bs in [2**i for i in range(11)]:
            for phase in ["prefill", "decode"]:
                data = generate_synthetic_routing(bs, phase)
                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"batch{bs}_{phase}.json")
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  {path}")
    else:
        collect_eplb(
            args.server_url,
            args.sharegpt_path,
            args.batch_sizes,
            args.max_tokens,
            args.concurrent,
        )


if __name__ == "__main__":
    main()
