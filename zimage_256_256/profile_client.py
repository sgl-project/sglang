#!/usr/bin/env python3
"""
Z-Image-Turbo profiling client — 配合 nsys profile 的 server 端使用。

用法:
    # 1. 先在 GPU 服务器上用 nsys profile 启动 server:
    nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
        --delay=30 --duration=120 -o profile_output -f true \
        sglang serve --model-path /path/to/Z-Image-Turbo \
            --port 30000 --warmup \
            --text-encoder-precisions bf16

    # 2. 等 server 启动后，运行本 client 脚本:
    python3 profile_client.py --port 30000 --size 256

    # 3. 不用 nsys profile 再跑一遍 server + client，记录 client 输出的总耗时
    #    这个时间传给 gputrc2graph.py 的 elapsed_nonprofiled_sec 参数
"""

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import aiohttp

# ============================================================
# 10 条 Prompt（每 2 条相同，共 5 组）
# ============================================================
PROMPTS = [
    # 组 1
    "A beautiful sunset over the ocean with golden clouds and calm waves",
    "A beautiful sunset over the ocean with golden clouds and calm waves",
    # 组 2
    "A futuristic cyberpunk city at night with neon lights and flying cars",
    "A futuristic cyberpunk city at night with neon lights and flying cars",
    # 组 3
    "A serene Japanese garden with cherry blossoms and a koi pond",
    "A serene Japanese garden with cherry blossoms and a koi pond",
    # 组 4
    "An astronaut riding a horse on the surface of Mars with Earth in the sky",
    "An astronaut riding a horse on the surface of Mars with Earth in the sky",
    # 组 5
    "A cozy library with warm lighting and tall wooden bookshelves full of books",
    "A cozy library with warm lighting and tall wooden bookshelves full of books",
]


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    idx: int,
    save_dir: str | None,
) -> dict:
    """发送单条图像生成请求，返回结果 dict"""
    payload = {
        "model": "Z-Image-Turbo",
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
        "seed": seed,
    }

    t0 = time.perf_counter()
    async with session.post(api_url, json=payload) as resp:
        if resp.status != 200:
            error_text = await resp.text()
            return {
                "idx": idx,
                "success": False,
                "error": f"HTTP {resp.status}: {error_text}",
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
        body = await resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000

    result = {
        "idx": idx,
        "success": True,
        "latency_ms": latency_ms,
        "prompt": prompt[:60],
        "inference_time_s": body.get("inference_time_s"),
        "peak_memory_mb": body.get("peak_memory_mb"),
    }

    # 可选：保存生成的图片
    if save_dir and body.get("data"):
        b64_data = body["data"][0].get("b64_json")
        if b64_data:
            img_path = os.path.join(save_dir, f"img_{idx:02d}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            result["saved_to"] = img_path

    return result


async def run_sequential(
    api_url: str,
    prompts: list[str],
    width: int,
    height: int,
    seed: int,
    save_dir: str | None,
    warmup: int,
):
    """逐条顺序发送请求（profiling 场景推荐顺序执行，避免并发干扰分析）"""
    timeout = aiohttp.ClientTimeout(total=300)  # 5min per request
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ------ Warmup 请求（不计入统计） ------
        if warmup > 0:
            print(f"\n{'='*60}")
            print(f" Warmup: sending {warmup} request(s)...")
            print(f"{'='*60}")
            for i in range(warmup):
                r = await send_request(
                    session, api_url, prompts[0], width, height, seed, -1, None
                )
                status = "OK" if r["success"] else f"FAIL: {r.get('error','')}"
                print(f"  warmup [{i+1}/{warmup}] {r['latency_ms']:.1f}ms — {status}")
            print()

        # ------ 正式请求 ------
        print(f"{'='*60}")
        print(f" Profiling: sending {len(prompts)} requests sequentially")
        print(f" Resolution: {width}x{height}  Seed: {seed}")
        print(f"{'='*60}\n")

        results = []
        total_start = time.perf_counter()

        for i, prompt in enumerate(prompts):
            r = await send_request(
                session, api_url, prompt, width, height, seed, i, save_dir
            )
            results.append(r)
            status = "OK" if r["success"] else f"FAIL: {r.get('error','')}"
            server_time = (
                f"server={r['inference_time_s']:.3f}s"
                if r.get("inference_time_s")
                else ""
            )
            print(
                f"  [{i+1:2d}/{len(prompts)}] {r['latency_ms']:8.1f}ms  "
                f"{server_time:20s}  {prompt[:50]}..."
            )

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000

    # ------ 统计汇总 ------
    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in ok]

    print(f"\n{'='*60}")
    print(f" Results Summary")
    print(f"{'='*60}")
    print(f"  Total requests:     {len(results)}")
    print(f"  Successful:         {len(ok)}")
    print(f"  Failed:             {len(fail)}")
    if latencies:
        print(f"  Total elapsed:      {total_elapsed_ms:.2f} ms")
        print(f"  Avg latency:        {sum(latencies)/len(latencies):.2f} ms")
        print(f"  Min latency:        {min(latencies):.2f} ms")
        print(f"  Max latency:        {max(latencies):.2f} ms")
        print(f"  Median latency:     {sorted(latencies)[len(latencies)//2]:.2f} ms")
    if fail:
        print(f"\n  Failures:")
        for r in fail:
            print(f"    [{r['idx']}] {r.get('error','unknown')}")

    print(f"\n{'='*60}")
    print(
        f"  ★ elapsed_nonprofiled_sec = {total_elapsed_ms/1000:.3f}"
    )
    print(f"    (传给 gputrc2graph.py 的 E2E 时间参数)")
    print(f"{'='*60}\n")

    return results


def wait_for_server(base_url: str, timeout: int = 300, interval: int = 5):
    """等待 server 启动就绪，并打印从脚本启动到 server ready 的总耗时"""
    import requests as req

    health_url = f"{base_url}/health"
    print(f"Waiting for server at {health_url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = req.get(health_url, timeout=5)
            if r.status_code == 200:
                ready_time = time.time()
                wait_sec = ready_time - start
                print(f"  Server ready! (waited {wait_sec:.1f}s)")

                # 如果有脚本启动时间戳，计算从脚本启动到 server ready 的总耗时
                script_start = os.environ.get("SCRIPT_START_TIME")
                if script_start:
                    total_startup = ready_time - float(script_start)
                    print(f"")
                    print(f"  ┌─────────────────────────────────────────────┐")
                    print(f"  │ Server startup time: {total_startup:8.1f}s             │")
                    print(f"  │ (from script launch to health OK)           │")
                    print(f"  │                                             │")
                    print(f"  │ ★ Recommended nsys --delay: {int(total_startup)+5:4d}s           │")
                    print(f"  │   (startup + 5s safety margin)              │")
                    print(f"  └─────────────────────────────────────────────┘")
                    print(f"")
                return True
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server not ready after {timeout}s")


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo profiling client — 配合 nsys profile 使用"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Server port (default: 30000)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        choices=[256, 512, 1024],
        help="Image size, will generate size×size (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to send (default: 10, max: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests before profiling (default: 2)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images to disk",
    )
    parser.add_argument(
        "--output-dir",
        default="./profile_images",
        help="Directory to save images (default: ./profile_images)",
    )
    parser.add_argument(
        "--wait-server",
        action="store_true",
        help="Wait for server to become ready before sending requests",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for waiting for server (default: 300)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/images/generations"

    # 等待 server
    if args.wait_server:
        wait_for_server(base_url, timeout=args.wait_timeout)

    # 准备保存目录
    save_dir = None
    if args.save_images:
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)

    # 取前 N 条 prompt
    prompts = PROMPTS[: args.num_prompts]

    # 运行
    asyncio.run(
        run_sequential(
            api_url=api_url,
            prompts=prompts,
            width=args.size,
            height=args.size,
            seed=args.seed,
            save_dir=save_dir,
            warmup=args.warmup,
        )
    )


if __name__ == "__main__":
    main()
