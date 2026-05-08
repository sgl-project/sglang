#!/usr/bin/env python3
"""E2E benchmark: ISL=4096, OSL=8, BS=1/4/8/16/32.
Records TTFT (time to first token) and TPOT (time per output token).
Uses streaming to measure TTFT accurately.
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER = "http://localhost:30000"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# ~4096 tokens of prefill text
PREFILL_TEXT = """Please analyze the following passage and provide a detailed summary.

The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project.

In 1973, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when funding for AI projects was hard to find.

In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research.

Beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting hiatus began. In the late 1990s and early 21st century, AI began to be used for logistics, data mining, medical diagnosis and other areas. The success was due to increasing computational power, greater emphasis on solving specific problems, new ties between AI and other fields and a commitment by researchers to mathematical methods and scientific standards.

Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997. In 2011, a question answering system, IBM Watson, won the quiz show Jeopardy! by a significant margin over its two greatest human champions.

In March 2016, AlphaGo, a program created by Google DeepMind, beat Lee Sedol, a top Go player, in a five-game match. This was the first time a computer Go program had beaten a human professional Go player on a full-sized board.

The advancement of deep learning techniques, particularly the transformer architecture introduced in 2017, led to unprecedented progress in natural language processing, computer vision, and generative AI. Large language models trained on vast corpora of text demonstrated emergent abilities that surprised even their creators, leading to a new era of AI applications.""" * 3


def send_request_streaming(text, max_tokens=8, temperature=0.0):
    """Send streaming request and measure TTFT and per-token times."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)

    ttft = None
    token_times = []
    total_tokens = 0

    for line in resp:
        line = line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                t_now = time.perf_counter()
                if ttft is None:
                    ttft = (t_now - t_start) * 1000  # ms
                token_times.append(t_now)
                total_tokens += 1
        except json.JSONDecodeError:
            continue

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000

    # Calculate TPOT: average time between consecutive tokens (excluding TTFT)
    tpot = None
    if len(token_times) > 1:
        inter_token_times = [(token_times[i+1] - token_times[i]) * 1000
                             for i in range(len(token_times) - 1)]
        tpot = sum(inter_token_times) / len(inter_token_times)

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "total_ms": total_ms,
        "tokens": total_tokens,
    }


def benchmark_batch(batch_size, warmup=2, iters=5, max_tokens=8):
    """Run benchmark at given batch size."""
    print(f"\n{'='*60}")
    print(f"  BS={batch_size}, ISL~4K, OSL={max_tokens}")
    print(f"{'='*60}")

    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        if batch_size == 1:
            send_request_streaming(PREFILL_TEXT, max_tokens=max_tokens)
        else:
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                list(ex.map(lambda _: send_request_streaming(PREFILL_TEXT, max_tokens=max_tokens),
                           range(batch_size)))

    all_results = []
    for i in range(iters):
        if batch_size == 1:
            r = send_request_streaming(PREFILL_TEXT, max_tokens=max_tokens)
            batch_results = [r]
        else:
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                futs = [ex.submit(send_request_streaming, PREFILL_TEXT, max_tokens=max_tokens)
                        for _ in range(batch_size)]
                batch_results = [f.result() for f in futs]

        # Aggregate
        ttfts = [r["ttft_ms"] for r in batch_results if r["ttft_ms"] is not None]
        tpots = [r["tpot_ms"] for r in batch_results if r["tpot_ms"] is not None]
        total_tokens = sum(r["tokens"] for r in batch_results)
        wall_ms = max(r["total_ms"] for r in batch_results)

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        avg_tpot = sum(tpots) / len(tpots) if tpots else None
        throughput = total_tokens / (wall_ms / 1000) if wall_ms > 0 else 0

        all_results.append({
            "iter": i + 1,
            "avg_ttft_ms": avg_ttft,
            "avg_tpot_ms": avg_tpot,
            "throughput_tps": throughput,
            "total_tokens": total_tokens,
            "wall_ms": wall_ms,
        })

        print(f"  Run {i+1}: TTFT={avg_ttft:.1f}ms, TPOT={avg_tpot:.1f}ms, "
              f"throughput={throughput:.2f} tok/s" if avg_ttft and avg_tpot else
              f"  Run {i+1}: total={wall_ms:.0f}ms, tokens={total_tokens}")

    # Summary
    valid = [r for r in all_results if r["avg_ttft_ms"] is not None]
    if valid:
        avg_ttft = sum(r["avg_ttft_ms"] for r in valid) / len(valid)
        avg_tpot = sum(r["avg_tpot_ms"] for r in valid) / len(valid)
        avg_tput = sum(r["throughput_tps"] for r in valid) / len(valid)
        print(f"  → Average: TTFT={avg_ttft:.1f}ms, TPOT={avg_tpot:.1f}ms, "
              f"throughput={avg_tput:.2f} tok/s")
        return {
            "batch_size": batch_size,
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_tpot_ms": round(avg_tpot, 1),
            "avg_throughput_tps": round(avg_tput, 2),
            "runs": all_results,
        }
    return {"batch_size": batch_size, "runs": all_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=8, help="OSL (output sequence length)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--output", type=str, default="bench_e2e_isl4k_osl8.json")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Health check
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=5)
        print("Server is healthy")
    except Exception as e:
        print(f"Server not available: {e}")
        sys.exit(1)

    print(f"\nE2E Benchmark: ISL~4K, OSL={args.max_tokens}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")

    results = {}
    for bs in batch_sizes:
        r = benchmark_batch(bs, warmup=args.warmup, iters=args.iters,
                           max_tokens=args.max_tokens)
        results[f"bs{bs}"] = r

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: ISL~4K, OSL={args.max_tokens}")
    print(f"{'='*70}")
    print(f"  {'BS':>4s}  {'TTFT(ms)':>10s}  {'TPOT(ms)':>10s}  {'Throughput':>12s}")
    print(f"  {'----':>4s}  {'--------':>10s}  {'--------':>10s}  {'----------':>12s}")
    for bs in batch_sizes:
        r = results[f"bs{bs}"]
        ttft = r.get("avg_ttft_ms", "N/A")
        tpot = r.get("avg_tpot_ms", "N/A")
        tput = r.get("avg_throughput_tps", "N/A")
        ttft_s = f"{ttft:.1f}" if isinstance(ttft, (int, float)) else ttft
        tpot_s = f"{tpot:.1f}" if isinstance(tpot, (int, float)) else tpot
        tput_s = f"{tput:.2f} tok/s" if isinstance(tput, (int, float)) else tput
        print(f"  {bs:4d}  {ttft_s:>10s}  {tpot_s:>10s}  {tput_s:>12s}")

    # Save results
    output = {
        "config": {
            "isl": "~4096",
            "osl": args.max_tokens,
            "batch_sizes": batch_sizes,
            "warmup": args.warmup,
            "iters": args.iters,
            "model": MODEL,
            "gpu": "RTX PRO 6000 (SM120)",
            "branch": "sm120-dsv4-enablement (rebased on main)",
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
