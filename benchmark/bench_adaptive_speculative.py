"""Benchmark adaptive speculative decoding against static baselines.

Run the same workload against one adaptive server and one or more static
servers, then compare throughput, latency, and acceptance length.

Workloads:
- low: steady-state low-acceptance generation
- high: steady-state high-acceptance generation
- transition: alternating low/high acceptance shifts to stress runtime switching
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HIGH_PROMPTS = [
    "Output exactly 256 new lines. Every line must be 1. Do not add numbering, punctuation, or commentary.",
    "Output exactly 256 new lines. Every line must be READY. Do not add numbering, punctuation, or commentary.",
]

LOW_PROMPTS = [
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. Make it emotionally resonant.",
    "Write 100 two-sentence biographies of eccentric inventors with unique names, hometowns, and inventions.",
    "Write a long travel diary from a botanist visiting a chain of floating islands. Every paragraph should introduce new flora, customs, weather, and political tensions.",
    "Write 80 newspaper headlines and subheads from 80 different alternate-history worlds. Each headline must introduce a different place, conflict, and technology.",
]

WORKLOADS = {
    "low": [
        ("low", LOW_PROMPTS),
    ],
    "high": [
        ("high", HIGH_PROMPTS),
    ],
    "transition": [
        ("low_1", LOW_PROMPTS),
        ("high_1", HIGH_PROMPTS),
        ("low_2", LOW_PROMPTS),
        ("high_2", HIGH_PROMPTS),
    ],
}


def build_phase_plan(workload: str, num_requests: int):
    return [
        (phase_name, prompts, num_requests)
        for phase_name, prompts in WORKLOADS[workload]
    ]


def send_request(base_url: str, prompt: str, max_tokens: int = 256):
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
                "return_logprob": False,
            },
            timeout=max(120, max_tokens),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e), "latency": time.perf_counter() - start}

    latency = time.perf_counter() - start
    meta = data.get("meta_info", {})
    completion_tokens = meta.get("completion_tokens", 0)
    spec_verify_ct = meta.get("spec_verify_ct", 0)
    accept_len = (
        completion_tokens / spec_verify_ct if spec_verify_ct > 0 else float("nan")
    )

    return {
        "latency": latency,
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct,
        "accept_length": accept_len,
    }


def run_phase(
    base_url: str,
    prompts,
    phase_name: str,
    num_requests: int,
    max_tokens: int,
    concurrency: int,
):
    expanded = (prompts * ((num_requests + len(prompts) - 1) // len(prompts)))[
        :num_requests
    ]

    print(
        f"\n--- Phase: {phase_name} ({num_requests} requests, concurrency={concurrency}) ---"
    )
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, base_url, p, max_tokens) for p in expanded]
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        print(f"  All {len(errors)} requests failed!")
        return {"phase": phase_name, "error": True}

    total_tokens = sum(r["completion_tokens"] for r in ok)
    total_verify = sum(r["spec_verify_ct"] for r in ok)
    avg_latency = sum(r["latency"] for r in ok) / len(ok)
    throughput = total_tokens / elapsed
    avg_accept_len = total_tokens / total_verify if total_verify > 0 else float("nan")

    stats = {
        "phase": phase_name,
        "num_requests": len(ok),
        "num_errors": len(errors),
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(throughput, 2),
        "avg_latency_s": round(avg_latency, 3),
        "avg_accept_length": round(avg_accept_len, 3),
    }

    print(
        f"  Throughput: {throughput:.1f} tok/s | "
        f"Avg latency: {avg_latency:.3f}s | "
        f"Avg accept_len: {avg_accept_len:.2f} | "
        f"Errors: {len(errors)}"
    )
    return stats


def summarize_phases(phase_stats):
    ok_stats = [s for s in phase_stats if not s.get("error")]
    if not ok_stats:
        return {"error": True}

    total_tokens = sum(s["total_tokens"] for s in ok_stats)
    total_elapsed = sum(s["elapsed_s"] for s in ok_stats)
    total_requests = sum(s["num_requests"] for s in ok_stats)

    weighted_latency = sum(s["avg_latency_s"] * s["num_requests"] for s in ok_stats)
    weighted_accept = sum(s["avg_accept_length"] * s["num_requests"] for s in ok_stats)

    return {
        "num_requests": total_requests,
        "total_tokens": total_tokens,
        "elapsed_s": round(total_elapsed, 2),
        "throughput_tok_s": round(total_tokens / total_elapsed, 2),
        "avg_latency_s": round(weighted_latency / total_requests, 3),
        "avg_accept_length": round(weighted_accept / total_requests, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark one workload for adaptive-vs-static speculative decoding"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        default="transition",
        help="Workload preset to run.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="Requests per phase.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent requests.",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup requests before the benchmark."
    )
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    base_url = f"http://{args.host}:{args.port}"

    print(f"Server: {base_url}")
    print(f"Workload: {args.workload}")

    phase_plan = build_phase_plan(args.workload, args.requests)
    if args.warmup > 0:
        print(f"\nWarming up with {args.warmup} requests...")
        warmup_prompts = phase_plan[0][1]
        run_phase(
            base_url,
            warmup_prompts,
            "warmup",
            args.warmup,
            args.max_tokens,
            args.concurrency,
        )

    phase_stats = []
    for phase_name, prompts, num_requests in phase_plan:
        phase_stats.append(
            run_phase(
                base_url,
                prompts,
                phase_name,
                num_requests,
                args.max_tokens,
                args.concurrency,
            )
        )

    overall = summarize_phases(phase_stats)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Phase':<10} {'Throughput':>12} {'Avg Latency':>12} {'Accept Len':>12}")
    print("-" * 50)
    for stats in phase_stats:
        if stats.get("error"):
            print(f"{stats['phase']:<10} {'ERROR':>12}")
            continue
        print(
            f"{stats['phase']:<10} "
            f"{stats['throughput_tok_s']:>10.1f}/s "
            f"{stats['avg_latency_s']:>10.3f}s "
            f"{stats['avg_accept_length']:>11.2f}"
        )

    if not overall.get("error"):
        print("-" * 50)
        print(
            f"{'OVERALL':<10} "
            f"{overall['throughput_tok_s']:>10.1f}/s "
            f"{overall['avg_latency_s']:>10.3f}s "
            f"{overall['avg_accept_length']:>11.2f}"
        )


if __name__ == "__main__":
    main()
