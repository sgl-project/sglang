"""
Needle-in-a-Haystack (NIAH) benchmark for TurboQuant KV cache.

Tests whether the model can retrieve specific facts embedded in long contexts
when using TurboQuant compressed KV cache.

Features:
- 5 different needle types (number, password, name, city, speed)
- Multiple context lengths and insertion depths
- Repeat testing with pass rate matrix
- Fuzzy answer matching

Usage:
    # Start server first:
    python -m sglang.launch_server --model-path <model> --port 30001 \
        --kv-cache-dtype turboquant_3bit --disable-cuda-graph

    # Quick test:
    python bench_niah_turboquant.py --port 30001

    # Full test with repeats:
    python bench_niah_turboquant.py --port 30001 \
        --context-lengths 512 1024 2048 4096 \
        --depth-percents 10 25 50 75 90 \
        --repeat 3 --verbose
"""

import argparse
import json
import re
import time
from collections import defaultdict

import requests

NEEDLES = [
    {
        "needle": "The special magic number is 9174826.",
        "question": "What is the special magic number mentioned in the text? Reply with ONLY the number.",
        "expected": ["9174826"],
    },
    {
        "needle": "The secret password for the vault is 'crystalline-7842'.",
        "question": "What is the secret password for the vault? Reply with ONLY the password.",
        "expected": ["crystalline-7842"],
    },
    {
        "needle": "The capital of the fictional country Zephyria is Windmere.",
        "question": "What is the capital of Zephyria? Reply with ONLY the city name.",
        "expected": ["Windmere", "windmere"],
    },
    {
        "needle": "Dr. Elena Chen won the Nobel Prize in Physics in 2031.",
        "question": "Who won the Nobel Prize in Physics in 2031? Reply with ONLY the full name.",
        "expected": ["Elena Chen", "elena chen"],
    },
    {
        "needle": "The maximum speed of the X-47 aircraft is 2,450 kilometers per hour.",
        "question": "What is the maximum speed of the X-47 aircraft in km/h? Reply with ONLY the number.",
        "expected": ["2450", "2,450"],
    },
]

FILLER_PARAGRAPH = (
    "The development of artificial intelligence has been one of the most significant "
    "technological advances of the 21st century. Machine learning algorithms have "
    "transformed industries ranging from healthcare to finance. Neural networks, "
    "inspired by biological brain structures, can now process images, understand "
    "natural language, and generate creative content. Researchers continue to push "
    "the boundaries of what these systems can achieve, while also grappling with "
    "important questions about safety, fairness, and alignment. The intersection "
    "of AI with other fields like robotics, quantum computing, and biotechnology "
    "promises even more transformative applications in the years ahead. "
)


def build_context(target_length: int, depth_percent: float, needle_text: str) -> str:
    target_chars = target_length * 4
    filler = FILLER_PARAGRAPH * (target_chars // len(FILLER_PARAGRAPH) + 1)

    insert_pos = int(len(filler) * depth_percent / 100.0)
    insert_pos = filler.rfind(". ", 0, insert_pos)
    if insert_pos == -1:
        insert_pos = 0
    else:
        insert_pos += 2

    context = filler[:insert_pos] + needle_text + " " + filler[insert_pos:]
    return context[:target_chars]


def query_server(port: int, context: str, question: str, model: str) -> str:
    url = f"http://localhost:{port}/v1/chat/completions"
    messages = [
        {"role": "user", "content": f"Read the following text carefully:\n\n{context}\n\n{question}"}
    ]
    payload = {"model": model, "messages": messages, "max_tokens": 64, "temperature": 0}
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def check_answer(response: str, expected_list: list[str]) -> bool:
    response_clean = re.sub(r"[^\w\s-]", "", response.lower().strip())
    for expected in expected_list:
        expected_clean = re.sub(r"[^\w\s-]", "", expected.lower().strip())
        if expected_clean in response_clean:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="NIAH benchmark for TurboQuant")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--depth-percents", type=int, nargs="+", default=[10, 25, 50, 75, 90])
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each test N times")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("NIAH Benchmark for TurboQuant")
    print(f"Server: localhost:{args.port}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Depth percents: {args.depth_percents}")
    print(f"Needles: {len(NEEDLES)} types, repeats: {args.repeat}")
    print("=" * 80)

    # results[ctx_len][depth] = (pass_count, total_count)
    matrix = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    all_results = []
    needle_idx = 0

    for ctx_len in args.context_lengths:
        for depth in args.depth_percents:
            for rep in range(args.repeat):
                needle = NEEDLES[needle_idx % len(NEEDLES)]
                needle_idx += 1

                context = build_context(ctx_len, depth, needle["needle"])
                t0 = time.time()
                response = query_server(args.port, context, needle["question"], args.model)
                elapsed = time.time() - t0

                is_correct = check_answer(response, needle["expected"])
                matrix[ctx_len][depth][1] += 1
                if is_correct:
                    matrix[ctx_len][depth][0] += 1

                status = "PASS" if is_correct else "FAIL"
                print(f"  ctx={ctx_len:5d}  depth={depth:3d}%  rep={rep+1}  {status}  ({elapsed:.1f}s)")
                if args.verbose or not is_correct:
                    print(f"    Needle: {needle['needle'][:60]}...")
                    print(f"    Response: {response[:200]}")

                all_results.append({
                    "context_length": ctx_len,
                    "depth_percent": depth,
                    "repeat": rep + 1,
                    "needle": needle["needle"][:60],
                    "correct": is_correct,
                    "response": response[:500],
                    "elapsed_s": round(elapsed, 2),
                })

    # Print pass rate matrix
    print("\n" + "=" * 80)
    print("NIAH Results Matrix:")
    print()
    header = f"{'depth%':>8s}"
    for ctx_len in args.context_lengths:
        header += f"  {ctx_len:>6d}"
    print(header)

    total_pass = 0
    total_count = 0
    for depth in args.depth_percents:
        row = f"{depth:>7d}%"
        for ctx_len in args.context_lengths:
            p, t = matrix[ctx_len][depth]
            total_pass += p
            total_count += t
            row += f"  {p:>2d}/{t:<2d} "
        print(row)

    print()
    pct = 100 * total_pass / max(total_count, 1)
    print(f"Overall: {total_pass}/{total_count} ({pct:.1f}%)")

    # Save results
    out_file = f"niah_results_{args.port}.json"
    with open(out_file, "w") as f:
        json.dump({
            "port": args.port,
            "model": args.model,
            "repeat": args.repeat,
            "results": all_results,
            "score": f"{total_pass}/{total_count} ({pct:.1f}%)",
        }, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
