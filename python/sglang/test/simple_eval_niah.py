"""
Needle-in-a-Haystack (NIAH) evaluation for TurboQuant KV cache compression.

Inserts a "needle" fact at various depths within a "haystack" of filler text,
then asks the model to retrieve it.  Evaluates whether KV cache compression
preserves the model's ability to retrieve information from long contexts.

Paper reference (TurboQuant, arXiv 2504.19874):
  - Llama-3.1-8B-Instruct, 4K-104K tokens, 4x compression
  - TurboQuant: 0.997 recall (same as full-precision baseline)

Usage:
  # Start server first:
  python -m sglang.launch_server --model-path <model> --kv-cache-dtype turboquant \\
      --turboquant-bits 3.5 --port 30001

  # Then run:
  python -m sglang.test.simple_eval_niah --port 30001
"""

import argparse
import json
import random
import string
import time
from typing import List, Optional

import requests


# The standard NIAH needle and question (from the original NIAH benchmark)
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich "
    "and sit in Dolores Park on a sunny day."
)
DEFAULT_QUESTION = (
    "What is the best thing to do in San Francisco? "
    "Answer with the exact sentence from the context."
)

# Filler text: repeating essay-like content to build the haystack
FILLER_PARAGRAPH = (
    "The history of artificial intelligence dates back to ancient myths and stories "
    "of artificial beings endowed with intelligence. The modern field of AI research "
    "was founded at a workshop at Dartmouth College in 1956. Researchers explored "
    "problem solving and symbolic methods. In the 1960s, the US Department of Defense "
    "took interest and began funding AI research. Early work focused on formal reasoning "
    "and search algorithms. Machine learning, a subset of AI, gained prominence in the "
    "1990s with the rise of statistical methods. Deep learning emerged in the 2010s, "
    "powered by large datasets and GPU computing. Today AI systems can translate languages, "
    "recognize images, play games, and generate text. The field continues to advance rapidly "
    "with new architectures and training methods being developed each year. "
)


def build_haystack(target_tokens: int, tokenizer_estimate: float = 4.0) -> str:
    """Build filler text of approximately target_tokens length."""
    target_chars = int(target_tokens * tokenizer_estimate)
    repeats = (target_chars // len(FILLER_PARAGRAPH)) + 1
    haystack = (FILLER_PARAGRAPH * repeats)[:target_chars]
    return haystack


def insert_needle(haystack: str, needle: str, depth_percent: float) -> str:
    """Insert needle at the specified depth (0=start, 100=end) in the haystack."""
    pos = int(len(haystack) * depth_percent / 100.0)
    # Find a sentence boundary near the position
    for i in range(pos, min(pos + 200, len(haystack))):
        if haystack[i] == ".":
            pos = i + 1
            break
    return haystack[:pos] + " " + needle + " " + haystack[pos:]


def query_server(
    base_url: str,
    model: str,
    context: str,
    question: str,
    max_tokens: int = 100,
) -> str:
    """Send a completion request to the SGLang server."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def score_response(response: str, needle: str) -> float:
    """Score whether the response contains the needle content."""
    # Extract key phrases from the needle
    key_phrases = [
        "sandwich",
        "Dolores Park",
        "sunny day",
        "San Francisco",
    ]
    matches = sum(1 for phrase in key_phrases if phrase.lower() in response.lower())
    return matches / len(key_phrases)


def run_niah(
    base_url: str = "http://localhost:30001",
    context_lengths: Optional[List[int]] = None,
    depth_percents: Optional[List[float]] = None,
    needle: str = DEFAULT_NEEDLE,
    question: str = DEFAULT_QUESTION,
    num_repeats: int = 1,
):
    """Run the full NIAH evaluation."""
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384]
    if depth_percents is None:
        depth_percents = [0, 25, 50, 75, 100]

    # Get model name
    info = requests.get(f"{base_url}/get_model_info", timeout=10).json()
    model = info["model_path"]
    print(f"Model: {model}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depth percents: {depth_percents}")
    print()

    results = []
    total_score = 0
    total_count = 0

    print(f"  {'Length':>8s}", end="")
    for d in depth_percents:
        print(f"  {d:>5.0f}%", end="")
    print("   Avg")
    print(f"  {'--------':>8s}", end="")
    for _ in depth_percents:
        print(f"  {'------':>6s}", end="")
    print("  ------")

    for ctx_len in context_lengths:
        row_scores = []
        print(f"  {ctx_len:>8d}", end="")
        for depth in depth_percents:
            scores = []
            for _ in range(num_repeats):
                haystack = build_haystack(ctx_len)
                context = insert_needle(haystack, needle, depth)
                try:
                    response = query_server(base_url, model, context, question)
                    s = score_response(response, needle)
                    scores.append(s)
                except Exception as e:
                    print(f"\n  ERROR at ctx={ctx_len}, depth={depth}: {e}")
                    scores.append(0.0)

            avg = sum(scores) / len(scores) if scores else 0
            row_scores.append(avg)
            total_score += avg
            total_count += 1
            icon = "1.0" if avg >= 0.75 else f"{avg:.1f}"
            print(f"  {icon:>6s}", end="")

            results.append({
                "context_length": ctx_len,
                "depth_percent": depth,
                "score": avg,
            })

        row_avg = sum(row_scores) / len(row_scores) if row_scores else 0
        print(f"  {row_avg:.3f}")

    overall = total_score / total_count if total_count else 0
    print(f"\n  OVERALL RECALL: {overall:.3f}")
    print(f"  Paper reference (TurboQuant, 4-bit): 0.997")
    print(f"  Paper reference (Full precision):     0.997")

    return overall, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack evaluation")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--depth-percents",
        type=float,
        nargs="+",
        default=[0, 25, 50, 75, 100],
        help="Needle depth percentages",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Wait for server
    print("Waiting for server...")
    for i in range(60):
        try:
            requests.get(f"{base_url}/health", timeout=2)
            print("Server ready.")
            break
        except Exception:
            time.sleep(2)
    else:
        print("Server not ready after 120s, aborting.")
        exit(1)

    overall, results = run_niah(
        base_url=base_url,
        context_lengths=args.context_lengths,
        depth_percents=args.depth_percents,
    )
