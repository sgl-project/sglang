#!/usr/bin/env python3
"""
Analyze attention patterns to find influential input tokens.

Usage:
    python analyze_attention.py "Your prompt here" --max-tokens 20
"""

import argparse
import json
import math
import requests
from collections import defaultdict


def analyze_attention(prompt: str, max_tokens: int = 20, api_base: str = "http://localhost:8000"):
    """Analyze which input tokens influence the most output tokens."""

    # Get completion with attention tokens
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "return_attention_tokens": True,
            "top_k_attention": 10,
        }
    )

    data = response.json()
    text = data["choices"][0]["text"]
    attention_tokens = data["choices"][0].get("attention_tokens", [])

    print(f"Prompt: \"{prompt}\"")
    print(f"Output: \"{text.strip()[:100]}{'...' if len(text) > 100 else ''}\"")
    print(f"Output tokens: {len(attention_tokens)}")
    print()

    # Tokenize prompt to get token texts
    tok_response = requests.post(
        f"{api_base}/v1/tokenize",
        json={"model": "default", "prompt": prompt}
    )
    prompt_token_ids = tok_response.json().get("tokens", [])
    prompt_len = len(prompt_token_ids)

    # Get individual token texts
    prompt_tokens = []
    for tid in prompt_token_ids:
        detok = requests.post(
            f"{api_base}/v1/detokenize",
            json={"model": "default", "tokens": [tid]}
        )
        prompt_tokens.append(detok.json().get("text", f"[{tid}]"))

    print(f"Prompt tokens ({prompt_len}):")
    for i, tok in enumerate(prompt_tokens):
        print(f"  {i:3d}: {repr(tok)}")
    print()

    # Calculate influence: how much each input token is attended to
    position_total_score = defaultdict(float)
    position_count = defaultdict(int)  # How many output tokens attend to it
    position_max_score = defaultdict(float)

    for entry in attention_tokens:
        positions = entry["token_positions"]
        scores = entry["attention_scores"]

        for pos, score in zip(positions, scores):
            if pos < prompt_len:  # Only count input tokens
                position_total_score[pos] += score
                position_count[pos] += 1
                position_max_score[pos] = max(position_max_score[pos], score)

    # Rank by total influence
    print("=" * 60)
    print("INPUT TOKEN INFLUENCE (ranked by total attention received)")
    print("=" * 60)
    print(f"{'Pos':>4} {'Token':<20} {'Total':>8} {'Count':>6} {'MaxScore':>8}")
    print("-" * 60)

    sorted_positions = sorted(
        position_total_score.items(),
        key=lambda x: -x[1]
    )

    for pos, total in sorted_positions[:15]:
        tok_text = prompt_tokens[pos] if pos < len(prompt_tokens) else f"[out_{pos}]"
        count = position_count[pos]
        max_score = position_max_score[pos]
        print(f"{pos:4d} {repr(tok_text):<20} {total:8.3f} {count:6d} {max_score:8.3f}")

    print()
    print("=" * 60)
    print("OUTPUT TOKEN ATTENTION BREAKDOWN")
    print("=" * 60)

    for i, entry in enumerate(attention_tokens[:5]):
        positions = entry["token_positions"]
        scores = entry["attention_scores"]
        logits = entry.get("topk_logits", [])
        logsumexp = entry.get("logsumexp_candidates", 0)

        # Calculate attention to prompt vs output
        prompt_attn = sum(s for p, s in zip(positions, scores) if p < prompt_len)
        output_attn = sum(s for p, s in zip(positions, scores) if p >= prompt_len)

        print(f"\nOutput token {i+1}:")
        print(f"  Prompt attention: {prompt_attn:.1%}")
        print(f"  Output attention: {output_attn:.1%}")
        print(f"  Top attended:")
        for j, (pos, score) in enumerate(zip(positions[:5], scores[:5])):
            if pos < len(prompt_tokens):
                tok_text = prompt_tokens[pos]
            else:
                tok_text = f"[out_{pos - prompt_len}]"

            # True probability
            if logits and logsumexp:
                true_prob = math.exp(logits[j] - logsumexp)
                print(f"    {pos:3d} {repr(tok_text):<15} score={score:.3f} true_prob={true_prob:.4f}")
            else:
                print(f"    {pos:3d} {repr(tok_text):<15} score={score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("prompt", help="The prompt to analyze")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max output tokens")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")

    args = parser.parse_args()
    analyze_attention(args.prompt, args.max_tokens, args.api_base)
