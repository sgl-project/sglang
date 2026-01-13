#!/usr/bin/env python3
"""
Needle Test for Attention Alignment Validation

This test verifies that attention visualization correctly aligns generated tokens
with their semantic sources.

Test: "The secret code is 4829. Please repeat the code. The secret code is"
Expected: Generated "4" should strongly attend to the "4" in "4829"

Usage:
    # Start server first:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --return-attention-tokens \
        --attention-tokens-top-k 16 \
        --port 8000 \
        --disable-cuda-graph

    # Then run this test:
    python needle_test.py --port 8000
"""

import argparse
import json
from typing import Any, Dict, List

import requests


def tokenize(text: str, base_url: str) -> List[Dict]:
    """Get tokens from the server's tokenizer."""
    try:
        resp = requests.post(
            f"{base_url}/tokenize",
            json={"text": text, "add_special_tokens": True},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Returns list of token IDs or token info
            token_ids = data.get("token_ids", data.get("tokens", []))
            # Also try to get text
            token_texts = data.get("token_texts", [])
            if token_texts:
                return [
                    {"id": tid, "text": txt} for tid, txt in zip(token_ids, token_texts)
                ]
            return [{"id": tid, "text": f"[{tid}]"} for tid in token_ids]
    except Exception as e:
        print(f"Tokenize failed: {e}")
    return []


def run_needle_test(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run the needle test and return results.

    The needle test checks:
    1. Given prompt with "4829", model should generate "4829"
    2. When generating first "4", attention should point to previous "4" in "4829"
    """
    prompt = "The secret code is 4829. Please repeat the code. The secret code is"

    print("=" * 60)
    print("NEEDLE TEST - Attention Alignment Validation")
    print("=" * 60)
    print(f"\nPrompt: {prompt}")
    print(f"Expected output: 4829 (or similar)")
    print("-" * 60)

    # Make request with attention capture
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.0,  # Deterministic
        "stream": False,  # Non-streaming to get attention in response
        "return_attention_tokens": True,  # Top-level flag
    }

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions", json=payload, timeout=120
        )

        if resp.status_code != 200:
            return {"status": "FAIL", "error": f"HTTP {resp.status_code}: {resp.text}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"status": "FAIL", "error": "No choices in response"}

        # Extract content
        message = choices[0].get("message", {})
        generated_text = message.get("content", "")

        # Extract attention data
        attention_data = choices[0].get("attention_tokens", [])

        # Tokenize generated text (approximate by splitting)
        generated_tokens = list(generated_text)  # Character-level fallback
        print(f"\nGenerated: {generated_text}")
        print(f"Tokens: {generated_tokens[:10]}...")  # First 10
        print(f"Attention entries: {len(attention_data)}")

        if not attention_data:
            return {
                "status": "FAIL",
                "error": "No attention data received. Check --return-attention-tokens flag.",
                "generated": generated_text,
            }

        # Tokenize the prompt to find "4" position
        prompt_tokens = tokenize(prompt, base_url)
        if verbose:
            print(f"\nPrompt tokens: {prompt_tokens}")

        # Find the position of "4" in "4829" (should be around position 6-8)
        needle_positions = []
        for i, tok in enumerate(prompt_tokens):
            if "4" in str(tok):
                needle_positions.append(i)

        print(f"\nNeedle '4' found at prompt positions: {needle_positions}")

        # Check first generated "4" attention
        first_four_idx = None
        for i, tok in enumerate(generated_tokens):
            if "4" in tok:
                first_four_idx = i
                break

        if first_four_idx is None:
            return {
                "status": "INCONCLUSIVE",
                "error": f"Model didn't generate '4'. Output: {generated_text}",
                "generated": generated_text,
            }

        print(f"\nFirst '4' generated at output position: {first_four_idx}")

        # Get attention for that token
        if first_four_idx < len(attention_data):
            attn = attention_data[first_four_idx]

            # Parse top-k positions
            top_k = []
            if "layers" in attn:
                # Multi-layer format
                for layer_id, layer_data in attn["layers"].items():
                    positions = layer_data.get("positions", [])
                    scores = layer_data.get("scores", [])
                    for pos, score in zip(positions, scores):
                        top_k.append(
                            {"position": pos, "score": score, "layer": layer_id}
                        )
            elif "positions" in attn:
                # Single layer format
                positions = attn.get("positions", [])
                scores = attn.get("scores", [])
                for pos, score in zip(positions, scores):
                    top_k.append({"position": pos, "score": score})

            if verbose:
                print(f"\nTop-K attention for first '4':")
                for item in sorted(top_k, key=lambda x: -x["score"])[:10]:
                    print(f"  Position {item['position']}: {item['score']:.4f}")

            # Check if any top-k position matches needle position
            top_positions = [
                item["position"]
                for item in sorted(top_k, key=lambda x: -x["score"])[:5]
            ]

            # Check overlap with needle positions
            hits = set(top_positions) & set(needle_positions)

            print(f"\nTop 5 attended positions: {top_positions}")
            print(f"Needle positions: {needle_positions}")
            print(f"Overlap: {hits}")

            if hits:
                print("\n" + "=" * 60)
                print("✅ PASS: Attention correctly points to semantic source")
                print("=" * 60)
                return {
                    "status": "PASS",
                    "generated": generated_text,
                    "needle_positions": needle_positions,
                    "top_attended": top_positions,
                    "overlap": list(hits),
                }
            else:
                print("\n" + "=" * 60)
                print("❌ FAIL: Attention does NOT point to semantic source")
                print("         This may indicate off-by-one alignment bug")
                print("=" * 60)
                return {
                    "status": "FAIL",
                    "error": "Attention misaligned - doesn't point to needle",
                    "generated": generated_text,
                    "needle_positions": needle_positions,
                    "top_attended": top_positions,
                }
        else:
            return {
                "status": "FAIL",
                "error": f"No attention for token {first_four_idx}",
                "generated": generated_text,
            }

    except requests.exceptions.ConnectionError:
        return {"status": "FAIL", "error": "Cannot connect to server. Is it running?"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Needle Test for Attention Alignment")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Check server health first
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"Server not healthy: {resp.status_code}")
            return 1
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to {base_url}")
        print("\nPlease start the server first:")
        print(f"  python -m sglang.launch_server \\")
        print(f"      --model-path Qwen/Qwen2.5-0.5B-Instruct \\")
        print(f"      --return-attention-tokens \\")
        print(f"      --attention-tokens-top-k 16 \\")
        print(f"      --port {args.port} \\")
        print(f"      --disable-cuda-graph")
        return 1

    result = run_needle_test(base_url, verbose=args.verbose)

    print(f"\n\nResult: {json.dumps(result, indent=2)}")

    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    exit(main())
