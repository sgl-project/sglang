"""
Smoke test for /v1/tokenize and /v1/detokenize round-trip.
"""

from __future__ import annotations

import httpx

MODEL = "Qwen/Qwen3-0.6B"
TEXT = "Hello, world!"


def test_tokenize_round_trip(router: str) -> None:
    """POST /v1/tokenize then /v1/detokenize must recover the original text."""
    # Tokenize
    tok_resp = httpx.post(
        f"{router}/v1/tokenize",
        json={"model": MODEL, "prompt": TEXT},
        timeout=30,
    )
    assert tok_resp.status_code == 200, tok_resp.text
    tokens = tok_resp.json()["tokens"]
    assert (
        isinstance(tokens, list) and len(tokens) > 0
    ), f"Expected non-empty token list, got: {tokens}"

    # Detokenize
    detok_resp = httpx.post(
        f"{router}/v1/detokenize",
        json={"model": MODEL, "tokens": tokens},
        timeout=30,
    )
    assert detok_resp.status_code == 200, detok_resp.text
    recovered = detok_resp.json()["text"]
    assert (
        TEXT in recovered or recovered in TEXT
    ), f"Round-trip mismatch: original={TEXT!r}, recovered={recovered!r}"
