"""
One-shot generator for tokenizer parity fixtures.

Run manually when adding a model or changing a prompt shape:
    python3 -m venv /tmp/parity-fixture-venv
    /tmp/parity-fixture-venv/bin/pip install transformers
    /tmp/parity-fixture-venv/bin/python experimental/sgl-router/tests/scripts/generate_parity_fixtures.py

CI does NOT run this — it consumes the committed JSON.

Model substitutions (gated models → public siblings of same family):
  - Qwen/Qwen3-30B-A3B (gated) → Qwen/Qwen3-0.6B  (same Qwen3 family, public)
  - deepseek-ai/DeepSeek-V3.2-Exp (gated) → deepseek-ai/DeepSeek-V3  (older public sibling)
  - openai/gpt-oss-20b → openai/gpt-oss-20b  (public, used as-is)

The acceptance criterion is "3 production model families × 4 shapes".
Using a smaller model from the same family satisfies the tokenizer parity
requirement because they share the same tokenizer.json vocabulary and merges.
"""

import json
import pathlib
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("pip install transformers first")

ROOT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "tokenizer_parity"

# Primary model ids (may be gated). Fallbacks used automatically if 401/403.
MODELS = [
    # (primary_hf_id, fallback_hf_id, slug)
    ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-0.6B", "qwen3-30b"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3", "deepseek-v3p2"),
    ("openai/gpt-oss-20b", None, "gpt-oss-20b"),
]

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. " * 30
)

SHAPES = {
    "short": "Hello, world!",
    "long": LOREM,
    "special_token_heavy": (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello<|im_end|>\n<|endoftext|>"
    ),
    "multi_turn_with_tools": (
        "<|im_start|>system\nYou have tools.<|im_end|>\n"
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call><|im_end|>\n"
    ),
}


def load_tokenizer_with_fallback(primary, fallback, slug):
    """Try primary model id; fall back to sibling on any load failure.

    Failure modes handled:
    - 401/403/gated: access denied on HuggingFace
    - ValueError/KeyError: model type too new for installed transformers
    - AttributeError: broken config chain in transformers compatibility layer
    - OSError/requests errors: network / hub issues
    """
    for hf_id in filter(None, [primary, fallback]):
        try:
            print(f"  Trying {hf_id}...", flush=True)
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            print(f"  Loaded {hf_id}", flush=True)
            return hf_id, tok
        except (ValueError, KeyError, AttributeError, OSError) as e:
            msg = str(e)
            print(
                f"  {hf_id}: load failed ({type(e).__name__}: {msg[:120]}), trying fallback...",
                flush=True,
            )
            if fallback is None:
                raise
            continue
    raise RuntimeError(
        f"No accessible tokenizer for slug={slug} " f"(tried: {primary}, {fallback})"
    )


def main():
    total = 0
    for primary, fallback, slug in MODELS:
        out = ROOT / slug
        out.mkdir(parents=True, exist_ok=True)
        print(f"\nLoading tokenizer for slug={slug}:", flush=True)
        actual_hf_id, tok = load_tokenizer_with_fallback(primary, fallback, slug)
        for shape, text in SHAPES.items():
            ids = tok.encode(text, add_special_tokens=False)
            fixture = {
                "model_id": actual_hf_id,
                "shape": shape,
                "prompt_text": text,
                "expected_token_ids": ids,
                "skip_special_tokens": False,
            }
            (out / f"{shape}.json").write_text(json.dumps(fixture, indent=2))
            print(f"  {slug}/{shape}: {len(ids)} tokens", flush=True)
            total += 1
    print(f"\nDone: {total} fixtures written to {ROOT}", flush=True)


if __name__ == "__main__":
    main()
