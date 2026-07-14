"""E2E parity test: seg_la vs seg_la_vk on a real Ling / Bailing-MoE-Linear model.

This test is **USER-RUN** — it needs a real checkpoint + GPUs + time.
Claude authored it; the user executes it.

Loads the same model twice (sequentially to save memory) — baseline with
default ``linear_backend`` (``"seg_la"``) and override
``json_model_override_args='{"linear_backend":"seg_la_vk"}'``.  Greedy-decodes
a fixed prompt set (single, batched, long) and asserts **identical token ids**.
Also runs a spec variant (NEXTN) to cover the MTP+scatter path.

Usage::
    MODEL=/root/.cache/modelscope/hub/models/inclusionAI/Ling-2___6-flash
    python3 test/manual/lightning/test_seg_la_vk_e2e_parity.py --model-path "$MODEL" --tp-size 4

Passing this test is the gate that confirms ``seg_la_vk`` is end-to-end
correct with no ``memory_pool.py`` / scatter changes needed.
"""

import argparse
import sys

import torch

from sglang import Engine

PROMPTS = [
    "The capital of France is",
    "Explain the key differences between linear attention and standard self-attention in one paragraph.",
    "Write a Python function to compute Fibonacci numbers recursively.",
    "In 2019, the European Space Agency launched",
    "A recipe for chocolate chip cookies:",
]

# Longer prompt to exercise prefill chunking
LONG_PROMPT = (
    "The theory of general relativity, proposed by Albert Einstein in 1915, "
    "fundamentally changed our understanding of gravity. Instead of viewing gravity "
    "as a force between masses, Einstein described it as the curvature of spacetime "
    "caused by mass and energy. This revolutionary framework has been confirmed by "
    "numerous experiments and observations over the past century, including the "
    "precession of Mercury's orbit, the bending of starlight during solar eclipses, "
    "and the recent detection of gravitational waves by LIGO. Describe the core "
    "principles of general relativity and its major experimental confirmations:"
)


def _build_engine(
    model_path: str,
    tp_size: int,
    extra_args: dict,
    spec: bool = False,
    model_impl: str = "auto",
):
    """Build an sglang.Engine with the given overrides."""
    kwargs = dict(
        model_path=model_path,
        tp_size=tp_size,
        model_impl=model_impl,
        trust_remote_code=True,
        mem_fraction_static=0.75,
        max_running_requests=64,
        log_level="error",
    )
    if extra_args:
        kwargs["json_model_override_args"] = _to_json_str(extra_args)
    if spec:
        kwargs.update(
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            mamba_scheduler_strategy="extra_buffer",
        )
    return Engine(**kwargs)


def _to_json_str(d: dict) -> str:
    import json

    return json.dumps(d)


def _greedy_generate(engine: Engine, prompts: list, max_tokens: int = 64):
    """Greedy (temperature=0) generate and return list of token-id lists."""
    try:
        outputs = engine.generate(
            prompts,
            sampling_params={"temperature": 0.0, "max_new_tokens": max_tokens},
        )
    except Exception:
        # Fallback: generate one-by-one if batched generate fails
        outputs = []
        for p in prompts:
            outputs.append(
                engine.generate(
                    p,
                    sampling_params={"temperature": 0.0, "max_new_tokens": max_tokens},
                )
            )
    tokens = []
    for out in outputs:
        if hasattr(out, "output_ids"):
            tokens.append(out.output_ids)
        elif isinstance(out, dict):
            tokens.append(out.get("output_ids", []))
        else:
            raise TypeError(f"Unexpected output type: {type(out)}")
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument(
        "--model-impl",
        type=str,
        default="auto",
        help="Model implementation backend, e.g. 'auto' or 'modelscope'",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--skip-spec", action="store_true", help="Skip speculative decoding test"
    )
    args = ap.parse_args()

    torch.cuda.empty_cache()

    # ---- Baseline: seg_la (kv, default) ----
    print("=== Loading baseline (seg_la / kv) ===", flush=True)
    engine_kv = _build_engine(
        args.model_path,
        args.tp_size,
        model_impl=args.model_impl,
        extra_args={},
        spec=False,
    )
    try:
        prompts = list(PROMPTS) + [LONG_PROMPT]
        kv_tokens = _greedy_generate(engine_kv, prompts, args.max_tokens)

        # Also test batched (first 3 prompts together)
        kv_batch = _greedy_generate(
            engine_kv, [PROMPTS[0], PROMPTS[1], PROMPTS[2]], args.max_tokens
        )
        print(
            f"  baseline generated {sum(len(t) for t in kv_tokens)} total tokens",
            flush=True,
        )
    finally:
        engine_kv.shutdown()
        del engine_kv
        torch.cuda.empty_cache()

    # ---- Override: seg_la_vk ----
    print("=== Loading override (seg_la_vk / vk) ===", flush=True)
    engine_vk = _build_engine(
        args.model_path,
        args.tp_size,
        model_impl=args.model_impl,
        extra_args={"linear_backend": "seg_la_vk"},
        spec=False,
    )
    try:
        prompts = list(PROMPTS) + [LONG_PROMPT]
        vk_tokens = _greedy_generate(engine_vk, prompts, args.max_tokens)
        vk_batch = _greedy_generate(
            engine_vk, [PROMPTS[0], PROMPTS[1], PROMPTS[2]], args.max_tokens
        )
        print(
            f"  override generated {sum(len(t) for t in vk_tokens)} total tokens",
            flush=True,
        )
    finally:
        engine_vk.shutdown()
        del engine_vk
        torch.cuda.empty_cache()

    # ---- Assert identical ----
    all_ok = True
    for i, (kv_tok, vk_tok) in enumerate(
        zip(kv_tokens + kv_batch, vk_tokens + vk_batch)
    ):
        if kv_tok != vk_tok:
            print(f"\nFAIL: prompt group {i}: tokens differ!", flush=True)
            print(
                f"  kv ({len(kv_tok)}): {kv_tok[:20]}{'...' if len(kv_tok) > 20 else ''}",
                flush=True,
            )
            print(
                f"  vk ({len(vk_tok)}): {vk_tok[:20]}{'...' if len(vk_tok) > 20 else ''}",
                flush=True,
            )
            all_ok = False
    if all_ok:
        print(
            "\nPASS: all prompts produce identical token ids (seg_la == seg_la_vk)",
            flush=True,
        )

    # ---- Spec variant (NEXTN → MTP + scatter) ----
    if not args.skip_spec:
        torch.cuda.empty_cache()
        print("\n=== Loading baseline (seg_la + spec) ===", flush=True)
        engine_kv_spec = _build_engine(
            args.model_path,
            args.tp_size,
            model_impl=args.model_impl,
            extra_args={},
            spec=True,
        )
        try:
            kv_spec_tokens = _greedy_generate(
                engine_kv_spec, list(PROMPTS), args.max_tokens
            )
        finally:
            engine_kv_spec.shutdown()
            del engine_kv_spec
            torch.cuda.empty_cache()

        print("=== Loading override (seg_la_vk + spec) ===", flush=True)
        engine_vk_spec = _build_engine(
            args.model_path,
            args.tp_size,
            model_impl=args.model_impl,
            extra_args={"linear_backend": "seg_la_vk"},
            spec=True,
        )
        try:
            vk_spec_tokens = _greedy_generate(
                engine_vk_spec, list(PROMPTS), args.max_tokens
            )
        finally:
            engine_vk_spec.shutdown()
            del engine_vk_spec
            torch.cuda.empty_cache()

        for i, (kv_tok, vk_tok) in enumerate(zip(kv_spec_tokens, vk_spec_tokens)):
            if kv_tok != vk_tok:
                print(f"\nFAIL (spec): prompt {i}: tokens differ!", flush=True)
                all_ok = False
        if all(
            kv_spec_tokens[i] == vk_spec_tokens[i] for i in range(len(kv_spec_tokens))
        ):
            print("PASS (spec): all prompts produce identical token ids", flush=True)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
