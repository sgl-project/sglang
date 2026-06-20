#!/usr/bin/env python3
"""Chain-of-Verification (CoVe) example for reducing LLM hallucinations with SGLang.

This script implements the "Factored CoVe" pattern from:
  Dhuliawala et al. (2023) "Chain-of-Verification Reduces Hallucination in Large Language Models"
  https://arxiv.org/abs/2309.11495

The key insight (Factored CoVe): the verification call runs in a **fresh session with
no shared history or KV-cache** from the original answer. This prevents the model from
simply repeating its earlier (possibly hallucinated) answer instead of genuinely checking it.

Flow
----
1. [Draft]    Send user query  → get initial answer from the model.
2. [Verify]   Open a *new, independent* chat (no history) and ask:
              "Does this answer actually address the question? If not, say so."
3. [Refine]   If the verifier flags a problem, ask it to produce a corrected answer
              (still within the same verify session so it has the critic context).
4. [Summarize] (Optional) Ask for a shorter final answer.

Usage
-----
1. Launch an SGLang-compatible server, e.g.:
     python -m sglang.launch_server \\
         --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

2. Run this script:
     python chain_of_verification.py --prompt "What year did the Titanic sink?"

   Or point at a different server / model:
     python chain_of_verification.py \\
         --base-url http://127.0.0.1:30002/v1 \\
         --model moonshot-v1-8k \\
         --prompt "Who invented the telephone?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = None  # auto-detected from /v1/models

VERIFY_SYSTEM_PROMPT = (
    "You are a strict fact-checker. "
    "You will be given a user question and a candidate answer. "
    "Decide whether the answer is accurate and directly addresses the question. "
    "Reply with exactly one of: PASS or FAIL, followed by a brief reason."
)

REFINE_INSTRUCTION = (
    "The previous answer was flagged as inaccurate or off-topic. "
    "Please provide a corrected, accurate answer to the original question."
)

SUMMARIZE_INSTRUCTION = (
    "Please give a concise, one-paragraph version of the verified answer above."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model(client: OpenAI, model: str | None) -> str:
    if model:
        return model
    models = client.models.list()
    if not models.data:
        raise RuntimeError("Server returned no models from /v1/models")
    return models.data[0].id


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    msg = response.choices[0].message
    # Reasoning models (e.g. Kimi-K2.5, Qwen3) may return the final answer in
    # `content` and the chain-of-thought in `reasoning_content`.  In some
    # configurations (greedy / temperature=0) `content` can be empty while the
    # useful text lives in `reasoning_content`, so fall back to it.
    content = msg.content or ""
    if not content.strip():
        content = getattr(msg, "reasoning_content", None) or ""
    return content


# ---------------------------------------------------------------------------
# CoVe pipeline
# ---------------------------------------------------------------------------


def chain_of_verification(
    client: OpenAI,
    model: str,
    user_query: str,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    summarize: bool = False,
    verbose: bool = True,
) -> str:
    """Run the Factored Chain-of-Verification pipeline.

    Parameters
    ----------
    client:       OpenAI-compatible client pointing at an SGLang server.
    model:        Model identifier.
    user_query:   The question or request from the user.
    max_tokens:   Token budget per call.
    temperature:  Sampling temperature.
    summarize:    If True, append an optional summarization step (Step 4).
    verbose:      Print intermediate results.

    Returns
    -------
    The final (verified / refined) answer string.
    """

    def _log(title: str, text: str) -> None:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{title}]")
            print(text)

    # ------------------------------------------------------------------
    # Step 1 — Draft: generate the initial answer
    # ------------------------------------------------------------------
    draft_messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_query},
    ]
    draft_answer = chat(client, model, draft_messages, max_tokens, temperature)
    _log("Step 1 · Draft answer", draft_answer)

    # ------------------------------------------------------------------
    # Step 2 — Verify (Factored): fresh session, no shared history/cache
    #
    # The verification session deliberately starts from scratch so the
    # model cannot attend to the draft answer's token embeddings.  This
    # mirrors the "factored" variant in the CoVe paper, which consistently
    # outperforms the joint variant.
    # ------------------------------------------------------------------
    verify_messages: list[dict[str, Any]] = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {user_query}\n\n"
                f"Candidate answer:\n{draft_answer}\n\n"
                "Does this answer accurately and completely address the question?"
            ),
        },
    ]
    verdict = chat(client, model, verify_messages, 256, temperature)
    _log("Step 2 · Verification verdict", verdict)

    passed = verdict.strip().upper().startswith("PASS")

    if passed:
        final_answer = draft_answer
        _log("Result", "Verification PASSED — using draft answer as final answer.")
    else:
        # ------------------------------------------------------------------
        # Step 3 — Refine: ask the verifier to correct its own critique
        #
        # We continue in the *verify* session (not the draft session) so the
        # model has context about *why* the draft failed.
        # ------------------------------------------------------------------
        verify_messages.append({"role": "assistant", "content": verdict})
        verify_messages.append({"role": "user", "content": REFINE_INSTRUCTION})
        refined_answer = chat(client, model, verify_messages, max_tokens, temperature)
        _log("Step 3 · Refined answer", refined_answer)
        final_answer = refined_answer

    # ------------------------------------------------------------------
    # Step 4 — Summarize (optional)
    # ------------------------------------------------------------------
    if summarize:
        summarize_messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": final_answer},
            {"role": "user", "content": SUMMARIZE_INSTRUCTION},
        ]
        summary = chat(client, model, summarize_messages, 256, temperature)
        _log("Step 4 · Summary (optional)", summary)
        final_answer = summary

    return final_answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chain-of-Verification (CoVe) hallucination reduction demo for SGLang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="SGLang OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name; auto-detected from /v1/models if omitted",
    )
    parser.add_argument(
        "--prompt",
        default="Who invented the telephone and in what year?",
        help="User question / prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Append an optional summarization step (Step 4)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate step output; only print the final answer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    try:
        model = resolve_model(client, args.model)
    except Exception as exc:
        print(f"Failed to connect to SGLang server: {exc}", file=sys.stderr)
        print(f"  Check server at {args.base_url}", file=sys.stderr)
        sys.exit(1)

    print(f"Model  : {model}")
    print(f"Query  : {args.prompt}")

    final = chain_of_verification(
        client=client,
        model=model,
        user_query=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        summarize=args.summarize,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(final)


if __name__ == "__main__":
    main()
