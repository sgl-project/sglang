"""
Top-nσ custom logit processor example.

Top-nσ ("Top-nσ: Not All Logits Are You Need", Tang et al., ACL 2025;
https://arxiv.org/abs/2411.07641) is a logit-space dynamic truncation method:

    threshold = max_logit - n * std_logit
    logits[logits < threshold] = -inf   # then sample as usual

Unlike SGLang's built-in ``top_k`` / ``top_p`` / ``min_p`` (which all filter in
*probability* space, after softmax), Top-nσ filters in *logit* space, before
softmax. The std-based threshold is dual-adaptive: it adapts both how many and
how loosely candidates survive, and because it never removes the argmax token
(threshold < max_logit whenever n > 0 and std > 0) greedy decoding is unaffected.

This file is a self-contained example of the SGLang custom logit processor
framework -- no framework changes. The processor is injected per request via the
``custom_logit_processor`` field, gated by ``--enable-custom-logit-processor``.

Usage:
    # 1. Launch a server with the custom logit processor gate enabled:
    python -m sglang.launch_server \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct \
        --port 30000 \
        --enable-custom-logit-processor

    # 2. Run the client demo (raw /generate + OpenAI chat.completions):
    python examples/runtime/custom_logit_processor/top_n_sigma.py

    # 3. Validate the truncation math without a server (needs torch only):
    python examples/runtime/custom_logit_processor/top_n_sigma.py --self-check

    # 4. Or run the companion pytest (imports this file, no server):
    pytest examples/runtime/custom_logit_processor/test_top_n_sigma.py
"""

import argparse
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

# Per-request key read from ``sampling_params.custom_params``.
PARAM_KEY = "top_n_sigma"


def apply_top_n_sigma(logits: torch.Tensor, n_values: torch.Tensor) -> torch.Tensor:
    """Vectorized Top-nσ truncation over a batch of logits.

    Args:
        logits: ``[batch, vocab]`` logits. The input is left unmodified; a new
            tensor is returned -- callers must use the return value.
        n_values: ``[batch]`` per-row nσ value. Rows whose value is ``<= 0`` or
            ``NaN`` are skipped (left unchanged).

    Rules (all vectorized, no per-row Python loop):
        * max/std are computed over the *finite* logits only. Real LLM logits
          routinely carry ``-inf`` (vocab padding, disallowed tokens, grammar
          constraints); reducing over them would drag std/max to ``-inf`` and,
          under an all-finite guard, skip the processor for nearly every request.
        * A row is filtered only when it has a valid ``n`` (> 0), at least two
          finite logits, and non-zero finite std -- otherwise it passes through
          untouched. (>= 2 finite logits are needed for an unbiased std.)
        * Argmax-invariant: ``threshold < max_logit`` for every filtered row, so
          the top-1 token always survives; already-``-inf`` tokens stay masked.
    """
    # Active rows: a positive, finite n and at least one finite logit.
    finite = torch.isfinite(logits)
    active = torch.isfinite(n_values) & (n_values > 0)
    active &= finite.any(dim=-1)
    if not bool(active.any()):
        return logits

    # max over finite logits (non-finite -> -inf so they never win the max).
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    max_logit = torch.where(finite, logits, neg_inf).max(dim=-1, keepdim=True).values

    # Unbiased (ddof=1) std over finite logits only, computed manually: torch has
    # no ``nanstd``/``nanmax``, so mask non-finite out of the mean/variance sums.
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    n_finite = finite.sum(dim=-1, keepdim=True)  # [batch, 1]
    mean = torch.where(finite, logits, zero).sum(dim=-1, keepdim=True) / n_finite.clamp(
        min=1
    )
    var = torch.where(finite, (logits - mean) ** 2, zero).sum(
        dim=-1, keepdim=True
    ) / (n_finite - 1).clamp(min=1)
    std_logit = var.sqrt()  # [batch, 1]
    # Need >= 2 finite logits for a defined std, and non-zero std to truncate.
    active &= (n_finite.squeeze(-1) >= 2) & (std_logit.squeeze(-1) > 0)
    if not bool(active.any()):
        return logits

    threshold = max_logit - n_values.unsqueeze(-1) * std_logit  # [batch, 1]
    mask = (logits < threshold) & active.unsqueeze(-1)
    return logits.masked_fill(mask, float("-inf"))


class TopNSigmaLogitProcessor(CustomLogitProcessor):
    """Top-nσ logit-space truncation, applied per request.

    Reads ``custom_params={"top_n_sigma": n}`` for each request in the batch.
    Requests without a positive numeric ``top_n_sigma`` are left unchanged, so
    the same processor can be attached to a mixed batch safely.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        # Build the per-row n vector; NaN marks "skip this row".
        n_values = torch.full(
            (logits.shape[0],), float("nan"), device=logits.device, dtype=logits.dtype
        )
        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            n = params.get(PARAM_KEY)
            # Reject None, non-numeric, and bool (bool is an int subclass).
            if isinstance(n, bool) or not isinstance(n, (int, float)):
                continue
            n_values[i] = float(n)

        return apply_top_n_sigma(logits, n_values)


def run_generate_example(base_url: str = "http://localhost:30000") -> None:
    """Raw ``/generate`` request using the processor."""
    import requests

    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "custom_logit_processor": TopNSigmaLogitProcessor.to_str(),
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": 32,
                # n = 1.0 keeps tokens within 1 std of the peak logit.
                "custom_params": {PARAM_KEY: 1.0},
            },
        },
        timeout=60,
    )
    print("[/generate]", response.json())


def run_openai_example(base_url: str = "http://127.0.0.1:30000/v1") -> None:
    """OpenAI-compatible chat.completions, with and without the processor.

    ``model="default"`` targets whichever model the server was launched with,
    so the example is not tied to a specific checkpoint.
    """
    import openai

    client = openai.Client(base_url=base_url, api_key="None")
    messages = [{"role": "user", "content": "List 3 countries and their capitals."}]

    filtered = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=1.0,
        max_tokens=32,
        extra_body={
            "custom_logit_processor": TopNSigmaLogitProcessor.to_str(),
            "custom_params": {PARAM_KEY: 1.0},
        },
    )
    print("[openai top_n_sigma=1.0]", filtered.choices[0].message.content)

    # Baseline: omit the processor to fall back to standard sampling.
    plain = client.chat.completions.create(
        model="default", messages=messages, temperature=1.0, max_tokens=32
    )
    print("[openai baseline]", plain.choices[0].message.content)


def self_check() -> None:
    """Assert the truncation math on a known input (needs torch, no server)."""
    # Row 0: n=1.0 -> keep tokens >= mean-ish threshold; Row 1: n=-1 -> untouched.
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0], [10.0, 9.0, 1.0, 0.0]])
    n_values = torch.tensor([1.0, -1.0])
    out = apply_top_n_sigma(logits.clone(), n_values)

    std = logits[0].std()
    threshold = logits[0].max() - 1.0 * std
    expected_keep = logits[0] >= threshold
    assert torch.isinf(out[0][~expected_keep]).all(), out[0]
    assert torch.isfinite(out[0][expected_keep]).all(), out[0]
    # Argmax-invariant: the top-1 token is always kept.
    assert torch.isfinite(out[0, logits[0].argmax()]), out[0]
    # Non-positive n leaves the row untouched.
    assert torch.equal(out[1], logits[1]), out[1]

    # std == 0 guard: all-equal logits pass through unchanged.
    flat = torch.zeros(1, 5)
    assert torch.equal(apply_top_n_sigma(flat.clone(), torch.tensor([2.0])), flat)

    # -inf logits are handled (max/std over finite only), NOT skipped: the row
    # is still truncated and the pre-existing -inf stays masked.
    logits_with_inf = torch.tensor([[10.0, 9.0, 1.0, float("-inf")]])
    out_inf = apply_top_n_sigma(logits_with_inf.clone(), torch.tensor([1.0]))
    expected_keep_inf = torch.tensor([True, True, False, False])
    assert torch.isinf(out_inf[0][~expected_keep_inf]).all(), out_inf
    assert torch.isfinite(out_inf[0][expected_keep_inf]).all(), out_inf
    # The processor stays ACTIVE with -inf masked tokens present -- it truncates
    # rather than being disabled/passed through (regression guard for the
    # all-finite reduction bug that would drag std/max to -inf and skip the row).
    assert not torch.equal(out_inf, logits_with_inf), out_inf

    # Single finite logit (rest -inf): std undefined -> row left unchanged.
    one_finite = torch.tensor([[5.0, float("-inf"), float("-inf"), float("-inf")]])
    assert torch.equal(apply_top_n_sigma(one_finite.clone(), torch.tensor([1.0])), one_finite)
    print("self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate the truncation math locally (no server needed).",
    )
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Only run the raw /generate example.",
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
    else:
        run_generate_example()
        if not args.skip_openai:
            run_openai_example()
