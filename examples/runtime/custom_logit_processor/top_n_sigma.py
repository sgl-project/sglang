"""
Example: Top-nσ (Top-n-sigma) logit truncation as a custom logit processor.

Top-nσ is a logit-space dynamic truncation method that uses the standard
deviation of the logit distribution to set a filtering threshold:

    threshold = max_logit - n * std_logit
    logits[logits < threshold] = -inf

Unlike probability-space filters (top_p, min_p), it operates *before* softmax
and adapts to the "peakiness" of the distribution:
  - Sharp distribution (model certain): small std -> narrow threshold -> fewer candidates
  - Flat distribution (model uncertain): large std -> wide threshold -> more candidates

Ref: Tang et al., "Top-nσ: Not All Logits Are You Need", ACL 2025.
     https://arxiv.org/abs/2411.07641

# Launch the server with the custom-logit-processor gate enabled.
#   python -m sglang.launch_server \
#       --model meta-llama/Meta-Llama-3-8B-Instruct \
#       --enable-custom-logit-processor

# Then run this script:
#   python examples/runtime/custom_logit_processor/top_n_sigma.py
"""

import requests
import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

url = "http://127.0.0.1:30000"
prompt = "The capital of France is"


class TopNSigmaLogitProcessor(CustomLogitProcessor):
    """Top-nσ logit truncation processor.

    Masks tokens whose logit value falls more than ``n`` standard deviations
    below the maximum logit, *before* temperature scaling and softmax.

    Implementation notes:
      - Fully vectorized across the batch (no per-row Python loop).
      - ``std`` is computed once and reused for both the ``std == 0`` guard and
        the threshold formula, avoiding a duplicate vocab-wide reduction.
      - Argmax-invariant by construction: the peak token always survives
        (it is zero standard deviations from itself).
    """

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits

        # Collect per-request n-sigma values and their batch indices.
        n_sigmas = []
        rows = []
        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            n = params.get("top_n_sigma")
            if n is None or not isinstance(n, (int, float)) or n <= 0:
                continue
            n_sigmas.append(n)
            rows.append(i)

        if not rows:
            return logits

        rows_t = torch.tensor(rows, device=logits.device, dtype=torch.long)
        selected = logits[rows_t]
        n_sigmas_t = torch.tensor(n_sigmas, device=logits.device, dtype=selected.dtype)

        # Compute std once; reuse it for both the edge-case guard and the
        # threshold formula to avoid scanning the vocab twice.
        std_all = selected.std(dim=-1, keepdim=True)

        # Skip rows with NaN/Inf logits or all-equal logits (std == 0).
        finite_mask = torch.isfinite(selected).all(dim=-1).unsqueeze(-1)
        nonzero_std_mask = std_all != 0
        process_mask = finite_mask & nonzero_std_mask

        max_logits = selected.max(dim=-1, keepdim=True).values
        thresholds = max_logits - n_sigmas_t.unsqueeze(-1) * std_all

        # Mask tokens below threshold, only on rows that pass the guards.
        selected.masked_fill_(
            (selected < thresholds) & process_mask, float("-inf")
        )
        # Advanced indexing produced a copy; write the modified rows back.
        logits[rows_t] = selected
        return logits


def run_generate():
    """Native /generate endpoint with top-nσ."""
    response = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "custom_logit_processor": TopNSigmaLogitProcessor.to_str(),
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 32,
                "custom_params": {"top_n_sigma": 2.0},
            },
        },
    ).json()
    print("=== /generate with top_n_sigma=2.0 ===")
    print(response["text"])


def run_openai():
    """OpenAI-compatible chat completions endpoint."""
    import openai

    client = openai.Client(base_url=url + "/v1", api_key="None")
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
        temperature=0.8,
        max_tokens=32,
        extra_body={
            "custom_logit_processor": TopNSigmaLogitProcessor.to_str(),
            "custom_params": {"top_n_sigma": 2.0},
        },
    )
    print("\n=== OpenAI chat with top_n_sigma=2.0 ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    run_generate()
    run_openai()
