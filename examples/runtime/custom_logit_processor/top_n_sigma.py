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

        batch_size = logits.size(0)

        # Pre-allocate per-row n-sigma and a boolean process_mask so we can
        # operate on the full logits tensor in-place, avoiding GPU copies that
        # advanced indexing would create.
        n_sigmas = [0.0] * batch_size
        process_mask = [False] * batch_size
        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            n = params.get("top_n_sigma")
            if n is None or not isinstance(n, (int, float)) or n <= 0:
                continue
            n_sigmas[i] = n
            process_mask[i] = True

        if not any(process_mask):
            return logits

        n_sigmas_t = torch.tensor(n_sigmas, device=logits.device, dtype=logits.dtype)
        process_mask_t = torch.tensor(process_mask, device=logits.device, dtype=torch.bool)

        # Compute std and max once over the full tensor (no row copies).
        std_all = logits.std(dim=-1, keepdim=True)
        max_logits = logits.max(dim=-1, keepdim=True).values

        # Build a combined mask: only rows that are (a) in the process list,
        # (b) have all-finite logits, and (c) have non-zero std.
        finite_mask = torch.isfinite(logits).all(dim=-1, keepdim=True)
        nonzero_std_mask = std_all != 0
        guard_mask = process_mask_t.unsqueeze(-1) & finite_mask & nonzero_std_mask

        thresholds = max_logits - n_sigmas_t.unsqueeze(-1) * std_all
        logits.masked_fill_(
            (logits < thresholds) & guard_mask, float("-inf")
        )
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

    client = openai.OpenAI(base_url=url + "/v1", api_key="None")
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
