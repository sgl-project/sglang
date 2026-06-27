"""Top-n-sigma custom logit processor example for SGLang.

Top-n-sigma ("Top-nσ: Not All Logits Are You Need", Tang et al., ACL 2025,
https://arxiv.org/abs/2411.07641) is a logit-space dynamic truncation method.
For each row of logits it keeps only the candidates whose logit is within ``n``
standard deviations of the maximum logit::

    threshold = max_logit - n * std_logit
    logits[logits < threshold] = -inf

Unlike SGLang's built-in probability-space filters (``top_k`` / ``top_p`` /
``min_p``), which run *after* softmax, Top-n-sigma runs *before* softmax -- which
is where the custom logit processor pipeline executes -- and adapts the
candidate set to the "peakiness" of the distribution. Because the maximum-logit
token always survives the threshold, greedy decoding is unaffected.

This is a self-contained example: the processor below requires no framework
changes and is injected per-request via the ``custom_logit_processor`` field
plus the ``--enable-custom-logit-processor`` server gate.

Launch a server with the custom logit processor enabled::

    python -m sglang.launch_server \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct \
        --port 30000 \
        --enable-custom-logit-processor

Then run this script::

    python examples/runtime/custom_logit_processor/top_n_sigma.py
"""

import requests
import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class TopNSigmaLogitProcessor(CustomLogitProcessor):
    """Apply Top-n-sigma truncation in logit space, vectorized over the batch.

    Per-request parameter (passed through ``custom_params``):
        top_n_sigma (float): how many standard deviations below the maximum
            logit to keep. Must be > 0, otherwise the request is left untouched.
    """

    def __call__(self, logits, custom_param_list=None):
        # No parameters supplied -> return logits unchanged.
        if not custom_param_list:
            return logits

        # Per-row n. 0.0 marks a row to skip: a missing, non-numeric, or
        # non-positive ``top_n_sigma`` (``bool`` is rejected explicitly since it
        # is a subclass of ``int``).
        n_per_row = []
        for params in custom_param_list:
            n = params.get("top_n_sigma") if params else None
            if isinstance(n, bool) or not isinstance(n, (int, float)) or n <= 0:
                n = 0.0
            n_per_row.append(float(n))

        n = torch.tensor(n_per_row, device=logits.device, dtype=torch.float32)
        n = n.unsqueeze(1)  # [batch, 1]
        if not bool((n > 0).any()):
            return logits

        # Reductions computed once (in fp32) and reused for both the std == 0
        # guard and the threshold formula.
        logits_f32 = logits.float()
        std = logits_f32.std(dim=-1, keepdim=True)  # [batch, 1]
        max_logit = logits_f32.amax(dim=-1, keepdim=True)  # [batch, 1]

        # A row is filtered only if it was requested (n > 0), its logits are
        # finite, and the distribution is not degenerate (std > 0).
        finite = torch.isfinite(logits_f32).all(dim=-1, keepdim=True)
        valid = (n > 0) & finite & torch.isfinite(std) & (std > 0)  # [batch, 1]

        # threshold = max_logit - n * std; mask everything strictly below it.
        threshold = max_logit - n * std  # [batch, 1]
        mask = valid & (logits_f32 < threshold)  # [batch, vocab]
        return logits.masked_fill(mask, float("-inf"))


def main():
    url = "http://127.0.0.1:30000/generate"
    prompt = "The capital of France is"

    # With Top-n-sigma truncation (keep tokens within 1 sigma of the max logit).
    response = requests.post(
        url,
        json={
            "text": prompt,
            "custom_logit_processor": TopNSigmaLogitProcessor().to_str(),
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": 32,
                "custom_params": {"top_n_sigma": 1.0},
            },
        },
    ).json()
    print("Top-n-sigma (n=1.0):", response["text"])

    # Baseline without the processor, for comparison.
    baseline = requests.post(
        url,
        json={
            "text": prompt,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 32},
        },
    ).json()
    print("Baseline:           ", baseline["text"])


if __name__ == "__main__":
    main()
