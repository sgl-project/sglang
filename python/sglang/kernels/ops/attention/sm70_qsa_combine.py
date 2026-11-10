"""Native split-attention merge for the measured SM70 QSA decode shapes."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "sm70_qsa_combine",
        cuda_files=["elementwise/sm70_qsa_combine.cuh"],
        cuda_wrappers=[("combine", "sglang::sm70_qsa_combine::combine")],
    )


def combine(partial, lse, lengths, selected_tokens, tokens_per_split=32):
    output = torch.empty(
        (partial.shape[0], 6, 256), dtype=partial.dtype, device=partial.device
    )
    _module().combine(partial, lse, lengths, output, selected_tokens, tokens_per_split)
    return output
