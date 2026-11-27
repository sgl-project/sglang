# type: ignore

"""Torch reference for Inkling-2's block attention-residual reduction.

The CuTe-DSL kernel in :mod:`inkling2_attn_res` relies on TMA, cluster launch
and PDL, so pre-Hopper devices use this implementation instead. It follows the
same math:

``vnorm_n = rsqrt(mean(x_n^2) + eps)``, ``logit = (q . x_n) * vnorm_n`` (queries
are pre-scaled by ``hidden_size**-0.5``), ``p = softmax(logit)`` and
``out = sum_n p_n * (vnorm_n if normalize_values else 1) * x_n``.
"""

import torch

_EPS = 1e-6


def attn_res_ref(
    blocks,
    partial_addend,
    queries,
    *,
    normalize_values=False,
    eps=_EPS,
):
    base = blocks[0]
    tokens, width = base.shape
    # The kernel accumulates in FP32 registers, so widen before adding the addend.
    sources = [block.float() for block in blocks]
    if partial_addend is not None:
        sources[-1] = sources[-1] + partial_addend.float()
    output = torch.empty(
        (queries.shape[0], tokens, width), dtype=torch.float32, device=base.device
    )
    if tokens == 0:
        return tuple(output.unbind(0))

    with torch.no_grad():
        logits = None
        for source in sources:
            x = source.float()
            vnorm = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / width + eps)
            term = (x @ queries.t()) * vnorm
            logits = (
                term.unsqueeze(-1)
                if logits is None
                else torch.cat([logits, term.unsqueeze(-1)], dim=-1)
            )
        probs = torch.softmax(logits, dim=-1)
        out = torch.zeros(
            (tokens, queries.shape[0], width),
            dtype=torch.float32,
            device=base.device,
        )
        for index, source in enumerate(sources):
            x = source.float()
            if normalize_values:
                x = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / width + eps)
            out += probs[..., index].unsqueeze(-1) * x.unsqueeze(-2)
        output.copy_(out.permute(1, 0, 2))
    return tuple(output.unbind(0))
