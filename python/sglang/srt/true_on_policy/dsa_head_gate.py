"""BF16 indexer head-gate projection shared by TOP serving and the trainer.

Registered as a custom op so graph capture and compilation keep this exact
kernel.
"""

import torch

from sglang.srt.utils.custom_op import register_custom_op


def _fake(x, weight):
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )


@register_custom_op(fake_impl=_fake)
def project_head_gate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or weight.shape[1] % 128
        or weight.shape[0] % 16
    ):
        raise NotImplementedError(
            "TOP head gate requires a BF16 projection with 128-aligned input and 16-aligned heads"
        )
    if x.requires_grad or weight.requires_grad:
        raise NotImplementedError("TOP head-gate backward is not implemented")
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        _matmul_persistent_deepgemm,
    )

    return _matmul_persistent_deepgemm(
        x.contiguous(), weight.t().contiguous(), torch.bfloat16
    ).float()
