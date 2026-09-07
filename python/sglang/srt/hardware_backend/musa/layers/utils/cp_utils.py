from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
        MusaFlashAttentionBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def musa_cp_attn_forward_extend(
    musa_fa_backend: "MusaFlashAttentionBackend",
    forward_batch: "ForwardBatch",
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Retained import for the MUSA backend; legacy CP execution is deprecated."""
    from sglang.srt.layers.utils.cp_utils import _deprecated_platform_cp

    _deprecated_platform_cp()
