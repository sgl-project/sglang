from typing import Dict, List

import torch

from sglang.srt.constrained.torch_ops.bitmask_ops import (
    apply_token_bitmask_inplace_torch,
)
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.srt.constrained.triton_ops.token_filter_ops import set_token_filter_triton
from sglang.srt.utils import is_hip

_is_hip = is_hip()
if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.srt.constrained.triton_ops.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )


def is_legacy_structural_tag(obj: Dict) -> bool:
    # test whether an object is a legacy structural tag
    # see `StructuralTagResponseFormat` at `sglang.srt.entrypoints.openai.protocol`
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False


def apply_packed_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Apply a packed int32 vocab mask to logits in-place.

    The packed mask uses one bit per token, where 1 means allowed and 0 means
    masked. By constrained-decoding policy, logits beyond the packed mask's
    token coverage are also masked out.
    """
    cutoff = vocab_mask.shape[-1] * 32
    if logits.device.type in {"cuda", "xpu", "musa"}:
        if _is_hip:
            apply_token_bitmask_inplace_cuda(logits, vocab_mask)
        else:
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
    elif logits.device.type in {"cpu", "npu"}:
        apply_token_bitmask_inplace_torch(logits, vocab_mask)
    else:
        raise RuntimeError(f"Unsupported device: {logits.device.type}")

    if logits.shape[-1] > cutoff:
        logits[..., cutoff:] = float("-inf")


def set_token_filter(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    """Set or clear specific tokens in a packed int32 vocab mask in-place.

    The packed mask uses one bit per token, where 1 means allowed and 0 means
    masked. If reset_vocab_mask is True, the row is reset to all masked when
    allowing specific tokens, or all allowed when blocking specific tokens.
    """
    if _is_hip or (vocab_mask.device.type != "cuda"):
        set_token_filter_torch(
            vocab_mask,
            token_ids,
            batch_idx,
            is_allowed=is_allowed,
            reset_vocab_mask=reset_vocab_mask,
        )
    else:
        set_token_filter_triton(
            vocab_mask,
            token_ids,
            batch_idx,
            is_allowed=is_allowed,
            reset_vocab_mask=reset_vocab_mask,
        )
