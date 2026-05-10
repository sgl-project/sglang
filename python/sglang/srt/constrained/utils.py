from typing import Dict, Optional

import torch


def is_legacy_structural_tag(obj: Dict) -> bool:
    # test whether an object is a legacy structural tag
    # see `StructuralTagResponseFormat` at `sglang.srt.entrypoints.openai.protocol`
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False


def is_packed_bitmask_allowed_token(
    vocab_mask: torch.Tensor,
    token_id: int,
    vocab_size: Optional[int] = None,
) -> bool:
    if vocab_size is not None and token_id >= vocab_size:
        return False

    packed_idx = token_id // 32
    if packed_idx >= vocab_mask.shape[-1]:
        return False
    packed_value = int(vocab_mask[packed_idx].item())
    return (packed_value & (1 << (token_id % 32))) != 0


def is_dense_bool_mask_allowed_token(
    vocab_mask: torch.Tensor,
    token_id: int,
    vocab_size: Optional[int] = None,
) -> bool:
    if vocab_size is not None and token_id >= vocab_size:
        return False
    if token_id >= vocab_mask.shape[-1]:
        return False
    return not bool(vocab_mask[token_id].item())
