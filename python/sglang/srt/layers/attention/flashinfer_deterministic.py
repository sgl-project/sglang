from typing import List, Optional, Sequence, Tuple

import torch


def deterministic_prefill_min_q(
    num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    gqa_group = max(num_qo_heads // num_kv_heads, 1)
    if head_dim >= 512:
        threshold = 32
    elif head_dim >= 256:
        threshold = 16
    else:
        threshold = 64
    return threshold // gqa_group + 1


def build_prefill_pad_indices(
    real_lens: Sequence[int], min_q: int, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    padded_lens: List[int] = [max(real, min_q) if real > 0 else 0 for real in real_lens]
    real_total = sum(real_lens)
    padded_total = sum(padded_lens)
    if padded_total == real_total:
        return None, None

    gather: List[int] = []
    out: List[int] = []
    real_off = 0
    pad_off = 0
    for real, padded in zip(real_lens, padded_lens):
        lead = padded - real
        gather.extend([real_total] * lead)
        gather.extend(range(real_off, real_off + real))
        out.extend(range(pad_off + lead, pad_off + padded))
        real_off += real
        pad_off += padded

    return (
        torch.tensor(gather, dtype=torch.long, device=device),
        torch.tensor(out, dtype=torch.long, device=device),
    )
