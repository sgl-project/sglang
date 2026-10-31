from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.consts import (
    canary_info,
    sampling_digest_from_values,
    to_signed_int64,
)
from sglang.kernels.ops.kv_canary.verify_ref import compute_slot_hash


def launch_canary_info_kernel_torch_reference(
    *,
    canary_buf: torch.Tensor,
    out_cache_loc: torch.Tensor,
    next_token_ids: torch.Tensor,
    accept_lens: Optional[torch.Tensor],
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    min_ps: torch.Tensor,
    out: torch.Tensor,
) -> None:
    buf_i64 = canary_buf.detach().cpu().contiguous().view(torch.int64)
    slot_rows = out_cache_loc.detach().cpu().tolist()
    token_rows = next_token_ids.detach().cpu().tolist()
    lengths = None if accept_lens is None else accept_lens.detach().cpu().tolist()
    temperature_list = temperatures.detach().cpu().tolist()
    top_p_list = top_ps.detach().cpu().tolist()
    top_k_list = top_ks.detach().cpu().tolist()
    min_p_list = min_ps.detach().cpu().tolist()
    values = []
    for row in range(out.shape[0]):
        sampling = sampling_digest_from_values(
            temperature=temperature_list[row],
            top_p=top_p_list[row],
            top_k=top_k_list[row],
            min_p=min_p_list[row],
        )
        values.append(
            [
                (
                    to_signed_int64(
                        canary_info(
                            compute_slot_hash(buf_i64, slot_rows[row][col]),
                            token_rows[row][col],
                            sampling,
                        )
                    )
                    if (lengths is None or col < lengths[row])
                    and slot_rows[row][col] >= 0
                    else 0
                )
                for col in range(out.shape[1])
            ]
        )
    out.copy_(torch.tensor(values, dtype=torch.int64).view(out.shape).to(out.device))
