"""Read the target KV rows selected by DFly2 without intermediate tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_dfly_kv(
    kv,
    request_ids,
    selections,
    out,
    valid,
    kv_stride: tl.constexpr,
    selection_stride: tl.constexpr,
    topk: tl.constexpr,
    kv_rows: tl.constexpr,
    latent_dim: tl.constexpr,
    scale: tl.constexpr,
    rows_per_block: tl.constexpr,
    channels_per_row: tl.constexpr,
):
    request = tl.program_id(0)
    row = tl.program_id(1) * rows_per_block + tl.arange(0, rows_per_block)
    channel = tl.arange(0, channels_per_row)
    slot_group = tl.load(request_ids + request).to(tl.int64)
    slot = tl.load(
        selections + slot_group * selection_stride + row,
        mask=row < topk,
        other=-1,
    )
    selected = (slot >= 0) & (slot < kv_rows)
    values = tl.load(
        kv + slot[:, None].to(tl.int64) * kv_stride + channel[None, :],
        mask=selected[:, None] & (channel[None, :] < latent_dim),
    ).to(tl.float32)
    values = tl.where(selected[:, None], values, 0.0)
    output_row = request * topk + row
    tl.store(
        out + output_row[:, None] * latent_dim + channel[None, :],
        values * scale,
        mask=(row[:, None] < topk) & (channel[None, :] < latent_dim),
    )
    tl.store(valid + output_row, selected, mask=row < topk)


def gather_dfly_kv(
    kv: torch.Tensor,
    request_ids: torch.Tensor,
    selections: torch.Tensor,
    *,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather and dequantize selected latents for one paired target layer.

    Selections are indexed by request pool slot. Missing entries are negative.
    """
    batch_size = request_ids.numel()
    topk = selections.shape[1]
    latent_dim = kv.shape[1]
    out = torch.empty((batch_size, topk, latent_dim), dtype=dtype, device=kv.device)
    valid = torch.empty((batch_size, topk), dtype=torch.bool, device=kv.device)
    if batch_size and topk:
        _gather_dfly_kv[(batch_size, triton.cdiv(topk, 4))](
            kv,
            request_ids,
            selections,
            out,
            valid,
            kv.stride(0),
            selections.stride(0),
            topk,
            kv.shape[0],
            latent_dim,
            scale,
            4,
            triton.next_power_of_2(latent_dim),
        )
    return out, valid
