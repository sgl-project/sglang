from __future__ import annotations

from typing import Tuple

import torch


def _as_i32_view(x_u8: torch.Tensor, words: int) -> torch.Tensor:
    assert x_u8.dtype == torch.uint8 and x_u8.is_contiguous()
    return x_u8.view(torch.int32).reshape(x_u8.shape[0], words)


def decode_plan_c(
    plan_c_u8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    i32 = _as_i32_view(plan_c_u8, 4)
    seq_len = i32[:, 0].to(torch.int64)
    word1 = i32[:, 1].to(torch.int64) & 0xFFFFFFFF
    ragged_id = (word1 & 0xFFFF).to(torch.int64)
    buffer_len = ((word1 >> 16) & 0xFFFF).to(torch.int64)
    rp0 = i32[:, 2].to(torch.int64)
    rp1 = i32[:, 3].to(torch.int64)
    return seq_len, ragged_id, buffer_len, rp0, rp1


def decode_plan_w(plan_w_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    i32 = _as_i32_view(plan_w_u8, 2).to(torch.int64)
    return i32[:, 0], i32[:, 1]


def decode_plan_d(
    plan_d_u8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    i32 = _as_i32_view(plan_d_u8, 4).to(torch.int64)
    return i32[:, 0], i32[:, 1], i32[:, 2], i32[:, 3]


def _infer_head_dim_from_inputs(
    kv_input: torch.Tensor, ape: torch.Tensor, kv_output: torch.Tensor
) -> int:
    assert (
        kv_output.ndim == 2
    ), f"kv_output must be [*, head_dim], got {kv_output.shape}"
    head_dim = int(kv_output.shape[1])
    assert (
        ape.shape[1] == head_dim
    ), f"ape head_dim {ape.shape[1]} != kv_output head_dim {head_dim}"
    assert (
        kv_input.shape[1] == 2 * head_dim
    ), f"kv_input last dim {kv_input.shape[1]} != 2*head_dim {2*head_dim}"
    return head_dim


def _flatten_slots_128(kv_buffer: torch.Tensor) -> torch.Tensor:
    assert kv_buffer.ndim == 3, f"expected [num_pages,128,elem], got {kv_buffer.shape}"
    num_pages, page_size, elem = kv_buffer.shape
    assert page_size == 128, f"c128 expects page_size=128, got {page_size}"
    return kv_buffer.reshape(num_pages * page_size, elem)


def _page_slice_128(kv_flat: torch.Tensor, page_idx: int) -> torch.Tensor:
    base = page_idx * 128
    return kv_flat[base : base + 128]


def c128_forward_torch(
    kv_buf_page: torch.Tensor,  # [128, head_dim*2]
    kv_input: torch.Tensor,  # [N, head_dim*2]
    ragged_id: int,  # position row index
    kv_out: torch.Tensor,  # [head_dim]
    ape: torch.Tensor,  # [128, head_dim]
    buffer_len: int,
) -> None:
    head_dim = kv_out.numel()
    dev = kv_input.device
    P = ragged_id

    kv_start = kv_input[P - 127 : P + 1]  # [128, 2*head_dim]
    if buffer_len <= 0:
        window = kv_start
    elif buffer_len >= 128:
        window = kv_buf_page
    else:
        m = (torch.arange(128, device=dev) < buffer_len)[:, None]
        window = torch.where(m, kv_buf_page, kv_start)

    kv = window[:, :head_dim]
    sc = window[:, head_dim : 2 * head_dim] + ape
    w = torch.softmax(sc, dim=0)
    kv_out.copy_((kv * w).sum(dim=0))


def flash_compress128_prefill(
    kv_buffer: torch.Tensor,  # [num_pages, 128, head_dim*2]
    kv_input: torch.Tensor,  # [N, head_dim*2]
    kv_output: torch.Tensor,  # [C, head_dim] (compact)
    ape: torch.Tensor,  # [128, head_dim]
    plan_c_u8: torch.Tensor,  # [C, 16]
    plan_w_u8: torch.Tensor,  # [W, 8]
) -> None:
    head_dim = _infer_head_dim_from_inputs(kv_input, ape, kv_output)

    kv_flat = _flatten_slots_128(kv_buffer)
    C = plan_c_u8.shape[0]
    assert kv_output.shape == (C, head_dim), (kv_output.shape, C, head_dim)

    seq_len, ragged_id, buffer_len, _rp0, rp1 = decode_plan_c(plan_c_u8)

    # compress
    for pid in range(C):
        if seq_len[pid].item() < 0:
            continue
        P = int(ragged_id[pid].item())
        bl = int(buffer_len[pid].item())
        page = int(rp1[pid].item())
        kv_buf_page = _page_slice_128(kv_flat, page)

        c128_forward_torch(
            kv_buf_page=kv_buf_page,
            kv_input=kv_input,
            ragged_id=P,
            kv_out=kv_output[pid],
            ape=ape,
            buffer_len=bl,
        )

    # write after compress
    rag_w, loc_w = decode_plan_w(plan_w_u8)
    for i in range(plan_w_u8.shape[0]):
        if rag_w[i].item() < 0:
            continue
        rid = int(rag_w[i].item())
        loc = int(loc_w[i].item())
        kv_flat[loc].copy_(kv_input[rid])


def flash_compress128_decode(
    kv_buffer: torch.Tensor,  # [num_pages, 128, head_dim*2]
    kv_input: torch.Tensor,  # [B, head_dim*2]
    kv_output: torch.Tensor,  # [B, head_dim]
    ape: torch.Tensor,  # [128, head_dim]
    plan_d_u8: torch.Tensor,  # [B, 16]
) -> None:
    head_dim = _infer_head_dim_from_inputs(kv_input, ape, kv_output)

    kv_flat = _flatten_slots_128(kv_buffer)
    B = kv_input.shape[0]
    assert kv_output.shape == (B, head_dim), (kv_output.shape, B, head_dim)

    seq_len, write_loc, _rp0, rp1 = decode_plan_d(plan_d_u8)

    for b in range(B):
        loc = int(write_loc[b].item())
        kv_flat[loc].copy_(kv_input[b])

        if loc % 128 != 127:
            continue

        page = int(rp1[b].item())
        kv_buf_page = _page_slice_128(kv_flat, page)

        c128_forward_torch(
            kv_buf_page=kv_buf_page,
            kv_input=kv_input,
            ragged_id=b,  # decode kv_src points at row b
            kv_out=kv_output[b],
            ape=ape,
            buffer_len=128,  # constant in flash_c128_decode
        )
