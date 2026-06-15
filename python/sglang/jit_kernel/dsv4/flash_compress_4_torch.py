from __future__ import annotations

from typing import Tuple

import torch


def _as_i32_view(x_u8: torch.Tensor, words: int) -> torch.Tensor:
    assert x_u8.dtype == torch.uint8 and x_u8.is_contiguous()
    return x_u8.view(torch.int32).reshape(x_u8.shape[0], words)


def decode_plan_c(
    plan_c_u8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    plan_c_u8: [C, 16] uint8
    Layout:
      u32 seq_len
      u16 ragged_id (low16 of word1)
      u16 buffer_len (high16 of word1)
      i32 read_page_0
      i32 read_page_1
    """
    i32 = _as_i32_view(plan_c_u8, 4)
    seq_len = i32[:, 0].to(torch.int64)
    word1 = i32[:, 1].to(torch.int64) & 0xFFFFFFFF
    ragged_id = (word1 & 0xFFFF).to(torch.int64)
    buffer_len = ((word1 >> 16) & 0xFFFF).to(torch.int64)
    rp0 = i32[:, 2].to(torch.int64)
    rp1 = i32[:, 3].to(torch.int64)
    return seq_len, ragged_id, buffer_len, rp0, rp1


def decode_plan_w(plan_w_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    plan_w_u8: [W, 8] uint8
    Final kernel layout: i32 ragged_id, i32 write_loc
    """
    i32 = _as_i32_view(plan_w_u8, 2).to(torch.int64)
    return i32[:, 0], i32[:, 1]


def decode_plan_d(
    plan_d_u8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    plan_d_u8: [B, 16] uint8
    Layout: i32 seq_len, i32 write_loc, i32 read_page_0, i32 read_page_1
    """
    i32 = _as_i32_view(plan_d_u8, 4).to(torch.int64)
    return i32[:, 0], i32[:, 1], i32[:, 2], i32[:, 3]


def _infer_head_dim_c4(
    kv_input: torch.Tensor, ape: torch.Tensor, kv_output: torch.Tensor
) -> int:
    assert (
        kv_output.ndim == 2
    ), f"kv_output must be [*, head_dim], got {kv_output.shape}"
    head_dim = int(kv_output.shape[1])
    assert ape.shape == (
        8,
        head_dim,
    ), f"ape must be [8, head_dim], got {ape.shape}, head_dim={head_dim}"
    assert (
        kv_input.shape[1] == 4 * head_dim
    ), f"kv_input last dim {kv_input.shape[1]} != 4*head_dim {4*head_dim}"
    return head_dim


def _flatten_slots(kv_buffer: torch.Tensor) -> torch.Tensor:
    """
    Expect kv_buffer shaped [num_pages, ratio(=4), elem] where elem = head_dim*4.
    Treat as [num_pages*4, elem] because write_loc indexes slots in ElementSize units.
    """
    assert kv_buffer.ndim == 3, f"expected [num_pages,4,elem], got {kv_buffer.shape}"
    num_pages, ratio, elem = kv_buffer.shape
    assert ratio == 4, f"c4 expects ratio=4, got {ratio}"
    return kv_buffer.reshape(num_pages * ratio, elem)


def _page_slice(kv_flat: torch.Tensor, page_idx: int) -> torch.Tensor:
    """
    kv_flat: [num_pages*4, elem] -> return [4, elem] window for a page
    """
    base = page_idx * 4
    return kv_flat[base : base + 4]


def c4_forward_torch(
    kv_buf_0: torch.Tensor,  # [4, elem]
    kv_buf_1: torch.Tensor,  # [4, elem]
    kv_input: torch.Tensor,  # [num_q, elem]
    ragged_id: int,
    ape: torch.Tensor,  # [8, head_dim]
    should_overlap: bool,
    buffer_len: int,
) -> torch.Tensor:
    """
    Returns [head_dim] float32
    """
    head_dim = int(ape.shape[1])
    dev = kv_input.device
    neg_inf = -torch.finfo(torch.float32).max
    P = ragged_id

    # overlap positions 0..3
    if should_overlap:
        kv_start_ov = kv_input[P - 7 : P - 3]
        kv_ov_in = kv_start_ov[:, :head_dim]
        sc_ov_in = kv_start_ov[:, 2 * head_dim : 3 * head_dim]
        kv_ov_buf = kv_buf_0[:, :head_dim]
        sc_ov_buf = kv_buf_0[:, 2 * head_dim : 3 * head_dim]

        if buffer_len <= 0:
            kv_ov, sc_ov = kv_ov_in, sc_ov_in
        elif buffer_len >= 4:
            kv_ov, sc_ov = kv_ov_buf, sc_ov_buf
        else:
            m = (torch.arange(4, device=dev) < buffer_len)[:, None]
            kv_ov = torch.where(m, kv_ov_buf, kv_ov_in)
            sc_ov = torch.where(m, sc_ov_buf, sc_ov_in)
    else:
        kv_ov = torch.zeros((4, head_dim), device=dev, dtype=torch.float32)
        sc_ov = torch.full((4, head_dim), neg_inf, device=dev, dtype=torch.float32)

    # fresh positions 4..7
    kv_start_fr = kv_input[P - 3 : P + 1]
    kv_fr_in = kv_start_fr[:, head_dim : 2 * head_dim]
    sc_fr_in = kv_start_fr[:, 3 * head_dim : 4 * head_dim]
    kv_fr_buf = kv_buf_1[:, head_dim : 2 * head_dim]
    sc_fr_buf = kv_buf_1[:, 3 * head_dim : 4 * head_dim]

    if buffer_len <= 4:
        kv_fr, sc_fr = kv_fr_in, sc_fr_in
    elif buffer_len >= 8:
        kv_fr, sc_fr = kv_fr_buf, sc_fr_buf
    else:
        m = ((torch.arange(4, device=dev) + 4) < buffer_len)[:, None]
        kv_fr = torch.where(m, kv_fr_buf, kv_fr_in)
        sc_fr = torch.where(m, sc_fr_buf, sc_fr_in)

    kv = torch.cat([kv_ov, kv_fr], dim=0)  # [8, head_dim]
    sc = torch.cat([sc_ov, sc_fr], dim=0) + ape  # [8, head_dim]
    w = torch.softmax(sc, dim=0)
    return (kv * w).sum(dim=0)


def flash_compress4_prefill(
    kv_buffer: torch.Tensor,  # [num_pages, 4, head_dim*4]
    kv_input: torch.Tensor,  # [num_q, head_dim*4]
    kv_output: torch.Tensor,  # [C, head_dim]
    ape: torch.Tensor,  # [8, head_dim]
    plan_c_u8: torch.Tensor,  # [C, 16]
    plan_w_u8: torch.Tensor,  # [W, 8]
) -> None:
    head_dim = _infer_head_dim_c4(kv_input, ape, kv_output)

    kv_flat = _flatten_slots(kv_buffer)
    C = plan_c_u8.shape[0]
    assert kv_output.shape == (C, head_dim)

    seq_len, ragged_id, buffer_len, rp0, rp1 = decode_plan_c(plan_c_u8)

    # compress
    for pid in range(C):
        if seq_len[pid].item() < 0:
            continue

        P = int(ragged_id[pid].item())
        bl = int(buffer_len[pid].item())
        need_overlap = int(seq_len[pid].item()) > 4

        kv_buf_0 = _page_slice(kv_flat, int(rp0[pid].item()))
        kv_buf_1 = _page_slice(kv_flat, int(rp1[pid].item()))

        kv_output[pid].copy_(
            c4_forward_torch(
                kv_buf_0=kv_buf_0,
                kv_buf_1=kv_buf_1,
                kv_input=kv_input,
                ragged_id=P,
                ape=ape,
                should_overlap=need_overlap,
                buffer_len=bl,
            )
        )

    # write after compress
    rag_w, loc_w = decode_plan_w(plan_w_u8)
    for i in range(plan_w_u8.shape[0]):
        if rag_w[i].item() < 0:
            continue
        rid = int(rag_w[i].item())
        loc = int(loc_w[i].item())
        kv_flat[loc].copy_(kv_input[rid])


def flash_compress4_decode(
    kv_buffer: torch.Tensor,  # [num_pages, 4, head_dim*4]
    kv_input: torch.Tensor,  # [B, head_dim*4]
    kv_output: torch.Tensor,  # [B, head_dim]
    ape: torch.Tensor,  # [8, head_dim]
    plan_d_u8: torch.Tensor,  # [B, 16]
) -> None:
    head_dim = _infer_head_dim_c4(kv_input, ape, kv_output)

    kv_flat = _flatten_slots(kv_buffer)
    B = kv_input.shape[0]
    assert kv_output.shape == (B, head_dim)

    seq_len, write_loc, rp0, rp1 = decode_plan_d(plan_d_u8)

    # Scatter every token's write in one shot (each token writes a distinct slot).
    kv_flat.index_copy_(0, write_loc, kv_input.to(kv_flat.dtype))

    # Compress fires for tokens whose seq_len just hit a multiple of the ratio.
    # Each active token reads its own (distinct) pages, so reading after all writes
    # matches the per-token write-then-read of the scalar path.
    active = (seq_len % 4 == 0).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return

    # buffer_len is constant 8 in decode: overlap (rows 0..3) comes from page rp0,
    # fresh (rows 4..7) from page rp1. Overlap is masked off when seq_len <= 4.
    buf0 = kv_buffer[rp0[active]].float()  # [A, 4, head_dim*4]
    buf1 = kv_buffer[rp1[active]].float()
    kv_ov = buf0[:, :, :head_dim]
    sc_ov = buf0[:, :, 2 * head_dim : 3 * head_dim]
    kv_fr = buf1[:, :, head_dim : 2 * head_dim]
    sc_fr = buf1[:, :, 3 * head_dim : 4 * head_dim]

    need_overlap = (seq_len[active] > 4)[:, None, None]
    neg_inf = -torch.finfo(torch.float32).max
    kv_ov = torch.where(need_overlap, kv_ov, torch.zeros_like(kv_ov))
    sc_ov = torch.where(need_overlap, sc_ov, torch.full_like(sc_ov, neg_inf))

    kv = torch.cat([kv_ov, kv_fr], dim=1)  # [A, 8, head_dim]
    sc = torch.cat([sc_ov, sc_fr], dim=1) + ape  # ape [8, head_dim] broadcasts
    w = torch.softmax(sc, dim=1)
    kv_output[active] = (kv * w).sum(dim=1).to(kv_output.dtype)
