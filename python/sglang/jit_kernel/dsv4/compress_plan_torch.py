from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def _compute_loc(
    swa_loc: torch.Tensor, swa_page_size: int, ring_size: int
) -> torch.Tensor:
    swa_loc = swa_loc.to(torch.int64)
    return (swa_loc // swa_page_size) * ring_size + (swa_loc % ring_size)


def _legacy_page(
    rid: torch.Tensor, position: torch.Tensor, compress_ratio: int
) -> torch.Tensor:
    rid = rid.to(torch.int64)
    position = position.to(torch.int64)
    if compress_ratio == 4:
        return rid * 2 + ((position // 4) & 1)
    return rid.clone()


def _legacy_loc(
    rid: torch.Tensor, position: torch.Tensor, compress_ratio: int
) -> torch.Tensor:
    return _legacy_page(rid, position, compress_ratio) * compress_ratio + (
        position % compress_ratio
    )


def _pack_decode_plans(
    seq_len: torch.Tensor, write_loc: torch.Tensor, rp0: torch.Tensor, rp1: torch.Tensor
) -> torch.Tensor:
    return (
        torch.stack(
            [
                seq_len.to(torch.int32),
                write_loc.to(torch.int32),
                rp0.to(torch.int32),
                rp1.to(torch.int32),
            ],
            dim=1,
        )
        .contiguous()
        .view(torch.uint8)
        .reshape(seq_len.shape[0], 16)
    )


def _prefill_buffer_len(j: int, compress_ratio: int) -> int:
    """How many of a compress event's window tokens come from the state buffer.

    Matches the CUDA JIT planner (c_plan.cuh): for the compress event at extend
    index ``j`` (0-based position within the current extend chunk),

        buffer_len = window_size - min(j + 1, window_size)

    where ``window_size`` is ``2 * compress_ratio`` for the overlapping c4 path
    and ``compress_ratio`` otherwise. The count decreases as ``j`` grows and
    reaches 0 once the whole window lives in ``kv_input`` (j + 1 >= window_size).
    It depends only on ``j``, not on ``prefix_len``.
    """
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return window_size - min(j + 1, window_size)


def plan_compress_prefill(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    pin_buffer: torch.Tensor,
    num_q_tokens: int,
    compress_ratio: int,
    swa_page_size: int,
    ring_size: int,
    use_cuda_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        req_pool_indices.device.type != "cpu" and req_pool_indices.dtype == torch.int64
    )
    assert req_to_token.device.type != "cpu" and req_to_token.dtype == torch.int32
    assert full_to_swa.device.type != "cpu" and full_to_swa.dtype == torch.int64
    assert (
        pin_buffer.device.type == "cpu"
        and pin_buffer.dtype == torch.uint8
        and pin_buffer.is_contiguous()
    )
    assert pin_buffer.numel() >= num_q_tokens * (16 + 8)

    device = req_pool_indices.device
    is_overlap = compress_ratio == 4
    mtp_pad = min(ring_size - compress_ratio, 4)

    seq_np = seq_lens.cpu().numpy().astype(np.int32)
    ext_np = extend_lens.cpu().numpy().astype(np.int32)

    c_seq: List[int] = []
    c_rid16: List[int] = []
    c_buf16: List[int] = []
    c_bid: List[int] = []
    w_packed32: List[int] = []
    w_pos1: List[int] = []

    counter = 0
    for b, (sl, el) in enumerate(zip(seq_np, ext_np)):
        sl = int(sl)
        el = int(el)
        prefix_len = sl - el

        last_c_pos = (sl // compress_ratio) * compress_ratio
        first_w_pos = min(
            last_c_pos - (compress_ratio if is_overlap else 0), sl - mtp_pad
        )

        for j in range(el):
            position = prefix_len + j
            ragged_id = counter + j

            if (position + 1) % compress_ratio == 0:
                buffer_len = _prefill_buffer_len(j, compress_ratio)

                c_seq.append(position + 1)
                c_rid16.append(ragged_id & 0xFFFF)
                c_buf16.append(int(buffer_len) & 0xFFFF)
                c_bid.append(b)

            do_write = position >= first_w_pos
            if not do_write and is_overlap:
                do_write = (position % swa_page_size) >= (
                    swa_page_size - compress_ratio
                )
            if do_write:
                w_packed32.append(((b & 0xFFFF) << 16) | (ragged_id & 0xFFFF))
                w_pos1.append(position + 1)

        counter += el

    num_c = len(c_seq)
    num_w = len(w_packed32)

    c_bytes = pin_buffer[: num_q_tokens * 16].reshape(num_q_tokens, 16)
    w_bytes = pin_buffer[num_q_tokens * 16 : num_q_tokens * 24].reshape(num_q_tokens, 8)
    plan_c_i32 = c_bytes.view(torch.int32).reshape(num_q_tokens, 4)
    plan_w_i32 = w_bytes.view(torch.int32).reshape(num_q_tokens, 2)

    plan_c_i32[:, 0] = -1
    plan_c_i32[:, 1] = 0
    plan_c_i32[:, 2] = -1
    plan_c_i32[:, 3] = -1
    plan_w_i32[:, 0] = -1
    plan_w_i32[:, 1] = -1

    if num_c:
        seq_t = torch.tensor(c_seq, dtype=torch.int32)
        rid16_t = torch.tensor(c_rid16, dtype=torch.int32)
        buf16_t = torch.tensor(c_buf16, dtype=torch.int32)
        bid_t = torch.tensor(c_bid, dtype=torch.int32)

        plan_c_i32[:num_c, 0] = seq_t
        plan_c_i32[:num_c, 1] = ((buf16_t & 0xFFFF) << 16) | (rid16_t & 0xFFFF)
        plan_c_i32[:num_c, 2] = -1
        plan_c_i32[:num_c, 3] = bid_t

    if num_w:
        plan_w_i32[:num_w, 0] = torch.tensor(w_packed32, dtype=torch.int32)
        plan_w_i32[:num_w, 1] = torch.tensor(w_pos1, dtype=torch.int32)

    plan_c = c_bytes[:num_c].to(device, non_blocking=True).clone()
    plan_w = w_bytes[:num_w].to(device, non_blocking=True).clone()

    # stage1 (kernel_1 semantics)
    if num_c:
        Ci32 = plan_c.view(torch.int32).reshape(num_c, 4)

        batch_id = Ci32[:, 3].long()
        pos1 = Ci32[:, 0].long() - 1
        pos0 = (pos1 - compress_ratio).clamp(min=0)

        buf_len = ((Ci32[:, 1] >> 16) & 0xFFFF).long()
        has_buf = buf_len > 0

        rid = req_pool_indices.index_select(0, batch_id).long()
        raw1 = req_to_token[rid, pos1].long()
        raw0 = req_to_token[rid, pos0].long()
        swa1 = full_to_swa.index_select(0, raw1).long()
        swa0 = full_to_swa.index_select(0, raw0).long()

        rp1 = (_compute_loc(swa1, swa_page_size, ring_size) // compress_ratio).to(
            torch.int32
        )
        rp0 = (_compute_loc(swa0, swa_page_size, ring_size) // compress_ratio).to(
            torch.int32
        )

        Ci32[:, 2] = torch.where(has_buf, rp0, torch.full_like(rp0, -1))
        Ci32[:, 3] = torch.where(has_buf, rp1, batch_id.to(torch.int32))

    if num_w:
        Wi32 = plan_w.view(torch.int32).reshape(num_w, 2)
        batch_id = (Wi32[:, 0] >> 16).long()
        pos = Wi32[:, 1].long() - 1

        rid = req_pool_indices.index_select(0, batch_id).long()
        raw = req_to_token[rid, pos].long()
        swa = full_to_swa.index_select(0, raw).long()
        loc = _compute_loc(swa, swa_page_size, ring_size).to(torch.int32)

        Wi32[:, 0] = (Wi32[:, 0] & 0xFFFF).to(torch.int32)
        Wi32[:, 1] = loc

    return plan_c, plan_w


def plan_compress_decode(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
    swa_page_size: int,
    ring_size: int,
) -> torch.Tensor:
    rid = req_pool_indices.long()
    pos1 = seq_lens.long() - 1
    pos0 = (pos1 - compress_ratio).clamp(min=0)

    raw1 = req_to_token[rid, pos1].long()
    raw0 = req_to_token[rid, pos0].long()
    swa1 = full_to_swa.index_select(0, raw1).long()
    swa0 = full_to_swa.index_select(0, raw0).long()

    loc1 = _compute_loc(swa1, swa_page_size, ring_size).to(torch.int32)
    loc0 = _compute_loc(swa0, swa_page_size, ring_size).to(torch.int32)

    return _pack_decode_plans(
        seq_lens, loc1, loc0 // compress_ratio, loc1 // compress_ratio
    )


def plan_compress_prefill_legacy(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    pin_buffer: torch.Tensor,
    num_q_tokens: int,
    compress_ratio: int,
    use_cuda_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        req_pool_indices.device.type != "cpu" and req_pool_indices.dtype == torch.int64
    )
    assert (
        pin_buffer.device.type == "cpu"
        and pin_buffer.dtype == torch.uint8
        and pin_buffer.is_contiguous()
    )
    assert pin_buffer.numel() >= num_q_tokens * (16 + 8)

    device = req_pool_indices.device
    is_overlap = compress_ratio == 4

    seq_np = seq_lens.cpu().numpy().astype(np.int32)
    ext_np = extend_lens.cpu().numpy().astype(np.int32)

    c_seq: List[int] = []
    c_rid16: List[int] = []
    c_buf16: List[int] = []
    c_bid: List[int] = []
    w_packed32: List[int] = []
    w_pos1: List[int] = []

    counter = 0
    for b, (sl, el) in enumerate(zip(seq_np, ext_np)):
        sl = int(sl)
        el = int(el)
        prefix_len = sl - el

        last_c_pos = (sl // compress_ratio) * compress_ratio
        first_w_pos = last_c_pos - (compress_ratio if is_overlap else 0)

        for j in range(el):
            position = prefix_len + j
            ragged_id = counter + j

            if (position + 1) % compress_ratio == 0:
                buffer_len = _prefill_buffer_len(j, compress_ratio)

                c_seq.append(position + 1)
                c_rid16.append(ragged_id & 0xFFFF)
                c_buf16.append(int(buffer_len) & 0xFFFF)
                c_bid.append(b)

            if position >= first_w_pos:
                w_packed32.append(((b & 0xFFFF) << 16) | (ragged_id & 0xFFFF))
                w_pos1.append(position + 1)

        counter += el

    num_c = len(c_seq)
    num_w = len(w_packed32)

    c_bytes = pin_buffer[: num_q_tokens * 16].reshape(num_q_tokens, 16)
    w_bytes = pin_buffer[num_q_tokens * 16 : num_q_tokens * 24].reshape(num_q_tokens, 8)
    plan_c_i32 = c_bytes.view(torch.int32).reshape(num_q_tokens, 4)
    plan_w_i32 = w_bytes.view(torch.int32).reshape(num_q_tokens, 2)

    plan_c_i32[:, 0] = -1
    plan_c_i32[:, 1] = 0
    plan_c_i32[:, 2] = -1
    plan_c_i32[:, 3] = -1
    plan_w_i32[:, 0] = -1
    plan_w_i32[:, 1] = -1

    if num_c:
        seq_t = torch.tensor(c_seq, dtype=torch.int32)
        rid16_t = torch.tensor(c_rid16, dtype=torch.int32)
        buf16_t = torch.tensor(c_buf16, dtype=torch.int32)
        bid_t = torch.tensor(c_bid, dtype=torch.int32)

        plan_c_i32[:num_c, 0] = seq_t
        plan_c_i32[:num_c, 1] = ((buf16_t & 0xFFFF) << 16) | (rid16_t & 0xFFFF)
        plan_c_i32[:num_c, 2] = -1
        plan_c_i32[:num_c, 3] = bid_t

    if num_w:
        plan_w_i32[:num_w, 0] = torch.tensor(w_packed32, dtype=torch.int32)
        plan_w_i32[:num_w, 1] = torch.tensor(w_pos1, dtype=torch.int32)

    plan_c = c_bytes[:num_c].to(device, non_blocking=True).clone()
    plan_w = w_bytes[:num_w].to(device, non_blocking=True).clone()

    if num_c:
        Ci32 = plan_c.view(torch.int32).reshape(num_c, 4)
        b = Ci32[:, 3].long()
        pos1 = Ci32[:, 0].long() - 1
        pos0 = (pos1 - compress_ratio).clamp(min=0)
        rid = req_pool_indices.index_select(0, b).long()
        Ci32[:, 2] = _legacy_page(rid, pos0, compress_ratio).to(torch.int32)
        Ci32[:, 3] = _legacy_page(rid, pos1, compress_ratio).to(torch.int32)

    if num_w:
        Wi32 = plan_w.view(torch.int32).reshape(num_w, 2)
        b = (Wi32[:, 0] >> 16).long()
        rid16 = (Wi32[:, 0] & 0xFFFF).to(torch.int32)
        pos = Wi32[:, 1].long() - 1
        rid = req_pool_indices.index_select(0, b).long()
        Wi32[:, 0] = rid16
        Wi32[:, 1] = _legacy_loc(rid, pos, compress_ratio).to(torch.int32)

    return plan_c, plan_w


def plan_compress_decode_legacy(
    req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, compress_ratio: int
) -> torch.Tensor:
    rid = req_pool_indices.long()
    pos1 = seq_lens.long() - 1
    pos0 = (pos1 - compress_ratio).clamp(min=0)
    write_loc = _legacy_loc(rid, pos1, compress_ratio).to(torch.int32)
    rp0 = _legacy_page(rid, pos0, compress_ratio).to(torch.int32)
    rp1 = _legacy_page(rid, pos1, compress_ratio).to(torch.int32)
    return _pack_decode_plans(seq_lens, write_loc, rp0, rp1)
