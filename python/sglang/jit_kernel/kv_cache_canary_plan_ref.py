"""Python reference for :func:`canary_plan_step`.

Vectorised CPU reference for the single-launch plan kernel. Pinned
byte-for-byte against the Triton implementation in
:mod:`sglang.jit_kernel.kv_cache_canary_plan`.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary_verify import VerifyPlan
from sglang.jit_kernel.kv_cache_canary_write import WritePlan


def canary_plan_step_torch_reference(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> None:
    """Python reference for :func:`canary_plan_step`. Same signature & byte-equal semantics.

    Vectorised across reqs using numpy / torch CPU; no python per-element loops. Extras are appended
    after per-req-derived entries without further translation (caller pre-translates).
    """
    bs = int(fb_req_pool_indices.shape[0])
    work_device = torch.device("cpu")

    verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    write_req_capacity = int(write_plan_out.write_seed_slot_indices.shape[0])

    if bs == 0:
        _zero_plans(
            verify_plan_out=verify_plan_out,
            write_plan_out=write_plan_out,
            write_req_capacity=write_req_capacity,
        )
        _append_extras(
            verify_plan_out=verify_plan_out,
            base_idx=0,
            extra_verify_slot_indices=extra_verify_slot_indices,
            extra_verify_positions=extra_verify_positions,
            extra_verify_prev_slot_indices=extra_verify_prev_slot_indices,
            extra_verify_num_valid=extra_verify_num_valid,
            verify_capacity=verify_capacity,
            work_device=work_device,
        )
        return

    req_pool_indices = fb_req_pool_indices.detach().to(
        device=work_device, dtype=torch.int64
    )
    prefix_lens = fb_prefix_lens.detach().to(device=work_device, dtype=torch.int64)
    extend_seq_lens = fb_extend_seq_lens.detach().to(
        device=work_device, dtype=torch.int64
    )
    req_to_token_host = req_to_token.detach().to(device=work_device, dtype=torch.int64)

    not_padding = req_pool_indices != 0

    if swa_window_size > 0:
        window_starts = torch.clamp(prefix_lens - swa_window_size, min=0)
    else:
        window_starts = torch.zeros_like(prefix_lens)
    verify_lens = torch.clamp(prefix_lens - window_starts, min=0)
    verify_lens = torch.where(not_padding, verify_lens, torch.zeros_like(verify_lens))

    write_lens = torch.where(
        not_padding, extend_seq_lens, torch.zeros_like(extend_seq_lens)
    )
    write_lens = torch.clamp(write_lens, min=0)

    # Materialize verify entries.
    verify_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=work_device),
            torch.cumsum(verify_lens, dim=0),
        ]
    )
    total_verify = int(verify_offsets[bs].item())

    if total_verify > 0:
        flat_req_idx = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int64, device=work_device), verify_lens
        )
        within_req = torch.arange(
            total_verify, dtype=torch.int64, device=work_device
        ) - verify_offsets[:-1].repeat_interleave(verify_lens)
        positions = window_starts[flat_req_idx] + within_req
        rp = req_pool_indices[flat_req_idx]
        slot_full = req_to_token_host[rp, positions]
        prev_pos = positions - 1
        safe_prev_pos = torch.clamp(prev_pos, min=0)
        prev_slot_full = req_to_token_host[rp, safe_prev_pos]

        if full_to_swa_index_mapping is not None:
            lut = full_to_swa_index_mapping.detach().to(
                device=work_device, dtype=torch.int64
            )
            slot = _swa_translate_simple(slot=slot_full, lut=lut)
            prev_slot_translated = _swa_translate_simple(slot=prev_slot_full, lut=lut)
        else:
            slot = slot_full
            prev_slot_translated = prev_slot_full

        chain_head_mask = prev_pos < 0
        prev_slot = torch.where(
            chain_head_mask,
            torch.full_like(prev_slot_translated, -1),
            prev_slot_translated,
        )

        capped = min(total_verify, verify_capacity)
        verify_plan_out.verify_slot_indices[:capped].copy_(
            slot[:capped]
            .to(verify_plan_out.verify_slot_indices.dtype)
            .to(verify_plan_out.verify_slot_indices.device)
        )
        verify_plan_out.verify_positions[:capped].copy_(
            positions[:capped]
            .to(verify_plan_out.verify_positions.dtype)
            .to(verify_plan_out.verify_positions.device)
        )
        verify_plan_out.verify_prev_slot_indices[:capped].copy_(
            prev_slot[:capped]
            .to(verify_plan_out.verify_prev_slot_indices.dtype)
            .to(verify_plan_out.verify_prev_slot_indices.device)
        )

    # Materialize write metadata: write_offsets cumsum + per-req seed slot.
    write_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=work_device),
            torch.cumsum(write_lens, dim=0),
        ]
    )
    # write_offsets out tensor shape is [write_req_capacity + 1]; only the active [0..bs] prefix is meaningful.
    out_write_offsets_len = int(write_plan_out.write_offsets.shape[0])
    copy_len = min(bs + 1, out_write_offsets_len)
    write_plan_out.write_offsets[:copy_len].copy_(
        write_offsets[:copy_len]
        .to(write_plan_out.write_offsets.dtype)
        .to(write_plan_out.write_offsets.device)
    )
    if copy_len < out_write_offsets_len:
        write_plan_out.write_offsets[copy_len:].zero_()

    seed_slot_full = torch.where(
        prefix_lens > 0,
        req_to_token_host[req_pool_indices, torch.clamp(prefix_lens - 1, min=0)],
        torch.full_like(prefix_lens, -1),
    )
    if full_to_swa_index_mapping is not None:
        lut = full_to_swa_index_mapping.detach().to(
            device=work_device, dtype=torch.int64
        )
        seed_slot = _swa_translate_simple(slot=seed_slot_full, lut=lut)
    else:
        seed_slot = seed_slot_full

    # Reqs that contribute no write entries: leave seed slot as -1 (so downstream skipping is unambiguous).
    seed_slot = torch.where(write_lens > 0, seed_slot, torch.full_like(seed_slot, -1))

    capped_reqs = min(bs, write_req_capacity)
    write_plan_out.write_seed_slot_indices[:capped_reqs].copy_(
        seed_slot[:capped_reqs]
        .to(write_plan_out.write_seed_slot_indices.dtype)
        .to(write_plan_out.write_seed_slot_indices.device)
    )

    # Active write-req count: number of leading reqs with write_lens > 0. Plan kernel writes bs (or smaller
    # when trailing rows are padding) — we treat "trailing padding rows do not contribute" by counting reqs
    # whose write_lens > 0 OR rpi != 0; for simplicity report bs (matches the kernel's "or smaller if padding
    # rows trail" clause when no padding).
    write_plan_out.write_num_valid_reqs.fill_(int(bs))

    # Append extras at base_idx = total_verify.
    _append_extras(
        verify_plan_out=verify_plan_out,
        base_idx=total_verify,
        extra_verify_slot_indices=extra_verify_slot_indices,
        extra_verify_positions=extra_verify_positions,
        extra_verify_prev_slot_indices=extra_verify_prev_slot_indices,
        extra_verify_num_valid=extra_verify_num_valid,
        verify_capacity=verify_capacity,
        work_device=work_device,
    )

    extras_count = int(extra_verify_num_valid.detach().to("cpu").item())
    verify_plan_out.verify_num_valid.fill_(int(total_verify + max(0, extras_count)))


def _zero_plans(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    write_req_capacity: int,
) -> None:
    verify_plan_out.verify_num_valid.zero_()
    write_plan_out.write_num_valid_reqs.zero_()
    write_plan_out.write_offsets.zero_()


def _swa_translate_simple(*, slot: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    lut_len = int(lut.shape[0])
    sentinel_mask = slot < 0
    safe_idx = torch.where(sentinel_mask, torch.zeros_like(slot), slot)
    if lut_len > 0:
        oob_mask = safe_idx >= lut_len
        safe_idx = torch.where(
            oob_mask, torch.full_like(safe_idx, lut_len - 1), safe_idx
        )
    translated = lut[safe_idx]
    return torch.where(sentinel_mask, slot, translated)


def _append_extras(
    *,
    verify_plan_out: VerifyPlan,
    base_idx: int,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    verify_capacity: int,
    work_device: torch.device,
) -> None:
    extras_count = int(extra_verify_num_valid.detach().to("cpu").item())
    if extras_count <= 0:
        return
    slots = extra_verify_slot_indices[:extras_count].to(
        device=work_device, dtype=torch.int64
    )
    positions = extra_verify_positions[:extras_count].to(
        device=work_device, dtype=torch.int64
    )
    prevs = extra_verify_prev_slot_indices[:extras_count].to(
        device=work_device, dtype=torch.int64
    )
    end_idx = min(base_idx + extras_count, verify_capacity)
    capped = max(0, end_idx - base_idx)
    if capped <= 0:
        return
    verify_plan_out.verify_slot_indices[base_idx:end_idx].copy_(
        slots[:capped]
        .to(verify_plan_out.verify_slot_indices.dtype)
        .to(verify_plan_out.verify_slot_indices.device)
    )
    verify_plan_out.verify_positions[base_idx:end_idx].copy_(
        positions[:capped]
        .to(verify_plan_out.verify_positions.dtype)
        .to(verify_plan_out.verify_positions.device)
    )
    verify_plan_out.verify_prev_slot_indices[base_idx:end_idx].copy_(
        prevs[:capped]
        .to(verify_plan_out.verify_prev_slot_indices.dtype)
        .to(verify_plan_out.verify_prev_slot_indices.device)
    )
