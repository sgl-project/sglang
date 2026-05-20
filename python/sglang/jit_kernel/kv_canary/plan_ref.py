"""Python reference for :func:`canary_plan_step`.

Per-req Python for-loops + scalar ops. Pinned byte-for-byte against the Triton implementation in
:mod:`sglang.jit_kernel.kv_canary.plan`.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan


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

    Extras are appended after per-req-derived entries without further translation (caller pre-translates).
    """
    bs = int(fb_req_pool_indices.shape[0])
    work_device = torch.device("cpu")

    verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    write_req_capacity = int(write_plan_out.write_seed_slot_indices.shape[0])

    if bs == 0:
        verify_plan_out.verify_num_valid.zero_()
        write_plan_out.write_num_valid_reqs.zero_()
        write_plan_out.write_offsets.zero_()
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

    req_pool_indices_host = fb_req_pool_indices.detach().to(device=work_device, dtype=torch.int64)
    prefix_lens_host = fb_prefix_lens.detach().to(device=work_device, dtype=torch.int64)
    extend_seq_lens_host = fb_extend_seq_lens.detach().to(device=work_device, dtype=torch.int64)
    req_to_token_host = req_to_token.detach().to(device=work_device, dtype=torch.int64)

    lut: Optional[torch.Tensor] = None
    if full_to_swa_index_mapping is not None:
        lut = full_to_swa_index_mapping.detach().to(device=work_device, dtype=torch.int64)

    total_verify = _materialize_verify_entries(
        verify_plan_out=verify_plan_out,
        req_pool_indices_host=req_pool_indices_host,
        prefix_lens_host=prefix_lens_host,
        req_to_token_host=req_to_token_host,
        swa_window_size=swa_window_size,
        lut=lut,
        verify_capacity=verify_capacity,
        work_device=work_device,
        bs=bs,
    )

    _materialize_write_metadata(
        write_plan_out=write_plan_out,
        req_pool_indices_host=req_pool_indices_host,
        prefix_lens_host=prefix_lens_host,
        extend_seq_lens_host=extend_seq_lens_host,
        req_to_token_host=req_to_token_host,
        lut=lut,
        write_req_capacity=write_req_capacity,
        work_device=work_device,
        bs=bs,
    )

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


def _swa_translate_slot(*, slot: int, lut: torch.Tensor) -> int:
    """Translate a single full-pool slot index to its SWA-pool equivalent via lut."""
    if slot < 0:
        return slot
    lut_len = int(lut.shape[0])
    safe_idx = min(slot, lut_len - 1) if lut_len > 0 else slot
    return int(lut[safe_idx].item())


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


def _materialize_verify_entries(
    *,
    verify_plan_out: VerifyPlan,
    req_pool_indices_host: torch.Tensor,
    prefix_lens_host: torch.Tensor,
    req_to_token_host: torch.Tensor,
    swa_window_size: int,
    lut: Optional[torch.Tensor],
    verify_capacity: int,
    work_device: torch.device,
    bs: int,
) -> int:
    out_slots: list[int] = []
    out_positions: list[int] = []
    out_prev_slots: list[int] = []

    for r in range(bs):
        rpi = int(req_pool_indices_host[r].item())
        prefix_len = int(prefix_lens_host[r].item())

        if rpi == 0:
            continue

        if swa_window_size > 0:
            window_start = max(0, prefix_len - swa_window_size)
        else:
            window_start = 0
        verify_len = max(0, prefix_len - window_start)

        for j in range(verify_len):
            position = window_start + j
            slot_full = int(req_to_token_host[rpi, position].item())

            if lut is not None:
                slot = _swa_translate_slot(slot=slot_full, lut=lut)
            else:
                slot = slot_full

            prev_position = position - 1
            if prev_position < 0:
                prev_slot = -1
            else:
                prev_slot_full = int(req_to_token_host[rpi, prev_position].item())
                if lut is not None:
                    prev_slot = _swa_translate_slot(slot=prev_slot_full, lut=lut)
                else:
                    prev_slot = prev_slot_full

            out_slots.append(slot)
            out_positions.append(position)
            out_prev_slots.append(prev_slot)

    total_verify = len(out_slots)
    if total_verify == 0:
        return 0

    capped = min(total_verify, verify_capacity)
    slots_t = torch.tensor(out_slots[:capped], dtype=torch.int64, device=work_device)
    positions_t = torch.tensor(out_positions[:capped], dtype=torch.int64, device=work_device)
    prev_slots_t = torch.tensor(out_prev_slots[:capped], dtype=torch.int64, device=work_device)

    verify_plan_out.verify_slot_indices[:capped].copy_(
        slots_t.to(verify_plan_out.verify_slot_indices.dtype).to(verify_plan_out.verify_slot_indices.device)
    )
    verify_plan_out.verify_positions[:capped].copy_(
        positions_t.to(verify_plan_out.verify_positions.dtype).to(verify_plan_out.verify_positions.device)
    )
    verify_plan_out.verify_prev_slot_indices[:capped].copy_(
        prev_slots_t.to(verify_plan_out.verify_prev_slot_indices.dtype).to(
            verify_plan_out.verify_prev_slot_indices.device
        )
    )

    return total_verify


def _materialize_write_metadata(
    *,
    write_plan_out: WritePlan,
    req_pool_indices_host: torch.Tensor,
    prefix_lens_host: torch.Tensor,
    extend_seq_lens_host: torch.Tensor,
    req_to_token_host: torch.Tensor,
    lut: Optional[torch.Tensor],
    write_req_capacity: int,
    work_device: torch.device,
    bs: int,
) -> None:
    out_write_offsets_len = int(write_plan_out.write_offsets.shape[0])
    max_seq_len = int(req_to_token_host.shape[1])

    write_offsets_list: list[int] = []
    seed_slots_list: list[int] = []

    running_offset = 0
    for r in range(bs):
        write_offsets_list.append(running_offset)

        rpi = int(req_pool_indices_host[r].item())
        extend_len = int(extend_seq_lens_host[r].item())

        if rpi == 0 or extend_len <= 0:
            write_len = 0
        else:
            write_len = max(0, extend_len)

        running_offset += write_len

    write_offsets_list.append(running_offset)

    copy_len = min(bs + 1, out_write_offsets_len)
    write_offsets_t = torch.tensor(write_offsets_list[:copy_len], dtype=torch.int64, device=work_device)
    write_plan_out.write_offsets[:copy_len].copy_(
        write_offsets_t.to(write_plan_out.write_offsets.dtype).to(write_plan_out.write_offsets.device)
    )
    if copy_len < out_write_offsets_len:
        write_plan_out.write_offsets[copy_len:].zero_()

    capped_reqs = min(bs, write_req_capacity)
    for r in range(capped_reqs):
        rpi = int(req_pool_indices_host[r].item())
        prefix_len = int(prefix_lens_host[r].item())
        extend_len = int(extend_seq_lens_host[r].item())

        if rpi == 0 or extend_len <= 0:
            seed_slots_list.append(-1)
            continue

        if prefix_len <= 0:
            seed_slots_list.append(-1)
            continue

        safe_seed_pos = min(prefix_len - 1, max(max_seq_len - 1, 0))
        seed_slot_full = int(req_to_token_host[rpi, safe_seed_pos].item())

        if lut is not None:
            seed_slot = _swa_translate_slot(slot=seed_slot_full, lut=lut)
        else:
            seed_slot = seed_slot_full

        seed_slots_list.append(seed_slot)

    if len(seed_slots_list) > 0:
        seed_slots_t = torch.tensor(seed_slots_list, dtype=torch.int64, device=work_device)
        write_plan_out.write_seed_slot_indices[:capped_reqs].copy_(
            seed_slots_t.to(write_plan_out.write_seed_slot_indices.dtype).to(
                write_plan_out.write_seed_slot_indices.device
            )
        )

    write_plan_out.write_num_valid_reqs.fill_(int(bs))
