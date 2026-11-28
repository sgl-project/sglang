from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.consts import REQ_POOL_IDX_PADDING
from sglang.kernels.ops.kv_canary.verify import VerifyPlan
from sglang.kernels.ops.kv_canary.write import WritePlan


def launch_canary_plan_kernels_torch_reference(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_capacity: int,
    req_to_verify_expected_tokens: Optional[torch.Tensor],
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor],
    kv_token_id_vs_position_offset: int,
) -> None:
    """Python reference for :func:`launch_canary_plan_kernels`. Same signature & byte-equal semantics."""
    bs = int(req_pool_indices.shape[0])
    work_device = torch.device("cpu")

    plan_verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if verify_capacity != plan_verify_capacity:
        raise ValueError(
            f"kv-canary: launch_canary_plan_kernels_torch_reference verify_capacity={verify_capacity} does not "
            f"match verify_plan_out.verify_slot_indices.shape[0]={plan_verify_capacity}"
        )
    write_req_capacity = int(write_plan_out.write_seed_slot_indices.shape[0])

    req_pool_indices_host = req_pool_indices.detach().to(
        device=work_device, dtype=torch.int64
    )
    prefix_lens_host = prefix_lens.detach().to(device=work_device, dtype=torch.int64)
    extend_seq_lens_host = extend_seq_lens.detach().to(
        device=work_device, dtype=torch.int64
    )
    req_to_token_host = req_to_token.detach().to(device=work_device, dtype=torch.int64)

    lut: Optional[torch.Tensor] = None
    if full_to_swa_index_mapping is not None:
        lut = full_to_swa_index_mapping.detach().to(device=work_device)

    expected_token_pool_host: Optional[torch.Tensor] = None
    req_to_verify_expected_tokens_valid_lens_host: Optional[torch.Tensor] = None
    if req_to_verify_expected_tokens is not None:
        expected_token_pool_host = req_to_verify_expected_tokens.detach().to(
            device=work_device, dtype=torch.int64
        )
        if req_to_verify_expected_tokens_valid_lens is None:
            raise ValueError(
                "kv-canary: launch_canary_plan_kernels_torch_reference requires "
                "req_to_verify_expected_tokens_valid_lens when req_to_verify_expected_tokens is set"
            )
        req_to_verify_expected_tokens_valid_lens_host = (
            req_to_verify_expected_tokens_valid_lens.detach().to(
                device=work_device, dtype=torch.int64
            )
        )

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
        expected_token_pool_host=expected_token_pool_host,
        req_to_verify_expected_tokens_valid_lens_host=req_to_verify_expected_tokens_valid_lens_host,
        kv_token_id_vs_position_offset=int(kv_token_id_vs_position_offset),
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

    _write_num_valid_and_enable(
        verify_plan_out=verify_plan_out,
        requested=total_verify,
        verify_capacity=verify_capacity,
    )


def _write_num_valid_and_enable(
    *,
    verify_plan_out: VerifyPlan,
    requested: int,
    verify_capacity: int,
) -> None:
    overflow = requested > verify_capacity
    clamped = verify_capacity if overflow else requested
    enable = 0 if overflow else 1
    verify_plan_out.verify_num_valid.fill_(int(clamped))
    verify_plan_out.enable.fill_(int(enable))


def _swa_translate_slot(*, slot: int, lut: torch.Tensor) -> int:
    if slot < 0:
        return slot
    lut_len = int(lut.shape[0])
    if slot >= lut_len:
        raise ValueError(
            f"kv-canary: SWA slot {slot} is outside full_to_swa_index_mapping length {lut_len}"
        )
    return int(lut[slot].item())


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
    expected_token_pool_host: Optional[torch.Tensor],
    req_to_verify_expected_tokens_valid_lens_host: Optional[torch.Tensor],
    kv_token_id_vs_position_offset: int,
) -> int:
    out_slots: list[int] = []
    out_positions: list[int] = []
    out_expected_input_ids: list[int] = []
    out_prev_slots: list[int] = []

    for r in range(bs):
        rpi = int(req_pool_indices_host[r].item())
        prefix_len = int(prefix_lens_host[r].item())

        if rpi == REQ_POOL_IDX_PADDING:
            continue

        if swa_window_size > 0:
            window_start = max(0, prefix_len - swa_window_size)
        else:
            window_start = 0
        verify_len = max(0, prefix_len - window_start)

        valid_len_r = (
            int(req_to_verify_expected_tokens_valid_lens_host[r].item())
            if req_to_verify_expected_tokens_valid_lens_host is not None
            else 0
        )

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

            expected_input_id = -1
            if expected_token_pool_host is not None:
                sot_pos = position + kv_token_id_vs_position_offset
                if 0 <= sot_pos < valid_len_r:
                    expected_input_id = int(
                        expected_token_pool_host[rpi, sot_pos].item()
                    )

            out_slots.append(slot)
            out_positions.append(position)
            out_expected_input_ids.append(expected_input_id)
            out_prev_slots.append(prev_slot)

    total_verify = len(out_slots)
    if total_verify == 0:
        return 0

    # On overflow CUDA plan_entries skips scatter (verify_enable=0); mirror that.
    if total_verify > verify_capacity:
        return total_verify

    slots_t = torch.tensor(out_slots, dtype=torch.int64, device=work_device)
    positions_t = torch.tensor(out_positions, dtype=torch.int64, device=work_device)
    expected_input_ids_t = torch.tensor(
        out_expected_input_ids, dtype=torch.int64, device=work_device
    )
    prev_slots_t = torch.tensor(out_prev_slots, dtype=torch.int64, device=work_device)

    verify_plan_out.verify_slot_indices[:total_verify].copy_(
        slots_t.to(verify_plan_out.verify_slot_indices.dtype).to(
            verify_plan_out.verify_slot_indices.device
        )
    )
    verify_plan_out.verify_expected_tokens[:total_verify].copy_(
        expected_input_ids_t.to(verify_plan_out.verify_expected_tokens.dtype).to(
            verify_plan_out.verify_expected_tokens.device
        )
    )
    verify_plan_out.verify_expected_positions[:total_verify].copy_(
        positions_t.to(verify_plan_out.verify_expected_positions.dtype).to(
            verify_plan_out.verify_expected_positions.device
        )
    )
    verify_plan_out.verify_prev_slot_indices[:total_verify].copy_(
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

        if rpi == REQ_POOL_IDX_PADDING or extend_len <= 0:
            write_len = 0
        else:
            write_len = max(0, extend_len)

        running_offset += write_len

    write_offsets_list.append(running_offset)

    copy_len = min(bs + 1, out_write_offsets_len)
    write_offsets_t = torch.tensor(
        write_offsets_list[:copy_len], dtype=torch.int64, device=work_device
    )
    write_plan_out.write_offsets[:copy_len].copy_(
        write_offsets_t.to(write_plan_out.write_offsets.dtype).to(
            write_plan_out.write_offsets.device
        )
    )
    if copy_len < out_write_offsets_len:
        write_plan_out.write_offsets[copy_len:].zero_()

    capped_reqs = min(bs, write_req_capacity)
    for r in range(capped_reqs):
        rpi = int(req_pool_indices_host[r].item())
        prefix_len = int(prefix_lens_host[r].item())
        extend_len = int(extend_seq_lens_host[r].item())

        if rpi == REQ_POOL_IDX_PADDING or extend_len <= 0:
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
        seed_slots_t = torch.tensor(
            seed_slots_list, dtype=torch.int64, device=work_device
        )
        write_plan_out.write_seed_slot_indices[:capped_reqs].copy_(
            seed_slots_t.to(write_plan_out.write_seed_slot_indices.dtype).to(
                write_plan_out.write_seed_slot_indices.device
            )
        )

    write_plan_out.write_num_valid_reqs.fill_(int(bs))
