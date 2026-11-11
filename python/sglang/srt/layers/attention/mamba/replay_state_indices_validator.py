"""Debug validation for Mamba CUDA-graph replay state indices.

The validator is intentionally CPU-only.  Replay metadata is prepared outside
the captured graph, so a diagnostic run may synchronize once here without
putting host reads or dynamic assertions into CUDA graph capture/replay.
"""

from __future__ import annotations

import torch


def validate_replay_state_indices_cpu(
    state_indices: torch.Tensor,
    *,
    valid_bs: int,
    total_bs: int,
    num_state_slots: int,
    pad_slot_id: int = -1,
) -> None:
    """Validate live and padded rows of a replay state-index buffer.

    Live rows must own distinct in-range slots in ``[0, num_state_slots)``.
    Slot zero is reserved for CUDA-graph dummy/idle traffic but is still a
    valid storage row.  All padded rows must carry exactly ``pad_slot_id`` so
    indexed state kernels skip them.
    """
    if state_indices.device.type != "cpu":
        raise ValueError("state_indices must be copied to CPU before validation")
    if state_indices.ndim != 1:
        raise ValueError("state_indices must be rank-1")
    if not 0 <= valid_bs <= total_bs <= state_indices.numel():
        raise ValueError(
            "expected 0 <= valid_bs <= total_bs <= state_indices.numel(), got "
            f"valid_bs={valid_bs} total_bs={total_bs} "
            f"numel={state_indices.numel()}"
        )
    if num_state_slots <= 1:
        raise ValueError(
            f"num_state_slots must include real slots, got {num_state_slots}"
        )

    indices = state_indices[:total_bs].to(dtype=torch.int64)
    live = indices[:valid_bs]
    padded = indices[valid_bs:]
    errors: list[str] = []

    live_in_range = (live >= 0) & (live < num_state_slots)
    if not bool(torch.all(live_in_range)):
        bad_rows = torch.nonzero(~live_in_range, as_tuple=False).flatten()
        errors.append(
            "live rows must contain in-range slots in "
            f"[0, {num_state_slots}); bad_rows={bad_rows.tolist()} "
            f"bad_values={live[bad_rows].tolist()}"
        )

    if live.numel() > 1:
        unique_live, counts = torch.unique(live, sorted=True, return_counts=True)
        duplicate_mask = counts > 1
        if bool(torch.any(duplicate_mask)):
            errors.append(
                "live rows must own unique slots; "
                f"duplicate_slots={unique_live[duplicate_mask].tolist()} "
                f"counts={counts[duplicate_mask].tolist()}"
            )

    bad_padding = padded != pad_slot_id
    if bool(torch.any(bad_padding)):
        bad_offsets = torch.nonzero(bad_padding, as_tuple=False).flatten()
        errors.append(
            f"padded rows must equal pad_slot_id={pad_slot_id}; "
            f"bad_rows={(bad_offsets + valid_bs).tolist()} "
            f"bad_values={padded[bad_offsets].tolist()}"
        )

    if errors:
        raise AssertionError(
            "Invalid Mamba replay state indices: "
            + "; ".join(errors)
            + f"; valid_bs={valid_bs} total_bs={total_bs} "
            f"indices={indices.tolist()}"
        )
