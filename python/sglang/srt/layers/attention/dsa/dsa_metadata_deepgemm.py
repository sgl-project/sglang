"""Refresh DeepGEMM schedules before an in-graph DSA verify replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.layers.attention.dsa.kpool_plan import _compute_pool_schedule_metadata

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata


def refresh_pool_schedule(metadata, pool_seqlens_per_q, *, slots_per_page):
    plan = metadata.kpool_write_plan
    if plan is None or plan.pool_schedule_metadata is None:
        return
    schedule = _compute_pool_schedule_metadata(
        pool_seqlens_per_q, slots_per_page=slots_per_page
    )
    if schedule is not None:
        plan.pool_schedule_metadata.copy_(schedule)


def build_verify_deepgemm_residual(
    backend,
    metadata: DSAMetadata,
    next_n: int,
    seq_lens: torch.Tensor,
    num_sms: int,
    ctx_lens_written: bool,
) -> Callable[[], None]:
    """Rebuild residual schedules from raw seq_lens before graph replay.
    Captured metadata still contains the previous replay's values then."""
    import deep_gemm

    schedule_dst = metadata.paged_mqa_schedule_metadata
    dg_get = deep_gemm.get_paged_mqa_logits_metadata
    pool_size = backend.dsa_index_kpool
    slots_per_page = backend._kpool_slots_per_page()
    kpool_plan = metadata.kpool_write_plan if pool_size > 1 else None
    has_pool_schedule = (
        kpool_plan is not None and kpool_plan.pool_schedule_metadata is not None
    )
    # Built lazily INSIDE the residual: this builder runs during stream
    # capture, where a torch.arange would be RECORDED (contents undefined
    # until the graph first replays) instead of executed; the residual
    # itself always runs out-of-graph, so its first call materializes the
    # constant eagerly.
    offsets_box: List[Optional[torch.Tensor]] = [None]

    def residual() -> None:
        offsets = offsets_box[0]
        if offsets is None:
            offsets = torch.arange(
                1, next_n + 1, dtype=torch.int32, device=seq_lens.device
            )
            offsets_box[0] = offsets
        seq_lens_i32 = seq_lens.to(torch.int32)
        if ctx_lens_written:
            src = (seq_lens_i32 + next_n).view(-1, 1).expand(-1, next_n).contiguous()
        else:
            src = (seq_lens_i32.view(-1, 1) + offsets).reshape(-1, 1).contiguous()
        schedule_dst.copy_(dg_get(src, 64, num_sms))
        if has_pool_schedule:
            pool_seqlens_per_q = torch.div(
                seq_lens_i32.view(-1, 1) + offsets,
                pool_size,
                rounding_mode="floor",
            ).reshape(-1)
            refresh_pool_schedule(
                metadata,
                pool_seqlens_per_q,
                slots_per_page=slots_per_page,
            )

    return residual
