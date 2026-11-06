"""Experimental request-owned SWA pages for the NPU PD decode path."""

import torch


def resolve_npu_swa_ring_size(window: int, page_size: int):
    from sglang.srt.environ import envs

    if not envs.SGLANG_DSV4_NPU_DECODE_SWA_RING.get():
        return None

    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_memory,
        get_parallel,
        get_schedule,
        get_spec,
        max_speculative_num_draft_tokens,
    )

    disagg, memory, parallel = get_disagg(), get_memory(), get_parallel()
    if (
        is_npu_arch35()
        or disagg.disaggregation_mode != "decode"
        or not memory.disable_radix_cache
        or disagg.disaggregation_decode_enable_radix_cache
        or disagg.disaggregation_decode_enable_offload_kvcache
        or disagg.disaggregation_decode_retraction_backup != "cpu_tensor"
        or memory.enable_hierarchical_cache
        or memory.enable_hisparse
        or memory.enable_unified_memory
        or get_exec().features.enable_encoder_swa_bounded_replay
        or parallel.pp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dcp_size != 1
        or get_schedule().max_running_requests is None
        or window != 128
        or page_size != 128
        or get_spec().speculative_algorithm not in (None, "DSPARK")
    ):
        raise ValueError(
            "NPU decode SWA rings require pre-A5 PD decode, no radix/offload/"
            "HiCache/HiSparse/unified memory/encoder replay, PP=CP=DCP=1, "
            "explicit cpu_tensor retraction backup and max-running-requests, "
            "window=page=128, and plain decode or DSpark."
        )
    draft = max_speculative_num_draft_tokens() or 1
    if draft > 6:
        raise ValueError(
            "NPU decode SWA rings currently support at most 6 draft tokens"
        )
    # Leave room for speculative writes and overlap before retiring old rows.
    return (window + 2 * draft + page_size - 1) // page_size * page_size


def npu_swa_ring_locations(req_indices, prefix_lens, seq_lens, total, ring_size):
    """Locations for flattened request-major appended rows; no device sync."""
    counts = (seq_lens - prefix_lens).to(torch.int64)
    starts = counts.cumsum(0) - counts
    requests = torch.repeat_interleave(
        req_indices.to(torch.int64), counts, output_size=total
    )
    offsets = torch.arange(total, device=seq_lens.device) - torch.repeat_interleave(
        starts, counts, output_size=total
    )
    positions = offsets + torch.repeat_interleave(
        prefix_lens.to(torch.int64), counts, output_size=total
    )
    return requests * ring_size + positions % ring_size
