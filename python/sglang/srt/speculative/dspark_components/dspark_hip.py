"""ROCm glue of DSparkWorkerV2 on gfx950: a decode step queues its draft before its host seq
lens land (overlap_utils_hip). The scheduler's glue calls it."""

from __future__ import annotations

from sglang.srt.runtime_context import get_parallel


def defers_seq_lens_cpu(worker) -> bool:
    """Whether a decode step can queue its draft before the host seq lens land: the
    draft reads them only on the GPU, and the host bound for the verify window is the
    reserved length."""
    return (
        worker._hosts_draft
        and not worker._draft_is_moe
        and not get_parallel().attn_dp_enabled
        and not worker.draft_model_runner.attn_backend.needs_cpu_seq_lens
    )
