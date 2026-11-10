"""Rebuild the trailing SWA window of a cached DeepSeek-V4 prefix by replaying
its last ``swa_recompute_len`` tokens. The replay reads compressed and indexer
entries from the cache and writes only SWA rows and the 4x compressor state."""

from typing import List, Optional

import msgspec

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.encoder_swa_replay import _build_replay_batch


class SWARecomputeOutput(msgspec.Struct):
    # Extend-shaped view of the replayed spans: prefix_lens, extend_lens, out_cache_loc.
    batch: ScheduleBatch
    logits_output: LogitsProcessorOutput


def swa_recompute_batch(batch: ScheduleBatch, runner) -> Optional[ScheduleBatch]:
    """One batch over every request that owes a replay, in batch order."""
    if batch.swa_recompute_starts is None:
        return None
    if batch.swa_recompute_replay is None:
        rows = [
            (i, start, batch.prefix_lens[i])
            for i, start in enumerate(batch.swa_recompute_starts)
            if start is not None
        ]
        # Each request's window is floored at its own start, so one forward
        # matches per-request replays.
        replay = _build_replay_batch(batch=batch, runner=runner, rows=rows)
        replay.swa_recompute_starts = None
        batch.swa_recompute_replay = replay
    return batch.swa_recompute_replay


def run_swa_recompute(
    worker, batch: ScheduleBatch, capture_hidden_mode
) -> List[SWARecomputeOutput]:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode

    runner = worker.model_runner
    replay = swa_recompute_batch(batch, runner)
    if replay is None:
        return []
    forward_batch = ForwardBatch.init_new(
        replay,
        runner,
        capture_hidden_mode=capture_hidden_mode or CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    forward_batch.swa_recompute = True
    out = runner.forward(forward_batch)
    # The scheduled extend publishes its own read-done event after this.
    runner.shared_read_done_event = None
    return [SWARecomputeOutput(batch=replay, logits_output=out.logits_output)]
