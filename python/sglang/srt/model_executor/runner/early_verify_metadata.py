"""Target-verify attention metadata staged during the draft forward
(SGLANG_OPT_DSV41_TARGET_METADATA_EARLY).

The verify inputs the metadata reads (positions, write locations, seq lens and
req pool indices) are final once the decode head has run. ``stage``, which the host
issues right after it launches the draft graph, copies them into the verify graph's
buffers and replays the runner's captured metadata glue graph on a side stream that
waits only for the head, so the GPU runs it beside the draft; the verify replay
orders itself after that stream (``take``). Issued after the draft launch, the stage
replaces the fill and glue replay the host would otherwise run before the verify,
instead of adding host time before the draft.

Contract:

- The side stream first waits for the head's event, recorded on the current stream
  after the previous verify replay, which has therefore finished reading the buffers
  it rewrites.
- The staged slots are private to the verify runner (never shared through the
  process-wide input-buffer pool), so the draft graph neither reads nor writes
  them; the glue graph writes only the verify backend's metadata.
- ``take`` accepts the staged work only for the same glue key and the very same
  source views; otherwise the current stream still waits for the side stream and
  the runner refills and re-runs the glue graph itself.
- Every staged record ends in a wait (``take`` or ``discard``) on the current
  stream before anyone else refills the buffers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import msgspec
import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    same_tensor_view,
)
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph

logger = logging.getLogger(__name__)


class VerifyInputSources(msgspec.Struct, frozen=True):
    # input_ids is not staged: the draft graph may write it in place (mid fold).
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


STAGED_SLOTS = frozenset(VerifyInputSources.__struct_fields__)


class _StagedVerifyInputs(msgspec.Struct, frozen=True):
    key: Any
    sources: VerifyInputSources


class EarlyVerifyMetadata:
    def __init__(self, *, device, log_every: int) -> None:
        self._stream = torch.cuda.Stream(device=device)
        self._done = torch.cuda.Event()
        self._staged: Optional[_StagedVerifyInputs] = None
        self._log_every = log_every
        self.staged_ct = 0
        self.taken_ct = 0
        self.rejected_ct = 0
        # Per attention variant: [replays that took the staged metadata, replays].
        self.variant_cts: Dict[Any, list] = {}

    def stage(
        self,
        *,
        glue: MetadataGlueGraph,
        key,
        registry: CudaGraphBufferRegistry,
        sources: VerifyInputSources,
        after: torch.cuda.Event,
    ) -> None:
        self.discard()
        self._stream.wait_event(after)
        slot_sources = msgspec.structs.asdict(sources)
        with torch.cuda.stream(self._stream):
            registry.copy_slots(slot_sources)
            glue.replay(key)
            self._done.record()
        for src in slot_sources.values():
            src.record_stream(self._stream)
        self._staged = _StagedVerifyInputs(key=key, sources=sources)
        self.staged_ct += 1
        if self.staged_ct == 1 or self.staged_ct % self._log_every == 0:
            logger.debug(
                "Target verify metadata staged on the side stream: %d staged, "
                "%d taken, %d rejected.",
                self.staged_ct,
                self.taken_ct,
                self.rejected_ct,
            )

    def take(self, *, key, forward_batch) -> bool:
        """True when this replay's inputs are the staged ones: the fill skips
        ``STAGED_SLOTS`` and the glue graph already ran for ``key``."""
        staged = self._staged
        if staged is None:
            return False
        self.discard()
        src = staged.sources
        matches = staged.key == key and all(
            cur is not None and same_tensor_view(cur, staged_src)
            for cur, staged_src in (
                (forward_batch.positions, src.positions),
                (forward_batch.out_cache_loc, src.out_cache_loc),
                (forward_batch.req_pool_indices, src.req_pool_indices),
                (forward_batch.seq_lens, src.seq_lens),
            )
        )
        if matches:
            self.taken_ct += 1
        else:
            self.rejected_ct += 1
            if self.rejected_ct == 1:
                logger.info(
                    "Staged target verify metadata did not match the replay "
                    "(key %s vs %s); refilling on the main stream.",
                    staged.key,
                    key,
                )
        return matches

    def count_variant(self, variant, *, taken: bool) -> None:
        cts = self.variant_cts.setdefault(variant, [0, 0])
        cts[0] += taken
        cts[1] += 1
        if cts[1] % self._log_every == 0:
            logger.debug(
                "Target verify replays of attention variant %s with staged metadata: "
                "%d of %d.",
                variant,
                cts[0],
                cts[1],
            )

    def discard(self) -> None:
        """Order the current stream after any staged side-stream work."""
        if self._staged is not None:
            torch.cuda.current_stream().wait_event(self._done)
            self._staged = None
