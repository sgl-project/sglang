"""The per-step decision, for every kernel layer at once.

Invariants: (1) pure decode or target-verify only; (2) rank-uniform: a function of what every rank sees alike (forward
mode, row count, batch size), never rank-local state; (3) graph-safe: under CUDA graphs the decision is baked in at
capture, so it must be a pure function of the captured batch -- nothing here may vary at replay for a captured size.
"""

from __future__ import annotations

from typing import Mapping, Optional

import msgspec
import torch

from sglang.srt.layers.mono.spec import Feature, MonoContext, MonoSpec, MonoStep


class Decision(msgspec.Struct, frozen=True):
    step: Optional[MonoStep]  # None: the model's own path
    reason: str  # why not, for metrics ("" when routed)


def decide(
    spec: MonoSpec,
    ctx: MonoContext,
    *,
    is_decode: bool,
    is_verify: bool,
    rows: int,
    batch_size: int,
) -> Decision:
    if not (is_decode or is_verify):
        return Decision(None, "not decode")
    if is_verify and Feature.SPEC_VERIFY not in spec.supports:
        return Decision(None, "verify unsupported")
    if rows not in spec.widths:
        return Decision(None, "width")
    if batch_size < 1 or rows % batch_size:
        return Decision(None, "rows not a multiple of the batch")
    step = MonoStep(
        rows=rows,
        batch_size=batch_size,
        is_verify=is_verify,
        query_len=rows // batch_size,
    )
    why = spec.eligible(ctx, step)
    if why:
        return Decision(None, why)
    return Decision(step, "")


def decide_for_batch(spec: MonoSpec, ctx: MonoContext, forward_batch) -> Decision:
    mode = forward_batch.forward_mode
    rows = (
        int(forward_batch.input_ids.shape[0])
        if forward_batch.input_ids is not None
        else 0
    )
    return decide(
        spec,
        ctx,
        is_decode=mode.is_decode(),
        is_verify=mode.is_target_verify(),
        rows=rows,
        batch_size=int(forward_batch.batch_size),
    )


class CacheBinding:
    """Re-binds the spec's caches when any tensor's storage moved, keyed on ``(data_ptr, numel)``: graph memory
    profiling runs real steps on throwaway caches before the real ones exist."""

    def __init__(self) -> None:
        self._key: Optional[tuple] = None

    def changed(self, caches: Mapping[str, torch.Tensor]) -> bool:
        key = tuple((n, t.data_ptr(), t.numel()) for n, t in sorted(caches.items()))
        if key == self._key:
            return False
        self._key = key
        return True
