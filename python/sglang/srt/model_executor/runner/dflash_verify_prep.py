# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash target-verify prep beside the draft graph (SGLANG_OPT_DFLASH_VERIFY_PREP_BESIDE_DRAFT).

The DFlash draft graph runner marks each decode round right before its graph replay (after its own
input and metadata prep). Everything the target verify's prep reads is final by then (verify positions
and write locations, prefix seq lens, req pool indices, mamba track positions), and the previous verify
replay was issued before the mark on the same stream. The target verify runner and
``prepare_mamba_track_for_verify`` then issue that prep on a side stream that waits only on the mark,
so the GPU runs it beside the draft graph; the verify replay waits on the side stream's last event.
Only target backends whose out-of-graph verify prep reads nothing the draft or the plan tables make
after the mark take this path (``unsupported_reason``).

Contract:

- The slots in ``VERIFY_PREP_SLOTS`` are private to the target verify runner (excluded from the
  process-wide input-buffer pool), so the draft graph neither reads nor writes them; the prep
  writes only those slots and the target backends' metadata, which the draft never touches.
- ``settle`` ends a round's side work: it orders the marked stream after it and clears the mark.
  Every path that reads side-stream results on the main stream settles first (the beside load,
  a fallback load, an eager verify or an eager draft through ``can_run_graph``), and the next mark
  settles a round that never reached its verify.
- Tensors made on the side stream are recorded on the main stream before they are released.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# The verify inputs the side stream copies; input_ids stays on the main stream (the draft makes it).
VERIFY_PREP_SLOTS = frozenset(
    {
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
        "mamba_track_indices",
    }
)


class DFlashVerifyPrep:
    def __init__(
        self,
        *,
        stream: torch.cuda.Stream,
    ) -> None:
        self._stream = stream
        self._main: Optional[torch.cuda.Stream] = None
        self._draft_start = torch.cuda.Event()
        self._done = torch.cuda.Event()
        self._marked = False
        self._issued = False
        self.beside_ct = 0
        self.fallback_ct = 0

    @property
    def marked(self) -> bool:
        return self._marked

    def mark_draft_start(self) -> None:
        """Called on the main stream right before the draft graph replay of a decode round."""
        self.settle()
        self._main = torch.cuda.current_stream()
        self._draft_start.record(self._main)
        self._marked = True

    @contextlib.contextmanager
    def side(self) -> Iterator[torch.cuda.Stream]:
        """Run the body on the side stream, after the mark; yields the main stream, on which
        the caller records the tensors it made here."""
        assert self._marked, "side work needs this round's draft-start mark"
        main = self._main
        if not self._issued:
            self._stream.wait_event(self._draft_start)
            self._issued = True
        with torch.cuda.stream(self._stream):
            yield main

    def settle(self) -> None:
        """Order the marked stream after this round's side work, and clear the mark."""
        if self._issued:
            self._done.record(self._stream)
            self._main.wait_event(self._done)
        self._marked = False
        self._issued = False

    def note(self, *, beside: bool) -> None:
        if beside:
            self.beside_ct += 1
        else:
            self.fallback_ct += 1
        total = self.beside_ct + self.fallback_ct
        first_fallback = not beside and self.fallback_ct == 1
        if total in (1, 100, 1000) or total % 10000 == 0 or first_fallback:
            logger.info(
                "DFlash verify prep beside the draft: %d rounds staged, %d on the main stream.",
                self.beside_ct,
                self.fallback_ct,
            )


# One per scheduler process (one device); the target verify runner installs it, the draft runner
# and prepare_mamba_track_for_verify look it up.
_PREP: Optional[DFlashVerifyPrep] = None


def install_dflash_verify_prep(
    *,
    enabled: bool,
    avoid_streams: Tuple[torch.cuda.Stream, ...],
) -> Optional[DFlashVerifyPrep]:
    """Called by every DFlash target verify runner: a runner that does not qualify uninstalls the
    prep of an earlier one, so marks and settles always follow the live runner."""
    from sglang.srt.managers.utils import allocate_distinct_stream

    global _PREP
    if not enabled:
        _PREP = None
        return None
    _PREP = DFlashVerifyPrep(
        stream=allocate_distinct_stream(torch.cuda, avoid_streams),
    )
    logger.info(
        "DFlash target verify prep runs on a side stream beside the draft graph."
    )
    return _PREP


def dflash_verify_prep() -> Optional[DFlashVerifyPrep]:
    return _PREP


def settle_dflash_verify_prep() -> None:
    if _PREP is not None:
        _PREP.settle()


def unsupported_reason(attn_backend, *, enable_unified_memory: bool) -> str:
    """Empty when the target backend's out-of-graph verify prep reads only state that is final
    before the draft replay (DSA full attention + mamba/KDA linear attention, no plan tables)."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
        MambaAttnBackendBase,
    )

    if enable_unified_memory:
        return (
            "unified memory: the draft builds the shared plan read table after the mark"
        )
    if not (
        isinstance(attn_backend, HybridLinearAttnBackend)
        and isinstance(attn_backend.full_attn_backend, DeepseekSparseAttnBackend)
        and isinstance(attn_backend.linear_attn_backend, MambaAttnBackendBase)
    ):
        return f"target backend {type(attn_backend).__name__} is not DSA + mamba/KDA"
    return ""


def marked_dflash_verify_prep() -> Optional[DFlashVerifyPrep]:
    """The prep when this round's draft replay marked it, else None."""
    return _PREP if _PREP is not None and _PREP.marked else None


def stage_beside_draft(
    *,
    prep: DFlashVerifyPrep,
    registry,
    sources: Dict[str, torch.Tensor],
    run_prep: Callable[[], None],
    check: bool,
) -> None:
    """Copy ``sources`` into their slots and run ``run_prep`` on the side stream; the current
    stream then waits for both."""
    with prep.side():
        registry.copy_slots(sources)
        run_prep()
        side = torch.cuda.current_stream()
        for src in sources.values():
            src.record_stream(side)
    prep.settle()
    prep.note(beside=True)
    if check:
        _check_staged_slots(registry=registry, sources=sources)


def _check_staged_slots(*, registry, sources: Dict[str, torch.Tensor]) -> None:
    for name, src in sources.items():
        buffer = registry.get_slot(name).buffer
        if _shares_pool_storage(buffer):
            raise RuntimeError(
                f"staged slot {name} aliases the process-wide input-buffer pool"
            )
        dst = buffer[: src.shape[0]]
        if not torch.equal(dst, src.to(dst.dtype)):
            raise RuntimeError(
                f"staged slot {name} differs from its source after the wait"
            )


def _shares_pool_storage(buffer: torch.Tensor) -> bool:
    from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool

    ptr = buffer.untyped_storage().data_ptr()
    return any(
        t.untyped_storage().data_ptr() == ptr
        for t in _forward_input_buffer_pool.values()
    )
