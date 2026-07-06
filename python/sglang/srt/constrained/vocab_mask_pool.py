# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Persistent, double-buffered vocab-mask pool for bitmask grammar backends.

Motivation
----------
The per-step structured-output hot path used to (a) allocate a fresh
``[bs, ceil(vocab/32)]`` int32 token bitmask on the host every decode step and
(b) issue a fresh host->device copy of it every step. Both show up on the
critical path between forward-end and sampling (see the py-spy profile:
``allocate_vocab_mask`` + ``move_vocab_mask``).

This pool removes the per-step allocation and reuses persistent pinned host and
device buffers sized to ``(max_bs, mask_width)``. Each step we reset the
needed rows to "all allowed", refill them, and issue a single ``copy_``.

Double buffering (why)
----------------------
The mask is consumed by the *delayed* sampler in overlap mode. The
host->device copy is issued with ``non_blocking=True`` (async), so after
``update_regex_vocab_mask`` returns to Python, the pinned host buffer may still
be in flight to the device. If the *next* step reused the same pinned buffer
and started refilling it (a pure CPU write) before that async copy drained, it
would corrupt the in-flight transfer for the previous step.

Device-side ordering is safe on its own: fill(copy) and apply
(``apply_token_bitmask_inplace``) for a given batch are enqueued back-to-back on
the same forward stream, and each step's ``launch_batch_sample_if_needed`` is a
distinct loop iteration, so per-batch device ops stay ordered. The only race is
the *host* pinned buffer being rewritten by the CPU underneath an outstanding
copy. We therefore alternate between two host+device slots per step: while the
copy out of slot ``s`` is draining, the next step fills slot ``s ^ 1``. Two
slots suffice because a slot is only reused two steps later, by which point its
copy has certainly completed (the intervening step both enqueued and, on the
overlap path, the sampler already consumed slot ``s ^ 1``).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from xgrammar import bitmask_dtype, get_bitmask_shape, reset_token_bitmask

# (vocab_size, device_str) -> VocabMaskPool
_POOLS: Dict[Tuple[int, str], "VocabMaskPool"] = {}


class VocabMaskPool:
    """Double-buffered persistent bitmask buffers for one (vocab, max_bs, device)."""

    NUM_SLOTS = 2

    def __init__(self, vocab_size: int, max_bs: int, device: str):
        self.vocab_size = vocab_size
        self.device = device
        self._pin = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.mask_width = get_bitmask_shape(1, vocab_size)[1]
        self.max_bs = 0
        self._host = []
        self._dev = []
        self._slot = 0
        self._ensure_capacity(max_bs)

    def _ensure_capacity(self, bs: int):
        """(Re)allocate the persistent buffers so they hold at least ``bs`` rows.

        The pool is normally sized once to cuda-graph-max-bs; this grow path is
        a safety net if a larger batch is ever observed.
        """
        if bs <= self.max_bs:
            return
        self.max_bs = bs
        shape = get_bitmask_shape(bs, self.vocab_size)  # (bs, mask_width)
        self._host = [
            torch.full(shape, -1, dtype=bitmask_dtype, pin_memory=self._pin)
            for _ in range(self.NUM_SLOTS)
        ]
        self._dev = [
            torch.empty(shape, dtype=bitmask_dtype, device=self.device)
            for _ in range(self.NUM_SLOTS)
        ]
        self._slot = 0

    def host_view(self, bs: int) -> torch.Tensor:
        """Reset the first ``bs`` rows of the current slot to all-allowed and
        return that host view for the caller to fill in place."""
        self._ensure_capacity(bs)
        host = self._host[self._slot]
        view = host[:bs]
        # Reset to "all tokens allowed" (-1 / all bits set), matching a fresh
        # allocate_token_bitmask. Rows for finished/terminated grammars are not
        # refilled, so they must be all-allowed here.
        reset_token_bitmask(view)
        return view

    def commit_to_device(self, bs: int) -> torch.Tensor:
        """Issue the single async host->device copy for the current slot and
        return the device view. Advances to the next slot for the next step."""
        host_view = self._host[self._slot][:bs]
        dev_view = self._dev[self._slot][:bs]
        dev_view.copy_(host_view, non_blocking=True)
        # Advance slot: next step fills the other buffer so this slot's async
        # copy can drain without the CPU rewriting it underneath.
        self._slot ^= 1
        return dev_view


def get_vocab_mask_pool(vocab_size: int, max_bs: int, device: str) -> VocabMaskPool:
    key = (vocab_size, str(device))
    pool = _POOLS.get(key)
    if pool is None:
        pool = VocabMaskPool(vocab_size, max_bs, device)
        _POOLS[key] = pool
    else:
        pool._ensure_capacity(max_bs)
    return pool
