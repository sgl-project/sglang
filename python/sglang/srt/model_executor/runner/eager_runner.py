# Copyright 2023-2026 SGLang Team
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
"""EagerRunner — the no-cuda-graph phase runner.

The eager dual of the cuda-graph runners. Where ``DecodeCudaGraphRunner`` and
``PrefillCudaGraphRunner`` capture a ``torch.cuda.CUDAGraph`` per shape and
replay it, ``EagerRunner`` runs ``model.forward`` live each iteration over one
static batch. It is used when CUDA graph is disabled for a phase
(``--disable-cuda-graph`` / ``--disable-decode-cuda-graph`` /
``--disable-prefill-cuda-graph`` / ``cuda_graph_config`` resolves a phase to
``disabled``).

When BOTH phases are disabled (the ``--disable-cuda-graph`` case),
``ModelRunner.decode_runner`` and ``ModelRunner.prefill_runner`` point at ONE
``EagerRunner`` instance, mode-dispatched on ``forward_batch.forward_mode``.
When only one phase is disabled, that phase's runner is an ``EagerRunner`` while
the other is a cuda-graph runner.

Design (delegate composition, lazy)
-----------------------------------
``EagerRunner`` is ONE class that subclasses :class:`BaseRunner` and holds BOTH
a decode-side and a prefill-side state set. Rather than re-derive (and risk
diverging from) the hundreds of lines of buffer / registry / attn-backend
construction those two runners already do correctly for the eager case, it
*composes* them: it lazily constructs

  - ``self._decode  = DecodeCudaGraphRunner(model_runner, eager=True)``
  - ``self._prefill = PrefillCudaGraphRunner(model_runner, eager=True)``

as private delegates (via ``ensure_decode`` / ``ensure_prefill``), and
dispatches ``load_batch`` / ``execute`` / ``can_run_graph`` /
``_autotune_buffers`` to the right delegate by ``forward_batch.forward_mode``.
The ``eager=True`` paths in those two classes no longer use any
``ExecutionBackend`` (``EagerBackend`` has been deleted): they reserve the
per-shape static batch (decode) or build it fresh per iteration (prefill) and
run the forward live, with ``self.backend`` left as ``None``.

This keeps behavior bit-for-bit identical to the previous
``DecodeCudaGraphRunner(eager=True)`` / ``PrefillCudaGraphRunner(eager=True)``
path while presenting a single ``EagerRunner`` to the rest of the system and
dropping the ``EagerBackend`` indirection.

Why the delegates are LAZY and built at their own init sites
------------------------------------------------------------
The prefill delegate's ``__init__`` reads ``model_runner.attention_layers`` /
``moe_layers`` / ``moe_fusions`` / ``dsa_indexers``, which ``ModelRunner`` only
populates inside ``init_prefill_cuda_graph`` (its layer-collection gate). The
decode delegate has no such dependency. ``init_decode_cuda_graph`` runs BEFORE
``init_prefill_cuda_graph``, so the prefill delegate cannot be constructed at
decode-init time. We therefore build each delegate at its own ``ModelRunner``
init site (``ensure_decode`` from ``init_decode_cuda_graph``, ``ensure_prefill``
from ``init_prefill_cuda_graph``); ``ensure_*`` is idempotent so the shared
instance can be threaded through both sites.

This means the decode delegate is constructed first, matching today's
construction order (today the eager decode runner is built in
``init_decode_cuda_graph`` and the eager prefill runner in
``init_prefill_cuda_graph``). The v2 "Decision 5" buffer-pool coalescing
(allocate the larger prefill buffer set first so decode coalesces onto it) is
NOT applied here because flipping to prefill-first would violate the
attention_layers ordering constraint above. The result is identical to today's
behavior (both sets allocated separately), just without the new optimization.

Warmup ordering
---------------
``warmup()`` (flashinfer autotune + PP-DeepGEMM) is run-once across both
delegates, gated by ``model_runner._kernel_warmed_up`` and triggered from each
delegate's ``prepare()``. The DECODE delegate carries the static buffers
flashinfer autotune needs; the prefill delegate's ``_autotune_buffers`` returns
``(None, None)`` and would trip ``BaseRunner.warmup``'s assertion. Because the
decode delegate is built first, it triggers the warmup, and the prefill
delegate's later ``warmup()`` is a no-op — preserving today's ordering. As a
safety net (e.g. an asymmetric config where only prefill is eager and the
prefill delegate would otherwise warm up alone), ``ensure_prefill`` suppresses
warmup when no decode delegate has been built yet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class EagerRunner(BaseRunner):
    """No-cuda-graph phase runner; mode-dispatched over decode + prefill state.

    Public surface (the :class:`BaseRunner` ABC):
      - can_run_graph(forward_batch) -> False (always; the dispatch gate that
        keeps callers from routing an eager batch into a graph-replay branch).
      - warmup() — inherited; run-once kernel warmup + flashinfer autotune.
      - load_batch(forward_batch, **kwargs) — dispatch on forward_mode to the
        decode / prefill delegate's eager load_batch.
      - execute(forward_batch, **kwargs) — dispatch on forward_mode to the
        decode / prefill delegate's eager execute (live model.forward).

    Lifecycle helpers (called by ModelRunner at the two init sites):
      - ensure_decode() — build the decode delegate if absent; idempotent.
      - ensure_prefill() — build the prefill delegate if absent; idempotent.

    Decode-side state (buffers, registry, reserved static batches, capture_bs,
    seq_len_fill_value, …) lives on ``self._decode``; prefill-side state
    (buffers, registry, max_num_tokens, attention/moe-layer snapshot, …) lives
    on ``self._prefill``. ``self.max_num_tokens`` is exposed as a property
    backed by the prefill delegate because ``forward_extend`` reads it on the
    runner to gate the eager prefill route.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self._decode: Optional[DecodeCudaGraphRunner] = None
        self._prefill: Optional[PrefillCudaGraphRunner] = None

    # ------------------------------------------------------------------ #
    # Lifecycle: lazy delegate construction
    # ------------------------------------------------------------------ #
    def ensure_decode(self) -> DecodeCudaGraphRunner:
        """Build the eager decode delegate if not already built (idempotent).

        Its __init__ -> prepare() -> warmup() runs the flashinfer autotune over
        the decode static buffers (run-once across both delegates), then reserves
        the per-shape static decode batches.
        """
        if self._decode is None:
            self._decode = DecodeCudaGraphRunner(self.model_runner, eager=True)
        return self._decode

    def ensure_prefill(self) -> PrefillCudaGraphRunner:
        """Build the eager prefill delegate if not already built (idempotent).

        Must be called after ModelRunner.init_prefill_cuda_graph has populated
        model_runner.attention_layers / moe_layers / moe_fusions / dsa_indexers,
        which the delegate snapshots at construction.

        If no decode delegate has been built yet, suppress this delegate's
        warmup: only the decode delegate carries the static buffers flashinfer
        autotune needs, so a prefill-only warmup would trip
        BaseRunner.warmup's assertion. (In the common both-disabled case the
        decode delegate is built first and already triggered warmup, so this
        suppression is a no-op there.)
        """
        if self._prefill is None:
            model_runner = self.model_runner
            if self._decode is None:
                prev_warmed = getattr(model_runner, "_kernel_warmed_up", False)
                model_runner._kernel_warmed_up = True
                try:
                    self._prefill = PrefillCudaGraphRunner(model_runner, eager=True)
                finally:
                    model_runner._kernel_warmed_up = prev_warmed
            else:
                self._prefill = PrefillCudaGraphRunner(model_runner, eager=True)
        return self._prefill

    @property
    def bs(self) -> Optional[int]:
        """Last padded decode batch size (set by the decode delegate's
        load_batch). Read by ModelRunner.forward for the experts-distribution
        capturer's ``cuda_graph_batch`` arg; ``None`` before the first decode
        load_batch. Mirrors the attribute the old eager DecodeCudaGraphRunner
        exposed directly.
        """
        if self._decode is None:
            return None
        return getattr(self._decode, "bs", None)

    @property
    def max_num_tokens(self) -> int:
        """Largest eager prefill token budget (forward_extend's eager gate).

        Reads through to the prefill delegate; callers only consult this on the
        prefill-eager path, where the prefill delegate has been built.
        """
        assert (
            self._prefill is not None
        ), "max_num_tokens read before the eager prefill delegate was built"
        return self._prefill.max_num_tokens

    # ------------------------------------------------------------------ #
    # Dispatch helpers
    # ------------------------------------------------------------------ #
    def _delegate_for(self, forward_batch: ForwardBatch):
        """Pick the decode or prefill delegate by forward mode.

        Mirrors ModelRunner._forward_raw routing: is_decode() -> forward_decode
        (decode delegate); everything else reaching an eager runner is an extend
        flavor -> forward_extend (prefill delegate). TARGET_VERIFY / DLLM_EXTEND
        are is_extend()=True, so they route to the prefill delegate the same way
        the inline eager prefill path handled them.
        """
        if forward_batch.forward_mode.is_decode():
            delegate = self._decode
            assert delegate is not None, "eager decode delegate not built"
            return delegate
        delegate = self._prefill
        assert delegate is not None, "eager prefill delegate not built"
        return delegate

    # ------------------------------------------------------------------ #
    # BaseRunner surface
    # ------------------------------------------------------------------ #
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # An eager runner never runs a graph. This is the graph-vs-eager
        # dispatch gate: returning False keeps callers (forward_extend /
        # _forward_raw) from routing this batch into a graph-replay branch.
        # (The deleted EagerBackend.can_run_graph returned True, but that was the
        # backend's "can I serve this batch" answer, a different question.)
        return False

    def _autotune_buffers(self) -> Tuple[Optional[Any], Optional[int]]:
        # Hand warmup()'s flashinfer-autotune dummy forward the decode delegate's
        # prepared static buffers (same as DecodeCudaGraphRunner._autotune_buffers
        # returns (self.buffers, self.max_bs)). Only reached if EagerRunner.warmup
        # is ever invoked directly; the delegates warm up via their own prepare().
        if self._decode is None:
            return None, None
        return self._decode.buffers, self._decode.max_bs

    def load_batch(self, forward_batch: ForwardBatch, *args, **kwargs) -> Any:
        # *args forwards positional extras the delegates accept (decode's
        # execute/load_batch take pp_proxy_tensors positionally; see the
        # forward_decode call site self.decode_runner.execute(fb, pp_proxy)).
        return self._delegate_for(forward_batch).load_batch(
            forward_batch, *args, **kwargs
        )

    def execute(self, forward_batch: ForwardBatch, *args, **kwargs) -> Any:
        return self._delegate_for(forward_batch).execute(forward_batch, *args, **kwargs)
