"""The draft context-KV commit of a static verify, deferred to the next draft forward
(SGLANG_DSPARK_DEFER_CONTEXT_KV).

The static verify graph stages the arguments of its draft-KV injection (target hidden,
write locations, positions, gated commit lengths, ring slots) instead of running it.
The next captured draft graph runs the same injection on a side stream forked at its
head (a capture head hook, before the in-graph metadata prep); the draft's first KV
store joins it (``join``, called from the draft attention backend's
``get_swa_out_cache_loc``, which both SGLang's draft attention and a megakernel's
reach before they store or read draft KV). Any other forward writes it eagerly first.

Contract:

- Every write disarms the staging (commit lengths 0), so a staging is written once.
- A forward that may write or read draft KV (prefill, draft extend, an eager draft)
  runs after the pending write; a draft graph whose rows miss staged ones is preceded
  by an eager write of every staged row.
- Nothing reads draft KV between the verify and the next draft forward.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class DeferredContextCommit:
    def __init__(
        self,
        *,
        max_bs: int,
        stride: int,
        hidden_width: int,
        hidden_dtype: torch.dtype,
        device,
        kv_injector,
        with_state_slot: bool,
    ) -> None:
        self.stride = stride
        rows = max_bs * stride
        self.hidden = torch.zeros(
            (rows, hidden_width), dtype=hidden_dtype, device=device
        )
        self.cache_loc = torch.zeros((rows,), dtype=torch.int64, device=device)
        self.positions = torch.zeros((rows,), dtype=torch.int64, device=device)
        # 0 for every request past the staged ones: their rows are not written.
        self.commit_lens = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        self.state_slot = (
            torch.zeros((rows,), dtype=torch.int64, device=device)
            if with_state_slot
            else None
        )
        self._kv_injector = kv_injector
        self._device = device
        # The fork's side stream and event, made at the first capture.
        self._stream: Optional[torch.cuda.Stream] = None
        self._done: Optional[torch.cuda.Event] = None
        # Host: requests a static verify staged since the last write (0: nothing pending).
        self.staged_bs = 0
        # Host, capture only: a fork the draft's first KV store has not joined yet.
        self._fork_unjoined = False
        # Draft graph batch sizes whose capture recorded the forked write.
        self._forked_bs: set[int] = set()
        self.num_written_in_graph = 0
        self.num_written_eagerly = 0

    @property
    def pending(self) -> bool:
        return self.staged_bs > 0

    def stage(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        state_slot: Optional[torch.Tensor],
    ) -> None:
        """Recorded in a static verify graph of ``commit_lens.shape[0]`` requests."""
        bs = commit_lens.shape[0]
        n = bs * self.stride
        self.hidden[:n].copy_(target_hidden[:n])
        self.cache_loc[:n].copy_(cache_loc[:n])
        self.positions[:n].copy_(positions[:n])
        self.commit_lens[:bs].copy_(commit_lens)
        self.commit_lens[bs:].zero_()
        if self.state_slot is not None:
            self.state_slot[:n].copy_(state_slot[:n])

    def note_static_verify(self, live_bs: int) -> None:
        """Host, after a static verify graph ran ``live_bs`` requests."""
        self.staged_bs = live_bs

    def write(self, bs: Optional[int] = None) -> None:
        """The staged injection over the first ``bs`` requests (default all), then a
        disarm, on the current stream."""
        bs = self.commit_lens.shape[0] if bs is None else bs
        n = bs * self.stride
        cache_loc = self.cache_loc[:n]
        self._kv_injector.inject_target_hidden(
            target_hidden=self.hidden[:n],
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(bs, self.stride),
            positions=self.positions[:n],
            commit_lens=self.commit_lens[:bs],
            state_slot=None if self.state_slot is None else self.state_slot[:n],
        )
        self.commit_lens.zero_()

    def flush(self) -> None:
        """Host: write a pending staging now (an eager forward follows)."""
        # A graph captured here would replay the write on every later replay.
        if self.pending and not _capturing():
            self.write()
            self.staged_bs = 0
            self.num_written_eagerly += 1

    def before_draft_forward(self, graph_bs: Optional[int]) -> None:
        """Host, before a draft forward: a graph of ``graph_bs`` requests (None: eager)
        writes only its own rows, so anything it would miss is written first."""
        if (
            graph_bs is None
            or graph_bs not in self._forked_bs
            or self.staged_bs > graph_bs
        ):
            self.flush()

    def after_draft_forward(self, graph_ran: bool) -> None:
        # A pending staging past before_draft_forward is one the graph writes.
        if graph_ran and self.pending:
            self.staged_bs = 0
            self.num_written_in_graph += 1

    def capture_head_hook(self, graph_runner, forward_batch) -> None:
        """First inside each draft decode graph capture (and its warmups)."""
        if not forward_batch.forward_mode.is_target_verify():
            return
        bs = forward_batch.batch_size
        if not _capturing():
            # Warmups write the disarmed staging, so its kernels compile before capture.
            if not self.pending:
                self.write(bs)
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self._device)
            self._done = torch.cuda.Event()
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self.write(bs)
            self._done.record()
        self._fork_unjoined = True
        self._forked_bs.add(bs)
        logger.info(
            "DSpark draft graph bs=%d: deferred context-KV commit forked at its head",
            bs,
        )

    def join(self) -> None:
        """At the draft's first KV store: the store waits for a forked write."""
        if self._fork_unjoined and _capturing():
            torch.cuda.current_stream().wait_event(self._done)
            self._fork_unjoined = False
            logger.info(
                "DSpark draft graph: deferred context-KV commit joined at the first KV store"
            )

    def disarm(self) -> None:
        """Drop whatever warmups or captures staged on placeholder inputs."""
        self.commit_lens.zero_()
        self.staged_bs = 0
        self._fork_unjoined = False


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def defer_context_kv_refusal(
    *,
    static_verify: bool,
    draft_is_moe: bool,
    unified_kv: bool,
    unified_memory: bool,
    hicache: bool,
    disaggregated: bool,
    attn_dp: bool,
) -> str:
    """Why the deferred commit cannot run here, or ""."""
    reasons = (
        (not static_verify, "the verify is not static (SGLANG_RAGGED_VERIFY_MODE)"),
        (not draft_is_moe, "the draft has no MoE layers"),
        (unified_kv, "unified KV writes the draft window through its ring"),
        (unified_memory, "the unified memory pool compacts KV between steps"),
        (hicache, "a HiCache storage backend moves KV between steps"),
        (disaggregated, "PD disaggregation transfers KV between steps"),
        (attn_dp, "attention DP runs idle draft forwards"),
    )
    return "; ".join(why for refused, why in reasons if refused)
