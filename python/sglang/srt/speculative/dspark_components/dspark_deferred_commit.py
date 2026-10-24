"""The draft context-KV commit of a static verify, split around the accept
(SGLANG_DSPARK_DEFER_CONTEXT_KV).

The commit projects the committed rows' target hidden into each draft stage's KV
(GEMMs, independent of the accept) and stores the committed rows (needs the accept's
commit lengths). In a static verify graph the projection is forked onto a side stream
at the target's logits processor (a forward pre-hook, which SGLang's forward and a
megakernel's head both call), so it overlaps the LM head and the accept; the static
epilogue joins it and stages the store's arguments (write locations, positions, gated
commit lengths, ring slots). The next captured draft graph runs the stores on a side
stream forked at its head (a capture head hook); the draft's first KV store joins
them (``join``, from the draft attention backend's ``get_swa_out_cache_loc``, which
SGLang's draft attention and a megakernel's both reach on the store's stream). Any
other forward runs the stores eagerly first.

Contract:

- Every write disarms the staging (commit lengths 0), so a staging is written once.
- A forward that may write or read draft KV (prefill, draft extend, an eager draft)
  runs after the pending write; a draft graph whose rows miss staged ones is preceded
  by an eager write of every staged row.
- Nothing reads draft KV between the verify and the next draft forward.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states

logger = logging.getLogger(__name__)

# Draft forwards between two counter lines.
_LOG_EVERY = 4096


class DeferredContextCommit:
    def __init__(
        self,
        *,
        max_bs: int,
        stride: int,
        device,
        kv_injector,
    ) -> None:
        self.stride = stride
        self._max_rows = max_bs * stride
        self._device = device
        self._kv_injector = kv_injector
        # Per draft stage [max_rows, kv width], made at the first (warmup) projection.
        self.kvs: Optional[List[torch.Tensor]] = None
        rows = self._max_rows
        self.cache_loc = torch.zeros((rows,), dtype=torch.int64, device=device)
        self.positions = torch.zeros((rows,), dtype=torch.int64, device=device)
        # 0 for every request past the staged ones: their rows are not written.
        self.commit_lens = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        self.state_slot = torch.zeros((rows,), dtype=torch.int64, device=device)
        # Host, set at the first staging: the injector writes a per-request SWA ring.
        self._with_state_slot = False
        # SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE: stage() in one Triton launch.
        self._one_launch_stage = (
            envs.SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE.get()
            and torch.device(device).type == "cuda"
        )
        # Side streams and events, made at the first use.
        self._proj_stream: Optional[torch.cuda.Stream] = None
        self._proj_done: Optional[torch.cuda.Event] = None
        self._store_stream: Optional[torch.cuda.Stream] = None
        self._store_done: Optional[torch.cuda.Event] = None
        # Host: requests a static verify staged since the last write (0: nothing pending).
        self.staged_bs = 0
        # Host: a forked projection the static epilogue has not joined yet.
        self._proj_unjoined = False
        # Host, capture only: forked stores the draft's first KV store has not joined yet.
        self._fork_unjoined = False
        # Draft graph batch sizes whose capture recorded the forked stores.
        self._forked_bs: set[int] = set()
        self.num_written_in_graph = 0
        self.num_written_eagerly = 0
        self._eager_reasons: dict[str, int] = {}
        self._draft_forwards = 0

    @property
    def pending(self) -> bool:
        return self.staged_bs > 0

    def _pool(self):
        return self._kv_injector.draft_model_runner.token_to_kv_pool

    # Verify side: the projection (forked at the logits processor) and the staging.

    def logits_pre_hook(self, module, args, kwargs) -> None:
        """Forward pre-hook on the target's logits processor: in a static verify
        graph (capture or warmup), fork the projection of the DSpark hidden."""
        from sglang.srt.model_executor.runner import get_is_capture_mode

        metadata = args[3] if len(args) > 3 else kwargs.get("logits_metadata")
        aux = args[4] if len(args) > 4 else kwargs.get("aux_hidden_states")
        if (
            aux is None
            or not metadata.forward_mode.is_target_verify()
            or not get_is_capture_mode()
        ):
            return
        if self._proj_stream is None:
            self._proj_stream = torch.cuda.Stream(device=self._device)
            self._proj_done = torch.cuda.Event()
        self._proj_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._proj_stream), torch.inference_mode():
            kvs = self._kv_injector.draft_model.project_commit_kv(
                main_hidden=pack_aux_hidden_states(aux), pool=self._pool()
            )
            if self.kvs is None:
                assert not _capturing(), "the commit KV buffers are made at a warmup"
                self.kvs = [kv.new_zeros((self._max_rows, kv.shape[1])) for kv in kvs]
            n = kvs[0].shape[0]
            for buf, kv in zip(self.kvs, kvs):
                buf[:n].copy_(kv)
            self._proj_done.record()
        self._proj_unjoined = True

    def join_projection(self) -> bool:
        """In the static epilogue: wait for this graph's projection; False if none ran."""
        if not self._proj_unjoined:
            return False
        torch.cuda.current_stream().wait_event(self._proj_done)
        self._proj_unjoined = False
        return True

    def stage(
        self,
        *,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        state_slot: Optional[torch.Tensor],
    ) -> None:
        """Recorded in a static verify graph of ``commit_lens.shape[0]`` requests."""
        bs = commit_lens.shape[0]
        n = bs * self.stride
        self._with_state_slot = state_slot is not None
        if self._one_launch_stage:
            from sglang.kernels.ops.speculative.dspark.stage_commit import stage_commit

            stage_commit(
                cache_loc_dst=self.cache_loc,
                positions_dst=self.positions,
                commit_lens_dst=self.commit_lens,
                state_slot_dst=self.state_slot if state_slot is not None else None,
                cache_loc=cache_loc,
                positions=positions,
                commit_lens=commit_lens,
                state_slot=state_slot,
                rows=n,
            )
            return
        self.cache_loc[:n].copy_(cache_loc[:n])
        self.positions[:n].copy_(positions[:n])
        self.commit_lens[:bs].copy_(commit_lens)
        self.commit_lens[bs:].zero_()
        if state_slot is not None:
            self.state_slot[:n].copy_(state_slot[:n])

    def note_static_verify(self, live_bs: int) -> None:
        """Host, after a static verify graph ran ``live_bs`` requests."""
        self.staged_bs = live_bs

    # Draft side: the gated stores.

    def write(self, bs: Optional[int] = None) -> None:
        """The staged stores over the first ``bs`` requests (default all), then a
        disarm, on the current stream."""
        if self.kvs is None:
            return
        bs = self.commit_lens.shape[0] if bs is None else bs
        n = bs * self.stride
        pool = self._pool()
        cache_loc = self.cache_loc[:n]
        positions = self.positions[:n]
        swa_loc = self._kv_injector.committed_swa_loc(
            pool=pool,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc.view(bs, self.stride),
            commit_lens=self.commit_lens[:bs],
            state_slot=self.state_slot[:n] if self._with_state_slot else None,
        )
        with torch.inference_mode():
            self._kv_injector.draft_model.store_commit_kv(
                kvs=[kv[:n] for kv in self.kvs],
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )
        self.commit_lens.zero_()

    def flush(self, reason: str = "prefill") -> None:
        """Host: write a pending staging now (an eager forward follows)."""
        # A graph captured here would replay the write on every later replay.
        if self.pending and not _capturing():
            self.write()
            self.staged_bs = 0
            self.num_written_eagerly += 1
            self._eager_reasons[reason] = self._eager_reasons.get(reason, 0) + 1

    def before_draft_forward(self, graph_bs: Optional[int]) -> None:
        """Host, before a draft forward: a graph of ``graph_bs`` requests (None: eager)
        writes only its own rows, so anything it would miss is written first."""
        self._draft_forwards += 1
        if self._draft_forwards % _LOG_EVERY == 0:
            logger.debug(
                "DSpark deferred context-KV commit: %d written in draft graphs, "
                "%d eagerly %s",
                self.num_written_in_graph,
                self.num_written_eagerly,
                self._eager_reasons,
            )
        if graph_bs is None:
            self.flush("eager draft")
        elif graph_bs not in self._forked_bs:
            self.flush("draft graph without the fork")
        elif self.staged_bs > graph_bs:
            self.flush("draft graph rows")

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
        if self._store_stream is None:
            self._store_stream = torch.cuda.Stream(device=self._device)
            self._store_done = torch.cuda.Event()
        self._store_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._store_stream):
            self.write(bs)
            self._store_done.record()
        self._fork_unjoined = True
        self._forked_bs.add(bs)
        logger.info(
            "DSpark draft graph bs=%d: deferred context-KV commit forked at its head",
            bs,
        )

    def join(self) -> None:
        """At the draft's first KV store: the store waits for the forked stores."""
        if self._fork_unjoined and _capturing():
            torch.cuda.current_stream().wait_event(self._store_done)
            self._fork_unjoined = False
            logger.info(
                "DSpark draft graph: deferred context-KV commit joined at the first KV store"
            )

    def disarm(self) -> None:
        """Drop whatever warmups or captures staged on placeholder inputs."""
        self.commit_lens.zero_()
        self.staged_bs = 0
        self._fork_unjoined = False
        self._proj_unjoined = False


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
