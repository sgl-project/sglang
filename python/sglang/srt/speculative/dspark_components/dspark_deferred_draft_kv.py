"""Defers the draft KV store of a static verify's committed tokens to the next
draft graph (SGLANG_DSPARK_DEFER_CONTEXT_KV).

The projection of the committed rows' target hidden forks onto a side stream at the
target's logits processor, overlapping the LM head and the accept; the static
epilogue joins it and stages the store. The next draft graph runs the store on a
side stream forked at its head and joined at its first KV store; an eager draft, or
a graph that would not cover every staged row, writes it eagerly first.

A draft forward writes only the staged rows of the requests it carries and that were
not retracted since; a prefill in between leaves the staging pending. Every write
disarms the staging (commit lengths 0), so a staging is written once.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

logger = logging.getLogger(__name__)

# Draft forwards between two counter lines.
_LOG_EVERY = 4096


class DeferredDraftKVStore:
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
        # Host: (req, retraction_count) per staged row; empty: nothing pending.
        self._staged_reqs: tuple = ()
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
        return bool(self._staged_reqs)

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

    def note_staged(self, reqs) -> None:
        """Host, after a static verify graph staged the rows of ``reqs``."""
        self._staged_reqs = tuple((req, req.retraction_count) for req in reqs)

    def _disarm_stale_rows(self, reqs) -> None:
        """Host, before a draft forward of ``reqs`` writes the staging: disarm the
        rows of the requests it does not carry or that were retracted since."""
        carried = set(reqs)
        for row, (req, retraction_count) in enumerate(self._staged_reqs):
            if req not in carried or req.retraction_count != retraction_count:
                self.commit_lens[row : row + 1].zero_()

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

    def _flush(self, reason: str) -> None:
        """Host: write a pending staging now (an eager forward follows)."""
        # A graph captured here would replay the write on every later replay.
        if self.pending and not _capturing():
            self.write()
            self._staged_reqs = ()
            self.num_written_eagerly += 1
            self._eager_reasons[reason] = self._eager_reasons.get(reason, 0) + 1

    def before_draft_forward(self, graph_bs: Optional[int], reqs) -> None:
        """Host, before a draft forward of ``reqs``: a graph of ``graph_bs`` requests
        (None: eager) writes only its own rows, so anything it would miss is
        written first; what stays pending is the graph's to write."""
        self._draft_forwards += 1
        if self._draft_forwards % _LOG_EVERY == 0:
            logger.debug(
                "DSpark deferred draft KV store: %d written in draft graphs, "
                "%d eagerly %s",
                self.num_written_in_graph,
                self.num_written_eagerly,
                self._eager_reasons,
            )
        self._disarm_stale_rows(reqs)
        if graph_bs is None:
            self._flush("eager draft")
        elif graph_bs not in self._forked_bs:
            self._flush("draft graph without the fork")
        elif len(self._staged_reqs) > graph_bs:
            self._flush("draft graph rows")
        elif self.pending:
            self._staged_reqs = ()
            self.num_written_in_graph += 1

    def attach_draft(self, draft_model_runner, proposer) -> None:
        """Draft side: the draft graph forks the staged write at its head and its
        first KV store joins it."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DeepseekV4MultiStepBackend,
        )

        proposer.deferred_draft_kv = self
        draft_model_runner.capture_head_hooks.append(self.capture_head_hook)
        backend = draft_model_runner.attn_backend
        backends = [backend]
        if isinstance(backend, DeepseekV4MultiStepBackend):
            backends.extend(backend.attn_backends)
        for b in backends:
            if isinstance(b, DeepseekV4AttnBackend):
                b.pre_kv_store_hook = self.join
        logger.info(
            "SGLANG_DSPARK_DEFER_CONTEXT_KV: the draft KV store runs at the "
            "head of the next draft graph, joined at its first KV store."
        )

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
            "DSpark draft graph bs=%d: deferred draft KV store forked at its head",
            bs,
        )

    def join(self) -> None:
        """At the draft's first KV store: the store waits for the forked stores."""
        if self._fork_unjoined and _capturing():
            torch.cuda.current_stream().wait_event(self._store_done)
            self._fork_unjoined = False
            logger.info(
                "DSpark draft graph: deferred draft KV store joined at the first KV store"
            )

    def disarm(self) -> None:
        """Drop whatever warmups or captures staged on placeholder inputs."""
        self.commit_lens.zero_()
        self._staged_reqs = ()
        self._fork_unjoined = False
        self._proj_unjoined = False


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def maybe_build_deferred_draft_kv(
    *,
    static_verify: bool,
    draft_is_moe: bool,
    max_bs: int,
    stride: int,
    device,
    kv_injector,
    target_model,
) -> Optional[DeferredDraftKVStore]:
    """SGLANG_DSPARK_DEFER_CONTEXT_KV: only where nothing moves or reads draft KV
    between a verify and the next draft forward but this worker's forwards."""
    if not envs.SGLANG_DSPARK_DEFER_CONTEXT_KV.get():
        return None
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )

    memory = get_memory()
    refusal = defer_context_kv_refusal(
        static_verify=static_verify,
        draft_is_moe=draft_is_moe,
        unified_kv=is_unified_kv_triton(),
        unified_memory=memory.enable_unified_memory,
        hicache=memory.enable_hierarchical_cache
        or memory.hicache_storage_backend is not None,
        disaggregated=get_disagg().disaggregation_mode != "null",
        attn_dp=get_parallel().attn_dp_enabled,
    )
    if refusal:
        logger.warning(
            "SGLANG_DSPARK_DEFER_CONTEXT_KV refused, the draft KV store stays "
            "in the verify graph: %s",
            refusal,
        )
        return None
    store = DeferredDraftKVStore(
        max_bs=max_bs, stride=stride, device=device, kv_injector=kv_injector
    )
    # The target's logits processor: SGLang's forward and a megakernel's head
    # both call it, with the DSpark hidden, once the hidden is final.
    target_model.logits_processor.register_forward_pre_hook(
        store.logits_pre_hook, with_kwargs=True
    )
    return store


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
    """Why the deferred draft KV store cannot run here, or ""."""
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
