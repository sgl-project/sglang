from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import numpy as np
import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_spec,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    # Local import: keep overlap_utils' module-level deps leaf-only so it stays
    # importable everywhere; spec_info pulls in the spec/schedule_batch graph.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if get_exec().overlap.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        return True
    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def draft_state_in_future_map(enable_overlap: bool) -> bool:
    """Whether the draft state one speculative forward hands to the next
    (bonus tokens, draft extras and new lengths, per row) travels through
    FutureMap, keyed by req_pool_idx: the producing worker writes it and the
    next step's worker gathers it. Every path but non-overlap pipeline
    parallelism, whose relay keeps it in slots of its own
    (speculative/pp_spec_relay.py)."""
    return enable_overlap or get_parallel().pp_size == 1


def decide_needs_confidence_relay() -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


def create_future_map(
    *,
    spec_algorithm: SpeculativeAlgorithm,
    device: str,
    req_to_token_pool: ReqToTokenPool,
    tp_worker,
    draft_worker,
) -> FutureMap:
    # FutureMap is always-on: input_ids relay used in both modes.
    # Workers without the spec_v2_attn_backends override fall back to
    # target-only so the helper still produces a safe decision (no
    # accidental opt-out for unaudited shapes).
    if draft_worker is not None:
        attn_backends = getattr(
            draft_worker,
            "spec_v2_attn_backends",
            (tp_worker.model_runner.attn_backend,),
        )
    else:
        attn_backends = (tp_worker.model_runner.attn_backend,)
    needs_cpu_seq_lens = decide_needs_cpu_seq_lens(attn_backends)
    needs_confidence_relay = decide_needs_confidence_relay()
    return spec_algorithm.create_future_map(
        device,
        req_to_token_pool,
        needs_cpu_seq_lens=needs_cpu_seq_lens,
        needs_confidence_relay=needs_confidence_relay,
    )


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            # Keep mix_running_indices (= the decode rows' req_pool_indices) alive on
            # the forward stream until this gather completes.
            # CPU has no cross-stream reuse hazard (and record_stream rejects CPU tensors).
            if batch.mix_running_indices.device.type != "cpu":
                batch.mix_running_indices.record_stream(
                    torch.get_device_module(batch.device).current_stream()
                )
            if batch.enable_overlap and not batch.spec_algorithm.is_none():
                future_map.resolve_mixed_spec_tails(batch)
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Speculative draft state is not resolved here: the worker gathers it from
    # FutureMap into its own draft input (FutureMap.gather_draft_input).


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):
    confidence: torch.Tensor
    generation: torch.Tensor


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape."""

    bonus_tokens: Optional[torch.Tensor]
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None
    # ngram delays the draft extend (ngram update)
    accept_tokens: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None
    # Whether the rows' draft state lets the draft replay its graphs.
    cuda_graph_compatible: bool = True

    @classmethod
    def from_ngram(cls, draft_input: NgramVerifyInput) -> RelayPayload:
        return cls(
            bonus_tokens=None,
            accept_tokens=draft_input.accept_tokens.reshape(
                -1, draft_input.draft_token_num
            ),
            accept_lens=draft_input.accept_lens,
        )

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
            dsa_topk_indices=draft_input.dsa_topk_indices,
            cuda_graph_compatible=draft_input.cuda_graph_compatible,
        )


class ConfidenceRelay(msgspec.Struct):
    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        depth = CONFIDENCE_RELAY_RING_DEPTH
        self.conf_ring = torch.empty(
            (depth, self.req_pool_size, gamma),
            dtype=torch.float32,
            pin_memory=self.device.type != "cpu",
        )
        self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
        self.copy_done = [
            torch.get_device_module(self.device).Event() for _ in range(depth)
        ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(self, batch: ScheduleBatch) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        # Only a ring copy an earlier event covers is read: the host never
        # waits on the forward stream, on any device.
        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


# Live publishes kept before dropping the ones no slot names anymore.
_MAX_LIVE_PUBLISHES = 64


class StepPublish:
    """The device commit of one forward's FutureMap writes: `seq_lens_ready`
    is recorded on the forward stream once the step's new lengths are in
    FutureMap, `ready` once all its slot writes are. A gather waits on the
    publishes of the steps that last wrote its slots.
    """

    __slots__ = ("step_id", "stream", "seq_lens_ready", "ready")

    def __init__(self, step_id: int, stream: Any, seq_lens_ready: Any):
        self.step_id = step_id
        self.stream = stream
        self.seq_lens_ready = seq_lens_ready
        self.ready = None


class FutureMap:
    """Pool-indexed relay for cross-step values, keyed by req_pool_idx: the
    producing forward writes its slots and the next step's worker gathers
    them. A slot is released only after the host commit of the last pending
    step that references it.
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]
        # Kept for the mixed-tail late binding (reserved-slot gather).
        self.req_to_token = req_to_token_pool.req_to_token
        self.device_module = torch.get_device_module(device)

        if _DEBUG_ASSERT:
            # Poisoned init: every row must be written before its first gather.
            self.output_tokens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
        else:
            self.output_tokens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = self.device_module.Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        # The confidence relay's ring copies run on every device.
        self.confidence_d2h_stream = self.fwd_prepare_d2h_stream
        if needs_confidence_relay and self.confidence_d2h_stream is None:
            self.confidence_d2h_stream = self.device_module.Stream()
        self.need_topk = False
        self.need_hidden_states = False
        self.topk_p_buf = None
        self.topk_index_buf = None
        self.hidden_states_buf = None
        self.draft_probs_buf = None
        self.dsa_topk_indices_buf = None
        # Host side: whether a slot's dsa_topk_indices_buf row holds a seed,
        # and whether its draft state lets the draft replay its graphs.
        self.dsa_topk_valid = np.zeros((self.req_pool_size,), dtype=bool)
        self.slot_graph_compatible = np.ones((self.req_pool_size,), dtype=bool)

        # ngram-only relay bufs
        self.accept_tokens_buf: Optional[torch.Tensor] = None
        self.accept_lens_buf: Optional[torch.Tensor] = None

        # Host side: the step that last published each slot (-1: none), and
        # the publishes some slot still names, by step id. Only speculative
        # steps publish; a non-speculative step's next reader is ordered
        # behind it on the forward stream.
        self.slot_writer = np.full((self.req_pool_size,), -1, dtype=np.int64)
        self.publishes: dict[int, StepPublish] = {}
        self.next_step_id = 0
        # The latest publish's seq_lens_ready; the confidence relay's ring
        # copies follow the publish order, not the slots.
        self.last_seq_lens_ready = None

        self.confidence_relay = ConfidenceRelay(
            device=torch.device(self.device),
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _maybe_init_forward_bufs(self, payload: RelayPayload) -> None:
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        # Prefill can omit spec extras; initialize each buffer when decode first
        # carries it instead of fixing the layout from the first payload.
        if not self.need_topk and (
            self.spec_algo.is_some()
            and self.spec_algo.need_topk()
            and payload.topk_p is not None
        ):
            self.need_topk = True
            topk_p0 = payload.topk_p[0]
            topk_index0 = payload.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )

        if not self.need_hidden_states and (
            self.spec_algo.is_some()
            and spec_need_hidden_states()
            and payload.hidden_states is not None
        ):
            self.need_hidden_states = True
            hidden_states0 = payload.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        if self.draft_probs_buf is None and payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

    def _maybe_init_dsa_topk_indices_buf(self, payload: RelayPayload) -> None:
        if self.dsa_topk_indices_buf is not None or payload.dsa_topk_indices is None:
            return
        seed0 = payload.dsa_topk_indices[0]
        self.dsa_topk_indices_buf = torch.empty(
            (self.req_pool_size, *seed0.shape),
            dtype=payload.dsa_topk_indices.dtype,
            device=self.device,
        )

    def _maybe_init_ngram_bufs(self, payload: RelayPayload) -> None:
        if self.accept_tokens_buf is not None:
            return
        # zeros, not empty: an unstashed row resolves to accept_len 0 (empty
        # splice at draft prep) instead of a garbage length.
        self.accept_tokens_buf = torch.zeros(
            (self.req_pool_size, payload.accept_tokens.shape[1]),
            dtype=payload.accept_tokens.dtype,
            device=self.device,
        )
        self.accept_lens_buf = torch.zeros(
            (self.req_pool_size,),
            dtype=payload.accept_lens.dtype,
            device=self.device,
        )

    def _host_indices(
        self, indices: torch.Tensor, indices_cpu: Optional[torch.Tensor]
    ) -> np.ndarray:
        if indices_cpu is None:
            # Syncs on a device tensor; callers that hold the host copy pass it.
            indices_cpu = indices.cpu()
        if isinstance(indices_cpu, torch.Tensor):
            return indices_cpu.numpy()
        return np.asarray(indices_cpu, dtype=np.int64)

    def writer_publishes(self, indices_cpu: np.ndarray) -> list[StepPublish]:
        """The publishes of the steps that last wrote these slots."""
        step_ids = np.unique(self.slot_writer[indices_cpu])
        return [self.publishes[int(i)] for i in step_ids if i >= 0]

    def _wait_writers(self, indices_cpu: np.ndarray, *, draft_state: bool) -> None:
        """Order the current stream after the last writers of these slots:
        their lengths, and with `draft_state` all of their slot writes."""
        stream = self.device_module.current_stream()
        for publish in self.writer_publishes(indices_cpu):
            if publish.stream == stream:
                continue  # stream-ordered already
            event = publish.seq_lens_ready
            if draft_state and publish.ready is not None:
                event = publish.ready
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                event.synchronize()
            else:
                stream.wait_event(event)

    def _host_seq_lens(
        self, seq_lens: torch.Tensor, indices_cpu: np.ndarray
    ) -> torch.Tensor:
        if self.fwd_prepare_d2h_stream is None:
            return seq_lens.cpu()  # bootstrap / non-CUDA
        # Don't sync the forward stream: gate a private stream on the writers'
        # publishes and copy into the static pinned buffer.
        for publish in self.writer_publishes(indices_cpu):
            self.fwd_prepare_d2h_stream.wait_event(publish.seq_lens_ready)
        with self.device_module.stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()
        return self.new_seq_lens_cpu_pinned[torch.from_numpy(indices_cpu)]

    def gather_seq_lens(
        self,
        indices: torch.Tensor,
        indices_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        """The committed lengths of these slots: on device, and on host with
        their sum when needs_cpu_seq_lens (else None, None: the backends take
        the GPU-only path)."""
        host_indices = self._host_indices(indices, indices_cpu)
        self._wait_writers(host_indices, draft_state=False)
        seq_lens = self.new_seq_lens_buf[indices]
        if not self.needs_cpu_seq_lens:
            return seq_lens, None, None
        seq_lens_cpu = self._host_seq_lens(seq_lens, host_indices)
        return seq_lens, seq_lens_cpu, int(seq_lens_cpu.sum())

    def gather_draft_input(
        self,
        carried: Optional[SpecInput],
        indices_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[SpecInput]:
        """The draft input of a step's next forward: a copy of the step's
        `carried` draft input with the per-row state gathered from the slots
        of its `future_indices`. `carried` is left as it is; one without keys
        (prefill, the pipeline-parallel relay) is copied as it is."""
        if carried is None:
            return None
        draft_input = copy.copy(carried)
        indices = carried.future_indices
        if indices is None or indices.shape[0] == 0:
            return draft_input
        host_indices = self._host_indices(indices, indices_cpu)
        self._wait_writers(host_indices, draft_state=True)

        if self.spec_algo.is_ngram():
            draft_input.accept_tokens = self.accept_tokens_buf[indices].flatten()
            draft_input.accept_lens = self.accept_lens_buf[indices]
            return draft_input

        # CPU has no Triton gather.
        on_cpu = indices.device.type == "cpu"
        if not on_cpu:
            # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
            # record_batch_in_overlap; record_stream here is redundant.
            indices.record_stream(self.device_module.current_stream())
        if self.need_topk:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            if on_cpu:
                topk_p = self.topk_p_buf[indices]
                topk_index = self.topk_index_buf[indices]
                bonus_tokens = self.output_tokens_buf[indices]
                hidden_states = (
                    None if hidden_states_buf is None else hidden_states_buf[indices]
                )
            else:
                topk_p, topk_index, bonus_tokens, hidden_states = gather_spec_extras(
                    indices,
                    self.topk_p_buf,
                    self.topk_index_buf,
                    self.output_tokens_buf,
                    hidden_states_buf,
                )
            draft_input.topk_p = topk_p
            draft_input.topk_index = topk_index
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        dsa_topk_available = self.dsa_topk_indices_buf is not None and bool(
            self.dsa_topk_valid[host_indices].all()
        )
        draft_input.future_dsa_topk_indices_available = dsa_topk_available
        draft_input.dsa_topk_indices = (
            self.dsa_topk_indices_buf[indices] if dsa_topk_available else None
        )
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )
        return draft_input

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(batch)

    def stash_bonus_tokens(
        self, indices: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> None:
        """Write only output_tokens_buf rows; for relays carrying no draft
        extras (stash() would lazy-init the spec bufs from the payload)."""
        self.output_tokens_buf[indices] = bonus_tokens.to(self.output_tokens_buf.dtype)

    def resolve_mixed_spec_tails(self, batch: ScheduleBatch) -> None:
        """Late-bind a spec mixed batch's decode tails (overlap): schedule-time
        lengths lag the in-flight step's accept count, so rebuild the tail rows
        from the published committed lengths behind their writers' publishes."""
        idx = batch.mix_running_indices
        n = int(idx.shape[0])
        if n == 0:
            return
        host_indices = self._host_indices(idx, batch.mix_running_indices_cpu)
        self._wait_writers(host_indices, draft_state=False)
        fresh = self.new_seq_lens_buf[idx]
        seq_lens = batch.seq_lens.clone()
        seq_lens[-n:] = fresh + 1
        batch.seq_lens = seq_lens
        out_cache_loc = batch.out_cache_loc.clone()
        out_cache_loc[-n:] = self.req_to_token[idx.long(), fresh.long()].to(
            out_cache_loc.dtype
        )
        batch.out_cache_loc = out_cache_loc

        fresh_cpu = self._host_seq_lens(fresh, host_indices)
        if batch.seq_lens_cpu is not None:
            seq_lens_cpu = batch.seq_lens_cpu.clone()
            seq_lens_cpu[-n:] = fresh_cpu + 1
            batch.seq_lens_cpu = seq_lens_cpu
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.prefix_lens = batch.prefix_lens[:-n] + [
            int(x) for x in fresh_cpu.tolist()
        ]

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        *,
        indices_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[StepPublish]:
        """Write the new lengths of these slots and, for a speculative step,
        record its publish: the event a gather of these slots waits on."""
        indices = future_indices
        if indices.shape[0] == 0:
            return None  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)
        if not self.spec_algo.is_some():
            return None
        stream = self.device_module.current_stream()
        publish = StepPublish(self.next_step_id, stream, self.device_module.Event())
        self.next_step_id += 1
        publish.seq_lens_ready.record(stream)
        self.slot_writer[self._host_indices(indices, indices_cpu)] = publish.step_id
        self.publishes[publish.step_id] = publish
        if len(self.publishes) > _MAX_LIVE_PUBLISHES:
            live = set(np.unique(self.slot_writer).tolist())
            self.publishes = {i: p for i, p in self.publishes.items() if i in live}
        self.last_seq_lens_ready = publish.seq_lens_ready
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.confidence_d2h_stream,
                publish_ready=publish.seq_lens_ready,
            )
        return publish

    def seal(self, publish: Optional[StepPublish]) -> None:
        """Record that every slot write of the publish's step is enqueued."""
        if publish is None:
            return
        publish.ready = self.device_module.Event()
        publish.ready.record(self.device_module.current_stream())

    def seed(
        self,
        future_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        payload: RelayPayload,
        *,
        indices_cpu: torch.Tensor,
    ) -> None:
        """Write the committed lengths and draft state of slots whose producing
        forward ran elsewhere (the prefill node of a PD pair), as that
        forward's device commit would have."""
        publish = self.publish(future_indices, seq_lens, indices_cpu=indices_cpu)
        self.stash(future_indices, payload, indices_cpu=indices_cpu)
        self.seal(publish)

    def cuda_graph_compatible(self, indices_cpu: torch.Tensor) -> bool:
        """Whether the draft state of all these slots lets the draft replay
        its graphs (the DP vote reduces it across ranks)."""
        host_indices = self._host_indices(indices_cpu, indices_cpu)
        return bool(self.slot_graph_compatible[host_indices].all())

    def release_slots(self, indices_cpu: torch.Tensor) -> None:
        """Forget the writers of slots being freed. Call it only once no
        pending step references the slots: at the host commit of the last
        one, so no in-flight forward still writes them."""
        host_indices = self._host_indices(indices_cpu, indices_cpu)
        self.slot_writer[host_indices] = -1
        self.dsa_topk_valid[host_indices] = False
        self.slot_graph_compatible[host_indices] = True

    def stash(
        self,
        future_indices: torch.Tensor,
        payload: RelayPayload,
        *,
        indices_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Write a step's draft state into its slots. With `indices_cpu`, also
        record which slots now hold a DSA top-k seed and whether their state
        lets the draft replay its graphs."""
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        if self.spec_algo.is_ngram():
            self._maybe_init_ngram_bufs(payload)
            self.accept_tokens_buf[indices] = payload.accept_tokens
            self.accept_lens_buf[indices] = payload.accept_lens
            return
        self._maybe_init_forward_bufs(payload)
        self._maybe_init_dsa_topk_indices_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )

        # Staging transitions refresh only the bonus token. Preserve top-k and
        # hidden state already published by the preceding target forward.
        if self.need_topk and payload.topk_p is not None:
            assert payload.topk_index is not None
            self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = payload.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states and payload.hidden_states is not None:
            self.hidden_states_buf[indices] = payload.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
        has_dsa_topk = (
            self.dsa_topk_indices_buf is not None
            and payload.dsa_topk_indices is not None
        )
        if has_dsa_topk:
            self.dsa_topk_indices_buf[indices] = payload.dsa_topk_indices.to(
                self.dsa_topk_indices_buf.dtype
            )
        if indices_cpu is not None:
            host_indices = self._host_indices(indices, indices_cpu)
            self.dsa_topk_valid[host_indices] = has_dsa_topk
            self.slot_graph_compatible[host_indices] = payload.cuda_graph_compatible
