"""DFLASH spec-v2 overlap scheduling data structures."""

import contextlib
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    mamba_track_positions_from_reqs,
)
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.invariants import (
    Bucket,
    Invariant,
    IsTrue,
    expect,
    resolve_level,
)

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Return an optional plan stream/context for overlap scheduling prep kernels."""
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


# A skipped H2D whose device track row no longer matches the host plan steers
# the verify's mamba state write at a stale ping-pong slot: no containment.
_PLAN_STAGING_CLEAN = Invariant(
    "dflash.plan_staging_clean", Bucket.FATAL_UNCONTAINABLE, IsTrue()
)

_PLAN_ROW_CUR_KV_LENS = 0
_PLAN_ROW_NXT_KV_LENS = 1
_PLAN_ROW_TRACK_POSITIONS = 2
_PLAN_ROWS = 3


class DecodePlanHostRows(NamedTuple):
    """This step's host-side plan rows, sliced to the batch."""

    seq_lens: torch.Tensor  # int64: committed lens (seeds batch.seq_lens_cpu)
    cur_kv_lens: torch.Tensor  # int32
    nxt_kv_lens: torch.Tensor  # int32
    track_positions: torch.Tensor  # int32


class DecodePlanDeviceRows(NamedTuple):
    """Device views of the plan rows, sliced to the batch."""

    cur_kv_lens: torch.Tensor
    nxt_kv_lens: torch.Tensor
    track_positions: Optional[torch.Tensor]


class DecodePlanStaging:
    """Host->device staging of the per-step DFLASH decode plan, one per device.

    Rows: cur/nxt_kv_lens (page-aligned allocation bounds, read on device only
    by alloc_for_spec_decode when num_needed_tokens > 0) and the mamba
    ping-pong track positions (read by every verify). Owned per device rather
    than per DFlashDraftInputV2 because the worker builds a fresh draft input
    every step; owning the buffers there meant three allocations and a fill
    kernel per step.

    The host rows are a two-deep ring. The H2D of step N reads its pinned slot
    asynchronously and may still be pending while the host prepares N+1 (the
    host runs ahead of the device); step N-1's copy is done by then because the
    overlap loop synchronizes copy_done(N-1) -- which fences forward(N-1) and
    the schedule-stream work before it -- before preparing N+1. The ring also
    keeps the host views handed out for step N (batch.seq_lens_cpu,
    nxt_kv_lens_cpu) intact while N's result is processed. A deeper pipeline
    needs a deeper ring: under SGLANG_INVARIANT_CHECK every shipped slot
    records an event and reacquiring a slot whose copy has not drained fails
    loudly; the default path carries no event traffic.

    The H2D is skipped when the device rows are already current: rows 0/1 are
    consumed only on allocation rounds, row 2 only changes with the track
    positions, and any change of request set or batch size re-ships all rows.
    Under SGLANG_INVARIANT_CHECK the copy is unconditional and a round the
    predicate called clean has its device track row compared against the host
    plan, so a missed writer fails loudly (dflash.plan_staging_clean).
    """

    RING_DEPTH = 2

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._device_module = torch.get_device_module(self.device.type)
        self._pin_memory = is_pin_memory_available(self.device)
        self._use_events = self.device.type != "cpu"
        self.capacity = 0
        self._slot = 0
        self._host_seq_lens: list[torch.Tensor] = []
        self._host_lens: list[torch.Tensor] = []
        self._slot_copied: list[Optional[torch.Event]] = []
        self.lens_gpu: Optional[torch.Tensor] = None
        self._device_stale = True
        self._shipped_track_positions: Optional[List[int]] = None
        self._shipped_req_ids: Optional[tuple[str, ...]] = None
        # Diagnostics: how sparse the gated H2D is over a decode.
        self.stage_ct = 0
        self.h2d_copy_ct = 0

    def _grow(self, bs: int) -> None:
        self.capacity = max(bs, 32, self.capacity * 2)
        # Old pinned slots may still back a pending copy; the caching host
        # allocator tracks that on free, and the old device buffer stays alive
        # through the views the in-flight batch snapshot holds.
        self._host_seq_lens = [
            torch.empty((self.capacity,), dtype=torch.int64, device="cpu")
            for _ in range(self.RING_DEPTH)
        ]
        self._host_lens = [
            torch.zeros(
                (_PLAN_ROWS, self.capacity),
                dtype=torch.int32,
                device="cpu",
                pin_memory=self._pin_memory,
            )
            for _ in range(self.RING_DEPTH)
        ]
        self._slot_copied = [None] * self.RING_DEPTH
        self.lens_gpu = torch.zeros(
            (_PLAN_ROWS, self.capacity), dtype=torch.int32, device=self.device
        )
        self._device_stale = True

    def acquire(self, bs: int) -> DecodePlanHostRows:
        """Take the next host slot for a step of `bs` requests."""
        if bs > self.capacity:
            self._grow(bs)
        self._slot = (self._slot + 1) % self.RING_DEPTH
        copied = self._slot_copied[self._slot]
        assert copied is None or copied.query(), (
            f"plan slot {self._slot} reacquired before its H2D drained: the ring "
            f"assumes copy_done(N-1) is synchronized before step N+1 is prepared"
        )
        lens = self._host_lens[self._slot]
        return DecodePlanHostRows(
            seq_lens=self._host_seq_lens[self._slot][:bs],
            cur_kv_lens=lens[_PLAN_ROW_CUR_KV_LENS, :bs],
            nxt_kv_lens=lens[_PLAN_ROW_NXT_KV_LENS, :bs],
            track_positions=lens[_PLAN_ROW_TRACK_POSITIONS, :bs],
        )

    def ship(
        self,
        *,
        bs: int,
        num_needed_tokens: int,
        track_positions: Optional[List[int]],
        req_ids: tuple[str, ...],
    ) -> DecodePlanDeviceRows:
        """Publish the acquired slot to the device on the current stream when
        its consumed rows changed since the last copy."""
        lens_gpu = self.lens_gpu
        assert lens_gpu is not None, "acquire() must precede ship()"
        self.stage_ct += 1
        dirty = (
            self._device_stale
            or num_needed_tokens > 0
            or req_ids != self._shipped_req_ids
            or track_positions != self._shipped_track_positions
        )
        checking = resolve_level() >= InvariantCheckLevel.WARN
        track_row = lens_gpu[_PLAN_ROW_TRACK_POSITIONS, :bs]
        previous_track_row = (
            track_row.clone()
            if checking and not dirty and track_positions is not None
            else None
        )
        if dirty or checking:
            lens_gpu.copy_(self._host_lens[self._slot], non_blocking=True)
            if checking and self._use_events:
                copied = self._device_module.Event()
                copied.record()
                self._slot_copied[self._slot] = copied
            self.h2d_copy_ct += 1
            self._device_stale = False
            self._shipped_req_ids = req_ids
            self._shipped_track_positions = (
                None if track_positions is None else list(track_positions)
            )
        if previous_track_row is not None:
            expect(
                _PLAN_STAGING_CLEAN,
                (previous_track_row == track_row).all(),
                msg="device track row diverged from a round the dirty predicate "
                "skipped",
            )
        return DecodePlanDeviceRows(
            cur_kv_lens=lens_gpu[_PLAN_ROW_CUR_KV_LENS, :bs],
            nxt_kv_lens=lens_gpu[_PLAN_ROW_NXT_KV_LENS, :bs],
            track_positions=None if track_positions is None else track_row,
        )


_DECODE_PLAN_STAGINGS: dict[str, DecodePlanStaging] = {}


def get_decode_plan_staging(device: torch.device | str) -> DecodePlanStaging:
    key = str(device)
    staging = _DECODE_PLAN_STAGINGS.get(key)
    if staging is None:
        staging = DecodePlanStaging(device)
        _DECODE_PLAN_STAGINGS[key] = staging
    return staging


def spec_track_positions(batch: ScheduleBatch) -> Optional[List[int]]:
    """This step's per-req mamba ping-pong write positions, or None when the
    extra-buffer cache is off. Lazy: the plan mamba_lazy_spec_prepare made;
    otherwise mamba_next_track_idx, which nothing advances between decode
    prep and the verify prep of the same step (results are processed after
    run_batch), so the verify may read the staged copy."""
    if not get_exec().mamba.enable_mamba_extra_buffer:
        return None
    if get_exec().mamba.enable_mamba_extra_buffer_lazy:
        positions = batch.mamba_lazy_spec_track_positions_cpu
        assert positions is not None and len(positions) == len(batch.reqs), (
            "lazy spec decode without a track plan: mamba_lazy_spec_prepare "
            "must run before DFlashDraftInputV2.prepare_for_decode"
        )
        return positions
    return mamba_track_positions_from_reqs(batch.reqs)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    # Legacy Eagle-shaped fields; DFLASH relays via FutureMap so these are unused.
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    nxt_kv_lens_cpu: Optional[torch.Tensor] = None
    nxt_kv_lens_sum: Optional[int] = None

    # Filled by scheduler after dispatch.
    future_indices: Optional[torch.Tensor] = None

    verify_token_budget: Optional[int] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)
        # Spec v2 draft state itself does not change token accounting.
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate headroom in the shared req_to_token pool for the next DFLASH step.

        DFLASH spec-v2 uses overlap scheduling's "over-allocation" approach: we reserve
        future KV slots ahead of time so the worker can gather `out_cache_loc` directly
        from `req_to_token` without allocator backup/restore. CPU metadata intentionally
        lags by one iteration; keep it separate from the reserved upper bound that backs
        the overallocated mapping.
        """
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)

        bs = batch.batch_size()
        if bs == 0:
            return

        batch.maybe_evict_swa()

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_spec().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )
        reserve = 2 * block_size
        page_size = batch.token_to_kv_pool_allocator.page_size

        cur_kv_lens, nxt_kv_lens, num_needed_tokens = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=reserve,
            page_size=page_size,
        )
        track_positions = spec_track_positions(batch)

        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        committed_lens: List[int] = []
        for i, req in enumerate(batch.reqs):
            committed_lens.append(int(req.kv.kv_committed_len))
            top_k = int(req.sampling_params.top_k)
            if top_k > max_top_k:
                max_top_k = top_k
            if i == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max(max_top_k, 1)
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        staging = get_decode_plan_staging(batch.device)
        host = staging.acquire(bs)
        host.seq_lens.copy_(torch.tensor(committed_lens, dtype=torch.int64))
        host.cur_kv_lens.copy_(torch.tensor(cur_kv_lens, dtype=torch.int32))
        host.nxt_kv_lens.copy_(torch.tensor(nxt_kv_lens, dtype=torch.int32))
        if track_positions is not None:
            host.track_positions.copy_(torch.tensor(track_positions, dtype=torch.int32))

        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        with plan_stream_ctx:
            if plan_stream is not None and caller_stream is not None:
                # `batch.seq_lens`, `batch.req_pool_indices`, and related tensors may
                # have just been rebuilt on the scheduler stream by filter/merge ops.
                # The plan stream must wait for those writes before reading them.
                plan_stream.wait_stream(caller_stream)

            device_rows = staging.ship(
                bs=bs,
                num_needed_tokens=num_needed_tokens,
                track_positions=track_positions,
                req_ids=tuple(req.rid for req in batch.reqs),
            )
            if device_rows.track_positions is not None:
                batch.mamba_spec_track_positions = device_rows.track_positions

            alloc_for_spec_decode(
                batch.tree_cache,
                batch.req_to_token_pool,
                reqs=batch.reqs,
                req_pool_indices=batch.req_pool_indices,
                cur_kv_lens=device_rows.cur_kv_lens,
                cur_kv_lens_cpu=host.cur_kv_lens,
                nxt_kv_lens=device_rows.nxt_kv_lens,
                nxt_kv_lens_cpu=host.nxt_kv_lens,
                num_needed_tokens=num_needed_tokens,
                batch=batch,
            )
        if caller_stream is not None:
            # Enqueue the dependency on the caller's stream, not inside the
            # plan-stream context, so forward work cannot observe partially
            # prepared req_to_token / KV allocation state.
            caller_stream.wait_stream(plan_stream)
        for req in batch.reqs:
            req.decode_batch_idx += 1
        # Seed committed; overlap's resolve overwrites it with the published value.
        batch.seq_lens_cpu = host.seq_lens
        batch.seq_lens_sum = sum(committed_lens)
        self.nxt_kv_lens_cpu = host.nxt_kv_lens
        self.nxt_kv_lens_sum = sum(nxt_kv_lens)

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        if self.nxt_kv_lens_cpu is not None:
            if new_indices_cpu is not None:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices_cpu]
            else:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices.cpu()]
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        if self.nxt_kv_lens_cpu is not None:
            assert spec_info.nxt_kv_lens_cpu is not None
            self.nxt_kv_lens_cpu = torch.cat(
                [self.nxt_kv_lens_cpu, spec_info.nxt_kv_lens_cpu]
            )
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())
        elif spec_info.nxt_kv_lens_cpu is not None:
            self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu
            self.nxt_kv_lens_sum = spec_info.nxt_kv_lens_sum

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
