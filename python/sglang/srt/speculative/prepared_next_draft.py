"""Stage the next EAGLE draft inputs in the preceding post-verify round."""

from __future__ import annotations

from functools import cache
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_resources,
    get_schedule,
    get_spec,
)


@cache
def prepared_next_draft_enabled() -> bool:
    """Whether generic next-round EAGLE runtime plumbing is safe to construct."""
    return (
        current_platform.is_cuda()
        and envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.get()
        and (get_spec().speculative_algorithm or "").upper() == "EAGLE"
        and (get_spec().speculative_eagle_topk or 1) == 1
        and not get_schedule().disable_overlap_schedule
        and get_parallel().moe_ep_size == 1
    )


def prepared_next_draft_reserve_configured() -> bool:
    """Safe pre-pool upper bound; graph initialization may only narrow it."""
    spec = get_spec()
    return (
        prepared_next_draft_enabled()
        and not spec.speculative_adaptive
        and get_parallel().pp_size == 1
        and not get_parallel().attn_dp_enabled
        and (spec.speculative_num_steps or 0) > 1
        and (spec.speculative_num_steps or 0)
        < (spec.speculative_num_draft_tokens or 0)
        and not get_exec().graph.disable_cuda_graph
        and not get_memory().enable_unified_memory
        and get_model().model_impl != "mindspore"
    )


class PreparedNextDraftInputs(msgspec.Struct, frozen=True, kw_only=True):
    """Inputs and KV write locations prepared for one subsequent draft round."""

    num_requests: int
    # The graph replay skips all inputs produced here.  Rebind the request rows
    # too: an eager preceding draft need not have populated the runner buffer.
    runner_request_pool_indices: torch.Tensor
    # This is the draft runner's static input.  It is deliberately distinct
    # from the scheduler-visible accepted lengths carried by PreparedRoundLease.
    runner_sequence_lengths: torch.Tensor
    positions: torch.Tensor
    mrope_positions: Optional[torch.Tensor]
    draft_out_cache_loc: torch.Tensor
    write_window: torch.Tensor


class PreparedRoundKey(msgspec.Struct, frozen=True, kw_only=True):
    """Host identity of a request batch, including req-pool row reuse."""

    request_slots: tuple[int, ...]
    request_generations: tuple[int, ...]


class PreparedRoundLease(msgspec.Struct, frozen=True, kw_only=True):
    """A fresh host identity plus the two owners' next-round device tensors."""

    key: PreparedRoundKey
    scheduler_sequence_lengths: torch.Tensor
    prepared: PreparedNextDraftInputs


class _PreparedNextDraftState(msgspec.Struct, frozen=True, kw_only=True):
    producer_key: Optional[PreparedRoundKey]
    has_runner_capacity: bool
    has_reserved_window: bool
    inputs: Optional[PreparedNextDraftInputs]
    scheduler_sequence_lengths: Optional[torch.Tensor]


def allocate_prepared_next_draft_inputs(
    *,
    num_requests: int,
    num_steps: int,
    window_size: int,
    with_mrope: bool,
    device,
    runner_buffers,
) -> Optional[PreparedNextDraftInputs]:
    """Bind draft fields to graph-runner inputs and own the full write window."""

    if num_requests <= 0 or num_steps <= 0 or num_steps > window_size:
        return None
    required = (
        ("req_pool_indices", num_requests),
        ("seq_lens", num_requests),
        ("positions", num_requests),
        ("out_cache_loc", num_requests * num_steps),
    )
    if any(
        not isinstance(getattr(runner_buffers, name, None), torch.Tensor)
        or getattr(runner_buffers, name).numel() < size
        for name, size in required
    ):
        return None
    if with_mrope:
        mrope = getattr(runner_buffers, "mrope_positions", None)
        if (
            not isinstance(mrope, torch.Tensor)
            or mrope.ndim != 2
            or mrope.shape[0] != 3
            or mrope.shape[1] < num_requests
        ):
            return None

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int64, device=device)

    return PreparedNextDraftInputs(
        num_requests=num_requests,
        runner_request_pool_indices=runner_buffers.req_pool_indices[:num_requests],
        runner_sequence_lengths=runner_buffers.seq_lens[:num_requests],
        positions=runner_buffers.positions[:num_requests],
        mrope_positions=(
            runner_buffers.mrope_positions[:, :num_requests] if with_mrope else None
        ),
        draft_out_cache_loc=runner_buffers.out_cache_loc[: num_requests * num_steps],
        write_window=zeros(num_requests * window_size),
    )


def prepare_next_draft_inputs(
    prepared: PreparedNextDraftInputs,
    *,
    req_pool_indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    num_steps: int,
    window_size: int,
    seq_len_fill_value: int,
) -> Optional[PreparedNextDraftInputs]:
    """Populate the next draft's fields, or decline to the stock path."""
    if not can_prepare_next_draft_inputs(
        prepared,
        req_pool_indices=req_pool_indices,
        new_seq_lens=new_seq_lens,
        req_to_token=req_to_token,
        num_steps=num_steps,
        window_size=window_size,
    ):
        return None

    from sglang.kernels.ops.speculative.cache_locs import prepare_draft_window

    prepare_draft_window(
        req_pool_indices,
        new_seq_lens,
        req_to_token,
        num_steps=num_steps,
        window=window_size,
        seq_len_fill_value=seq_len_fill_value,
        req_pool_indices_out=prepared.runner_request_pool_indices,
        seq_lens_out=prepared.runner_sequence_lengths,
        positions=prepared.positions,
        mrope=prepared.mrope_positions,
        draft_out_cache_loc=prepared.draft_out_cache_loc,
        write_virtual=prepared.write_window,
    )
    return prepared


def can_prepare_next_draft_inputs(
    prepared: PreparedNextDraftInputs,
    *,
    req_pool_indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    num_steps: int,
    window_size: int,
) -> bool:
    """Validate the Triton producer contract before it can assert or launch."""
    from sglang.srt.platforms import current_platform

    if (
        not current_platform.is_cuda()
        or not current_platform.capabilities.supports_triton
        or num_steps <= 0
        or window_size <= 0
        or num_steps > window_size
    ):
        return False

    inputs = (req_pool_indices, new_seq_lens, req_to_token)
    if any(not isinstance(t, torch.Tensor) for t in inputs):
        return False
    rows = prepared.num_requests
    if (
        rows <= 0
        or req_pool_indices.ndim != 1
        or new_seq_lens.ndim != 1
        or req_pool_indices.numel() != rows
        or new_seq_lens.numel() != rows
        or req_to_token.ndim != 2
        or req_to_token.shape[0] <= 0
        or req_to_token.shape[1] < window_size
        or any(t.dtype not in (torch.int32, torch.int64) for t in inputs)
        or any(not t.is_contiguous() for t in inputs)
        or any(not t.is_cuda for t in inputs)
    ):
        return False

    outputs = (
        prepared.runner_request_pool_indices,
        prepared.runner_sequence_lengths,
        prepared.positions,
        prepared.draft_out_cache_loc,
        prepared.write_window,
    )
    if (
        any(not isinstance(t, torch.Tensor) for t in outputs)
        or any(t.dtype != torch.int64 for t in outputs)
        or any(not t.is_contiguous() for t in outputs)
        or any(not t.is_cuda for t in outputs)
        or prepared.runner_request_pool_indices.ndim != 1
        or prepared.runner_request_pool_indices.numel() != rows
        or prepared.runner_sequence_lengths.ndim != 1
        or prepared.runner_sequence_lengths.numel() != rows
        or prepared.positions.ndim != 1
        or prepared.positions.numel() != rows
        or prepared.draft_out_cache_loc.ndim != 1
        or prepared.draft_out_cache_loc.numel() != rows * num_steps
        or prepared.write_window.ndim != 1
        or prepared.write_window.numel() != rows * window_size
    ):
        return False

    mrope = prepared.mrope_positions
    if mrope is not None and (
        not isinstance(mrope, torch.Tensor)
        or mrope.dtype != torch.int64
        or not mrope.is_cuda
        or mrope.ndim != 2
        or mrope.shape != (3, rows)
        or mrope.stride(1) != 1
    ):
        return False

    tensors = inputs + outputs + (() if mrope is None else (mrope,))
    device = req_pool_indices.device
    if any(t.device != device for t in tensors):
        return False

    # Every output is independently owned.  In particular, the draft runner's
    # static request/sequence inputs must not alias their producer inputs.
    output_ptrs = [t.data_ptr() for t in outputs]
    if mrope is not None:
        output_ptrs.append(mrope.data_ptr())
    return (
        len(output_ptrs) == len(set(output_ptrs))
        and prepared.runner_request_pool_indices.data_ptr()
        != req_pool_indices.data_ptr()
        and prepared.runner_sequence_lengths.data_ptr() != new_seq_lens.data_ptr()
    )


def can_use_prepared_next_draft_inputs(
    prepared: PreparedNextDraftInputs,
    *,
    num_requests: int,
    plans_window: bool,
    with_mrope: bool,
) -> bool:
    """Whether the prepared buffers exactly describe this draft round."""
    return (
        plans_window
        and prepared.num_requests == num_requests
        and prepared.runner_request_pool_indices.ndim == 1
        and prepared.runner_request_pool_indices.numel() == num_requests
        and (prepared.mrope_positions is not None) == with_mrope
    )


class NextDraftInputStager:
    """Resolve one fresh lease for scheduler and worker consumers."""

    def __init__(
        self,
        *,
        req_to_token_pool: Any,
        window_size: int,
        graph_sizes,
        with_mrope: bool,
        plans_window: bool = True,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.window_size = window_size
        self.graph_sizes = frozenset(graph_sizes or ())
        self.with_mrope = with_mrope
        self.plans_window = plans_window
        self._state: Optional[_PreparedNextDraftState] = None
        self._lease: Optional[PreparedRoundLease] = None
        self._resolution_attempted = False

    def note_decode(
        self,
        batch,
        *,
        prepared: Optional[PreparedNextDraftInputs],
        new_seq_lens: Optional[torch.Tensor],
    ) -> None:
        required = 3 * self.window_size
        self._state = _PreparedNextDraftState(
            producer_key=self._batch_key(batch),
            # A graph replay is not required for consumption, but the producer
            # must have bound this size to the draft runner's static buffers.
            has_runner_capacity=len(batch.reqs) in self.graph_sizes,
            has_reserved_window=all(
                req.kv.kv_allocated_len >= req.kv.kv_committed_len + required
                for req in batch.reqs
            ),
            inputs=prepared,
            scheduler_sequence_lengths=new_seq_lens,
        )
        self._lease = None
        self._resolution_attempted = False

    def forget(self) -> None:
        self._state = None
        self._lease = None
        self._resolution_attempted = False

    def _batch_key(self, batch) -> Optional[PreparedRoundKey]:
        generations = getattr(self.req_to_token_pool, "req_generation", None)
        if (
            not isinstance(generations, torch.Tensor)
            or generations.device.type != "cpu"
            or generations.ndim != 1
            or generations.dtype not in (torch.int32, torch.int64)
        ):
            return None
        try:
            slots = tuple(int(req.kv.req_pool_idx) for req in batch.reqs)
        except (TypeError, ValueError):
            return None
        if any(slot < 0 or slot >= generations.numel() for slot in slots):
            return None
        return PreparedRoundKey(
            request_slots=slots,
            request_generations=tuple(int(generations[slot]) for slot in slots),
        )

    def _batch_uses_mrope(self, batch) -> bool:
        return self.with_mrope and all(
            item is None for item in getattr(batch, "multimodal_inputs", ())
        )

    def _resolve(self, batch) -> Optional[PreparedRoundLease]:
        if self._resolution_attempted:
            return self._lease
        self._resolution_attempted = True
        state = self._state
        consumer_key = self._batch_key(batch)
        if (
            state is None
            or state.producer_key is None
            or consumer_key != state.producer_key
            or not state.has_runner_capacity
            or not state.has_reserved_window
            or not all(
                req.kv.kv_allocated_len
                >= req.kv.kv_committed_len + 3 * self.window_size
                for req in batch.reqs
            )
            or state.inputs is None
            or state.scheduler_sequence_lengths is None
            or state.scheduler_sequence_lengths.ndim != 1
            or state.scheduler_sequence_lengths.numel() != len(batch.reqs)
            or not can_use_prepared_next_draft_inputs(
                state.inputs,
                num_requests=len(batch.reqs),
                plans_window=self.plans_window,
                with_mrope=self._batch_uses_mrope(batch),
            )
            or state.inputs.runner_sequence_lengths.data_ptr()
            == state.scheduler_sequence_lengths.data_ptr()
        ):
            self._state = None
            return None
        self._lease = PreparedRoundLease(
            key=consumer_key,
            scheduler_sequence_lengths=state.scheduler_sequence_lengths,
            prepared=state.inputs,
        )
        return self._lease

    def take_scheduler_sequence_lengths(self, batch) -> Optional[torch.Tensor]:
        lease = self._resolve(batch)
        return None if lease is None else lease.scheduler_sequence_lengths

    def take_inputs(self, batch) -> Optional[PreparedNextDraftInputs]:
        lease = self._resolve(batch)
        prepared = None if lease is None else lease.prepared
        self.forget()
        return prepared


def set_prepared_next_draft_reserve_enabled(enabled: bool) -> None:
    """Resolve the pre-pool upper bound to the graph initialization result."""
    assert not enabled or prepared_next_draft_reserve_configured()
    get_resources().prepared_next_draft_reserve_enabled = enabled


def prepared_next_draft_reserve_enabled() -> bool:
    """Whether row sizing or live allocation needs the third KV window."""
    resolved = get_resources().prepared_next_draft_reserve_enabled
    return (
        prepared_next_draft_reserve_configured()
        if resolved is None
        else resolved
    )


class PreparedNextDraftProducer:
    """One producer shared by captured post-verify graphs and eager rounds."""

    _EAGER_RING_DEPTH = 2

    def __init__(
        self,
        *,
        num_steps: int,
        window_size: int,
        seq_len_fill_value: int,
        with_mrope: bool,
        req_to_token: torch.Tensor,
        runner_buffers: Any,
        capture_sizes,
    ) -> None:
        self.num_steps = num_steps
        self.window_size = window_size
        self.seq_len_fill_value = seq_len_fill_value
        self.with_mrope = with_mrope
        self.req_to_token = req_to_token
        self.runner_buffers = runner_buffers
        self.capture_sizes = frozenset(capture_sizes or ())
        self._eager_rings: dict[tuple[int, torch.device], list[Any]] = {}

    def allocate(
        self, *, num_requests: int, device
    ) -> Optional[PreparedNextDraftInputs]:
        if num_requests not in self.capture_sizes:
            return None
        return allocate_prepared_next_draft_inputs(
            num_requests=num_requests,
            num_steps=self.num_steps,
            window_size=self.window_size,
            with_mrope=self.with_mrope,
            device=device,
            runner_buffers=self.runner_buffers,
        )

    def prepare(
        self,
        prepared: Optional[PreparedNextDraftInputs],
        *,
        req_pool_indices: torch.Tensor,
        scheduler_sequence_lengths: torch.Tensor,
    ) -> Optional[PreparedNextDraftInputs]:
        if prepared is None:
            return None
        return prepare_next_draft_inputs(
            prepared,
            req_pool_indices=req_pool_indices,
            new_seq_lens=scheduler_sequence_lengths,
            req_to_token=self.req_to_token,
            num_steps=self.num_steps,
            window_size=self.window_size,
            seq_len_fill_value=self.seq_len_fill_value,
        )

    def prepare_eager(
        self,
        *,
        req_pool_indices: torch.Tensor,
        scheduler_sequence_lengths: torch.Tensor,
    ) -> Optional[PreparedNextDraftInputs]:
        num_requests = int(req_pool_indices.numel())
        key = (num_requests, req_pool_indices.device)
        ring = self._eager_rings.get(key)
        if ring is None:
            prepared = [
                self.allocate(num_requests=num_requests, device=req_pool_indices.device)
                for _ in range(self._EAGER_RING_DEPTH)
            ]
            ring = self._eager_rings[key] = [prepared, 0]
        prepared = ring[0][ring[1]]
        ring[1] = (ring[1] + 1) % self._EAGER_RING_DEPTH
        return self.prepare(
            prepared,
            req_pool_indices=req_pool_indices,
            scheduler_sequence_lengths=scheduler_sequence_lengths,
        )
