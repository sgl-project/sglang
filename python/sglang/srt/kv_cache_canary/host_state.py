from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List

import torch

from sglang.jit_kernel.kv_cache_canary import VIOLATION_FIELDS
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.fingerprint import mix_step

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RequestState:
    prev_hash_tail: int
    k_req: int


class CanaryHostState:
    """Per-request canary high-water-mark and chain-hash tail state.

    Indexed by ``req_pool_idx`` (the stable per-request index assigned by
    ``ReqToTokenPool``). Lives entirely on the host; the GPU side only sees the
    expected fingerprints we compute here.
    """

    def __init__(self, *, config: CanaryConfig, num_req_slots: int) -> None:
        self._config = config
        self._num_req_slots = num_req_slots
        self._states: Dict[int, _RequestState] = {}
        self._lock = threading.Lock()

    def reset_request(self, req_pool_idx: int) -> None:
        with self._lock:
            self._states.pop(req_pool_idx, None)

    def _get_or_init(self, req_pool_idx: int) -> _RequestState:
        state = self._states.get(req_pool_idx)
        if state is None:
            state = _RequestState(prev_hash_tail=self._config.seed, k_req=0)
            self._states[req_pool_idx] = state
        return state

    def plan_batch(
        self,
        *,
        req_pool_indices: List[int],
        req_token_counts: List[int],
        req_start_positions: List[int],
        input_tokens_per_req: List[List[int]],
    ) -> "BatchPlan":
        """Compute expected (req_id, token_id, position, prev_hash) per token slot.

        Parameters
        ----------
        req_pool_indices : per-request stable index
        req_token_counts : number of new tokens (positions) for each req this step
        req_start_positions : first new position for each req this step (decode → seq_len; extend → prefix_len)
        input_tokens_per_req : the token ids being inserted for each req (length = req_token_counts[i])
        """
        if len(req_pool_indices) != len(req_token_counts):
            raise RuntimeError(
                "kv-canary: req_pool_indices and req_token_counts length mismatch"
            )
        if len(req_pool_indices) != len(input_tokens_per_req):
            raise RuntimeError(
                "kv-canary: req_pool_indices and input_tokens_per_req length mismatch"
            )

        total_slots = sum(req_token_counts)
        expected_req_ids = [0] * total_slots
        expected_token_ids = [0] * total_slots
        expected_positions = [0] * total_slots
        expected_prev_hashes = [0] * total_slots
        verify_mask = [0] * total_slots
        # For host bookkeeping: post-plan, what each request's state should advance to.
        next_state: Dict[int, _RequestState] = {}

        cursor = 0
        with self._lock:
            for req_pool_idx, count, start_pos, tokens in zip(
                req_pool_indices,
                req_token_counts,
                req_start_positions,
                input_tokens_per_req,
            ):
                state = self._get_or_init(req_pool_idx)
                prev_hash = state.prev_hash_tail
                k_req = state.k_req
                for offset in range(count):
                    pos = start_pos + offset
                    token_id = tokens[offset]
                    expected_req_ids[cursor] = req_pool_idx
                    expected_token_ids[cursor] = token_id
                    expected_positions[cursor] = pos
                    expected_prev_hashes[cursor] = _to_signed_int64(prev_hash)
                    verify_mask[cursor] = 1 if pos < k_req else 0
                    prev_hash = mix_step(prev_hash, token_id, pos)
                    cursor += 1
                new_k_req = max(k_req, start_pos + count)
                next_state[req_pool_idx] = _RequestState(
                    prev_hash_tail=prev_hash, k_req=new_k_req
                )

        return BatchPlan(
            expected_req_ids=expected_req_ids,
            expected_token_ids=expected_token_ids,
            expected_positions=expected_positions,
            expected_prev_hashes=expected_prev_hashes,
            verify_mask=verify_mask,
            next_state=next_state,
        )

    def commit_plan(self, plan: "BatchPlan") -> None:
        with self._lock:
            for req_pool_idx, new_state in plan.next_state.items():
                self._states[req_pool_idx] = new_state


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlan:
    expected_req_ids: List[int]
    expected_token_ids: List[int]
    expected_positions: List[int]
    expected_prev_hashes: List[int]
    verify_mask: List[int]
    next_state: Dict[int, _RequestState]


def _to_signed_int64(unsigned_value: int) -> int:
    mask = (1 << 64) - 1
    value = unsigned_value & mask
    if value >= (1 << 63):
        value -= 1 << 64
    return value


@dataclass(slots=True)
class CanaryDeviceState:
    """GPU-resident state shared across head/tail kernel invocations."""

    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor
    first_violation: torch.Tensor
    first_violation_set: torch.Tensor
    is_errored: torch.Tensor
    slot_run_counter_head: torch.Tensor
    slot_run_counter_tail: torch.Tensor
    kernel_run_counter_head: torch.Tensor
    kernel_run_counter_tail: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryDeviceState":
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
            first_violation=torch.zeros(
                VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            first_violation_set=torch.zeros(1, dtype=torch.int32, device=device),
            is_errored=torch.zeros(1, dtype=torch.uint8, device=device),
            slot_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            slot_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
        )
