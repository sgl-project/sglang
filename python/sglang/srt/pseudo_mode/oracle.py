"""CPU-side deterministic oracle for pseudo-mode testing.

The oracle answers two independent questions about a given request:

- ``predict_input_token(req_id, position)``: what token sglang must feed
  ``model.forward`` at ``(req_id, position)``. Pure lookup over
  ``origin_input_ids`` (prompt range) and ``output_history`` (decoded
  range). Zero internal scheduler awareness.
- ``predict_output_token(req_id, step)``: what token the sampler must
  emit at ``(req_id, step)``. Pure function of ``(seed, req_id, step)``
  modulo ``vocab_size``. Zero internal state read.

The only mutator is :meth:`PseudoOracle.commit_step`, called after a
forward step actually produces an output token, so the next
``predict_input_token`` query can resolve through ``output_history``.

The single piece of scheduler state the oracle mirrors is the
``req_pool_idx -> req_id`` map (sglang's batch composition uses integer
req-pool indices; tests use human-readable string ids). The map is
maintained by install.py via admit / finish hooks.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary_ref import splitmix64_mix

if TYPE_CHECKING:
    from sglang.srt.kv_cache_canary.host_state import BatchPlan
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_U64_MASK = (1 << 64) - 1


@dataclass(slots=True, kw_only=True)
class _ReqState:
    """Per-request oracle state.

    ``committed_chunks`` is the number of prompt tokens already supplied
    via prior chunked-prefill forwards; it advances only when
    :meth:`PseudoOracle.register_chunk_commit` is called. ``in_decode``
    flips to True once the request leaves the extend phase entirely
    (committed_chunks == len(origin_input_ids)).
    """

    origin_input_ids: List[int]
    output_history: List[int] = field(default_factory=list)
    eos_at: int
    committed_chunks: int = 0
    in_decode: bool = False


class PseudoOracle:
    """Contract-only oracle. Mirrors what tokens SHOULD flow, not WHEN/HOW scheduler schedules.

    Determinism contract: given identical ``(seed, vocab_size, eos_id)``
    plus an identical sequence of admit / commit_step / finish calls,
    two oracle instances on two machines produce byte-identical answers
    to every ``predict_*`` query.
    """

    def __init__(
        self,
        *,
        seed: int = 0xC0FFEE,
        vocab_size: int,
        eos_id: int,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(
                f"pseudo-oracle: vocab_size must be positive, got {vocab_size}"
            )
        self._seed: int = seed & _U64_MASK
        self._vocab_size: int = vocab_size
        self._eos_id: int = eos_id
        self._reqs: Dict[str, _ReqState] = {}
        self._req_pool_to_id: Dict[int, str] = {}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def admit(
        self,
        *,
        req_id: str,
        origin_input_ids: List[int],
        max_new_tokens: int,
        eos_at: Optional[int] = None,
    ) -> None:
        if req_id in self._reqs:
            raise ValueError(f"pseudo-oracle: req_id {req_id!r} already admitted")
        if not origin_input_ids:
            raise ValueError(
                f"pseudo-oracle: admit {req_id!r} with empty origin_input_ids"
            )
        if max_new_tokens < 0:
            raise ValueError(
                f"pseudo-oracle: admit {req_id!r} max_new_tokens {max_new_tokens} < 0"
            )
        effective_eos_at = eos_at if eos_at is not None else max_new_tokens
        if effective_eos_at < 0:
            raise ValueError(
                f"pseudo-oracle: admit {req_id!r} eos_at {effective_eos_at} < 0"
            )
        self._reqs[req_id] = _ReqState(
            origin_input_ids=list(origin_input_ids),
            eos_at=effective_eos_at,
        )

    def register_req_pool_mapping(self, *, req_pool_idx: int, req_id: str) -> None:
        if req_id not in self._reqs:
            raise KeyError(
                f"pseudo-oracle: register_req_pool_mapping for unknown req_id {req_id!r}"
            )
        existing = self._req_pool_to_id.get(req_pool_idx)
        if existing is not None and existing != req_id:
            raise ValueError(
                f"pseudo-oracle: req_pool_idx {req_pool_idx} already maps to "
                f"{existing!r}, cannot re-map to {req_id!r}"
            )
        self._req_pool_to_id[req_pool_idx] = req_id

    def predict_input_token(self, *, req_id: str, position: int) -> int:
        state = self._reqs[req_id]
        prefill_len = len(state.origin_input_ids)
        if position < 0:
            raise ValueError(
                f"pseudo-oracle: predict_input_token {req_id!r} position {position} < 0"
            )
        if position < prefill_len:
            return state.origin_input_ids[position]
        offset = position - prefill_len
        if offset >= len(state.output_history):
            raise IndexError(
                f"pseudo-oracle: predict_input_token {req_id!r} position {position} "
                f"requires output_history[{offset}] but only "
                f"{len(state.output_history)} tokens committed"
            )
        return state.output_history[offset]

    def predict_output_token(self, *, req_id: str, step: int) -> int:
        state = self._reqs[req_id]
        if step < 0:
            raise ValueError(
                f"pseudo-oracle: predict_output_token {req_id!r} step {step} < 0"
            )
        if step >= state.eos_at:
            return self._eos_id
        return self._output_token_hash(req_id=req_id, step=step) % self._vocab_size

    def predict_input_tokens_for_plan(
        self,
        *,
        plan: "BatchPlan",
        forward_batch: "ForwardBatch",
    ) -> Tuple[List[int], List[int]]:
        """Vectorised per-write-entry oracle answers.

        Critically, ``expected_positions`` is recomputed *independently*
        from per-req committed state — never copied from
        ``plan.write_positions`` or ``forward_batch.positions``. That
        gives the head kernel a real witness for
        ``INPUT_POSITION_MISMATCH``.
        """
        is_decode = forward_batch.forward_mode.is_decode()
        expected_tokens: List[int] = []
        expected_positions: List[int] = []
        for wreq_idx in range(plan.num_write_reqs):
            start = plan.write_req_entry_starts[wreq_idx]
            count = plan.write_req_entry_counts[wreq_idx]
            if count == 0:
                continue
            req_pool_idx = plan.write_req_ids[start]
            req_id = self._req_pool_to_id[req_pool_idx]
            expected_k_req = self._expected_k_req(req_id=req_id, is_decode=is_decode)
            for j in range(count):
                pos = expected_k_req + j
                expected_positions.append(pos)
                expected_tokens.append(
                    self.predict_input_token(req_id=req_id, position=pos)
                )
        return expected_tokens, expected_positions

    def predict_next_tokens_for_active_batch(
        self,
        *,
        forward_batch: "ForwardBatch",
        device: torch.device,
    ) -> torch.Tensor:
        req_pool_indices = forward_batch.req_pool_indices.tolist()
        out = torch.empty(len(req_pool_indices), dtype=torch.int64, device=device)
        for i, req_pool_idx in enumerate(req_pool_indices):
            req_id = self._req_pool_to_id[int(req_pool_idx)]
            step = len(self._reqs[req_id].output_history)
            out[i] = self.predict_output_token(req_id=req_id, step=step)
        return out

    def register_chunk_commit(self, *, req_id: str, chunk_size: int) -> None:
        """Advance per-req chunk bookkeeping after a chunked-prefill step.

        Called after sglang has consumed ``chunk_size`` prompt tokens
        for ``req_id`` in an extend forward. The final chunk transition
        sets ``in_decode = True``; subsequent decode steps don't call
        this hook.
        """
        if chunk_size <= 0:
            raise ValueError(
                f"pseudo-oracle: register_chunk_commit {req_id!r} chunk_size "
                f"{chunk_size} must be > 0"
            )
        state = self._reqs[req_id]
        new_committed = state.committed_chunks + chunk_size
        prefill_len = len(state.origin_input_ids)
        if new_committed > prefill_len:
            raise ValueError(
                f"pseudo-oracle: register_chunk_commit {req_id!r} would advance "
                f"committed_chunks to {new_committed} but origin_input_ids has "
                f"only {prefill_len} tokens"
            )
        state.committed_chunks = new_committed
        if new_committed == prefill_len:
            state.in_decode = True

    def commit_step(self, *, req_id: str, output_token: int) -> None:
        state = self._reqs[req_id]
        if not state.in_decode and state.committed_chunks != len(state.origin_input_ids):
            raise RuntimeError(
                f"pseudo-oracle: commit_step {req_id!r} before all prompt chunks "
                f"committed ({state.committed_chunks}/{len(state.origin_input_ids)})"
            )
        state.in_decode = True
        state.output_history.append(int(output_token))

    def finish(self, *, req_id: str) -> None:
        if req_id not in self._reqs:
            raise KeyError(f"pseudo-oracle: finish unknown req_id {req_id!r}")
        stale_pool_indices = [
            idx for idx, mapped in self._req_pool_to_id.items() if mapped == req_id
        ]
        for idx in stale_pool_indices:
            del self._req_pool_to_id[idx]

    def _expected_k_req(self, *, req_id: str, is_decode: bool) -> int:
        state = self._reqs[req_id]
        if is_decode:
            return len(state.origin_input_ids) + len(state.output_history) - 1
        return state.committed_chunks

    def _output_token_hash(self, *, req_id: str, step: int) -> int:
        req_id_hash = self._hash_req_id(req_id)
        chained = splitmix64_mix(
            prev_hash=self._seed, token_id=req_id_hash, position=0
        )
        return splitmix64_mix(prev_hash=chained, token_id=step, position=0)

    @staticmethod
    def _hash_req_id(req_id: str) -> int:
        digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)
