# SPDX-License-Identifier: Apache-2.0
"""
Candidate-trajectory execution contract for opt-in multi-candidate world/action
models (phase 1 of the proposal: internal types and validation, no model
enabled).

This lets a model/pipeline adapter declare that several independent
denoising outputs generated from one conditioning context are *candidates for
one logical prediction* rather than independent public media outputs. The
runtime owns candidate identity, RNG-stream independence, atomic batch
admission, and the validated reduction step; the model owns the reduction
rule itself (registered via :func:`register_candidate_reducer`).

This is intentionally narrow, matching the RFC:
  - it does not average images or videos;
  - it does not change default ``num_outputs_per_prompt`` behavior; it only
    formalizes an opt-in path for models with a validated reducer;
  - ``count=1`` must be identical to the existing (non-candidate) path.

The existing per-request RNG handling in ``LatentPreparationStage`` already
draws request-owned random latents separately (``Req.generator`` may be a
list of ``torch.Generator``, one per output) and only batches deterministic
work afterward -- :func:`derive_candidate_generators` follows the same
precedent for the candidate axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_KNOWN_SEED_POLICIES = frozenset({"per_candidate", "shared"})

# A reducer receives a tensor whose leading dimension is the candidate axis
# (shape ``(count, ...)``) and returns a reduced tensor with that axis
# removed (shape ``(...)``).
CandidateReducerFn = Callable[[torch.Tensor], torch.Tensor]

_REDUCER_REGISTRY: Dict[str, CandidateReducerFn] = {}


class CandidateContractError(RuntimeError):
    """Raised when the candidate-trajectory contract is violated."""


def register_candidate_reducer(
    name: str,
) -> Callable[[CandidateReducerFn], CandidateReducerFn]:
    """Register a model-owned reduction rule under ``name``.

    The framework intentionally does not hard-code a universal action
    reducer beyond ``"mean"``: quality validation for any given reducer is
    model-specific, per the RFC's correctness/quality-gates section.
    """

    def _decorator(fn: CandidateReducerFn) -> CandidateReducerFn:
        if name in _REDUCER_REGISTRY:
            raise CandidateContractError(f"reducer {name!r} is already registered")
        _REDUCER_REGISTRY[name] = fn
        return fn

    return _decorator


def _reducer_none(candidates: torch.Tensor) -> torch.Tensor:
    if candidates.shape[0] != 1:
        raise CandidateContractError(
            "reducer 'none' requires exactly one candidate; got "
            f"{candidates.shape[0]}. Use a real reducer for count > 1."
        )
    return candidates[0]


def _reducer_mean(candidates: torch.Tensor) -> torch.Tensor:
    return candidates.mean(dim=0)


_REDUCER_REGISTRY["none"] = _reducer_none
_REDUCER_REGISTRY["mean"] = _reducer_mean


def get_candidate_reducer(name: str) -> CandidateReducerFn:
    try:
        return _REDUCER_REGISTRY[name]
    except KeyError:
        raise CandidateContractError(
            f"unknown reducer {name!r}; registered reducers: {sorted(_REDUCER_REGISTRY)}"
        ) from None


@dataclass(frozen=True)
class CandidateTrajectorySpec:
    """Declares how many candidate trajectories to generate and how to
    reduce them to one served result.

    Attributes:
        count: Number of independent candidate trajectories to generate from
            one conditioning context. ``count=1`` must preserve the existing
            (non-candidate) output exactly.
        reducer: Name of a registered reducer ("none", "mean", or a
            model-registered reducer) applied to the candidate axis.
        return_candidates: If True, the raw per-candidate tensors are
            returned alongside the reduced result. Off by default, per the
            RFC's "don't expose raw candidates by default" guidance.
        seed_policy: "per_candidate" gives every candidate an independent,
            stably-derived RNG stream (the default). "shared" gives every
            candidate the same stream, for models whose declared
            equivalence rule requires shared randomness.
    """

    count: int = 1
    reducer: str = "none"
    return_candidates: bool = False
    seed_policy: str = "per_candidate"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.count == 1 and self.reducer != "none":
            raise ValueError(
                "count=1 must use reducer='none' to preserve the existing path"
            )
        if self.count > 1 and self.reducer == "none":
            raise ValueError("count > 1 requires a real reducer, not 'none'")
        if self.seed_policy not in _KNOWN_SEED_POLICIES:
            raise ValueError(
                f"unknown seed_policy {self.seed_policy!r}; expected one of "
                f"{sorted(_KNOWN_SEED_POLICIES)}"
            )
        # Fail fast on an unregistered reducer rather than at reduction time.
        get_candidate_reducer(self.reducer)


@dataclass(frozen=True)
class CandidateGroup:
    """Identity of one logical request's candidate batch.

    ``candidate_ids`` is stable across dynamic batching and output slicing:
    callers key per-candidate state (seeds, partial reducer state, ...) by
    ``(request_id, candidate_id)`` so unrelated co-batched requests never
    collide.
    """

    request_id: str
    candidate_ids: Tuple[int, ...]
    public_output: str = "action"

    def __post_init__(self) -> None:
        if len(self.candidate_ids) == 0:
            raise ValueError("candidate_ids must be non-empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError(f"candidate_ids must be unique, got {self.candidate_ids}")
        if list(self.candidate_ids) != sorted(self.candidate_ids):
            raise ValueError(
                f"candidate_ids must be sorted for stable ordering, got {self.candidate_ids}"
            )


def build_candidate_group(
    request_id: str, spec: CandidateTrajectorySpec, public_output: str = "action"
) -> CandidateGroup:
    """Build the identity for one request's candidate batch from its spec."""
    return CandidateGroup(
        request_id=request_id,
        candidate_ids=tuple(range(spec.count)),
        public_output=public_output,
    )


def _mix_seed(base_seed: int, candidate_id: int) -> int:
    """Deterministically derive a per-candidate seed from (base_seed, candidate_id).

    Uses a splitmix64-style bit mixer rather than plain addition so nearby
    base seeds or candidate indices don't produce correlated streams.
    Restricted to torch.Generator's accepted 64-bit unsigned range.
    """
    MASK64 = (1 << 64) - 1
    z = (base_seed + candidate_id * 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return z & MASK64


def derive_candidate_generators(
    base_seed: int,
    spec: CandidateTrajectorySpec,
    device: torch.device | str = "cpu",
) -> list:
    """Create one independent ``torch.Generator`` per declared candidate.

    Under ``seed_policy="per_candidate"`` each candidate ``i`` receives a
    stream derived from ``(base_seed, i)`` per the RFC's RNG/equivalence
    rules: for a fixed seed list, batched execution must reproduce the same
    candidate set as sequential execution, and candidate ordering is stable.
    Under ``seed_policy="shared"`` every candidate gets the same stream.
    """
    generators = []
    for candidate_id in range(spec.count):
        gen = torch.Generator(device=device)
        if spec.seed_policy == "per_candidate":
            gen.manual_seed(_mix_seed(base_seed, candidate_id))
        else:
            gen.manual_seed(base_seed & ((1 << 64) - 1))
        generators.append(gen)
    return generators


def validate_candidate_admission(
    spec: CandidateTrajectorySpec, max_execution_batch: int
) -> None:
    """Admit a candidate group atomically, or fail clearly.

    Phase 1 does not support microbatching a candidate group: either the
    whole group fits the configured execution batch, or admission fails so
    the caller can apply a documented sequential fallback. Silently
    truncating the group would violate the "one logical request" contract.
    """
    if spec.count > max_execution_batch:
        raise CandidateContractError(
            f"candidate group of size {spec.count} does not fit the configured "
            f"execution batch of {max_execution_batch}; group admission must be "
            "atomic in phase 1 (no microbatching). Use a sequential fallback."
        )


def reduce_candidates(
    candidates: torch.Tensor, spec: CandidateTrajectorySpec
) -> torch.Tensor:
    """Apply ``spec``'s reducer along the candidate axis (dim 0).

    Raises if ``candidates.shape[0] != spec.count`` so a partial/failed
    candidate group can never silently produce a "reduced" result -- per
    the RFC's cancellation/failure correctness gate, a failed candidate
    must not yield a partial aggregate.
    """
    if candidates.shape[0] != spec.count:
        raise CandidateContractError(
            f"expected {spec.count} candidates on dim 0, got {candidates.shape[0]}; "
            "a partial candidate group must not be reduced"
        )
    reducer = get_candidate_reducer(spec.reducer)
    return reducer(candidates)
