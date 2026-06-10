"""Register the SUFFIX and HYBRID_SUFFIX_MTP speculative-decoding algorithms.

Both are registered as *plugin* algorithms via ``SpeculativeAlgorithm.register``
rather than builtin enum values.

- ``_SuffixLike.is_ngram()`` returns ``True`` so SUFFIX transparently reuses
  every existing NGRAM dispatch branch in the scheduler / cuda-graph runner /
  model runner — the verify and KV-cache path is identical; only the draft
  source differs (see ``SuffixWorker``).
- ``_HybridLike.is_ngram()`` and ``is_eagle()`` both return ``True`` so the
  hybrid worker can dispatch internally to either the SUFFIX (NGRAM-style) or
  MTP (EAGLE-style) path on a per-batch basis. The decision is made by the
  per-batch selector defined in
  :mod:`sglang.srt.speculative.hybrid_backend_selector`.

Factories are imported lazily so ``arctic_inference`` is required only when
SUFFIX / HYBRID_SUFFIX_MTP is actually selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class _SuffixLike(CustomSpecAlgo):
    """SUFFIX dispatches like NGRAM (shared verify / KV-cache path)."""

    def is_ngram(self) -> bool:
        return True

    def is_suffix(self) -> bool:
        return True

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ):
        # CustomSpecAlgo base does not provide this (the scheduler calls it
        # unconditionally); mirror SpeculativeAlgorithm.create_future_map.
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)


def _suffix_factory(server_args: "ServerArgs") -> Type:
    # With overlap scheduling enabled, dispatch the V2 worker (BaseSpecWorker
    # contract — takes the per-step batch and runs alongside the scheduler's
    # plan/predict streams). Otherwise stay on the V1 NGRAMWorker subclass.
    if not server_args.disable_overlap_schedule:
        from sglang.srt.speculative.suffix_worker_v2 import SuffixWorkerV2

        return SuffixWorkerV2
    from sglang.srt.speculative.suffix_worker import SuffixWorker

    return SuffixWorker


def _validate_suffix_args(server_args: "ServerArgs") -> None:
    required = (
        "speculative_suffix_max_tree_depth",
        "speculative_suffix_max_cached_requests",
        "speculative_suffix_max_spec_factor",
        "speculative_suffix_min_token_prob",
    )
    missing = [a for a in required if not hasattr(server_args, a)]
    if missing:
        raise ValueError(f"SUFFIX speculative decoding requires server args {missing}.")


class _HybridLike(CustomSpecAlgo):
    """HYBRID_SUFFIX_MTP behaves like NGRAM on the SUFFIX path and like EAGLE
    on the MTP path. Returning True for both means existing builtin-only
    branches still trigger; the hybrid worker dispatches per-batch internally
    via :class:`HybridBackendSelector`.
    """

    def is_ngram(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        # The MTP path is EAGLE-based (uses the MTP head as draft model);
        # return True so EAGLE-specific paths (draft_extend cuda graph, MoE
        # A2A, ...) still trigger for those steps.
        return True

    def is_hybrid_suffix_mtp(self) -> bool:
        return True

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)


def _hybrid_factory(server_args: "ServerArgs") -> Type:
    # HYBRID_SUFFIX_MTP is V2-only: it inherits EAGLEWorkerV2 and dispatches
    # per-batch between SUFFIX V2 and MTP V2 paths. A V1 variant is not
    # provided because the SUFFIX V1 path (NGRAMWorker subclass) cannot live
    # under the V2 overlap scheduler, and there is no benefit to a hybrid
    # that's V1-only.
    from sglang.srt.speculative.hybrid_suffix_mtp_worker_v2 import (
        HybridSuffixMTPWorkerV2,
    )

    return HybridSuffixMTPWorkerV2


# Install default ``is_*()`` stubs on SpeculativeAlgorithm + CustomSpecAlgo so
# code that probes ``spec_algorithm.is_suffix()`` / ``is_hybrid_suffix_mtp()``
# on builtin enum values returns False without AttributeError. Our plugin
# subclasses above override these to True. Idempotent.
def _is_false(self):
    return False


for _cls in (SpeculativeAlgorithm, CustomSpecAlgo):
    if not hasattr(_cls, "is_suffix"):
        _cls.is_suffix = _is_false
    if not hasattr(_cls, "is_hybrid_suffix_mtp"):
        _cls.is_hybrid_suffix_mtp = _is_false


SpeculativeAlgorithm.register(
    "SUFFIX",
    supports_overlap=True,
    validate_server_args=_validate_suffix_args,
    spec_class=_SuffixLike,
)(_suffix_factory)

SpeculativeAlgorithm.register(
    "HYBRID_SUFFIX_MTP",
    supports_overlap=True,
    validate_server_args=_validate_suffix_args,
    spec_class=_HybridLike,
)(_hybrid_factory)
