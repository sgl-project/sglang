"""Register the SUFFIX speculative-decoding algorithm.

SUFFIX is registered as a *plugin* algorithm (``SpeculativeAlgorithm.register``)
rather than a builtin enum value. ``_SuffixLike.is_ngram()`` returns ``True`` so
SUFFIX transparently reuses every existing NGRAM dispatch branch in the
scheduler / cuda-graph runner / model runner — the verify and KV-cache path is
identical; only the draft source differs (see ``SuffixWorker``). The factory is
imported lazily so ``arctic_inference`` is required only when SUFFIX is actually
selected.
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


SpeculativeAlgorithm.register(
    "SUFFIX",
    supports_overlap=True,
    validate_server_args=_validate_suffix_args,
    spec_class=_SuffixLike,
)(_suffix_factory)
