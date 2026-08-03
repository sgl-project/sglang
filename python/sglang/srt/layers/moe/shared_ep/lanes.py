"""Deterministic SharedEP state-lane geometry and resource keys."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Every lane owns two VMM mappings and two system-scope GPU epochs. Keep the
# initial production envelope small and explicit instead of allowing an
# unbounded speculative setting to multiply persistent peer mappings.
SHARED_EP_MAX_STATE_LANES = 8

_MODEL_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class SharedEpLaneProtocol:
    """Static lane dimensions captured by every MoE layer on every rank."""

    tbo_width: int
    generation_width: int

    @property
    def lane_count(self) -> int:
        return self.tbo_width * self.generation_width

    def lane_id(
        self,
        *,
        generation_index: int = 0,
        tbo_subbatch_index: int | None = None,
    ) -> int:
        subbatch = 0 if tbo_subbatch_index is None else int(tbo_subbatch_index)
        generation = int(generation_index)
        if not 0 <= subbatch < self.tbo_width:
            raise ValueError(
                "SharedEP TBO subbatch index is outside the admitted lane "
                f"protocol: {subbatch} not in [0, {self.tbo_width})"
            )
        if not 0 <= generation < self.generation_width:
            raise ValueError(
                "SharedEP generation index is outside the admitted lane "
                f"protocol: {generation} not in [0, {self.generation_width})"
            )
        return generation * self.tbo_width + subbatch


def _max_speculative_draft_tokens(server_args) -> int:
    if not getattr(server_args, "speculative_algorithm", None):
        return 1
    value = getattr(server_args, "max_speculative_num_draft_tokens", None)
    if value is None:
        value = getattr(server_args, "speculative_num_draft_tokens", None)
    return max(1, int(value or 1))


def compute_shared_ep_lane_protocol(server_args) -> SharedEpLaneProtocol:
    """Derive a rank-identical, fixed-size lane protocol from server config."""

    protocol = SharedEpLaneProtocol(
        tbo_width=2 if getattr(server_args, "enable_two_batch_overlap", False) else 1,
        generation_width=_max_speculative_draft_tokens(server_args),
    )
    if protocol.lane_count > SHARED_EP_MAX_STATE_LANES:
        raise ValueError(
            "SharedEP state-lane requirement exceeds the fixed release cap: "
            f"{protocol.tbo_width} TBO lane(s) * "
            f"{protocol.generation_width} generation lane(s) = "
            f"{protocol.lane_count}, cap is {SHARED_EP_MAX_STATE_LANES}. "
            "Reduce --speculative-num-draft-tokens or disable TBO."
        )
    return protocol


def validate_shared_ep_model_namespace(model_namespace: str) -> str:
    namespace = str(model_namespace)
    if not _MODEL_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "SharedEP model namespace must be a deterministic lowercase "
            f"identifier, got {namespace!r}"
        )
    return namespace


def shared_ep_state_resource_key(
    *,
    runtime_name: str,
    profile_name: str,
    ep_size: int,
    model_namespace: str,
    lane_id: int,
) -> str:
    """Build the process resource key without rank-local hashes or object IDs."""

    namespace = validate_shared_ep_model_namespace(model_namespace)
    lane = int(lane_id)
    if lane < 0:
        raise ValueError(f"SharedEP lane ID must be non-negative, got {lane}")
    return (
        f"shared_ep:{runtime_name}:{profile_name}:ep{int(ep_size)}:"
        f"model={namespace}:lane={lane}"
    )
