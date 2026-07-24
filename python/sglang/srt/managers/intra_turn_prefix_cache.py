from __future__ import annotations

import dataclasses
from array import array
from typing import List, Optional

from sglang.srt.managers.schedule_batch import Req


@dataclasses.dataclass
class IntraTurnPrefixStagePlan:
    """One synthetic-prefix stage shared by a group of requests."""

    shared_token_ids: array[int]
    reqs: List[Req]
    common_prefix_len: int
    original_prefix_len: int
    estimated_saved_tokens: int

    @property
    def stage_new_tokens(self) -> int:
        return self.common_prefix_len - self.original_prefix_len


@dataclasses.dataclass
class IntraTurnPrefixPlan:
    stage: IntraTurnPrefixStagePlan


def build_intra_turn_prefix_plan(
    reqs: List[Req],
    *,
    min_shared_new_tokens: int,
) -> Optional[IntraTurnPrefixPlan]:
    """Find one shared uncached prefix worth materializing in a synthetic stage."""

    candidates = [req for req in reqs if _is_eligible_req(req)]
    if len(candidates) < 2:
        return None

    best_stage: Optional[IntraTurnPrefixStagePlan] = None
    fill_ids_by_req = {id(req): req.get_fill_ids() for req in candidates}

    for i, seed in enumerate(candidates):
        seed_ids = fill_ids_by_req[id(seed)]
        seed_prefix_len = len(seed.prefix_indices)
        for other in candidates[i + 1 :]:
            if seed.extra_key != other.extra_key:
                continue

            other_prefix_len = len(other.prefix_indices)
            if seed_prefix_len != other_prefix_len:
                continue

            other_ids = fill_ids_by_req[id(other)]
            common_len = _common_prefix_len(seed_ids, other_ids)
            shared_new_tokens = common_len - seed_prefix_len
            if shared_new_tokens < min_shared_new_tokens:
                continue

            group = [
                req
                for req in candidates
                if (
                    req.extra_key == seed.extra_key
                    and len(req.prefix_indices) == seed_prefix_len
                    and _has_prefix(fill_ids_by_req[id(req)], seed_ids, common_len)
                )
            ]
            if len(group) < 2:
                continue

            estimated_saved = shared_new_tokens * (len(group) - 1)
            if (
                best_stage is not None
                and estimated_saved <= best_stage.estimated_saved_tokens
            ):
                continue

            best_stage = IntraTurnPrefixStagePlan(
                shared_token_ids=array("q", seed_ids[:common_len]),
                reqs=group,
                common_prefix_len=common_len,
                original_prefix_len=seed_prefix_len,
                estimated_saved_tokens=estimated_saved,
            )

    if best_stage is None:
        return None
    return IntraTurnPrefixPlan(stage=best_stage)


def _is_eligible_req(req: Req) -> bool:
    if req.extend_range is None:
        return False
    if req.is_dllm() or req.sampling_params.max_new_tokens == 0:
        return False
    if req.return_logprob or req.return_hidden_states:
        return False
    if req.input_embeds is not None or req.positional_embed_overrides is not None:
        return False
    if req.multimodal_inputs is not None or req.token_type_ids is not None:
        return False
    if req.session is not None:
        return False
    if getattr(req, "skip_radix_cache_insert", False):
        return False
    return len(req.get_fill_ids()) > len(req.prefix_indices)


def _common_prefix_len(a: array[int], b: array[int]) -> int:
    end = min(len(a), len(b))
    idx = 0
    while idx < end and a[idx] == b[idx]:
        idx += 1
    return idx


def _has_prefix(
    candidate: array[int], prefix_source: array[int], prefix_len: int
) -> bool:
    if len(candidate) < prefix_len:
        return False
    for idx in range(prefix_len):
        if candidate[idx] != prefix_source[idx]:
            return False
    return True
