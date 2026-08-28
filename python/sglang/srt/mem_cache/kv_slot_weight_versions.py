from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

from sglang.srt.utils.weight_versions import (
    UNKNOWN_WEIGHT_VERSION,
    WeightVersionSpan,
    WeightVersionSpans,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class KvSlotWeightVersions:
    def __init__(
        self, *, num_slots: int, device: str, req_to_token_pool: ReqToTokenPool
    ):
        self._slot_version_ids = torch.zeros(
            (num_slots,), dtype=torch.int32, device=device
        )
        self._req_to_token_pool = req_to_token_pool
        self._version_str_by_id: List[str] = [UNKNOWN_WEIGHT_VERSION]
        self._version_id_by_str: Dict[str, int] = {UNKNOWN_WEIGHT_VERSION: 0}

    def record(self, *, slot_indices: torch.Tensor, version: str) -> None:
        self._slot_version_ids[slot_indices] = self._intern(version=version)

    def record_req(self, req: Req) -> None:
        num_prompt_tokens = min(
            len(req.origin_input_ids), req.effective_kv_committed_len()
        )
        req.prefill_weight_versions = self.lookup_spans(
            self._req_to_token_pool.req_to_token[req.req_pool_idx, :num_prompt_tokens]
        )

    def lookup_spans(self, slot_indices: torch.Tensor) -> WeightVersionSpans:
        version_ids = self._slot_version_ids[slot_indices]
        if len(version_ids) == 0:
            return []

        is_run_start = torch.ones_like(version_ids, dtype=torch.bool)
        is_run_start[1:] = version_ids[1:] != version_ids[:-1]
        run_starts = is_run_start.nonzero().flatten()
        run_bounds = torch.cat([run_starts, run_starts.new_tensor([len(version_ids)])])
        run_version_ids = version_ids[run_starts].tolist()
        run_bounds_list = run_bounds.tolist()

        return [
            WeightVersionSpan(
                version=self._version_str_by_id[version_id], start=start, end=end
            )
            for version_id, start, end in zip(
                run_version_ids, run_bounds_list, run_bounds_list[1:]
            )
        ]

    def _intern(self, version: str) -> int:
        if (version_id := self._version_id_by_str.get(version)) is not None:
            return version_id

        version_id = len(self._version_str_by_id)
        self._version_str_by_id.append(version)
        self._version_id_by_str[version] = version_id
        return version_id
