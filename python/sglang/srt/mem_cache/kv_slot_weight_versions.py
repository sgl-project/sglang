from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

from sglang.srt.utils.weight_versions import WeightVersionSpan, WeightVersionSpans

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


_UNWRITTEN_VERSION_ID = -1


class KvSlotWeightVersions:
    def __init__(
        self, *, num_slots: int, device: str, req_to_token_pool: ReqToTokenPool
    ):
        self._slot_version_ids = torch.full(
            (num_slots,), _UNWRITTEN_VERSION_ID, dtype=torch.int32, device=device
        )
        self._req_to_token_pool = req_to_token_pool
        self._version_str_by_id: List[str] = []
        self._version_id_by_str: Dict[str, int] = {}

    def record(self, *, slot_indices: torch.Tensor, version: str) -> None:
        self._slot_version_ids[slot_indices] = self._intern(version=version)

    def fill_req_prefill_weight_versions(self, req: Req) -> None:
        num_prompt_tokens = min(
            len(req.origin_input_ids), req.effective_kv_committed_len()
        )
        req.prefill_weight_versions = self._lookup_spans(
            self._req_to_token_pool.req_to_token[req.req_pool_idx, :num_prompt_tokens]
        )

    def _lookup_spans(self, slot_indices: torch.Tensor) -> WeightVersionSpans:
        version_ids = self._slot_version_ids[slot_indices]
        if len(version_ids) == 0:
            return []
        if (is_unwritten := version_ids == _UNWRITTEN_VERSION_ID).any():
            raise ValueError(
                "KV slots without a recorded weight version were looked up: "
                f"{slot_indices[is_unwritten].tolist()}"
            )

        version_changes_at: List[int] = (
            (version_ids[1:] != version_ids[:-1]).nonzero().flatten().tolist()
        )
        run_starts = [0] + [position + 1 for position in version_changes_at]
        run_ends = run_starts[1:] + [len(version_ids)]
        run_version_ids: List[int] = version_ids[run_starts].tolist()

        return [
            WeightVersionSpan(
                version=self._version_str_by_id[version_id], start=start, end=end
            )
            for version_id, start, end in zip(
                run_version_ids, run_starts, run_ends, strict=True
            )
        ]

    def _intern(self, version: str) -> int:
        if (version_id := self._version_id_by_str.get(version)) is not None:
            return version_id

        version_id = len(self._version_str_by_id)
        self._version_str_by_id.append(version)
        self._version_id_by_str[version] = version_id
        return version_id
