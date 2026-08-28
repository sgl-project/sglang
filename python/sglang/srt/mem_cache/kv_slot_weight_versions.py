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
        version_ids: List[int] = self._slot_version_ids[slot_indices].tolist()
        if _UNWRITTEN_VERSION_ID in version_ids:
            unwritten_slots = [
                slot
                for slot, version_id in zip(
                    slot_indices.tolist(), version_ids, strict=True
                )
                if version_id == _UNWRITTEN_VERSION_ID
            ]
            raise ValueError(
                "KV slots without a recorded weight version were looked up: "
                f"{unwritten_slots}"
            )

        spans: WeightVersionSpans = []
        for position, version_id in enumerate(version_ids):
            version = self._version_str_by_id[version_id]
            if spans and spans[-1].version == version:
                spans[-1].end = position + 1
            else:
                spans.append(
                    WeightVersionSpan(version=version, start=position, end=position + 1)
                )
        return spans

    def _intern(self, version: str) -> int:
        if (version_id := self._version_id_by_str.get(version)) is not None:
            return version_id

        version_id = len(self._version_str_by_id)
        self._version_str_by_id.append(version)
        self._version_id_by_str[version] = version_id
        return version_id
