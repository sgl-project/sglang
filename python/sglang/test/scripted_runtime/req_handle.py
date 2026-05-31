from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


@dataclass(frozen=True, slots=True)
class ScriptedReqHandle:
    rid: str
    context: "ScriptedContext"

    @property
    def req(self) -> Optional["Req"]:
        return self.context.find_req_by_rid(self.rid)

    @property
    def finished(self) -> bool:
        return self.context.is_finished(self.rid)

    @property
    def is_chunking(self) -> bool:
        return self.context.is_chunking(self.rid)

    @property
    def output_tokens(self) -> List[int]:
        req = self.req
        return [] if req is None else list(req.output_ids)

    @property
    def finish_reason(self) -> Optional[Dict[str, Any]]:
        req = self.req
        if req is None or req.finished_reason is None:
            return None
        return req.finished_reason.to_json()

    @property
    def error_message(self) -> Optional[str]:
        reason = self.finish_reason
        return reason.get("message") if reason is not None else None

    @property
    def row_idx(self) -> Optional[int]:
        req = self.req
        return None if req is None else req.req_pool_idx

    @property
    def num_input_tokens(self) -> int:
        req = self.req
        return 0 if req is None else len(req.origin_input_ids)

    @property
    def fill_ids_len(self) -> int:
        req = self.req
        return 0 if req is None else len(req.fill_ids)

    @property
    def extend_input_len(self) -> int:
        req = self.req
        return 0 if req is None else req.extend_input_len

    @property
    def kv_committed_len(self) -> int:
        req = self.req
        return 0 if req is None else req.kv_committed_len

    @property
    def cached_tokens(self) -> int:
        req = self.req
        return 0 if req is None else req.cached_tokens

    @property
    def cached_tokens_snapshot(self) -> int:
        return self.cached_tokens

    @property
    def prefix_indices_len(self) -> int:
        req = self.req
        return 0 if req is None else len(req.prefix_indices)

    @property
    def host_hit_length(self) -> int:
        req = self.req
        return 0 if req is None else req.host_hit_length

    @property
    def inflight_middle_chunks(self) -> int:
        req = self.req
        return 0 if req is None else req.inflight_middle_chunks

    @property
    def spec_verify_count(self) -> int:
        req = self.req
        return 0 if req is None else req.spec_verify_ct

    @property
    def total_tokens(self) -> int:
        req = self.req
        return 0 if req is None else len(req.origin_input_ids) + len(req.output_ids)

    @property
    def kv_pages(self) -> int:
        req = self.req
        if req is None:
            return 0
        page_size = self.context._scheduler.page_size
        return (req.kv_allocated_len + page_size - 1) // page_size

    @property
    def last_node_id(self) -> Optional[int]:
        req = self.req
        if req is None or req.last_node is None:
            return None
        return req.last_node.id

    @property
    def lora_path(self) -> Optional[str]:
        req = self.req
        return None if req is None else req.lora_id
