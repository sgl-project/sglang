from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.test.scripted_runtime.context.radix import _node_lock_ref

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


@dataclass(frozen=True, slots=True)
class ScriptedReqHandle:
    rid: str
    context: ScriptedContext

    @property
    def req(self) -> Optional[Req]:
        return self.context.find_req_by_rid(self.rid)

    @property
    def finished(self) -> bool:
        return self.context.is_finished(self.rid)

    @property
    def is_chunking(self) -> bool:
        return self.context.is_chunking(self.rid)

    @property
    def chunks_done(self) -> int:
        return self.context.chunks_done(self.rid)

    @property
    def status(self) -> str:
        return self.context.status(self.rid)

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.context.remaining_prompt_tokens(self.rid)

    @property
    def kv_pages(self) -> int:
        req = self.req
        if req is None or req.req_pool_idx is None:
            return 0
        page_size = self.context.scheduler.page_size
        return (req.kv.kv_allocated_len + page_size - 1) // page_size

    @property
    def lock_refs(self) -> int:
        req = self.req
        if req is None:
            return 0
        node = req.last_node
        if node is None:
            return 0
        return _node_lock_ref(node)
