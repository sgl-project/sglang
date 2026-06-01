from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ReqLogprob
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
    def chunks_done(self) -> int:
        return self.context.chunks_done(self.rid)

    @property
    def start_send_idx(self) -> int:
        return self.req.start_send_idx

    @property
    def tmp_end_idx(self) -> int:
        return self.req.tmp_end_idx

    @property
    def status(self) -> str:
        return self.context.status(self.rid)

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.context.remaining_prompt_tokens(self.rid)

    @property
    def logprobs(self) -> Optional["ReqLogprob"]:
        return self.context.logprobs(self.rid)

    @property
    def num_input_logprobs(self) -> int:
        return self.context.num_input_logprobs(self.rid)

    @property
    def kv_pages(self) -> int:
        page_size = self.context._scheduler.page_size
        return (self.req.kv_allocated_len + page_size - 1) // page_size

    @property
    def lock_refs(self) -> int:
        node = self.req.last_node
        return node.lock_ref if node is not None else 0
