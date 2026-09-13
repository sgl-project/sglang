from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.test.scripted_runtime.context import queries
from sglang.test.scripted_runtime.context.radix import _node_lock_ref, resolve_node

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


class _RequestEpoch(msgspec.Struct, kw_only=True):
    rid: str
    post_future: Future
    batch_start: int
    excluded_reqs: tuple[Req, ...]
    req: Optional[Req] = None
    closed: bool = False
    abort_requested: bool = False


class ScriptedReqHandle(msgspec.Struct, frozen=True):
    rid: str
    context: ScriptedContext
    _epoch: Optional[_RequestEpoch] = None

    @property
    def req(self) -> Optional[Req]:
        return queries.find_req_by_rid(self.context, rid=self.rid, epoch=self._epoch)

    @property
    def finished(self) -> bool:
        return queries.is_finished(self.context, rid=self.rid, epoch=self._epoch)

    @property
    def is_chunking(self) -> bool:
        return queries.is_chunking(self.context, rid=self.rid, epoch=self._epoch)

    @property
    def chunks_done(self) -> int:
        return queries.chunks_done(self.context, rid=self.rid, epoch=self._epoch)

    @property
    def status(self) -> str:
        return queries.status(self.context, rid=self.rid, epoch=self._epoch)

    @property
    def remaining_prompt_tokens(self) -> int:
        return queries.remaining_prompt_tokens(
            self.context, rid=self.rid, epoch=self._epoch
        )

    @property
    def kv_pages(self) -> int:
        req = self.req
        if req is None or not req.kv.holds_kv:
            return 0
        page_size = self.context.scheduler.page_size
        return (req.kv.kv_allocated_len + page_size - 1) // page_size

    @property
    def lock_refs(self) -> int:
        req = self.req
        if req is None:
            return 0
        node = resolve_node(self.context.scheduler.tree_cache, req.last_node)
        if node is None:
            return 0
        return _node_lock_ref(node)
