from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class ContextQueriesMixin:

    def find_req_by_rid(self, rid: str) -> Optional["Req"]:
        s = self._scheduler
        chunked = s.chunked_req
        if chunked is not None and chunked.rid == rid:
            return chunked
        for r in s.waiting_queue:
            if r.rid == rid:
                return r
        if s.running_batch is not None:
            for r in s.running_batch.reqs:
                if r.rid == rid:
                    return r
        last_batch = getattr(s, "last_batch", None)
        if last_batch is not None:
            for r in last_batch.reqs:
                if r.rid == rid:
                    return r
        return None

    def is_finished(self, rid: str) -> bool:
        req = self.find_req_by_rid(rid)
        if req is None:
            return rid in self._req_handles
        return req.finished()

    def is_chunking(self, rid: str) -> bool:
        s = self._scheduler
        return s.chunked_req is not None and s.chunked_req.rid == rid
