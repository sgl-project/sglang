# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Callable, List, Optional

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier

logger = logging.getLogger(__name__)


class SchedulerInputBlocker:
    def __init__(self, noop: bool):
        self._state = _State.UNBLOCKED
        self._block_depth = 0
        self._pending_reqs = []
        self._deferred_reqs = []
        self._noop = noop
        self._global_unblock_barrier = PollBasedBarrier(noop=noop)

    def has_pending(self) -> bool:
        return (
            self._state != _State.UNBLOCKED
            or bool(self._deferred_reqs)
            or self._global_unblock_barrier.global_pending
        )

    def needs_polling(self) -> bool:
        return (
            self._state == _State.GLOBAL_UNBLOCK_BARRIER
            or bool(self._deferred_reqs)
            or self._global_unblock_barrier.global_polling
        )

    def handle(self, recv_reqs: Optional[List[Any]]):
        assert (recv_reqs is None) == self._noop

        if not self._noop:
            recv_reqs = self._deferred_reqs + recv_reqs
            self._deferred_reqs = []
            output_reqs = []
            for index, recv_req in enumerate(recv_reqs):
                if self._state == _State.GLOBAL_UNBLOCK_BARRIER:
                    # Each guard cohort must finish the same barrier on every rank.
                    self._deferred_reqs.extend(recv_reqs[index:])
                    break
                output_reqs += self._handle_recv_req(recv_req)

        global_arrived_unblock_barrier = (
            self._global_unblock_barrier.poll_global_arrived(
                pending=self._state != _State.UNBLOCKED or bool(self._deferred_reqs),
                polling=self._state == _State.GLOBAL_UNBLOCK_BARRIER
                or bool(self._deferred_reqs),
            )
        )
        if (
            self._state == _State.GLOBAL_UNBLOCK_BARRIER
            and global_arrived_unblock_barrier
        ):
            output_reqs += self._handle_arrive_unblock_barrier()

        if not self._noop:
            return output_reqs

    def _handle_recv_req(self, recv_req):
        if isinstance(recv_req, BlockReqInput):
            if recv_req.req_type == BlockReqType.BLOCK:
                self._execute_block_req()
                return []
            elif recv_req.req_type == BlockReqType.UNBLOCK:
                self._execute_unblock_req()
                return []
            else:
                raise NotImplementedError(f"{recv_req=}")
        else:
            if self._state == _State.UNBLOCKED:
                return [recv_req]
            else:
                self._pending_reqs.append(recv_req)
                return []

    def _execute_block_req(self):
        logger.info("Handle block req")
        if self._block_depth == 0:
            self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)
        self._block_depth += 1

    def _execute_unblock_req(self):
        logger.info("Handle unblock req")
        assert self._block_depth > 0
        self._block_depth -= 1
        if self._block_depth:
            return
        self._change_state(
            original=_State.BLOCKED, target=_State.GLOBAL_UNBLOCK_BARRIER
        )
        self._global_unblock_barrier.local_arrive()

    def _handle_arrive_unblock_barrier(self):
        logger.info(f"Arrived at unblock barrier ({len(self._pending_reqs)=})")
        self._change_state(
            original=_State.GLOBAL_UNBLOCK_BARRIER, target=_State.UNBLOCKED
        )
        output_reqs = [*self._pending_reqs]
        self._pending_reqs.clear()
        return output_reqs

    def _change_state(self, original: "_State", target: "_State"):
        assert self._state == original, f"{self._state=} {original=} {target=}"
        self._state = target


class _State(Enum):
    UNBLOCKED = auto()
    BLOCKED = auto()
    GLOBAL_UNBLOCK_BARRIER = auto()


@contextmanager
def input_blocker_guard_region(dispatch_to_scheduler: Callable[[BlockReqInput], None]):
    dispatch_to_scheduler(BlockReqInput(req_type=BlockReqType.BLOCK))
    try:
        yield
    finally:
        dispatch_to_scheduler(BlockReqInput(req_type=BlockReqType.UNBLOCK))
