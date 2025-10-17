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
from typing import Any, List, Optional

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier

logger = logging.getLogger(__name__)


class SchedulerInputBlocker:
    def __init__(self, noop: bool):
        self._state = _State.UNBLOCKED
        self._pending_reqs = []
        self._noop = noop
        self._global_unblock_barrier = PollBasedBarrier(noop=noop)

    def handle(self, recv_reqs: Optional[List[Any]]):
        assert (recv_reqs is None) == self._noop

        if not self._noop:
            output_reqs = []
            for recv_req in recv_reqs:
                output_reqs += self._handle_recv_req(recv_req)

        global_arrived_unblock_barrier = (
            self._global_unblock_barrier.poll_global_arrived()
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
            if recv_req.type == BlockReqType.BLOCK:
                self._execute_block_req()
                return []
            elif recv_req.type == BlockReqType.UNBLOCK:
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
        self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)

    def _execute_unblock_req(self):
        logger.info("Handle unblock req")
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
def input_blocker_guard_region(send_to_scheduler):
    send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.BLOCK))
    try:
        yield
    finally:
        send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.UNBLOCK))
