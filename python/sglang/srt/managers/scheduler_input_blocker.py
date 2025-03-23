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
from enum import Enum, auto
from typing import List, Optional, Any

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType


class SchedulerInputBlocker:
    def __init__(self):
        self._state = _State.UNBLOCKED

    def handle(self, recv_reqs: Optional[List[Any]]):
        output_reqs = []
        for recv_req in recv_reqs:
            output_reqs += self._handle_recv_req(recv_req)

    def _handle_recv_req(self, recv_req):
        if isinstance(recv_req, BlockReqInput):
            if recv_req.type == BlockReqType.BLOCK:
                self._execute_block_request()
                return []
            elif recv_req.type == BlockReqType.UNBLOCK:
                self._execute_unblock_request()
                return []
            else:
                raise NotImplementedError(f"{recv_req=}")
        else:
            return TODO

    def _execute_block_request(self):
        TODO

    def _execute_unblock_request(self):
        TODO


class _State(Enum):
    UNBLOCKED = auto()
    BLOCKED = auto()
    AWAITING_GLOBAL_UNBLOCK = auto()
