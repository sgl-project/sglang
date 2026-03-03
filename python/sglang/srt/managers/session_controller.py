# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class SessionReqNode:
    def __init__(
        self,
        req: Req,
        parent: Optional["SessionReqNode"] = None,
        children=None,
    ):
        self.req = req
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = [] if not children else children

    def clear_children(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)
        self.children = []

    def clear(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)

        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()
        del req_dict[self.req.rid]

    def abort(self):
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()

    def __str__(self):
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        if len(self.children) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.children[0].req.rid
            ret = self.children[0]._str_helper(prefix)
            for child in self.children[1:]:
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.streaming = streaming
        self.timeout = timeout
        self.last_active_time: float = time.monotonic()
        self.req_nodes: Dict[str, SessionReqNode] = {}

    def is_timed_out(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
    ):
        assert req.session_params is not None
        self.last_active_time = time.monotonic()
        session_params = req.session_params

        last_req_node = None
        last_req = None
        abort = False
        abort_message = ""
        if self.streaming:
            # Streaming sessions: only simple appends allowed; reject otherwise.
            if session_params.replace:
                abort = True
                abort_message = "Streaming sessions do not support replace."
            elif session_params.drop_previous_output:
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset and session_params.offset != 0:
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif self.req_nodes:
                assert len(self.req_nodes) == 1
                _, last_req_node = self.req_nodes.popitem()
                last_req = last_req_node.req
        elif session_params.replace:
            if session_params.rid is None:
                for _, req_node in self.req_nodes.items():
                    req_node.clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_children(self.req_nodes)
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        if last_req is not None:
            # trim bos token if it is an append
            if (
                tokenizer is not None
                and req.input_ids
                and req.input_ids[0] == tokenizer.bos_token_id
            ):
                req.input_ids = req.input_ids[1:]

            input_ids = (
                last_req.origin_input_ids
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]
            )

            if session_params.drop_previous_output:
                input_ids = last_req.origin_input_ids[:]

            if session_params.offset and session_params.offset != 0:
                input_ids = input_ids[: session_params.offset] + req.input_ids
            else:
                input_ids += req.input_ids

            input_ids_unpadded = (
                last_req.origin_input_ids_unpadded
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]
            )
            if session_params.drop_previous_output:
                input_ids_unpadded = last_req.origin_input_ids_unpadded[:]

            if session_params.offset and session_params.offset != 0:
                input_ids_unpadded = (
                    input_ids_unpadded[: session_params.offset] + req.input_ids
                )
            else:
                input_ids_unpadded += req.input_ids
        else:
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        new_req = Req(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            priority=req.priority,
            routing_key=req.routing_key,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.multimodal_inputs
        new_req.tokenizer = tokenizer

        if abort:
            new_req.set_finish_with_abort(abort_message)
        elif self.streaming:
            if last_req is not None:
                last_req.session = None
            self.req_nodes[req.rid] = SessionReqNode(new_req)
        else:
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node

        return new_req


class SessionController:
    def __init__(self, tree_cache: BasePrefixCache):
        self.sessions: Dict[str, Session] = {}
        self._last_reap_time: float = 0.0
        self.tree_cache = tree_cache

    def __contains__(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
            )
            return OpenSessionReqOutput(session_id, True)

    def close(self, recv_req: CloseSessionReqInput):
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id)

    def _close(self, session_id: str):
        session = self.sessions[session_id]
        if session.streaming and session.req_nodes:
            assert len(session.req_nodes) == 1
            req = next(iter(session.req_nodes.values())).req
            if not req.finished():
                req.session = None
        if isinstance(self.tree_cache, SessionAwareCache):
            self.tree_cache.release_session(session_id)
        del self.sessions[session_id]

    def maybe_reap(self, now: float, interval: float = 1.0):
        # reap sessions every second
        if now - self._last_reap_time > interval:
            self._last_reap_time = now
            timed_out = [
                sid for sid, session in self.sessions.items() if session.is_timed_out()
            ]
            for sid in timed_out:
                logger.info(f"Session {sid} timed out, closing.")
                self._close(sid)

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
