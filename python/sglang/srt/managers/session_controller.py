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

import logging
import uuid
from typing import Dict, Optional

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req


class SessionReqNode:
    def __init__(self, req, parent=None, childs=None):
        self.req = req
        self.parent = parent
        if parent is not None:
            parent.childs.append(self)
        self.childs = [] if not childs else childs

    def clear_childs(self, req_dict):
        for req_node in self.childs:
            req_node.clear(req_dict)
        self.childs = []

    def clear(self, req_dict):
        for req_node in self.childs:
            req_node.clear(req_dict)

        if self.req.finished_reason is None:
            self.req.to_abort = True
        del req_dict[self.req.rid]

    def abort(self):
        if self.req.finished_reason is None:
            self.req.to_abort = True

    def __str__(self):
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        if len(self.childs) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.childs[0].req.rid
            ret = self.childs[0]._str_helper(prefix)
            for child in self.childs[1:]:
                prefix = " " * len(origin_prefix) + r" \- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    def __init__(self, capacity_of_str_len: int, session_id: Optional[str] = None):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.req_nodes: Dict[str, SessionReqNode] = {}

    def create_req(self, req: TokenizedGenerateReqInput, tokenizer):
        assert req.session_params is not None
        session_params = req.session_params

        last_req_node = None
        last_req = None
        abort = False
        if session_params.replace:
            if session_params.rid is None:
                for _, req_node in self.req_nodes.items():
                    req_node.clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_childs(self.req_nodes)
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    if not last_req.finished():
                        logging.warning(
                            "The request in a session is appending to a request that hasn't finished."
                        )
                        abort = True

        if last_req is not None:
            # trim bos token if it is an append
            if tokenizer is not None and req.input_ids[0] == tokenizer.bos_token_id:
                req.input_ids = req.input_ids[1:]

            input_ids = (
                last_req.origin_input_ids
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]
            )
            if session_params.offset and session_params.offset != 0:
                input_ids = input_ids[: session_params.offset] + req.input_ids
            else:
                input_ids += req.input_ids
            input_ids_unpadded = (
                last_req.origin_input_ids_unpadded
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]
            )
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
            lora_path=req.lora_path,
            session_id=self.session_id,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.mm_inputs
        new_req.tokenizer = tokenizer
        if abort:
            new_req.to_abort = True
        else:
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node

        return new_req
