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

import copy
import uuid
from dataclasses import dataclass
from typing import Optional

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, List, Req


class Session:
    def __init__(self, capacity_of_str_len: int, session_id: str = None):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.reqs: List[Req] = []

    def create_req(self, req: TokenizedGenerateReqInput, tokenizer):
        if req.session_rid is not None:
            while len(self.reqs) > 0:
                if self.reqs[-1].rid == req.session_rid:
                    break
                self.reqs = self.reqs[:-1]
        else:
            self.reqs = []
        if len(self.reqs) > 0:
            input_ids = (
                self.reqs[-1].origin_input_ids
                + self.reqs[-1].output_ids[
                    : self.reqs[-1].sampling_params.max_new_tokens
                ]
                + req.input_ids
            )
            input_ids_unpadded = (
                self.reqs[-1].origin_input_ids_unpadded
                + self.reqs[-1].output_ids[
                    : self.reqs[-1].sampling_params.max_new_tokens
                ]
                + req.input_ids
            )
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
        )
        if len(self.reqs) > 0:
            new_req.image_inputs = self.reqs[-1].image_inputs
        new_req.tokenizer = tokenizer
        if req.session_rid is not None and len(self.reqs) == 0:
            new_req.finished_reason = FINISH_ABORT(
                f"Invalid request: requested session rid {req.session_rid} does not exist in the session history"
            )
        else:
            self.reqs.append(new_req)
        return new_req
