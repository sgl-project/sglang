"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
from collections import OrderedDict
from typing import List, Union

import zmq

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    GetMemPoolSizeReqOutput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR, FINISH_MATCHED_TOKEN
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_zmq_socket, kill_parent_process
from sglang.utils import find_printable_text, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    vid: int
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name
        )

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.decode_status = LimitedCapacityDict()

    def trim_eos(self, output: Union[str, List[int]], finished_reason, no_stop_trim):
        if no_stop_trim:
            return output

        # Trim stop str. TODO(lmzheng): handle the case where multiple stop strs are hit
        if isinstance(finished_reason, FINISH_MATCHED_STR) and isinstance(output, str):
            pos = output.find(finished_reason.matched)
            return output[:pos] if pos != -1 else output
        if isinstance(finished_reason, FINISH_MATCHED_TOKEN) and isinstance(
            output, list
        ):
            assert len(output) > 0
            return output[:-1]
        return output

    def event_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            if isinstance(recv_obj, BatchEmbeddingOut):
                # If it is embedding model, no detokenization is needed.
                self.send_to_tokenizer.send_pyobj(
                    BatchEmbeddingOut(
                        rids=recv_obj.rids,
                        embeddings=recv_obj.embeddings,
                        meta_info=recv_obj.meta_info,
                        finished_reason=recv_obj.finished_reason,
                    )
                )
                continue
            elif isinstance(recv_obj, UpdateWeightReqOutput):
                # If it is a weight update request, no detokenization is needed.
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue
            elif isinstance(recv_obj, GetMemPoolSizeReqOutput):
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue
            else:
                assert isinstance(recv_obj, BatchTokenIDOut)

            bs = len(recv_obj.rids)

            # Initialize decode status
            read_ids, surr_ids = [], []
            for i in range(bs):
                rid = recv_obj.rids[i]
                vid = recv_obj.vids[i]
                if rid not in self.decode_status or self.decode_status[rid].vid != vid:
                    s = DecodeStatus(
                        vid=vid,
                        decoded_text=recv_obj.decoded_texts[i],
                        decode_ids=recv_obj.decode_ids[i],
                        surr_offset=0,
                        read_offset=recv_obj.read_offsets[i],
                    )
                    self.decode_status[rid] = s
                else:
                    s = self.decode_status[rid]
                    s.decode_ids = recv_obj.decode_ids[i]

                read_ids.append(
                    self.trim_eos(
                        s.decode_ids[s.surr_offset :],
                        recv_obj.finished_reason[i],
                        recv_obj.no_stop_trim[i],
                    )
                )
                surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

            # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
            surr_texts = self.tokenizer.batch_decode(
                surr_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )
            read_texts = self.tokenizer.batch_decode(
                read_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )

            # Incremental decoding
            output_strs = []
            for i in range(bs):
                s = self.decode_status[recv_obj.rids[i]]
                new_text = read_texts[i][len(surr_texts[i]) :]
                if recv_obj.finished_reason[i] is None:
                    # Streaming chunk: update the decode status
                    if len(new_text) > 0 and not new_text.endswith("�"):
                        s.decoded_text = s.decoded_text + new_text
                        s.surr_offset = s.read_offset
                        s.read_offset = len(s.decode_ids)
                        new_text = ""
                    else:
                        new_text = find_printable_text(new_text)

                output_strs.append(
                    self.trim_eos(
                        s.decoded_text + new_text,
                        recv_obj.finished_reason[i],
                        recv_obj.no_stop_trim[i],
                    )
                )

            self.send_to_tokenizer.send_pyobj(
                BatchStrOut(
                    rids=recv_obj.rids,
                    output_strs=output_strs,
                    meta_info=recv_obj.meta_info,
                    finished_reason=recv_obj.finished_reason,
                )
            )


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity=1 << 15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    configure_logger(server_args)

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
