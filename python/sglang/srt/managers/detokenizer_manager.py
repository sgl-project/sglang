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

import asyncio
import dataclasses
from typing import List

import uvloop
import zmq
import zmq.asyncio

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import find_printable_text, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


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
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{port_args.detokenizer_port}")

        self.send_to_tokenizer = context.socket(zmq.PUSH)
        self.send_to_tokenizer.connect(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.decode_status = {}

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = await self.recv_from_router.recv_pyobj()

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
            elif self.tokenizer is None:
                # If the tokenizer is skipped, no detokenization is needed
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue

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

                read_ids.append(s.decode_ids[s.surr_offset :])
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

                output_strs.append(s.decoded_text + new_text)

                # Trim stop str. TODO(lmzheng): handle the case where multiple stop strs are hit
                if isinstance(recv_obj.finished_reason[i], FINISH_MATCHED_STR):
                    pos = output_strs[i].find(recv_obj.finished_reason[i].matched)
                    if pos != -1:
                        output_strs[i] = output_strs[i][:pos]

            self.send_to_tokenizer.send_pyobj(
                BatchStrOut(
                    rids=recv_obj.rids,
                    output_strs=output_strs,
                    meta_info=recv_obj.meta_info,
                    finished_reason=recv_obj.finished_reason,
                )
            )


def start_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    try:
        manager = DetokenizerManager(server_args, port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(manager.handle_loop())
