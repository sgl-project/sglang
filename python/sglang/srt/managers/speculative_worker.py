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

"""A speculative draft worker."""
import logging
from typing import List

import torch

from sglang.global_config import global_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.speculative_utils import SpecDraftInput, SpecInfoPipline
from sglang.srt.managers.tp_worker import ModelTpServer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class SpecDraftServer(ModelTpServer):
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        spec_queue: SpecInfoPipline,
    ):
        super().__init__(gpu_id, tp_rank, server_args, nccl_port, spec_queue)

    def exposed_step(self, recv_reqs: List):
        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(
                    recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    self.handle_generate_request(recv_req)
                    self.forward_extend_step()
                    break
                elif isinstance(recv_req, SpecDraftInput):
                    self.forward_draft_step()
                    break
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

        except Exception:
            logger.error("Exception in ModelTpServer:\n" + get_exception_traceback())
            raise

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret

    @torch.inference_mode()
    def forward_extend_step(self):
        new_batch = self.get_new_prefill_batch()

        if new_batch is not None:
            # Run a new prefill batch
            draft_input = self.spec_queue.draft_input_queue.get()
            draft_input.init()
            new_batch.spec_draft_input = draft_input
            self.forward_prefill_batch(new_batch)
            new_batch.spec_draft_input.prepare_for_decode(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
            self.forward_decode_step()

    @torch.inference_mode()
    def forward_decode_step(self):
        assert self.running_batch is not None, "Running Batch should not be None."
        # Run a few decode batches continuously for reducing overhead
        for _ in range(self.server_args.num_speculative_tokens):
            self.num_generated_tokens += len(self.running_batch.reqs)
            self.forward_decode_batch(self.running_batch)

            # Print stats
            if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                self.print_decode_stats()

            if self.running_batch.is_empty():
                self.running_batch = None
                break

            if self.out_pyobjs and self.running_batch.has_stream():
                break
        verify_input = self.running_batch.spec_draft_input.prepare_for_verify(
            self.running_batch
        )
        print(verify_input.draft_token)
        print(verify_input.retrive_index)

    def forward_prefill_batch(self, batch: ScheduleBatch):
        # Only implement EAGLE currently.
        if self.server_args.speculative_algorithm == "EAGLE":
            batch.prepare_for_extend(self.model_config.vocab_size)

            if self.model_runner.is_generation:
                # Forward and sample the next tokens
                if batch.extend_num_tokens != 0:
                    sample_output, logits_output = self.model_runner.forward(batch)
        else:
            super().forward_prefill_batch(batch)

    def forward_decode_batch(self, batch: ScheduleBatch):
        # don't need check memory, draft runner have same kv cache capacity
        # with target runner. target target will decides which req need be retracted.

        # Update batch tensors
        batch.forward_mode = ForwardMode.SPECDECODE

        # Forward and sample the next tokens
        sample_output, logits_output = self.model_runner.forward(batch)
        batch.spec_draft_input.prepare_for_decode(batch)
