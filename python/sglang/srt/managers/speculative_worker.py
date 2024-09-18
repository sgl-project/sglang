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
from typing import List

import torch

from sglang.global_config import global_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.speculative_utils import SpecInfoPipline
from sglang.srt.managers.tp_worker import ModelTpServer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SpecDraftServer(ModelTpServer):
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
        spec_queue: SpecInfoPipline,
    ):
        super().__init__(
            gpu_id, tp_rank, server_args, nccl_port, model_overide_args, spec_queue
        )

    @torch.inference_mode()
    def forward_step(self):
        new_batch = self.get_new_prefill_batch()

        if new_batch is not None:
            # Run a new prefill batch
            draft_input = self.spec_queue.draft_input_queue.get()
            new_batch.spec_draft_input = draft_input
            self.forward_prefill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run a decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(global_config.num_continue_decode_steps):
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
            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio
