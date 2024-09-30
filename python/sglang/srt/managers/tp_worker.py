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

"""A tensor parallel worker."""

import json
import logging

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.io_struct import UpdateWeightReqInput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, is_multimodal_model, set_random_seed

logger = logging.getLogger(__name__)


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
    ):
        # Parse args
        self.tp_rank = tp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=json.loads(server_args.json_model_override_args),
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size,
        )
        self.max_req_input_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def get_token_and_memory_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_input_len,
            self.random_seed,
        )

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        return logits_output, next_token_ids

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings.tolist()
        return embeddings

    def update_weights(self, recv_req: UpdateWeightReqInput):
        success, message = self.model_runner.update_weights(
            recv_req.model_path, recv_req.load_format
        )
        return success, message
