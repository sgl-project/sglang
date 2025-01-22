from typing import List, Optional, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import EAGLEDraftInput
from sglang.srt.utils import rank0_print


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Do not capture cuda graph in `super().__init__()`
        # We will capture it later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        super().__init__(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            server_args=server_args,
            nccl_port=nccl_port,
            dp_rank=dp_rank,
            is_draft_worker=True,
        )
        self.target_worker = target_worker
        self.server_args = server_args
        self.finish_extend_len = []

        # Share the embedding and lm_head
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.model_runner.model.set_embed_and_head(embed, head)
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.model_runner.init_cuda_graphs()

    def forward_draft_decode(self, batch: ScheduleBatch):
        batch.spec_info.prepare_for_decode(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)

    def forward_draft_extend(self, batch: ScheduleBatch):
        self._set_mem_pool(batch, self.model_runner)
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._set_mem_pool(batch, self.target_worker.model_runner)

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            # Draft
            self._set_mem_pool(batch, self.model_runner)
            for i in range(self.server_args.speculative_num_steps):
                self.forward_draft_decode(batch)
            batch.spec_info.clear_draft_cache(batch)
            self._set_mem_pool(batch, self.target_worker.model_runner)

            # Verify
            (
                next_draft_input,
                logits_output,
                verified_id,
                self.finish_extend_len,
                accept_length_cpu,
                model_worker_batch,
            ) = self.verify(batch)
            next_draft_input.load_server_args(self.server_args)
            batch.spec_info = next_draft_input
            # if it is None, means all requsets are finished
            if batch.spec_info.verified_id is not None:
                self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verified_id,
                model_worker_batch,
                sum(accept_length_cpu),
            )

        else:
            # Forward with the target model and get hidden states.
            # We need the full hidden states to prefill the KV cache of the draft model.
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Forward with the draft model.
            spec_info = EAGLEDraftInput()
            spec_info.load_server_args(self.server_args)
            spec_info.hidden_states = logits_output.hidden_states
            spec_info.verified_id = next_token_ids
            batch.spec_info = spec_info
            self.forward_draft_extend(batch)
            return logits_output, next_token_ids, model_worker_batch, 0

    def verify(self, batch: ScheduleBatch):
        verify_input = batch.spec_info.prepare_for_verify(batch)
        verify_input.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        verify_input.hidden_states = logits_output.hidden_states
        res = verify_input.verify(batch, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return res + (model_worker_batch,)

    def _set_mem_pool(self, batch: ScheduleBatch, runner: ModelRunner):
        batch.token_to_kv_pool = runner.token_to_kv_pool
        batch.req_to_token_pool = runner.req_to_token_pool

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        seq_lens_backup = batch.seq_lens

        self._set_mem_pool(batch, self.model_runner)
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._set_mem_pool(batch, self.target_worker.model_runner)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ):
        sample_output = torch.softmax(
            logits_output.next_token_logits, dim=-1
        )  # TODO(kavioyu): Support more sampling methods
        spec_info = forward_batch.spec_info
        spec_info.sample_output = sample_output
        spec_info.hidden_states = logits_output.hidden_states
        spec_info.prev_mode = forward_batch.forward_mode

    # Don't support prefix share now.
    def finish_request(self, reqs: Union[Req, List[Req]]):
        if not isinstance(reqs, List):
            reqs = [reqs]
        for req in reqs:
            if req.rid not in self.finish_extend_len:
                continue
            req_len = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                - self.finish_extend_len[req.rid]
                - 1
            )
            kv_indices = self.model_runner.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][:req_len]
            self.model_runner.token_to_kv_pool.free(kv_indices)
            self.model_runner.req_to_token_pool.free(req.req_pool_idx)
