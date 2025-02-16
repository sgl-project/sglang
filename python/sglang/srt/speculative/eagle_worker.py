import logging
import time
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
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    assign_draft_cache_locs,
    fast_topk,
    select_top_k_tokens,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


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
        self.finish_extend_len = []

        # Parse arguments
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.server_args = server_args

        # Share the embedding and lm_head
        if not self.speculative_algorithm.is_nextn():
            embed, head = self.target_worker.model_runner.model.get_embed_and_head()
            self.model_runner.model.set_embed_and_head(embed, head)
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph

        # Create multi-step attn backends and cuda graph runners
        if server_args.attention_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif server_args.attention_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        else:
            raise ValueError(
                f"EAGLE is not supportted in attention backend {server_args.attention_backend}"
            )

        self.model_runner.draft_attn_backend = self.draft_attn_backend
        self.init_cuda_graphs()

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        logger.info(f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s")

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            # Draft
            spec_info: EagleVerifyInput = self.draft(batch)

            # Verify
            (
                next_draft_input,
                logits_output,
                verified_id,
                self.finish_extend_len,
                accept_length_cpu,
                model_worker_batch,
            ) = self.verify(batch, spec_info)
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
            batch.spec_info = EagleDraftInput(
                hidden_states=logits_output.hidden_states,
                verified_id=next_token_ids,
            )
            self.forward_draft_extend(batch)
            return logits_output, next_token_ids, model_worker_batch, 0

    def draft(self, batch: ScheduleBatch):
        self._set_mem_pool(batch, self.model_runner)

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Allocate cache locations
        out_cache_loc = batch.alloc_token_slots(
            num_seqs * self.topk * self.speculative_num_steps
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )

        if can_cuda_graph:
            score_list, token_list, parents_list = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # Initialize attention backend
            self.draft_attn_backend.init_forward_metadata(forward_batch)

            # Run forward steps
            score_list, token_list, parents_list = self.draft_forward(forward_batch)

        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
            batch.sampling_info.is_all_greedy,
        )

        # Free cache locations
        batch.token_to_kv_pool.free(out_cache_loc)
        self._set_mem_pool(batch, self.target_worker.model_runner)
        return ret

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # we don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[
                forward_batch.batch_size
                * self.topk
                * i : forward_batch.batch_size
                * self.topk
                * (i + 1)
            ]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(batch, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return res + (model_worker_batch,)

    def forward_draft_extend(self, batch: ScheduleBatch):
        self._set_mem_pool(batch, self.model_runner)
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._set_mem_pool(batch, self.target_worker.model_runner)

    def _set_mem_pool(self, batch: ScheduleBatch, runner: ModelRunner):
        batch.token_to_kv_pool = runner.token_to_kv_pool
        batch.req_to_token_pool = runner.req_to_token_pool

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        seq_lens_backup = batch.seq_lens
        req_pool_indices_backup = batch.req_pool_indices

        self._set_mem_pool(batch, self.model_runner)
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(batch, self.speculative_num_steps)
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
        batch.req_pool_indices = req_pool_indices_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        spec_info = forward_batch.spec_info
        spec_info.topk_p, spec_info.topk_index = fast_topk(probs, self.topk, dim=-1)
        spec_info.hidden_states = logits_output.hidden_states

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
