import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info_v2 import move_accepted_tokens_to_target_kvcache
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.spec_utils import detect_nan, generate_token_bitmask

logger = logging.getLogger(__name__)


class NGRAMWorkerV2(NGRAMWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
        )
        self.server_args = server_args
        self.enable_nan_detection = server_args.enable_nan_detection
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # Set constant
        NgramVerifyInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        _, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        self._prepare_for_speculative_decoding(model_worker_batch, is_spec_v2=True)
        verify_input: NgramVerifyInput = model_worker_batch.spec_info
        accept_length = None
        new_seq_lens, verify_done = None, None

        if model_worker_batch.forward_mode.is_target_verify():
            # Prepare grammar data on CPU if needed
            if model_worker_batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrive_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrive_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrive_next_token.shape
                ).cpu()

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            # Generate vocab mask for constrained decoding
            vocab_mask = None
            if model_worker_batch.has_grammar:
                # Generate the logit mask for structured output.
                vocab_mask = generate_token_bitmask(
                    model_worker_batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    model_worker_batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                    # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    model_worker_batch.sampling_info.vocab_mask = None

            # Sample
            if self.enable_nan_detection:
                detect_nan(logits_output)
            (
                predict,
                accept_length,
                accept_index,
            ) = verify_input.sample(model_worker_batch, logits_output, vocab_mask)
            new_seq_lens = model_worker_batch.seq_lens + accept_length
            move_accepted_tokens_to_target_kvcache(
                model_worker_batch,
                accept_index,
                accept_length,
                self.token_to_kv_pool_allocator,
                self.draft_token_num,
            )
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

            self._update_ngram_cache(model_worker_batch)
            model_worker_batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            new_seq_lens = model_worker_batch.seq_lens
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

        # Construct the next draft input
        next_draft_input = NgramVerifyInput(
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_length,
            next_draft_input=next_draft_input,
        )
