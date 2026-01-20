from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker


class NgramVerifyInputV2(NgramVerifyInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,
        draft_token_num: int,
    ):
        super().__init__(
            draft_token,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_token_num,
        )
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_next_token = retrive_next_token
        self.retrive_next_sibling = retrive_next_sibling
        self.draft_token_num = draft_token_num
        self.device = self.custom_mask.device
        self.verify_done: Optional[torch.cuda.Event] = None

    # def verify(
    #     self,
    #     batch: ScheduleBatch,
    #     logits_output: LogitsProcessorOutput,
    #     page_size: int,
    #     vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    #     is_spec_v2: bool = True,
    # ):
    #     return super().verify(
    #         batch, logits_output, page_size, vocab_mask, is_spec_v2=True
    #     )


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
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0
        accept_lens = None

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            verify_input: NgramVerifyInputV2 = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size, is_spec_v2=True
            )
            # Store accept_lens for per-request metrics
            accept_lens = verify_input.accept_length
            # TODO csy ngram v2 logprob not supported yet
            # if batch.return_logprob:
            #     add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            self._update_ngram_cache(batch)
            batch.forward_mode = ForwardMode.DECODE
            batch.logits_output = logits_output  # to be used as parameter in scheduler update_running_batch

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            next_draft_input=model_worker_batch.spec_info,
        )
