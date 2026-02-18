import logging
from typing import Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs_func,
    move_accepted_tokens_to_target_kvcache,
)
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import USE_FULL_MASK, NGRAMWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.token_to_kv_pool = self.token_to_kv_pool_allocator._kvcache
        self.count = 0

    def _prepare_draft_tokens_v2(
        self, batch: ModelWorkerBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        stride = self.draft_token_num

        prev_token_ids, prev_accept_lens = (
            batch.spec_info.verified_tokens,
            batch.spec_info.accept_lens,
        )
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()

        self.ngram_cache.synchronize()
        batch_tokens = []
        assert len(batch.reqs) == len(self.prev_accept_lens)
        i = 0
        for req in batch.reqs:
            # TODO grammar doesn't overlap, output_ids will be normal, deal with it!
            # prev_token_id and prev_accept_lens are filtered and merged, so here should not encounter index out of bound
            prev_tokens = self.prev_token_ids[
                i * stride : i * stride + self.prev_accept_lens[i]
            ]
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids,
                req.output_ids + prev_tokens,
                self.max_match_window_size,
            )
            batch_tokens.append(check_token)
            i += 1
        req_drafts, mask = self.ngram_cache.batch_get(batch_tokens)
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding_v2(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_extend():
            return

        bs = len(batch.reqs)

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens_v2(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            for i in range(bs):
                seq_len = batch.seq_lens_cpu[i]
                req_mask = torch.ones((self.draft_token_num, seq_len)).to(
                    device=self.device, non_blocking=True
                )
                req_mask = torch.cat(
                    (
                        req_mask,
                        torch.from_numpy(mask[i]).to(
                            device=self.device, non_blocking=True
                        ),
                    ),
                    dim=1,
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + self.draft_token_num,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=self.device,
        )
        batch.spec_info = NgramVerifyInput(
            server_args=self.server_args,
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=self.draft_token_num,
            is_spec_v2=True,
        )

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(model_worker_batch.seq_lens)
        self._prepare_for_speculative_decoding_v2(model_worker_batch)
        verify_input: NgramVerifyInput = model_worker_batch.spec_info
        accept_length = torch.tensor([1] * bs, dtype=torch.int32).to(
            device=self.device, non_blocking=True
        )

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
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

            verified_tokens = predict[accept_index].flatten()

            # copy kvcache will not use the new_seq_lens
            move_accepted_tokens_to_target_kvcache(
                model_worker_batch,
                accept_index,
                accept_length,
                self.token_to_kv_pool_allocator,
                self.draft_token_num,
            )
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

            verified_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32
            ).to(device=self.device, non_blocking=True)
            verified_tokens[:, 0] = predict
            verified_tokens = verified_tokens.flatten()
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

        # Construct the next draft input
        next_draft_input = NgramVerifyInput(
            server_args=self.server_args,
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            verified_tokens=verified_tokens,
            accept_lens=accept_length,
            is_spec_v2=True,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_length,
            next_draft_input=next_draft_input,
        )
