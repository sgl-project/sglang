from typing import List, Union

import torch
import triton

from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import assign_req_to_token_pool
from sglang.srt.speculative.spec_info import SpecInfo


class NGramWorker:

    def __init__(
        self,
        target_worker: TpModelWorker,
        server_args: ServerArgs,
    ):
        self.target_worker = target_worker
        self.draft_token_num = server_args.speculative_num_draft_tokens
        self.ngram_window_size = server_args.speculative_ngram_window_size
        # Don't support prefix share now.

    def finish_request(self, reqs: Union[Req, List[Req]]):
        pass
        # if not isinstance(reqs, List):
        #     reqs = [reqs]
        # for req in reqs:
        # req_len = (
        #     len(req.origin_input_ids)
        #     + len(req.output_ids)
        #     - self.finish_extend_len[req.rid]
        #     - 1
        # )
        # kv_indices = self.model_runner.req_to_token_pool.req_to_token[
        #     req.req_pool_idx
        # ][:req_len]
        # self.model_runner.token_to_kv_pool.free(kv_indices)
        # self.model_runner.req_to_token_pool.free(req.req_pool_idx)

    def find_candidate_pred_tokens(self, input_tokens: torch.Tensor):
        input_length = input_tokens.size(1)
        draft_tokens_start_pos = input_length
        for ngram_size in range(self.ngram_window_size, 0, -1):
            ngram = input_tokens[0, -ngram_size:].tolist()
            windows = input_tokens.unfold(dimension=1, size=ngram_size, step=1)
            ngram_tensor = torch.tensor(ngram, device=input_tokens.device).unsqueeze(0)
            matches = (windows == ngram_tensor).all(dim=2)
            match_indices = matches.nonzero(as_tuple=True)[1]
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.draft_token_num
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_tokens[0, start_idx:end_idx]

        return torch.tensor([], dtype=torch.long, device=input_tokens.device)

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, ForwardBatch]:
        if batch.forward_mode.is_decode():
            draft_tokens = self.find_candidate_pred_tokens(
                torch.tensor([req.fill_ids for req in batch.reqs], device=batch.device)
            )
            batch.spec_info.draft_tokens = draft_tokens
            (
                logits_output,
                verified_id,
                self.finish_extend_len,
                model_worker_batch,
            ) = self.verify(batch)
            next_spec_info = NGramSpecInfo(
                draft_token_num=self.draft_token_num,
                verified_id=verified_id,
                draft_tokens=draft_tokens,
                positions=batch.spec_info.positions,
            )
            return logits_output, verified_id, model_worker_batch, next_spec_info
        else:
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            ngramSpecInfo = NGramSpecInfo(
                draft_token_num=self.draft_token_num,
                verified_id=next_token_ids,
                draft_tokens=None,
                positions=None,
            )
            return logits_output, next_token_ids, model_worker_batch, ngramSpecInfo

    def verify(self, batch: ScheduleBatch):
        batch.spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = torch.cat(
            [predict, torch.full([1], -1, dtype=torch.long, device="cuda")], dim=-1
        )
        draft_tokens = torch.cat(
            [
                batch.spec_info.draft_tokens,
                torch.full([1], -1, dtype=torch.long, device="cuda"),
            ],
            dim=-1,
        )

        accept_mask = draft_tokens[:, 1:] == target_predict[:, :-1]
        accept_length = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)
        accept_length_cpu = accept_length.tolist()
        accept_index = torch.cat(
            [torch.arange(length, device="cuda") for length in accept_length]
        )
        verified_id = predict[accept_index]
        verified_id_cpu = verified_id.tolist()

        # TODO: Verify whether this is correct mem free idx
        evict_mask = torch.full_like(
            batch.spec_info.draft_tokens, True, dtype=torch.bool
        )
        evict_mask[accept_mask] = False
        mem_need_free_idx = batch.out_cache_loc[evict_mask]
        batch.token_to_kv_pool.free(mem_need_free_idx)

        # TODO: Verify whether this is correct  bs
        # from remote_pdb import set_trace
        # set_trace(host = "127.0.0.1", port=7728)
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_mask + 1,
            batch.out_cache_loc[accept_index],
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

        batch.seq_lens.add_(accept_mask + 1)
        new_accept_index = []
        unfinished_index = []
        finished_extend_len = {}  # {rid:accept_length + 1}

        low = 0
        for i, (req, verified_len) in enumerate(zip(batch.reqs, accept_length_cpu)):
            req.output_ids.extend(verified_id_cpu[low : low + verified_len + 1])
            req.check_finished()
            if not req.finished():
                new_accept_index.append(accept_index[low : low + verified_len + 1])
                unfinished_index.append(i)
            low += verified_len + 1
            finished_extend_len[req.rid] = verified_len + 1

        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        batch.forward_mode = ForwardMode.DECODE
        return (
            logits_output,
            verified_id,
            finished_extend_len,
            model_worker_batch,
        )


class NGramSpecInfo(SpecInfo):
    def __init__(
        self,
        draft_token_num: int,
        verified_id: torch.Tensor,
        draft_tokens: torch.Tensor,
        positions: torch.Tensor,
    ):
        self.draft_token_num = draft_token_num
        self.verified_id = verified_id
        self.draft_tokens = draft_tokens
        self.positions = positions

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")

        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def prepare_for_verify(self, batch: ScheduleBatch):
        self.draft_tokens = torch.cat(
            (self.verified_id.unsqueeze(1), self.draft_tokens), dim=1
        ).flatten()
        self.positions = torch.tensor(
            [
                list(range(seq_len, seq_len + self.draft_tokens.size(0)))
                for seq_len in batch.seq_lens.tolist()
            ],
            device=batch.seq_lens.device,
        ).flatten()
        batch.input_ids = self.draft_tokens
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )
