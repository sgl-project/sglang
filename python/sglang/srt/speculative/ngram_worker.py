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
        self.max_num_draft_tokens = server_args.speculative_num_draft_tokens
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

    def find_candidate_pred_tokens(self, batch: ScheduleBatch):
        input_tokens = torch.cat(
            [
                torch.tensor([req.fill_ids for req in batch.reqs], device=batch.device),
                batch.spec_info.verified_ids,
            ],
            dim=-1,
        )
        input_length = input_tokens.size(1)
        for ngram_size in range(self.ngram_window_size, 0, -1):
            ngram = input_tokens[0, -ngram_size:].tolist()
            windows = input_tokens.unfold(dimension=1, size=ngram_size, step=1)
            ngram_tensor = torch.tensor(ngram, device=input_tokens.device).unsqueeze(0)
            matches = (windows == ngram_tensor).all(dim=2)
            match_indices = matches.nonzero(as_tuple=True)[1]
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.max_num_draft_tokens
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_tokens[:, start_idx:end_idx]

        return torch.tensor([], dtype=torch.long, device=input_tokens.device)

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, ForwardBatch]:
        if batch.forward_mode.is_decode():
            draft_tokens = self.find_candidate_pred_tokens(batch)
            batch.spec_info.draft_tokens = draft_tokens
            (
                logits_output,
                verified_ids,
                self.finish_extend_len,
                model_worker_batch,
            ) = self.verify(batch)
            next_spec_info = NGramSpecInfo(
                max_num_draft_tokens=self.max_num_draft_tokens,
                verified_ids=verified_ids,
                draft_tokens=draft_tokens,
                positions=batch.spec_info.positions,
            )
            return logits_output, verified_ids, model_worker_batch, next_spec_info
        else:
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            ngramSpecInfo = NGramSpecInfo(
                max_num_draft_tokens=self.max_num_draft_tokens,
                verified_ids=next_token_ids.unsqueeze(1),  # bs * 1
                draft_tokens=None,
                positions=None,
            )
            return logits_output, next_token_ids, model_worker_batch, ngramSpecInfo

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.spec_info.draft_tokens = torch.cat(
            (batch.spec_info.verified_ids[:, -1:], batch.spec_info.draft_tokens), dim=1
        ).flatten()
        batch.spec_info.positions = torch.tensor(
            [
                list(range(seq_len, seq_len + batch.spec_info.draft_tokens.size(0)))
                for seq_len in batch.seq_lens.tolist()
            ],
            device=batch.seq_lens.device,
        ).flatten()
        batch.input_ids = batch.spec_info.draft_tokens
        seq_lens_cpu = batch.seq_lens.item()
        input_ids_cpu = batch.input_ids.numel()
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        # should use assign_req_to_token_pool
        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_cpu : (seq_lens_cpu + input_ids_cpu)
        ] = batch.out_cache_loc
        batch.seq_lens.add_(batch.input_ids.numel())

    def verify(self, batch: ScheduleBatch):
        self.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        # == forward_batch_generation ==
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        predict = torch.argmax(logits_output.next_token_logits, dim=-1)

        # == compare draft tokes and predict tokens ==
        target_predict = torch.cat(
            [
                torch.full(
                    [1],
                    batch.spec_info.draft_tokens[0],
                    dtype=torch.long,
                    device="cuda",
                ),
                predict,
            ],
            dim=-1,
        ).unsqueeze(0)
        draft_tokens = torch.cat(
            [
                torch.full([1], -1, dtype=torch.long, device="cuda"),
                batch.spec_info.draft_tokens,
            ],
            dim=-1,
        ).unsqueeze(0)
        accept_mask = draft_tokens[:, 1:] == target_predict[:, :-1]
        accept_mask = torch.cat(
            [accept_mask, torch.zeros((1, 1), dtype=torch.bool, device="cuda")], dim=1
        )
        accept_length = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)
        accept_length_cpu = accept_length.tolist()
        accept_index = torch.cat(
            [torch.arange(length, device="cuda") for length in accept_length]
        )
        verified_ids = predict[accept_index]
        verified_ids_cpu = verified_ids.tolist()

        # == update batch req_to_token_pool and token_to_kv_pool ==
        mem_need_free_idx = batch.out_cache_loc[accept_length:]
        batch.token_to_kv_pool.free(mem_need_free_idx)
        batch.seq_lens.sub_(batch.input_ids.shape[0] - accept_index.shape[0])
        # should use `assign_req_to_token_pool`?
        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens :
        ] = torch.full_like(
            batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices, batch.seq_lens :
            ],
            0,
        )
        if batch.input_ids.shape[0] != accept_index.shape[0]:
            batch.out_cache_loc = batch.out_cache_loc[:-accept_length]

        finished_extend_len = {}  # {rid:accept_length + 1}
        low = 0
        for i, (req, verified_len) in enumerate(zip(batch.reqs, accept_length_cpu)):
            req.output_ids.extend(verified_ids_cpu[low : low + verified_len + 1])
            low += verified_len + 1
            finished_extend_len[req.rid] = verified_len + 1

        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        batch.forward_mode = ForwardMode.DECODE
        return (
            logits_output,
            verified_ids.unsqueeze(0),
            finished_extend_len,
            model_worker_batch,
        )


class NGramSpecInfo(SpecInfo):
    def __init__(
        self,
        max_num_draft_tokens: int,
        verified_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        positions: torch.Tensor,
    ):
        self.max_num_draft_tokens = max_num_draft_tokens
        self.verified_ids = verified_ids
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
            (1 + batch_size) * self.max_num_draft_tokens,
            step=self.max_num_draft_tokens,
            dtype=torch.int32,
            device="cuda",
        )

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )

        paged_kernel_lens = paged_kernel_lens + self.max_num_draft_tokens
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
