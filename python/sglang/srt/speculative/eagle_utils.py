from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.build_eagle_tree import (
    build_tree_kernel,
    build_tree_kernel_efficient,
)
from sglang.srt.utils import is_cuda_available

if is_cuda_available():
    from sgl_kernel import tree_speculative_sampling_target_only

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    def prepare_for_extend(self, batch: ScheduleBatch):
        req_pool_indices = batch.alloc_req_slots(len(batch.reqs))
        out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        batch.out_cache_loc = out_cache_loc

        pt = 0
        for i, req in enumerate(batch.reqs):
            req.req_pool_idx = req_pool_indices[i]
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                batch.req_to_token_pool.req_to_token[req.req_pool_idx][
                    :pre_len
                ] = req.prefix_indices

            batch.req_to_token_pool.req_to_token[req.req_pool_idx, pre_len:seq_len] = (
                out_cache_loc[pt : pt + req.extend_input_len]
            )

            pt += req.extend_input_len

        # TODO: support batching inputs
        assert len(batch.extend_lens) == 1
        batch.input_ids = torch.concat((batch.input_ids[1:], self.verified_id))

    def prepare_extend_after_decode(self, batch: ScheduleBatch, speculative_num_steps):
        batch.out_cache_loc = batch.alloc_token_slots(self.verified_id.numel())
        accept_length_cpu = batch.spec_info.accept_length_cpu
        batch.extend_lens = [x + 1 for x in accept_length_cpu]
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend
        seq_lens_cpu = batch.seq_lens.tolist()

        pt = 0
        i = 0
        for req in batch.reqs:
            if req.finished():
                continue
            # assert seq_len - pre_len == req.extend_input_len
            input_len = batch.extend_lens[i]
            seq_len = seq_lens_cpu[i]
            batch.req_to_token_pool.req_to_token[req.req_pool_idx][
                seq_len - input_len : seq_len
            ] = batch.out_cache_loc[pt : pt + input_len]
            pt += input_len
            i += 1
        assert pt == batch.out_cache_loc.shape[0]

        self.positions = torch.empty_like(self.verified_id)
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.long)
        self.accept_length.add_(1)

        create_extend_spec_info[(self.accept_length.numel(),)](
            self.verified_id,
            batch.seq_lens,
            self.accept_length,
            torch.cumsum(self.accept_length, axis=0, dtype=torch.int),
            self.positions,
            new_verified_id,
            triton.next_power_of_2(speculative_num_steps + 1),
        )

        batch.seq_lens_sum = sum(seq_lens_cpu)
        batch.input_ids = self.verified_id
        self.verified_id = new_verified_id

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        bs = self.accept_length.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor):
        self.topk_p = self.topk_p[: len(new_indices)]
        self.topk_index = self.topk_index[: len(new_indices)]
        self.hidden_states = self.hidden_states[: len(new_indices)]
        self.verified_id = self.verified_id[: len(new_indices)]

    def merge_batch(self, spec_info: EagleDraftInput):
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclasses.dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    draft_token_num: int
    spec_steps: int
    capture_hidden_mode: CaptureHiddenMode

    @classmethod
    def create(
        cls,
        verified_id: torch.Tensor,
        score_list: List[torch.Tensor],
        token_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        topk: int,
        spec_steps: int,
        num_verify_tokens: int,
        is_all_greedy: bool,
    ):
        if is_all_greedy:
            tree_mask, position, retrive_index, retrive_cum_len, draft_tokens = (
                build_tree_kernel(
                    verified_id,
                    score_list,  # b, n, topk; n= 1 + (num_steps-1) * self.topk
                    token_list,
                    parents_list,
                    seq_lens,
                    seq_lens_sum,
                    topk,
                    spec_steps,
                    num_verify_tokens,
                )
            )

            return cls(
                draft_tokens,
                tree_mask,
                position,
                retrive_index,
                None,
                None,
                retrive_cum_len,
                num_verify_tokens,
                spec_steps,
                CaptureHiddenMode.FULL,
            )
        else:
            (
                tree_mask,
                position,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                draft_tokens,
            ) = build_tree_kernel_efficient(
                verified_id,
                score_list,
                token_list,
                parents_list,
                seq_lens,
                seq_lens_sum,
                topk,
                spec_steps,
                num_verify_tokens,
            )

            return cls(
                draft_tokens,
                tree_mask,
                position,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                None,
                num_verify_tokens,
                spec_steps,
                CaptureHiddenMode.FULL,
            )

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.input_ids = self.draft_token
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

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
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def verify(self, batch: ScheduleBatch, logits_output: torch.Tensor) -> torch.Tensor:
        draft_token = torch.cat(
            [self.draft_token, torch.full([1], -1, dtype=torch.int32, device="cuda")],
            dim=-1,
        )
        candidates = draft_token[self.retrive_index]
        if batch.sampling_info.is_all_greedy:
            # temp == 0
            bs = self.retrive_cum_len.numel() - 1
            predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            predict = torch.cat(
                [predict, torch.full([1], -1, dtype=torch.int32, device="cuda")], dim=-1
            )
            target_predict = predict[self.retrive_index]
            # logits = logits_output.next_token_logits[self.retrive_index]
            # target_predict = torch.argmax(logits[:, :-1], dim=-1)
            accept_mask = candidates[:, 1:] == target_predict[:, :-1]

            accept_mask = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)
            max_draft_len = self.retrive_index.shape[-1]
            accept_index = torch.full(
                (bs, max_draft_len), -1, dtype=torch.int32, device="cuda"
            )
            accept_length = torch.empty((bs,), dtype=torch.int, device="cuda")
            extract_index = torch.full((bs * 2,), 0, dtype=torch.int, device="cuda")
            eagle_verify_retrive[(bs,)](
                self.retrive_index.contiguous(),
                accept_mask.contiguous(),
                self.retrive_cum_len,
                accept_index,
                accept_length,
                extract_index,
                max_draft_len,
                self.draft_token_num,
                triton.next_power_of_2(max_draft_len),
            )
        else:
            # temp > 0
            bs = self.retrive_index.shape[0]
            predict_shape = list(logits_output.next_token_logits.shape)[:-1]
            predict_shape[-1] += 1
            target_logits = logits_output.next_token_logits[self.retrive_index]
            predict = torch.full(predict_shape, -1, dtype=torch.int32, device="cuda")
            accept_index = torch.full(
                (bs, self.spec_steps + 1), -1, dtype=torch.int32, device="cuda"
            )
            accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")
            expanded_temperature = batch.sampling_info.temperatures.unsqueeze(1)
            target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
            draft_probs = torch.full_like(
                target_probs, 0, dtype=torch.float32, device="cuda"
            )
            coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates.to(torch.int32),
                retrive_index=self.retrive_index.to(torch.int32),
                retrive_next_token=self.retrive_next_token.to(torch.int32),
                retrive_next_sibling=self.retrive_next_sibling.to(torch.int32),
                uniform_samples=coins,
                target_probs=target_probs,
                draft_probs=draft_probs,
                deterministic=True,
            )

        new_accept_index = []
        unfinished_index = []
        finished_extend_len = {}  # {rid:accept_length + 1}
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            new_accept_index_ = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                # if not found_finished:
                req.output_ids.append(id)
                finished_extend_len[req.rid] = j + 1
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    new_accept_index_.append(idx)
            if not req.finished():
                new_accept_index.extend(new_accept_index_)
                unfinished_index.append(i)
            req.spec_verify_ct += 1
        accept_length = (accept_index != -1).sum(dim=1) - 1

        accept_index = accept_index[accept_index != -1]
        accept_length_cpu = accept_length.tolist()
        verified_id = predict[accept_index]

        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        mem_need_free_idx = batch.out_cache_loc[evict_mask]
        batch.token_to_kv_pool.free(mem_need_free_idx)
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length + 1,
            batch.out_cache_loc[accept_index],
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )
        batch.seq_lens.add_(accept_length + 1)

        draft_input = EagleDraftInput()
        if len(new_accept_index) > 0:
            new_accept_index = torch.tensor(new_accept_index, device="cuda")
            draft_input.hidden_states = batch.spec_info.hidden_states[new_accept_index]
            draft_input.verified_id = predict[new_accept_index]
            draft_input.accept_length = accept_length[unfinished_index]
            draft_input.accept_length_cpu = [
                accept_length_cpu[i] for i in unfinished_index
            ]
            if has_finished:
                draft_input.seq_lens_for_draft_extend = batch.seq_lens[unfinished_index]
                draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices[
                    unfinished_index
                ]
            else:
                draft_input.seq_lens_for_draft_extend = batch.seq_lens
                draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices

        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        return (
            draft_input,
            logits_output,
            verified_id,
            finished_extend_len,
            accept_length_cpu,
        )


@triton.jit
def eagle_verify_retrive(
    retrive_index,
    accept_mask,
    retrive_cum_len,
    accept_index,
    accept_length,
    extract_index,
    max_len: tl.constexpr,
    draft_token_num: tl.constexpr,
    max_len_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    retrive_end = tl.load(retrive_cum_len + pid + 1)
    retrive_start = tl.load(retrive_cum_len + pid)
    retrive_len = retrive_end - retrive_start
    accept_ptr = accept_mask + retrive_start
    accept_offset = tl.arange(0, draft_token_num)
    accept_load_mask = accept_offset < retrive_len
    accept_len_list = tl.load(
        accept_ptr + accept_offset, mask=accept_load_mask, other=-1
    )

    accept_len = tl.max(accept_len_list)
    max_index = tl.argmax(accept_len_list, axis=0, tie_break_left=True)
    # triton is not support argmax with tie_break_right, so I need implement it by some way
    mask_max = accept_len_list == accept_len

    count_mask = tl.full(shape=[draft_token_num], value=0, dtype=tl.int32)
    count = tl.sum(tl.where(mask_max, 1, count_mask))
    if count > 1:
        index = tl.arange(0, draft_token_num)
        mask_left = index != max_index
        remained_index = tl.where(mask_max and mask_left, index, 0)
        max_index = tl.max(remained_index)

    tl.store(accept_length + pid, accept_len)
    retrive_index_ptr = retrive_index + (retrive_start + max_index) * max_len
    retrive_offset = tl.arange(0, max_len_upper)
    retrive_load_mask = retrive_offset < accept_len + 1
    data = tl.load(retrive_index_ptr + retrive_offset, mask=retrive_load_mask)

    tl.store(
        accept_index + pid * max_len + retrive_offset, data, mask=retrive_load_mask
    )

    extract_load_ptr = accept_index + pid * max_len + accept_len
    if accept_len == max_len - 1:
        extract_data = tl.load(extract_load_ptr - 1)
        tl.store(extract_index + pid * 2, extract_data)
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2 + 1, extract_data)

    else:
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2, extract_data)


@triton.jit
def create_extend_spec_info(
    verified_id,
    seq_len,
    accept_len,
    accept_len_cum,
    positions,
    new_verified_id,
    accept_len_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = 0 if pid == 0 else tl.load(accept_len_cum + pid - 1)
    seq_length = tl.load(seq_len + pid)
    accept_length = tl.load(accept_len + pid)
    positions_ptr = positions + offset
    data = tl.arange(0, accept_len_upper)
    mask = data < accept_length
    tl.store(positions_ptr + data, seq_length - accept_length + data, mask)

    offset = tl.load(accept_len_cum + pid) - 1
    verified_id_data = tl.load(verified_id + offset)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(seq_lens + pid)
    kv_end = tl.load(seq_lens + pid) + topk * speculative_num_steps
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    num_loop = tl.cdiv(topk * speculative_num_steps, BLOCK_SIZE)
    for i in range(num_loop):
        save_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE + kv_start
        load_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    num_seqs: tl.constexpr,
    topk: tl.constexpr,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    extend_data = tl.load(
        token_pool_ptr + seq_len + tl.arange(0, iter_upper) * topk + topk_id,
        mask=extend_offset < iters,
    )
    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@torch.compile
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device="cuda")
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )

    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)

        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
            0, hidden_states.shape[0], step=topk, device="cuda"
        ).repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        max_value, max_index = torch.max(values, dim=dim)
        return max_value.unsqueeze(1), max_index.unsqueeze(1)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)
