from __future__ import annotations

from typing import TYPE_CHECKING, List, Type

import torch
import triton
import triton.language as tl

from .build_egale_tree import build_tree_kernel
from sglang.srt.model_executor.forward_batch_info import ForwardMode, ForwardBatch

if TYPE_CHECKING:
    from python.sglang.srt.layers.sampler import SampleOutput
    from python.sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs


# Copy from sglang.srt.layers.flashinfer_utils.create_flashinfer_kv_indices_triton due to import error
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    max_context_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset

    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


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
    max_index = tl.argmax(accept_len_list, axis=0)
    accept_len = tl.max(accept_len_list)

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


class SpecInput:
    pass


class SpecDraftInput(SpecInput):
    def prepare_for_extend(self, batch):
        raise NotImplementedError()

    def prepare_for_decode(self, batch):
        raise NotImplementedError()

    def generate_attn_arg(
        self,
        req_pool_indices: List,
        paged_kernel_lens: List,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError()

    def clear():
        pass


class SpecVerifyInput(SpecInput):
    pass


class SpecDraftInfoFactory:
    def __init__(self):
        self.factory = {}

    def register(self, name: str) -> SpecDraftInput:
        def wrapper(info: Type[SpecDraftInput]) -> Type[SpecDraftInput]:
            self.factory[name] = info
            return info

        return wrapper

    def get(self, name):
        if name is None:
            return None
        return self.factory[name]


DraftInfoFactory = SpecDraftInfoFactory()


@DraftInfoFactory.register("EAGLE")
class EAGLEDraftInput(SpecDraftInput):
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None
    positions: torch.Tensor = None
    evict_mask: torch.Tensor = None

    def init(self, server_args: ServerArgs):
        self.prev_mode = ForwardMode.DECODE
        self.sample_output = None
        self.topk: int = 10
        self.num_verify_token: int = server_args.num_draft_tokens

        self.scores: torch.Tensor = None
        self.score_list: List[torch.Tensor] = []
        self.token_list: List[torch.Tensor] = []
        self.parents_list: List[torch.Tensor] = []
        self.cache_list: List[torch.Tenor] = []
        self.iter = 0
        self.root_token: int = None
        assert self.topk <= 10, "topk should <= 10"

    def prepare_for_extend(self, batch: ForwardBatch):
        req_pool_indices = batch.alloc_req_slots(len(batch.reqs))
        out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        
        pt=0
        for i, req in enumerate(batch.reqs):
            req.req_pool_idx = req_pool_indices[i]
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                batch.req_to_token_pool.req_to_token[req.req_pool_idx][
                    :pre_len
                ] = req.prefix_indices

            batch.req_to_token_pool.req_to_token[req.req_pool_idx][pre_len:seq_len] = (
                out_cache_loc[pt : pt + req.extend_input_len]
            )

            pt += req.extend_input_len
        
        seq_lens = [0] + batch.extend_lens
        input_ids = batch.input_ids.tolist()
        verified_id = batch.spec_info.verified_id.tolist()
        model_input_ids = []
        for i in range(len(seq_lens) - 1):
            model_input_ids.extend(
                input_ids[seq_lens[i] + 1 : seq_lens[i + 1]] + [verified_id[i]]
            )
        batch.input_ids = torch.tensor(
            model_input_ids, dtype=torch.int32, device="cuda"
        )

    def capture_for_decode(self, sample_output: SampleOutput, prev_mode: ForwardMode):
        self.sample_output = sample_output
        self.prev_mode = prev_mode

    def prepare_for_decode(self, batch: ScheduleBatch):
        prob = self.sample_output  # b * (1/topk), vocab
        top = torch.topk(prob, self.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values  # b * (1/topk), topk
        if self.prev_mode == ForwardMode.DECODE:
            scores = torch.mul(
                self.scores.unsqueeze(2), topk_p.reshape(-1, self.topk, self.topk)
            )  # (b, topk) mul (b * topk ,topk) -> b, topk, topk
            topk_cs = torch.topk(
                scores.flatten(start_dim=1), self.topk, dim=-1
            )  # (b, topk)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            self.scores = topk_cs_p

            selected_input_index = topk_cs_index.flatten() // self.topk  # b* topk

            batch.spec_info.hidden_states = batch.spec_info.hidden_states[
                selected_input_index, :
            ]
            batch.input_ids = torch.gather(
                topk_index.reshape(-1, self.topk**2), index=topk_cs_index, dim=1
            ).flatten()
            batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
            self.score_list.append(scores)
            self.token_list.append(topk_index)
            self.parents_list.append(
                topk_cs_index.flatten() + (self.topk**2 * (self.iter - 1) + self.topk)
            )

        elif self.prev_mode == ForwardMode.EXTEND:
            self.scores = topk_p  # b, top_k
            self.score_list.append(topk_p.unsqueeze(1))
            self.token_list.append(topk_index)
            batch.spec_info.hidden_states = (
                batch.spec_info.hidden_states.repeat_interleave(self.topk, 0)
            )
            batch.input_ids = topk_index.flatten()
            batch.out_cache_loc = batch.alloc_token_slots(topk_index.numel())
            self.parents_list.append(
                torch.arange(-1, self.topk, dtype=torch.int, device="cuda")
            )
        self.cache_list.append(batch.out_cache_loc)
        self.positions = (
            batch.seq_lens[:, None]
            + torch.ones([1, self.topk], device="cuda", dtype=torch.long) * self.iter - 1
        ).flatten()

        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices,
            batch.seq_lens
            + self.topk * self.iter : batch.seq_lens
            + self.topk * (self.iter + 1),
        ] = batch.out_cache_loc
        self.iter += 1

    def prepare_for_verify(self, batch: ScheduleBatch):
        score_list = torch.cat(self.score_list, dim=1).view(-1)  # b, 1/topk, topk
        ss_token_list = torch.cat(self.token_list, dim=0).view(
            -1
        )  # b * (self.topk+depth*self.topk)
        top_scores = torch.topk(score_list, self.num_verify_token - 1, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((self.verified_id, draft_tokens), dim=0)

        parent_list = torch.cat(self.parents_list[:-1], dim=0)

        tree_mask, position, retrive_index, retrive_cum_len = build_tree_kernel(
            parent_list,
            top_scores_index,
            batch.seq_lens,
            self.topk,
            self.iter - 1,
            self.num_verify_token,
        )

        # out_cache = torch.cat(self.cache_list, dim=0)
        # mem_need_free_idx = out_cache[top_scores_index]
        # batch.token_to_kv_pool.free(mem_need_free_idx)
        
        return EagleVerifyInput(
            draft_tokens,
            tree_mask,
            position-1,
            retrive_index,
            retrive_cum_len,
            self.num_verify_token,
        )

    def prepare_new_draft_stage(self, batch: ScheduleBatch):
        batch.input_ids = self.verified_id

    def generate_attn_arg(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
    ):
        req_pool_indices = req_pool_indices.tolist()
        paged_kernel_lens = paged_kernel_lens.tolist()
        bs = self.topk * len(req_pool_indices)
        seq_len = self.positions.reshape(-1).contiguous()
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(seq_len + 1, dim=0)
        kv_last_page_len = torch.ones((bs,), dtype=torch.int32, device="cuda")
        kv_indices_list = []
        # TODO: reimplement it by triton @kavioyu
        for i in range(len(req_pool_indices)):
            for k in range(self.topk):
                index = torch.arange(self.iter) * self.topk + k
                kv_indices_list.append(
                    req_to_token_pool.req_to_token[
                        req_pool_indices[i], : paged_kernel_lens[i]
                    ]
                )
                kv_indices_list.append(
                    req_to_token_pool.req_to_token[
                        req_pool_indices[i], paged_kernel_lens[i] + index
                    ]
                )
        kv_indices = torch.cat(kv_indices_list, dim=0).contiguous()
        return kv_indices, cum_kv_seq_len, kv_last_page_len, None

    def clear(self):
        self.iter = 0
        self.score_list.clear()
        self.positions = None


class EagleVerifyInput(SpecVerifyInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_cum_len: torch.Tensor,
        draft_token_num: int,
    ):
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_cum_len = retrive_cum_len
        self.draft_token_num = draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.input_ids = self.draft_token
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices,
            batch.seq_lens : batch.seq_lens + self.draft_token_num,
        ] = batch.out_cache_loc

    def generate_attn_arg(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
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
        
        paged_kernel_lens = paged_kernel_lens.add_(self.draft_token_num)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")
        
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token_pool.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token_pool.req_to_token.size(1),
        )

        return kv_indices, cum_kv_seq_len, kv_last_page_len, qo_indptr

    def verify(self, batch: ScheduleBatch, logits_output: torch.Tensor) -> torch.Tensor:
        predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = predict[self.retrive_index]
        candidates = self.draft_token[self.retrive_index]
        # logits = logits_output.next_token_logits[self.retrive_index]
        # target_predict = torch.argmax(logits[:, :-1], dim=-1)
        accept_mask = candidates[:, 1:] == target_predict[:, :-1]
        accept_mask = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)
        bs = self.retrive_cum_len.numel() - 1

        max_draft_len = self.retrive_index.shape[-1]
        accept_index = torch.full(
            (bs, max_draft_len), -1, dtype=torch.long, device="cuda"
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
        accept_index = accept_index[accept_index != -1]
        extract_index = extract_index[extract_index != 0]

        batch.spec_info.verified_id = predict[extract_index]
        batch.spec_info.hidden_states = batch.spec_info.hidden_states[
            extract_index
        ]

        accept_length_cpu = accept_length.tolist()
        verified_id_cpu = predict[accept_index].tolist()
        print(verified_id_cpu)

        low = 0
        for req, verified_len in zip(batch.reqs, accept_length_cpu):
            req.output_ids.extend(verified_id_cpu[low : low + verified_len + 1])
            low += verified_len

        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        mem_need_free_idx = batch.out_cache_loc[evict_mask]

        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices,
            batch.seq_lens : batch.seq_lens + self.draft_token_num,
        ] = batch.out_cache_loc

        batch.token_to_kv_pool.free(mem_need_free_idx)
        batch.spec_info.evict_mask = evict_mask

        return batch.spec_info.verified_id

