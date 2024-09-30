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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Type

import torch
import triton

from python.sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    InputMetadata,
)

if TYPE_CHECKING:
    from python.sglang.srt.layers.sampler import SampleOutput
    from python.sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class SpecDraftInput:
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


class SpecVerifyInput:
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
        return self.factory[name]


DraftInfoFactory = SpecDraftInfoFactory()


@DraftInfoFactory.register("EAGLE")
class EAGLEDraftInput(SpecDraftInput):
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None

    def init(self):
        self.prev_mode = None
        self.sample_output = None
        self.topk: int = 10
        self.num_verify_token: int = 64

        self.scores: torch.Tensor = None
        self.score_list: List[torch.Tensor] = []
        self.token_list: List[torch.Tensor] = []
        self.parents_list: List[torch.Tensor] = []
        self.iter = 0
        self.root_token: int = None

    positions: torch.Tensor = None

    def prepare_for_extend(self, batch):
        seq_lens = [0] + batch.seq_lens.tolist()
        input_ids = batch.input_ids.tolist()
        verified_id = self.verified_id.tolist()
        model_input_ids = []
        for i in range(len(seq_lens) - 1):
            model_input_ids.extend(
                input_ids[seq_lens[i] + 1 : seq_lens[i + 1]] + [verified_id[i]]
            )
        batch.input_ids = torch.tensor(
            model_input_ids, dtype=torch.int32, device="cuda"
        )
        self.verified_id = self.verified_id.clone()
        # need consider bs @kavioyu

    def capture_for_decode(self, sample_output: SampleOutput, prev_mode: ForwardMode):
        self.sample_output = sample_output
        self.prev_mode = prev_mode

    def prepare_for_decode(self, batch: ScheduleBatch):
        prob = self.sample_output  # b * (1/topk), vocab
        top = torch.topk(prob, self.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values  # b * (1/topk), topk
        if self.prev_mode == ForwardMode.SPECDECODE:
            scores = torch.mul(
                self.scores.unsqueeze(2), topk_p.reshape(-1, self.topk, self.topk)
            )  # (b, topk) mul (b * topk ,topk) -> b, topk, topk
            topk_cs = torch.topk(
                scores.flatten(start_dim=1), self.topk, dim=-1
            )  # (b, topk)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            self.scores = topk_cs_p

            selected_input_index = topk_cs_index.flatten() // self.topk  # b* topk

            batch.spec_draft_input.hidden_states = batch.spec_draft_input.hidden_states[
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

        elif self.prev_mode == ForwardMode.SPECEXTEND:
            self.scores = topk_p  # b, top_k
            self.score_list.append(topk_p.unsqueeze(1))
            self.token_list.append(topk_index)
            batch.spec_draft_input.hidden_states = (
                batch.spec_draft_input.hidden_states.repeat_interleave(self.topk, 0)
            )
            batch.input_ids = topk_index.flatten()
            batch.out_cache_loc = batch.alloc_token_slots(topk_index.numel())
            self.parents_list.append(
                torch.arange(-1, self.topk, dtype=torch.int, device="cuda")
            )

        self.positions = (
            batch.seq_lens[:, None]
            + torch.ones([1, self.topk], device="cuda") * self.iter
        ).flatten()

        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices,
            batch.seq_lens
            + self.topk * self.iter : batch.seq_lens
            + self.topk * (self.iter + 1),
        ] = batch.out_cache_loc
        self.iter += 1

    def prepare_for_verify(self):
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
        torch.save(top_scores, "score_list.pth")
        torch.save(top_scores, "scores.pth")
        print(parent_list.shape)
        print(parent_list)
        print(top_scores_index.shape)
        print(top_scores_index)

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
        return kv_indices, cum_kv_seq_len, kv_last_page_len

    def clear(self):
        self.iter = 0
        self.score_list.clear()
        self.positions = None


class SpecInfoPipline:
    def __init__(self):
        ctx = torch.multiprocessing.get_context("forkserver")
        self.draft_input_queue = ctx.Queue()
        self.draft_output_queue = ctx.Queue()
        self.max_total_num_tokens = ctx.Value("i", -1)
