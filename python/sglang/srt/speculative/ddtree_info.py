from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class DDTreeVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    tree_budget: int

    child_maps: List[Dict[int, Dict[int, int]]] = field(default_factory=list)
    actual_tree_sizes: Optional[torch.Tensor] = None

    custom_mask: Optional[torch.Tensor] = None
    topk: int = 1

    accepted_indices: List[List[int]] = field(default_factory=list)
    next_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)

    def get_spec_adjust_token_coefficient(self):
        return (self.draft_token_num, self.draft_token_num)

    def prepare_for_verify(self, batch, page_size):
        bs = len(batch.reqs)
        q_len = self.draft_token_num

        allocator = batch.token_to_kv_pool_allocator
        out_cache_loc = allocator.alloc(bs * q_len)

        for i, req in enumerate(batch.reqs):
            start_idx = i * q_len
            batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, batch.seq_lens[i]:batch.seq_lens[i] + q_len
            ] = out_cache_loc[start_idx:start_idx + q_len]

        batch.out_cache_loc = out_cache_loc

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(self, *, batch, logits_output, page_size):
        from sglang.srt.speculative.ddtree_utils import (
            follow_verified_tree,
            compact_ddtree_kv_cache,
        )

        bs = len(batch.reqs)

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        self.accepted_indices, self.next_tokens = follow_verified_tree(
            self.child_maps, target_predict
        )

        commit_lens = []
        num_correct_drafts_per_req = []
        for i, req in enumerate(batch.reqs):
            accepted = self.accepted_indices[i]
            num_accepted = len(accepted) - 1

            for idx in accepted[1:]:
                token_id = int(self.draft_token[i * self.draft_token_num + idx])
                req.output_ids.append(token_id)
                req.check_finished()
                if req.finished():
                    break

            if not req.finished():
                bonus = int(self.next_tokens[i].item())
                req.output_ids.append(bonus)
                req.check_finished()

            commit_lens.append(len(accepted))
            num_correct_drafts_per_req.append(num_accepted)

        commit_lens_tensor = torch.tensor(commit_lens, dtype=torch.long, device=batch.device)
        past_lengths = batch.seq_lens.clone()

        batch.seq_lens += commit_lens_tensor

        for layer in batch.model_runner.model.layers:
            attn_layer = layer.self_attn.attn
            compact_ddtree_kv_cache(
                batch.token_to_kv_pool,
                attn_layer,
                batch.out_cache_loc.view(bs, self.draft_token_num),
                self.accepted_indices,
                past_lengths,
                self.actual_tree_sizes,
            )

        hidden = logits_output.hidden_states
        if hidden is not None:
            hidden = hidden.view(bs, self.draft_token_num, -1)
            segments = []
            for i, n in enumerate(commit_lens):
                segments.append(hidden[i, :n, :])
            next_target_hidden = torch.cat(segments, dim=0)
        else:
            next_target_hidden = None

        num_correct_drafts_cpu = num_correct_drafts_per_req

        return self.next_tokens, commit_lens_tensor, next_target_hidden, num_correct_drafts_cpu
