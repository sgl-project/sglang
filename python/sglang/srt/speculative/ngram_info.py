from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class NgramVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor = None,
        custom_mask: torch.Tensor = None,
        positions: torch.Tensor = None,
        retrieve_index: torch.Tensor = None,
        retrieve_next_token: torch.Tensor = None,
        retrieve_next_sibling: torch.Tensor = None,
        draft_token_num: int = None,
        grammar: BaseGrammarObject = None,
        future_indices: Optional[torch.Tensor] = None,
        new_seq_lens: Optional[torch.Tensor] = None,
        accept_tokens: Optional[torch.Tensor] = None,
        accept_lens: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.NGRAM_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.draft_token_num = draft_token_num
        self.num_tokens_per_req = draft_token_num
        self.num_tokens_for_logprob_per_req = draft_token_num
        self.grammar = grammar

        # Inputs for V2 overlap worker
        self.future_indices = future_indices
        self.new_seq_lens = new_seq_lens
        self.accept_tokens = accept_tokens
        self.accept_lens = accept_lens

        self.device = (
            custom_mask.device if custom_mask is not None else new_seq_lens.device
        )

    @property
    def max_tree_depth(self) -> int:
        # NGRAM trees are node-budgeted with no depth cap: the corpus BFS only
        # stops on the node budget, so a single long match can chain all
        # draft_token_num nodes (spec_steps is meaningless for this tree).
        return self.draft_token_num

    @property
    def tree_topk(self) -> int:
        # Irregular tree: per-level branching follows the corpus matches.
        return -1

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=self.device,
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

        # Pad custom_mask when CUDA graph pads batch size beyond the actual number of requests.
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * bs
        )
        custom_mask = self.custom_mask
        if custom_mask.numel() < mask_numel:
            custom_mask = torch.cat(
                [
                    custom_mask,
                    torch.full(
                        (mask_numel - custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, self.qo_indptr, custom_mask

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
        if self.new_seq_lens is not None:
            self.new_seq_lens = self.new_seq_lens[new_indices]
        self.accept_tokens = self.accept_tokens.reshape(-1, self.draft_token_num)[
            new_indices, :
        ]
        self.accept_tokens = self.accept_tokens.flatten()
        self.accept_lens = self.accept_lens[new_indices]

    def merge_batch(self, spec_info: NgramVerifyInput):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                (self.future_indices, spec_info.future_indices), dim=0
            )
        if self.new_seq_lens is not None:
            assert spec_info.new_seq_lens is not None
            self.new_seq_lens = torch.cat(
                (self.new_seq_lens, spec_info.new_seq_lens), dim=0
            )
        self.accept_tokens = torch.cat(
            (self.accept_tokens, spec_info.accept_tokens), dim=0
        )
        self.accept_lens = torch.cat((self.accept_lens, spec_info.accept_lens), dim=0)
