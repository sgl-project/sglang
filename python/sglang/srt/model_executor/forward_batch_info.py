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

"""ModelRunner runs the forward passes of the models."""
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    batch_size: int
    total_num_tokens: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool

    # For extend
    extend_seq_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    extend_no_prefix: bool

    # Output location of the KV cache
    out_cache_loc: torch.Tensor = None

    # Output options
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None

    # Trition attention backend
    triton_max_seq_len: int = 0
    triton_max_extend_len: int = 0
    triton_start_loc: torch.Tensor = None
    triton_prefix_lens: torch.Tensor = None

    # FlashInfer attention backend
    flashinfer_prefill_wrapper_ragged: "BatchPrefillWithRaggedKVCacheWrapper" = None
    flashinfer_prefill_wrapper_paged: "BatchPrefillWithPagedKVCacheWrapper" = None
    flashinfer_decode_wrapper: "BatchDecodeWithPagedKVCacheWrapper" = None
    flashinfer_use_ragged: bool = False

    @classmethod
    def create(
        cls,
        model_runner,
        forward_mode,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
        top_logprobs_nums=None,
        return_logprob=False,
        skip_flashinfer_init=False,
    ):
        flashinfer_use_ragged = False
        if not skip_flashinfer_init and not model_runner.server_args.disable_flashinfer:
            if forward_mode != ForwardMode.DECODE and int(torch.sum(seq_lens)) > 4096:
                flashinfer_use_ragged = True
            init_flashinfer_args(
                forward_mode,
                model_runner,
                req_pool_indices,
                seq_lens,
                prefix_lens,
                model_runner.flashinfer_decode_wrapper,
                flashinfer_use_ragged,
            )

        batch_size = len(req_pool_indices)

        if forward_mode == ForwardMode.DECODE:
            positions = ((seq_lens - 1) + position_ids_offsets).to(torch.int64)
            extend_seq_lens = extend_start_loc = extend_no_prefix = None
            if not model_runner.server_args.disable_flashinfer:
                # This variable is not needed in this case,
                # we do not compute it to make it compatbile with cuda graph.
                total_num_tokens = None
            else:
                total_num_tokens = int(torch.sum(seq_lens))
        else:
            seq_lens_cpu = seq_lens.cpu().numpy()
            prefix_lens_cpu = prefix_lens.cpu().numpy()
            position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
            positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(
                            prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                            seq_lens_cpu[i] + position_ids_offsets_cpu[i],
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                ),
                device="cuda",
            )
            extend_seq_lens = seq_lens - prefix_lens
            extend_start_loc = torch.zeros_like(seq_lens)
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
            extend_no_prefix = torch.all(prefix_lens == 0)
            total_num_tokens = int(torch.sum(seq_lens))

        ret = cls(
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            positions=positions,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            extend_no_prefix=extend_no_prefix,
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
            flashinfer_prefill_wrapper_ragged=model_runner.flashinfer_prefill_wrapper_ragged,
            flashinfer_prefill_wrapper_paged=model_runner.flashinfer_prefill_wrapper_paged,
            flashinfer_decode_wrapper=model_runner.flashinfer_decode_wrapper,
            flashinfer_use_ragged=flashinfer_use_ragged,
        )

        if model_runner.server_args.disable_flashinfer:
            (
                ret.triton_max_seq_len,
                ret.triton_max_extend_len,
                ret.triton_start_loc,
                ret.triton_prefix_lens,
            ) = init_triton_args(forward_mode, seq_lens, prefix_lens)

        return ret


def init_flashinfer_args(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper,
    flashinfer_use_ragged=False,
):
    """Init auxiliary variables for FlashInfer attention backend."""
    num_qo_heads = model_runner.model_config.num_attention_heads // model_runner.tp_size
    num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
    head_dim = model_runner.model_config.head_dim
    batch_size = len(req_pool_indices)
    total_num_tokens = int(torch.sum(seq_lens))

    if flashinfer_use_ragged:
        paged_kernel_lens = prefix_lens
    else:
        paged_kernel_lens = seq_lens

    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
    paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
    kv_indices = torch.cat(
        [
            model_runner.req_to_token_pool.req_to_token[
                req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
            ]
            for i in range(batch_size)
        ],
        dim=0,
    ).contiguous()
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    if forward_mode == ForwardMode.DECODE:
        flashinfer_decode_wrapper.end_forward()
        flashinfer_decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
        )
    else:
        # extend part
        qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

        if flashinfer_use_ragged:
            model_runner.flashinfer_prefill_wrapper_ragged.end_forward()
            model_runner.flashinfer_prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
            )

        # cached part
        model_runner.flashinfer_prefill_wrapper_paged.end_forward()
        model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
        )


def init_triton_args(forward_mode, seq_lens, prefix_lens):
    """Init auxiliary variables for triton attention backend."""
    batch_size = len(seq_lens)
    max_seq_len = int(torch.max(seq_lens))
    start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    if forward_mode == ForwardMode.DECODE:
        max_extend_len = None
    else:
        extend_seq_lens = seq_lens - prefix_lens
        max_extend_len = int(torch.max(extend_seq_lens))

    return max_seq_len, max_extend_len, start_loc, prefix_lens
