from __future__ import annotations

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
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


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
    sampling_info: SamplingBatchInfo
    batch_size: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool

    # Output location of the KV cache
    out_cache_loc: torch.Tensor

    total_num_tokens: int = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: torch.Tensor = None
    extend_prefix_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    extend_no_prefix: bool = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None
    extend_seq_lens_cpu: List[int] = None
    logprob_start_lens_cpu: List[int] = None

    # For multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[List[int]]] = None
    image_offsets: List[List[int]] = None

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

    def init_multimuldal_info(self, batch: ScheduleBatch):
        reqs = batch.reqs
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_sizes for r in reqs]
        self.image_offsets = [r.image_offsets for r in reqs]

    def compute_positions(self, batch: ScheduleBatch):
        position_ids_offsets = batch.position_ids_offsets

        if self.forward_mode == ForwardMode.DECODE:
            if True:
                self.positions = self.seq_lens - 1
            else:
                # Deprecated
                self.positions = (self.seq_lens - 1) + position_ids_offsets
        else:
            if True:
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(batch.prefix_lens_cpu[i], len(req.fill_ids))
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )
            else:
                # Deprecated
                position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(
                                batch.prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                                len(req.fill_ids) + position_ids_offsets_cpu[i],
                            )
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )

        # Positions should be in long type
        self.positions = self.positions.to(torch.int64)

    def compute_extend_infos(self, batch: ScheduleBatch):
        if self.forward_mode == ForwardMode.DECODE:
            self.extend_seq_lens = self.extend_start_loc = self.extend_no_prefix = None
            self.extend_seq_lens_cpu = self.logprob_start_lens_cpu = None
        else:
            extend_lens_cpu = [
                len(r.fill_ids) - batch.prefix_lens_cpu[i]
                for i, r in enumerate(batch.reqs)
            ]
            self.extend_seq_lens = torch.tensor(extend_lens_cpu, device="cuda")
            self.extend_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
            self.extend_start_loc = torch.zeros_like(self.seq_lens)
            self.extend_start_loc[1:] = torch.cumsum(self.extend_seq_lens[:-1], dim=0)
            self.extend_no_prefix = all(l == 0 for l in batch.prefix_lens_cpu)

            self.extend_seq_lens_cpu = extend_lens_cpu
            self.logprob_start_lens_cpu = [
                (
                    min(
                        req.logprob_start_len - batch.prefix_lens_cpu[i],
                        extend_lens_cpu[i] - 1,
                    )
                    if req.logprob_start_len >= batch.prefix_lens_cpu[i]
                    else extend_lens_cpu[i] - 1  # Fake extend, actually decode
                )
                for i, req in enumerate(batch.reqs)
            ]

    @classmethod
    def from_schedule_batch(
        cls,
        model_runner: "ModelRunner",
        batch: ScheduleBatch,
        forward_mode: ForwardMode,
    ):
        ret = cls(
            forward_mode=forward_mode,
            sampling_info=batch.sampling_info,
            batch_size=batch.batch_size(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        )

        ret.sampling_info.prepare_penalties()

        ret.compute_positions(batch)

        ret.compute_extend_infos(batch)

        if (
            forward_mode != ForwardMode.DECODE
            or model_runner.server_args.disable_flashinfer
        ):
            ret.total_num_tokens = int(torch.sum(ret.seq_lens))

        if forward_mode != ForwardMode.DECODE:
            ret.init_multimuldal_info(batch)

        if model_runner.server_args.disable_flashinfer:
            ret.init_triton_args(batch)

        flashinfer_use_ragged = False
        if not model_runner.server_args.disable_flashinfer:
            if (
                forward_mode != ForwardMode.DECODE
                and int(torch.sum(ret.seq_lens)) > 4096
                and model_runner.sliding_window_size is None
            ):
                flashinfer_use_ragged = True
            ret.init_flashinfer_handlers(
                model_runner, batch.prefix_lens_cpu, flashinfer_use_ragged
            )

        return ret

    def init_triton_args(self, batch: ScheduleBatch):
        """Init auxiliary variables for triton attention backend."""
        self.triton_max_seq_len = int(torch.max(self.seq_lens))
        self.triton_start_loc = torch.zeros_like(self.seq_lens, dtype=torch.int32)
        self.triton_start_loc[1:] = torch.cumsum(self.seq_lens[:-1], dim=0)

        if self.forward_mode == ForwardMode.DECODE:
            self.triton_max_extend_len = None
        else:
            self.triton_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
            extend_seq_lens = self.seq_lens - self.triton_prefix_lens
            self.triton_max_extend_len = int(torch.max(extend_seq_lens))

    def init_flashinfer_handlers(
        self,
        model_runner,
        prefix_lens_cpu,
        flashinfer_use_ragged,
    ):
        if self.forward_mode == ForwardMode.DECODE:
            prefix_lens = None
        else:
            prefix_lens = self.extend_prefix_lens

        update_flashinfer_indices(
            self.forward_mode,
            model_runner,
            self.req_pool_indices,
            self.seq_lens,
            prefix_lens,
            flashinfer_use_ragged=flashinfer_use_ragged,
        )

        (
            self.flashinfer_prefill_wrapper_ragged,
            self.flashinfer_prefill_wrapper_paged,
            self.flashinfer_decode_wrapper,
            self.flashinfer_use_ragged,
        ) = (
            model_runner.flashinfer_prefill_wrapper_ragged,
            model_runner.flashinfer_prefill_wrapper_paged,
            model_runner.flashinfer_decode_wrapper,
            flashinfer_use_ragged,
        )


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    max_context_len,
    kv_indices_ptr,
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


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper=None,
    flashinfer_use_ragged=False,
):
    """Init auxiliary variables for FlashInfer attention backend."""
    num_qo_heads = model_runner.model_config.num_attention_heads // model_runner.tp_size
    num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
    head_dim = model_runner.model_config.head_dim
    batch_size = len(req_pool_indices)

    if model_runner.sliding_window_size is None:
        if flashinfer_use_ragged:
            paged_kernel_lens = prefix_lens
        else:
            paged_kernel_lens = seq_lens

        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(batch_size,)](
            model_runner.req_to_token_pool.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            model_runner.req_to_token_pool.req_to_token.size(1),
            kv_indices,
        )

        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

        if forward_mode == ForwardMode.DECODE:
            # CUDA graph uses different flashinfer_decode_wrapper
            if flashinfer_decode_wrapper is None:
                flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

            flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                data_type=model_runner.kv_cache_dtype,
                q_data_type=model_runner.dtype,
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
    else:
        # window attention use paged only
        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
        for wrapper_id in range(2):
            if wrapper_id == 0:
                if forward_mode == ForwardMode.DECODE:
                    paged_kernel_lens = torch.minimum(
                        seq_lens, torch.tensor(model_runner.sliding_window_size + 1)
                    )
                else:
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        torch.tensor(model_runner.sliding_window_size)
                        + seq_lens
                        - prefix_lens,
                    )
            else:
                paged_kernel_lens = seq_lens

            kv_start_idx = seq_lens - paged_kernel_lens

            kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

            kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
            create_flashinfer_kv_indices_triton[(batch_size,)](
                model_runner.req_to_token_pool.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                model_runner.req_to_token_pool.req_to_token.size(1),
                kv_indices,
            )

            if forward_mode == ForwardMode.DECODE:
                # CUDA graph uses different flashinfer_decode_wrapper
                if flashinfer_decode_wrapper is None:
                    flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

                flashinfer_decode_wrapper[wrapper_id].end_forward()
                flashinfer_decode_wrapper[wrapper_id].begin_forward(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                    data_type=model_runner.kv_cache_dtype,
                    q_data_type=model_runner.dtype,
                )
            else:
                # extend part
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device="cuda"
                )
                qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].end_forward()
                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
