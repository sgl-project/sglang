# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Static-buffer dataclasses used by the CUDA graph runners.

DecodeInputBuffers backs the decode-phase capture/replay path.
PrefillInputBuffers backs the prefill-phase capture/replay path.

Both subclass ForwardInputBuffers so that buffer-pool sharing works
the same way as for non-cuda-graph forward paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import NgramEmbeddingInfo
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers

_has_foreach_copy = hasattr(torch, "_foreach_copy_")


def _grouped_foreach_copy_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    """Call torch._foreach_copy_ grouped by (dst_dtype, src_dtype) pairs."""

    def foreach_copy(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
        if _has_foreach_copy:
            torch._foreach_copy_(dsts, srcs)
        else:
            for dst, src in zip(dsts, srcs):
                dst.copy_(src)

    groups: Dict[Tuple[torch.dtype, torch.dtype], Tuple[List, List]] = {}
    for dst, src in zip(dsts, srcs):
        key = (dst.dtype, src.dtype)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(dst)
        groups[key][1].append(src)
    for group_dsts, group_srcs in groups.values():
        foreach_copy(group_dsts, group_srcs)


def _allocate_pp_proxy_tensors(
    *,
    max_num_tokens: int,
    max_hidden_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    hc_hidden_size: Optional[int] = None,
    pp_proxy_topk_size: Optional[int] = None,
    pp_proxy_residual_num_blocks: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Allocate the stable buffers consumed by an incoming PP proxy."""
    is_mhc = hc_hidden_size is not None
    pp_hidden_size = hc_hidden_size if is_mhc else hidden_size
    pp_proxy_tensors = {
        "hidden_states": torch.zeros((max_hidden_tokens, pp_hidden_size), dtype=dtype),
    }
    if not is_mhc:
        # Only Kimi K3 supplies num_blocks: its PP bank is token-major
        # [T, blocks, H]. Other models use the phase-specific hidden-token bound.
        residual_shape = (
            (max_num_tokens, pp_proxy_residual_num_blocks, hidden_size)
            if pp_proxy_residual_num_blocks is not None
            else (max_hidden_tokens, hidden_size)
        )
        pp_proxy_tensors["residual"] = torch.zeros(residual_shape, dtype=dtype)
    if pp_proxy_topk_size is not None:
        pp_proxy_tensors["topk_indices"] = torch.zeros(
            (max_num_tokens, pp_proxy_topk_size), dtype=torch.int32
        )
    return pp_proxy_tensors


@dataclass
class DecodeInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    input_embeds: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    num_token_non_padded: torch.Tensor
    custom_mask: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    global_num_tokens_gpu: torch.Tensor
    global_num_tokens_for_logprob_gpu: torch.Tensor
    encoder_lens: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]
    ngram_embedding_info: Optional[NgramEmbeddingInfo]
    rids_int: Optional[torch.Tensor]
    bootstrap_room_ids_int: Optional[torch.Tensor]

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_num_token: int,
        hidden_size: int,
        next_token_logits_buffer: torch.Tensor,
        dtype: torch.dtype,
        dp_size: int,
        pp_size: int,
        is_encoder_decoder: bool,
        require_mlp_tp_gather: bool,
        seq_len_fill_value: int,
        encoder_len_fill_value: int,
        num_tokens_per_req: int,
        cache_loc_dtype: torch.dtype,
        enable_mamba_track: bool,
        ne_token_table: Optional[torch.Tensor] = None,
        hc_hidden_size: Optional[int] = None,
        pp_proxy_topk_size: Optional[int] = None,
        pp_proxy_residual_num_blocks: Optional[int] = None,
    ) -> DecodeInputBuffers:
        with torch.device(device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = torch.ones(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_req,
                dtype=torch.bool,
            )
            mamba_track_indices = (
                torch.zeros((max_bs,), dtype=torch.int64)
                if enable_mamba_track
                else None
            )
            mamba_track_mask = (
                torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
            )

            pp_proxy_tensors = (
                _allocate_pp_proxy_tensors(
                    max_num_tokens=max_num_token,
                    max_hidden_tokens=max_num_token,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    hc_hidden_size=hc_hidden_size,
                    pp_proxy_topk_size=pp_proxy_topk_size,
                    pp_proxy_residual_num_blocks=pp_proxy_residual_num_blocks,
                )
                if pp_size > 1
                else None
            )

            if is_encoder_decoder:
                encoder_lens = torch.full(
                    (max_bs,), encoder_len_fill_value, dtype=torch.int32
                )
            else:
                encoder_lens = None

            if require_mlp_tp_gather:
                global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros(
                    (dp_size,), dtype=torch.int32
                )
            else:
                global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

            ngram_embedding_info = (
                NgramEmbeddingInfo(
                    token_table=ne_token_table,
                    column_starts=torch.zeros([max_bs], dtype=torch.int32),
                    req_lens=torch.ones([max_bs], dtype=torch.int32),
                    out_column_starts=torch.zeros([max_bs], dtype=torch.int32),
                    out_req_lens=torch.ones([max_bs], dtype=torch.int32),
                    skip_token_table_update=torch.zeros([max_bs], dtype=torch.bool),
                )
                if ne_token_table is not None
                else None
            )

            if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
                rids_int = torch.zeros((max_bs,), dtype=torch.int64)
                bootstrap_room_ids_int = torch.full((max_bs,), -1, dtype=torch.int64)
            else:
                rids_int = None
                bootstrap_room_ids_int = None

        seq_lens_cpu = torch.full(
            (max_bs,),
            seq_len_fill_value,
            dtype=torch.int64,
            device="cpu",
        )

        return cls(
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            num_token_non_padded=num_token_non_padded,
            custom_mask=custom_mask,
            next_token_logits_buffer=next_token_logits_buffer,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            encoder_lens=encoder_lens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            pp_proxy_tensors=pp_proxy_tensors,
            ngram_embedding_info=ngram_embedding_info,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
        )


@dataclass
class PrefillInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    num_token_non_padded: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    mamba_track_seqlens: Optional[torch.Tensor]
    positions: torch.Tensor
    input_embeds: Optional[torch.Tensor]
    mrope_positions: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_num_tokens: int,
        cache_loc_dtype: torch.dtype,
        is_multimodal: bool,
        hidden_size: int,
        dtype: torch.dtype,
        enable_mamba_track: bool,
        pp_size: int = 1,
        is_first_pp_rank: bool = False,
        hc_hidden_size: Optional[int] = None,
        pp_proxy_topk_size: Optional[int] = None,
        pp_proxy_residual_num_blocks: Optional[int] = None,
    ) -> PrefillInputBuffers:
        with torch.device(device):
            input_ids = torch.zeros((max_num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros((max_num_tokens,), dtype=cache_loc_dtype)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            mamba_track_indices = (
                torch.zeros((max_bs,), dtype=torch.int64)
                if enable_mamba_track
                else None
            )
            mamba_track_mask = (
                torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
            )
            mamba_track_seqlens = (
                torch.zeros((max_bs,), dtype=torch.int32)
                if enable_mamba_track
                else None
            )
            positions = torch.zeros((max_num_tokens,), dtype=torch.int64)

            if is_multimodal:
                input_embeds = torch.zeros((max_num_tokens, hidden_size), dtype=dtype)
                mrope_positions = torch.zeros((3, max_num_tokens), dtype=torch.int64)
            else:
                input_embeds = None
                mrope_positions = None

            pp_proxy_tensors = (
                _allocate_pp_proxy_tensors(
                    max_num_tokens=max_num_tokens,
                    max_hidden_tokens=max_num_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    hc_hidden_size=hc_hidden_size,
                    pp_proxy_topk_size=pp_proxy_topk_size,
                    pp_proxy_residual_num_blocks=pp_proxy_residual_num_blocks,
                )
                if pp_size > 1 and not is_first_pp_rank
                else None
            )

        return cls(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            num_token_non_padded=num_token_non_padded,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            positions=positions,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
