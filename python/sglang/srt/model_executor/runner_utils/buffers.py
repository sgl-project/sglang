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
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    NgramEmbeddingInfo,
    PPProxyTensors,
    compute_local_num_token_non_padded,
)
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


@dataclass
class DecodeInputBuffers(ForwardInputBuffers):

    input_ids: torch.Tensor
    input_embeds: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    # Shared-KV-pool cuda-graph WRITE-location buffer (int64, matching the
    # v2p table dtype). Holds the full-physical write locations under the shared
    # pool, refreshed IN-GRAPH each replay by the backend's
    # `init_forward_metadata_in_graph`. `None` for non-shared pools (then the
    # in-graph translate is a no-op, keeping baseline cg_on byte-identical).
    out_cache_loc_full_physical: Optional[torch.Tensor]
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
        vocab_size: int,
        dtype: torch.dtype,
        dp_size: int,
        pp_size: int,
        is_encoder_decoder: bool,
        require_mlp_tp_gather: bool,
        seq_len_fill_value: int,
        encoder_len_fill_value: int,
        num_tokens_per_bs: int,
        cache_loc_dtype: torch.dtype,
        enable_mamba_track: bool,
        ne_token_table: Optional[torch.Tensor] = None,
        hc_hidden_size: Optional[int] = None,
        pp_proxy_topk_size: Optional[int] = None,
        enable_shared_kv_pool: bool = False,
    ) -> DecodeInputBuffers:
        with torch.device(device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            # Shared pool: a separate int64 full-physical write buffer is
            # allocated (matching the v2p table), refreshed in-graph each replay.
            # The SWA write loc rides the backend `swa_out_cache_loc` rail, so no
            # SWA buffer is allocated here. None for non-shared pools.
            if enable_shared_kv_pool:
                out_cache_loc_full_physical = torch.zeros(
                    (max_num_token,), dtype=torch.int64
                )
            else:
                out_cache_loc_full_physical = None
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = torch.ones(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
                dtype=torch.bool,
            )
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size),
                dtype=torch.float,
            )
            mamba_track_indices = (
                torch.zeros((max_bs,), dtype=torch.int64)
                if enable_mamba_track
                else None
            )
            mamba_track_mask = (
                torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
            )

            if pp_size > 1:
                is_mhc = hc_hidden_size is not None
                hs = hc_hidden_size if is_mhc else hidden_size
                pp_proxy_tensors = {
                    "hidden_states": torch.zeros((max_bs, hs), dtype=dtype),
                }
                if not is_mhc:
                    pp_proxy_tensors["residual"] = torch.zeros(
                        (max_bs, hidden_size), dtype=dtype
                    )
                if pp_proxy_topk_size is not None:
                    pp_proxy_tensors["topk_indices"] = torch.zeros(
                        (max_num_token, pp_proxy_topk_size), dtype=torch.int32
                    )
            else:
                pp_proxy_tensors = None

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
            out_cache_loc_full_physical=out_cache_loc_full_physical,
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

    def populate_from_forward_batch(
        self,
        *,
        forward_batch: ForwardBatch,
        raw_bs: int,
        raw_num_token: int,
        bs: int,
        seq_len_fill_value: int,
        require_gathered_buffer: bool,
        num_tokens_per_bs: int,
        dsa_enable_prefill_cp: bool,
        enable_num_token_non_padded_flag: bool,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if bs != raw_bs:
            self.seq_lens.fill_(seq_len_fill_value)
            self.out_cache_loc.zero_()
            if self.mamba_track_indices is not None:
                self.mamba_track_indices.zero_()
            if self.mamba_track_mask is not None:
                self.mamba_track_mask.fill_(False)

        # Build batched copy lists for all GPU tensors.
        dsts = [
            self.input_ids[:raw_num_token],
            self.req_pool_indices[:raw_bs],
            self.seq_lens[:raw_bs],
            self.out_cache_loc[:raw_num_token],
            self.positions[:raw_num_token],
        ]
        srcs = [
            forward_batch.input_ids,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
            forward_batch.positions,
        ]

        if self.ngram_embedding_info is not None:
            ngram_embedding_info = forward_batch.ngram_embedding_info
            self.ngram_embedding_info.column_starts[:raw_bs].copy_(
                ngram_embedding_info.column_starts
            )
            self.ngram_embedding_info.req_lens[:raw_bs].copy_(
                ngram_embedding_info.req_lens
            )

        if (
            self.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            dsts.append(self.mamba_track_indices[:raw_bs])
            srcs.append(forward_batch.mamba_track_indices)
        if (
            self.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            dsts.append(self.mamba_track_mask[:raw_bs])
            srcs.append(forward_batch.mamba_track_mask)

        if self.encoder_lens is not None and forward_batch.encoder_lens is not None:
            dsts.append(self.encoder_lens[:raw_bs])
            srcs.append(forward_batch.encoder_lens)

        if forward_batch.mrope_positions is not None:
            dsts.append(self.mrope_positions[:, :raw_num_token])
            srcs.append(forward_batch.mrope_positions)

        if self.rids_int is not None and forward_batch.rids_int is not None:
            dsts.append(self.rids_int[:raw_bs])
            srcs.append(forward_batch.rids_int)
        if (
            self.bootstrap_room_ids_int is not None
            and forward_batch.bootstrap_room_ids_int is not None
        ):
            dsts.append(self.bootstrap_room_ids_int[:raw_bs])
            srcs.append(forward_batch.bootstrap_room_ids_int)

        if require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * num_tokens_per_bs)

        if enable_num_token_non_padded_flag:
            if require_gathered_buffer and not dsa_enable_prefill_cp:
                num_tokens_per_dp = bs * num_tokens_per_bs
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=forward_batch.num_token_non_padded,
                    num_tokens_per_dp=num_tokens_per_dp,
                )
                dsts.append(self.num_token_non_padded)
                srcs.append(local)
            else:
                dsts.append(self.num_token_non_padded)
                srcs.append(forward_batch.num_token_non_padded)

        # Pipeline-parallel proxy tensors.
        if pp_proxy_tensors is not None and self.pp_proxy_tensors is not None:
            for key, buf in self.pp_proxy_tensors.items():
                src = pp_proxy_tensors.tensors[key]
                dim = src.shape[0]
                dsts.append(buf[:dim])
                srcs.append(src)

        # Batch all GPU copies, grouped by dtype pair.
        _grouped_foreach_copy_(dsts, srcs)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)


@dataclass
class PrefillInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    mamba_track_seqlens: Optional[torch.Tensor]
    positions: torch.Tensor
    input_embeds: Optional[torch.Tensor]
    mrope_positions: Optional[torch.Tensor]

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
    ) -> PrefillInputBuffers:
        with torch.device(device):
            input_ids = torch.zeros((max_num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros((max_num_tokens,), dtype=cache_loc_dtype)
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

        return cls(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            positions=positions,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
        )

    def populate_from_forward_batch(
        self,
        *,
        forward_batch: ForwardBatch,
        raw_num_tokens: int,
        static_num_tokens: int,
        is_multimodal: bool,
    ) -> None:
        """Copy serving-batch values into static buffers and zero out
        the padding region between raw_num_tokens and
        static_num_tokens.
        """
        if static_num_tokens != raw_num_tokens:
            self.out_cache_loc.zero_()
            self.input_ids[raw_num_tokens:static_num_tokens].zero_()
            self.positions[raw_num_tokens:static_num_tokens].zero_()
            if is_multimodal:
                self.input_embeds[raw_num_tokens:static_num_tokens].zero_()
            if forward_batch.mrope_positions is not None:
                self.mrope_positions[:, raw_num_tokens:static_num_tokens].zero_()

        bs = forward_batch.batch_size

        self.input_ids[:raw_num_tokens].copy_(forward_batch.input_ids)
        self.positions[:raw_num_tokens].copy_(forward_batch.positions)
        self.out_cache_loc[:raw_num_tokens].copy_(forward_batch.out_cache_loc)

        if (
            self.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            self.mamba_track_indices[:bs].copy_(forward_batch.mamba_track_indices)
        if (
            self.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            self.mamba_track_mask[:bs].copy_(forward_batch.mamba_track_mask)
        if (
            self.mamba_track_seqlens is not None
            and forward_batch.mamba_track_seqlens is not None
        ):
            self.mamba_track_seqlens[:bs].copy_(forward_batch.mamba_track_seqlens)

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_num_tokens].copy_(
                forward_batch.mrope_positions
            )
