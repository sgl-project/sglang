from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Dict, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
    compute_local_num_token_non_padded,
)


def _keep_larger(
    existing: Optional[torch.Tensor],
    new: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if existing is None:
        return new
    if new is None:
        return existing
    return existing if existing.numel() >= new.numel() else new


@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor
    input_embeds: Optional[torch.Tensor]
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    out_cache_loc_swa: Optional[torch.Tensor]
    positions: torch.Tensor
    mrope_positions: Optional[torch.Tensor]
    num_token_non_padded: torch.Tensor
    custom_mask: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    mamba_track_seqlens: Optional[torch.Tensor]
    global_num_tokens_gpu: torch.Tensor
    global_num_tokens_for_logprob_gpu: torch.Tensor
    encoder_lens: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]

    # singleton
    _graph_input_buffers: ClassVar[Optional["GraphInputBuffers"]] = None

    @classmethod
    def get_graph_input_buffers(cls) -> Optional["GraphInputBuffers"]:
        return cls._graph_input_buffers

    @classmethod
    def _set_graph_input_buffers(cls, buffers: "GraphInputBuffers") -> None:
        existing = cls._graph_input_buffers
        if existing is not None:
            for f in fields(cls):
                old_val = getattr(existing, f.name)
                new_val = getattr(buffers, f.name)
                if isinstance(old_val, torch.Tensor) or isinstance(
                    new_val, torch.Tensor
                ):
                    setattr(
                        buffers,
                        f.name,
                        _keep_larger(old_val, new_val),
                    )
        cls._graph_input_buffers = buffers

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
        enable_swa_out_cache: bool = False,
        is_pcg: bool = False,
        is_multimodal: bool = False,
    ) -> "GraphInputBuffers":

        if is_pcg:
            custom_mask_size = max_bs + max_num_token
            pp_proxy_token_dim = max_num_token
        else:
            custom_mask_size = (
                max_bs * seq_len_fill_value + max_num_token
            ) * num_tokens_per_bs
            pp_proxy_token_dim = max_bs

        with torch.device(device):
            # For PCG, Only create input_embeds and mrope_positions for multimodal
            # model to save memory
            # 1. In multimodal, we only compile and capture the language model part.
            # 2. The embedder is outside of the graph, but cuda graph requires the
            # input embeds to have a fixed memory address.
            # 3. Input embeds is a pre-allocated buffer. In model.forward, we copy
            # the embed output to this buffer.
            alloc_embeds = (not is_pcg) or is_multimodal

            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            input_embeds = (
                torch.zeros((max_num_token, hidden_size), dtype=dtype)
                if alloc_embeds
                else None
            )
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            out_cache_loc_swa = (
                torch.zeros((max_num_token,), dtype=cache_loc_dtype)
                if enable_swa_out_cache
                else None
            )
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = (
                torch.zeros((3, max_num_token), dtype=torch.int64)
                if alloc_embeds
                else None
            )
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = torch.ones((custom_mask_size,), dtype=torch.bool)
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size), dtype=torch.float
            )
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

            if pp_size > 1:
                pp_proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (pp_proxy_token_dim, hidden_size), dtype=dtype
                    ),
                    "residual": torch.zeros(
                        (pp_proxy_token_dim, hidden_size), dtype=dtype
                    ),
                }
            else:
                pp_proxy_tensors = None

            encoder_lens = (
                torch.full((max_bs,), encoder_len_fill_value, dtype=torch.int32)
                if is_encoder_decoder
                else None
            )

            if require_mlp_tp_gather:
                global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros(
                    (dp_size,), dtype=torch.int32
                )
            else:
                global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

        # Keep seq_lens_cpu as a true CPU tensor, like the old implementation.
        seq_lens_cpu = torch.full(
            (max_bs,), seq_len_fill_value, dtype=torch.int32, device="cpu"
        )

        buffers = cls(
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            positions=positions,
            mrope_positions=mrope_positions,
            num_token_non_padded=num_token_non_padded,
            custom_mask=custom_mask,
            next_token_logits_buffer=next_token_logits_buffer,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            encoder_lens=encoder_lens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        cls._set_graph_input_buffers(buffers)
        return buffers

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
        nsa_enable_prefill_cp: bool,
        enable_num_token_non_padded_flag: bool,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Optional[torch.Tensor]:
        if bs != raw_bs:
            self.seq_lens.fill_(seq_len_fill_value)
            self.out_cache_loc.zero_()
            if self.mamba_track_indices is not None:
                self.mamba_track_indices.zero_()
            if self.mamba_track_mask is not None:
                self.mamba_track_mask.fill_(False)

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        if (
            self.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            self.mamba_track_indices[:raw_bs].copy_(forward_batch.mamba_track_indices)
        if (
            self.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            self.mamba_track_mask[:raw_bs].copy_(forward_batch.mamba_track_mask)

        seq_lens_cpu: Optional[torch.Tensor] = None
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            seq_lens_cpu = self.seq_lens_cpu[:bs]

        if self.encoder_lens is not None and forward_batch.encoder_lens is not None:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)

        if (
            forward_batch.mrope_positions is not None
            # and self.mrope_positions is not None
        ):
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)

        if require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * num_tokens_per_bs)

        if enable_num_token_non_padded_flag:
            if require_gathered_buffer and not nsa_enable_prefill_cp:
                num_tokens_per_dp = bs * num_tokens_per_bs
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=forward_batch.num_token_non_padded,
                    num_tokens_per_dp=num_tokens_per_dp,
                )
                self.num_token_non_padded.copy_(local)
            else:
                self.num_token_non_padded.copy_(forward_batch.num_token_non_padded)

        # Pipeline-parallel proxy tensors.
        if pp_proxy_tensors is not None and self.pp_proxy_tensors is not None:
            for key, buf in self.pp_proxy_tensors.items():
                src = pp_proxy_tensors.tensors[key]
                dim = src.shape[0]
                buf[:dim].copy_(src)

        return seq_lens_cpu
