from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import os

import numpy as np


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None


class AscendAttnBackend(AttentionBackend):

    def gen_attention_mask(self, max_seq_len: int, dtype=torch.float16):
        mask_flag = torch.tril(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        ).view(max_seq_len, max_seq_len)
        mask_flag = ~mask_flag
        if dtype == torch.float16:
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        self.mask = (
            torch.masked_fill(
                torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
            )
            .to(dtype)
            .to(self.device)
        )
        self.mask_len = max_seq_len

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = ForwardMetadata()
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.native_attn = TorchNativeAttnBackend(model_runner)
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        if not self.use_fia:
            self.gen_attention_mask(128, model_runner.dtype)
        mask_length = 2048
        self.fia_mask = ~torch.tril(
            torch.ones(
                (mask_length, mask_length),
                dtype=torch.bool,
                device=model_runner.device,
            )
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : forward_batch.seq_lens.max()
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        self.forward_metadata.seq_lens_list_cumsum = np.cumsum(
            forward_batch.extend_seq_lens_cpu
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )

            else:
                query = q.view(-1, layer.tp_q_head_num * layer.qk_head_dim)
                attn_output = torch.empty(
                    (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )

                torch_npu._npu_flash_attention_qlens(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    mask=self.mask,
                    block_table=self.forward_metadata.block_tables,
                    seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    scale_value=layer.scaling,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    out=attn_output,
                )
        else:
            assert (
                layer.qk_head_dim != layer.v_head_dim
            ), "FIA only supports qk_head_dim != v_head_dim"
            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=layer.tp_q_head_num,
                input_layout="TND",
                atten_mask=self.fia_mask,
                sparse_mode=3,
                actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                scale=layer.scaling,
                next_tokens=0,
            )

        return attn_output

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            num_tokens = q.shape[0]

            if self.use_fia:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        forward_batch.batch_size,
                        -1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim
                    ),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                    scale=layer.scaling,
                )
            else:
                query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    out=attn_output,
                )

            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                kv_c = kv_c.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )
                k_pe = k_pe.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
            else:
                assert (
                    get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE") == False
                )  # _npu_paged_attention_mla not support graph mode
                q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)
