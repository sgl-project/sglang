from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

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
    seq_lens_cpu_list_cumsum: Optional[List[int]] = None
    seq_lens_cpu_list: Optional[List[int]] = None


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.tril(
        torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
    ).view(max_seq_len, max_seq_len)
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    if dtype == torch.float16:
        mask_value = torch.finfo(torch.float32).min
    else:
        mask_value = 1
    attn_mask = torch.masked_fill(
        torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
    ).to(dtype)
    return attn_mask


class AttentionMaskBuilder:
    def __init__(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
    ):
        attn_mask = _generate_attn_mask(max_seq_len, dtype)
        self._seq_len_cached = attn_mask.shape[0]
        self.attn_mask_cache = attn_mask

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        self._update_attn_cache(max_seq_len, dtype, device)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def _update_attn_cache(self, seqlen: int, dtype: torch.dtype, device: torch.device):
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            self.attn_mask_cache = _generate_attn_mask(seqlen, dtype)
        if self.attn_mask_cache.device != device:
            self.attn_mask_cache = self.attn_mask_cache.to(device)


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
        self.forward_metadata = None
        self.device = model_runner.device
        self.attn_mask_builder = AttentionMaskBuilder(8192, model_runner.dtype)
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()

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

        self.forward_metadata.seq_lens_cpu_list_cumsum = torch.cumsum(
            forward_batch.seq_lens_cpu, dim=0
        ).tolist()

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, self.max_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        has_prefixcache = sum(forward_batch.extend_prefix_lens_cpu) > 0

        if not self.use_mla:
            query = q.view(-1, layer.tp_q_head_num * layer.qk_head_dim)
            output = torch.empty(
                (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=query.dtype,
                device=query.device,
            )
            mask = self.attn_mask_builder.get_attn_mask(128, query.dtype, query.device)

            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=mask,
                block_table=self.forward_metadata.block_tables,
                seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                scale_value=layer.scaling,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                out=output,
            )
            return output
        elif has_prefixcache:
            raise NotImplementedError(
                f"prefixcache attention is not implemented, seqlen len: {forward_batch.seq_lens_cpu}, \
                    extend_prefix_lens: {forward_batch.extend_prefix_lens_cpu}."
                f"Please set args: --disable-radix-cache and --chunked-prefill-size."
            )
        else:
            mask = self.attn_mask_builder.get_attn_mask(2048, q.dtype, q.device).to(
                torch.int8
            )
            num_token_padding = q.shape[0]
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]

            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                atten_mask=mask,
                sparse_mode=3,
                actual_seq_lengths=self.forward_metadata.seq_lens_cpu_list_cumsum,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_list_cumsum,
                num_heads=layer.tp_q_head_num,
                input_layout="TND",
                scale=layer.scaling,
                next_tokens=0,
            )

            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
            if num_token_padding != forward_batch.num_token_non_padded_cpu:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.view(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            num_tokens = query.shape[0]
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
            )
            output = torch.empty(
                (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_rope_cache = k_rope.view(
                -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
            )
            c_kv_cache = c_kv.view(
                -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
            )

            q_nope = q.view(-1, layer.tp_q_head_num, 1, self.kv_lora_rank)
            q_rope = q_rope.view(-1, layer.tp_q_head_num, 1, self.qk_rope_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.zeros_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        if self.graph_mode:
            return self.forward_decode_graph(
                q, k, v, layer, forward_batch, q_rope=q_rope, k_rope=k_rope
            )

        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            num_tokens = query.shape[0]
            output = torch.empty(
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
                out=output,
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if (self.use_fia) and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
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
                q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank)
                q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
                num_tokens = q_nope.shape[0]
                kv_cache, kv_rope_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )

                attn_output = torch_npu.atb.npu_multi_head_latent_attention(
                    q_nope,
                    q_rope,
                    kv_cache,
                    kv_rope_cache,
                    self.forward_metadata.block_tables,
                    self.forward_metadata.seq_lens_cpu_int,
                    layer.tp_q_head_num,
                    layer.scaling,
                    layer.tp_k_head_num,
                    cache_mode="krope_ctkv",
                )

            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)
