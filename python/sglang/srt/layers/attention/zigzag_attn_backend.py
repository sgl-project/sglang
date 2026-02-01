
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List

import torch
import numpy as np
import math
import inspect

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.zigzag_ops.zigzag_attn_interface import zigzag_attn_with_kvcache, zigzag_attn_varlen_func
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size, get_attention_tp_rank
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

@dataclass
class ForwardMetadata:
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    cache_seqlens_int32: torch.Tensor = None
    page_table: torch.Tensor = None
    max_seq_len_q: int = 1
    max_seq_len_k: int = 1

PAGE_SIZE = 64    

class ZigZagAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
        speculative_num_steps: int = 0,
    ):
        super().__init__()
        self.forward_metadata: ForwardMetadata = None
        self.max_context_len = model_runner.model_config.context_len

        model_config = model_runner.model_config
        self.speculative_step_id = speculative_step_id
        self.speculative_num_steps = speculative_num_steps
        self.num_q_head = model_config.num_attention_heads
        self.num_k_head = model_config.num_key_value_heads
        self.tp_size = get_attention_tp_size()
        self.local_q_head = self.num_q_head // self.tp_size
        self.local_k_head = self.num_k_head // self.tp_size
        self.max_seq_len = model_runner.server_args.context_length
        if model_runner.server_args.speculative_algorithm is None:
            self.draft_token_num = 0
        else:
            self.draft_token_num = model_runner.server_args.speculative_num_draft_tokens
        
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # duo attn config
        duo_attention_config = model_config.hf_config.streaming_sparse_attention
        self.sink_size = duo_attention_config.get('sink_size', 128)
        self.recent_size = duo_attention_config.get('recent_size', 256)
        self.sparsity = duo_attention_config.get('sparsity', 0.5)
        self.layers_type = duo_attention_config.get('layer_type', "")

        self.block_size = 128
        num_sink_blocks = math.ceil(self.sink_size / self.block_size)
        num_recent_blocks = math.ceil(self.recent_size / self.block_size)
        self.streaming_info = torch.tensor(
            [[num_sink_blocks, num_recent_blocks]] * self.local_q_head,
            device=model_runner.device,
            dtype=torch.int32,
        )
        self.head_mask = torch.full((self.local_q_head,), -1, device=model_runner.device, dtype=torch.int32)
        if model_runner.is_draft_worker:
            self.layers_type = "1" * model_config.num_attention_layers
        
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            # For MLA, k is the latent cache which is shared by the key and value.
            # Plus, it encompasses the `kv_lora_rank` and `k_rope`.
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None, # The None input here is not used within the function anyway.
            )
        
        q_head = layer.tp_q_head_num
        streaming_info = self.streaming_info if self.layers_type[layer.layer_id] == '1' else None
        
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            if not any(forward_batch.extend_prefix_lens_cpu):
                # No prefix prefill, no absorb, MHA.
                # `save_kv_cache` op has been done before entering the backend,
                # since it requires accessing the latent cache which would better 
                # not be involved in the backend during prefill.
                assert not save_kv_cache
                bs = forward_batch.batch_size
                cu_seqlens_q = self.forward_metadata.cu_seqlens_q
                cu_seqlens_k = self.forward_metadata.cu_seqlens_k
                q = q.view(-1, q_head, layer.qk_head_dim) # E.g., 128 + 64.
                v = v.contiguous()
                o = zigzag_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata.max_seq_len_q,
                    max_seqlen_k=self.forward_metadata.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=True,
                    streaming_info=streaming_info,
                    head_mask_type=self.head_mask,
                )
                o = o.view(-1, q_head * layer.v_head_dim) # E.g., 128.
            else:
                # Prefix extend, absorb, MQA.
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                # Recover the shape [num_pages, page_size, ...].
                k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size))
                page_table = self.forward_metadata.page_table
                q = q.view(-1, 1, q_head, layer.qk_head_dim) # E.g., 512 + 64.
                o = zigzag_attn_with_kvcache(
                    q,
                    k_buffer,
                    self.forward_metadata.cache_seqlens_int32,
                    page_table,
                    layer.v_head_dim,
                    layer.scaling,
                    causal=True,
                    streaming_info=streaming_info,
                    head_mask_type=self.head_mask,
                )
                o = o.view(-1, q_head * layer.v_head_dim) # E.g., 512.
        else:
            # MTP.
            assert (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ), f"{forward_batch.forward_mode=}"
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)) # Recover the shape [num_pages, page_size, ...].
            bs = forward_batch.batch_size
            page_table = self.forward_metadata.page_table
            q = q.view(bs, -1, q_head, layer.head_dim)
            q_seqlen = q.shape[1]
            o = zigzag_attn_with_kvcache(
                q,
                k_buffer,
                self.forward_metadata.cache_seqlens_int32,
                page_table,
                layer.v_head_dim,
                layer.scaling,
                causal=True,
                streaming_info=streaming_info,
                head_mask_type=self.head_mask,
            )
            o = o.view(-1, q_head * layer.v_head_dim)
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None,
            )

        bs = forward_batch.batch_size
        q_head = layer.tp_q_head_num
        kv_head = layer.tp_k_head_num
        streaming_info = self.streaming_info if self.layers_type[layer.layer_id] == '1' else None

        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size))
        page_table = self.forward_metadata.page_table
        q = q.view(-1, 1, q_head, layer.qk_head_dim)
        o = zigzag_attn_with_kvcache(
            q,
            k_buffer,
            self.forward_metadata.cache_seqlens_int32,
            page_table,
            layer.v_head_dim,
            layer.scaling,
            causal=True,
            streaming_info=streaming_info,
            head_mask_type=self.head_mask,
        )
        o = o.view(-1, q_head * layer.v_head_dim)
        return o


    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        metadata = ForwardMetadata()
        
        if forward_batch.forward_mode.is_decode_or_idle():
            metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.max_seq_len_q = 1
            metadata.cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
        elif forward_batch.forward_mode.is_target_verify():
            metadata.cache_seqlens_int32 = (
                forward_batch.seq_lens + self.draft_token_num
            ).to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + self.draft_token_num
            metadata.max_seq_len_q = self.draft_token_num
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * self.draft_token_num + 1,
                self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                # extend_seq_lens = forward_batch.extend_seq_lens
                # metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                # metadata.cu_seqlens_q = torch.nn.functional.pad(
                #     torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                # )
                seqlens_expanded = torch.cat(
                    [
                        torch.arange(
                            kv_len - qo_len + 1, kv_len + 1, dtype=torch.int32, device=device,
                        )
                        for qo_len, kv_len in zip(
                            forward_batch.extend_seq_lens_cpu,
                            forward_batch.seq_lens_cpu.tolist(),
                        )
                    ]
                )
                metadata.cache_seqlens_int32 = seqlens_expanded
                total_q_tokens = sum(forward_batch.extend_seq_lens_cpu)
                metadata.max_seq_len_q = 1
                metadata.cu_seqlens_q = torch.arange(
                    0, total_q_tokens + 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        max_page_len = (metadata.max_seq_len_k + PAGE_SIZE - 1) // PAGE_SIZE
        page_table = torch.full(
            (bs, max_page_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None,
            page_table,
            self.req_to_token.stride(0),
            max_page_len,
        )
        if forward_batch.forward_mode == ForwardMode.EXTEND and any(forward_batch.extend_prefix_lens_cpu):
            page_table = page_table.repeat_interleave(forward_batch.extend_seq_lens, dim=0)
        metadata.page_table = page_table
        self.forward_metadata = metadata

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        self.page_table = cuda_graph_kv_indices
        self.cu_seqlens_q = torch.zeros(max_bs+1, dtype=torch.int32, device="cuda")
        self.cu_seqlens_k = torch.zeros(max_bs+1, dtype=torch.int32, device="cuda")
        self.cache_seqlens_int32 = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
    
    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        device = seq_lens.device
        max_seq_len_q = 1
        if forward_mode.is_decode_or_idle():
            self.cache_seqlens_int32[:bs] = seq_lens
            self.cu_seqlens_q[:bs+1] = torch.arange(
                0, bs + 1, dtype=torch.int32, device=device
            )
            self.cu_seqlens_k[:bs+1] = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.draft_token_num
            self.cache_seqlens_int32[:bs] = seq_lens
            self.cu_seqlens_q[:bs+1] = torch.arange(
                0,
                bs * self.draft_token_num + 1,
                self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            self.cu_seqlens_k[:bs+1] = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            max_seq_len_q = self.draft_token_num

        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            self.page_table,
            self.req_to_token.stride(0),
            self.page_table.stride(0),
        )
        max_seq_len_k = seq_lens.max().item()
        max_page_len = (max_seq_len_k + PAGE_SIZE - 1) // PAGE_SIZE
        self.forward_metadata = ForwardMetadata(
            self.cu_seqlens_q[:bs+1],
            self.cu_seqlens_k[:bs+1],
            self.cache_seqlens_int32[:bs],
            self.page_table[:bs],
            max_seq_len_q,
            max_seq_len_k,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        max_seq_len_q = 1
        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            self.cache_seqlens_int32[:bs] = seq_lens
            self.cu_seqlens_q[:bs+1] = torch.arange(
                0, bs + 1, dtype=torch.int32, device=device
            )
            self.cu_seqlens_k[:bs+1] = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.draft_token_num
            self.cache_seqlens_int32[:bs] = seq_lens
            self.cu_seqlens_q[:bs+1] = torch.arange(
                0,
                bs * self.draft_token_num + 1,
                self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            self.cu_seqlens_k[:bs+1] = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            max_seq_len_q = self.draft_token_num

        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            self.page_table,
            self.req_to_token.stride(0),
            self.page_table.stride(0),
        )
        max_seq_len_k = seq_lens.max().item()
        max_page_len = (max_seq_len_k + PAGE_SIZE - 1) // PAGE_SIZE
        self.forward_metadata = ForwardMetadata(
            self.cu_seqlens_q[:bs+1],
            self.cu_seqlens_k[:bs+1],
            self.cache_seqlens_int32[:bs],
            self.page_table[:bs],
            max_seq_len_q,
            max_seq_len_k,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


class ZigZagAttnMultiStepBackend:
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[ZigZagAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                ZigZagAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )
