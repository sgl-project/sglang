from __future__ import annotations

"""
Attention backend powered by HPC-Ops (https://github.com/Tencent/hpc-ops),
a production-grade operator library for LLM inference developed by the
Tencent Hunyuan AI Infra team.

The backend wraps the paged MHA kernels ``attention_with_kvcache_prefill_bf16``
(extend) and ``attention_decode_bf16`` (decode). It reads the SGLang
token-major KV pool as a paged NHD cache ``(num_pages, page_size, num_kv_heads,
head_dim)`` without any copy, so the only hard requirements are the kernel
constraints:

- NVIDIA Hopper or newer (sm90+)
- ``--page-size 64``
- bf16 model dtype and bf16 KV cache
- head_dim == 128 and num_q_heads // num_kv_heads in {4, 8}
- default softmax scaling (``head_dim ** -0.5``), no sliding window,
  no logit cap, decoder-only attention

Note that the HPC-Ops kernels are currently tuned primarily for H20: on other
GPUs (H100/H200/B200, ...) the speedup over the default attention backend may
be limited or absent.

Enable it explicitly with ``--attention-backend hpc_ops``.
"""

import functools
import importlib.util
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.trtllm_mha_graph_metadata import (
    update_trtllm_mha_graph_metadata,
)
from sglang.srt.layers.attention.triton_ops.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# (num_q_heads // num_kv_heads) values supported by the HPC-Ops MHA kernels.
_SUPPORTED_GQA_GROUP_SIZES = (4, 8)
_SUPPORTED_HEAD_DIM = 128
_REQUIRED_PAGE_SIZE = 64


@functools.cache
def has_hpc_ops() -> bool:
    """Return True if the ``hpc`` package (HPC-Ops) is installed."""
    return importlib.util.find_spec("hpc") is not None


class HPCOpsMetadata(msgspec.Struct):
    """Per-forward-batch metadata consumed by the HPC-Ops kernels."""

    # KV cache length per request (including tokens written this iteration).
    # shape: [batch_size], int32
    cache_seqlens_int32: Optional[torch.Tensor] = None
    # Cumulative query lengths, extend mode only. shape: [batch_size + 1], int32
    cu_seqlens_q: Optional[torch.Tensor] = None
    # Cumulative KV lengths. Only used as a scratch output of the fused
    # cuda-graph metadata kernel. shape: [batch_size + 1], int32
    cu_seqlens_k: Optional[torch.Tensor] = None
    # Paged KV block table. shape: [batch_size, max_num_pages], int32
    page_table: Optional[torch.Tensor] = None
    # Maximum query length among the batch (1 for decode).
    max_seq_len_q: int = 1


class HPCOpsAttnBackend(AttentionBackend):
    """HPC-Ops paged MHA attention backend (bf16 KV cache)."""

    # The page table is built on-device from seq_lens, so the D2H sync for
    # seq_lens_cpu is not needed (same as trtllm_mha / triton).
    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        if not has_hpc_ops():
            raise ImportError(
                "The hpc_ops attention backend requires the `hpc` package. "
                "Install it from https://github.com/Tencent/hpc-ops"
            )

        self.page_size = model_runner.page_size
        if self.page_size != _REQUIRED_PAGE_SIZE:
            raise ValueError(
                f"The hpc_ops attention backend requires --page-size "
                f"{_REQUIRED_PAGE_SIZE}, got {self.page_size}."
            )
        if model_runner.kv_cache_dtype != torch.bfloat16:
            raise ValueError(
                "The hpc_ops attention backend only supports the bf16 KV cache "
                f"for now, got {model_runner.kv_cache_dtype}."
            )

        config = model_runner.model_config
        head_dim = config.head_dim
        num_q_heads = config.num_attention_heads // model_runner.tp_size
        num_kv_heads = config.get_num_kv_heads(model_runner.tp_size)
        gqa_group_size = num_q_heads // num_kv_heads
        if head_dim != _SUPPORTED_HEAD_DIM or gqa_group_size not in (
            _SUPPORTED_GQA_GROUP_SIZES
        ):
            raise ValueError(
                "The hpc_ops attention backend requires head_dim == "
                f"{_SUPPORTED_HEAD_DIM} and num_q_heads // num_kv_heads in "
                f"{_SUPPORTED_GQA_GROUP_SIZES}, got head_dim={head_dim}, "
                f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}."
            )

        self.max_context_len = model_runner.model_config.context_len
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # CUDA graph state (allocated in init_cuda_graph_state).
        self.decode_cuda_graph_metadata = {}

        self.forward_metadata: Optional[HPCOpsMetadata] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build metadata for the eager (non-CUDA-graph) path."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        metadata = HPCOpsMetadata()
        metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata.max_seq_len_q = 1
        else:
            assert forward_batch.forward_mode.is_extend(), (
                "The hpc_ops attention backend does not support forward mode "
                f"{forward_batch.forward_mode}."
            )
            max_q = max(forward_batch.extend_seq_lens_cpu)
            metadata.max_seq_len_q = (
                int(max_q.item()) if isinstance(max_q, torch.Tensor) else int(max_q)
            )
            metadata.cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )

        metadata.page_table = torch.empty(
            (batch_size, self.max_num_pages), dtype=torch.int32, device=device
        )
        build_trtllm_mha_page_table(
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            cache_seqlens=metadata.cache_seqlens_int32,
            page_table=metadata.page_table,
            page_size=self.page_size,
        )
        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs, self.max_num_pages, dtype=torch.int32, device=self.device
            ),
        }

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        bs = forward_batch.batch_size
        assert forward_batch.forward_mode.is_decode_or_idle(), (
            "The hpc_ops attention backend only supports CUDA graphs for "
            f"normal decode, got {forward_batch.forward_mode}."
        )
        if in_capture:
            metadata = HPCOpsMetadata()
            metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                "cache_seqlens"
            ][:bs]
            metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                : bs + 1
            ]
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            metadata.max_seq_len_q = 1
            self.decode_cuda_graph_metadata[bs] = metadata

        self.forward_metadata = self.decode_cuda_graph_metadata[bs]

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Rebuild cache_seqlens and the page table on-device (graph-recordable)."""
        bs = forward_batch.batch_size
        metadata = self.decode_cuda_graph_metadata[bs]
        update_trtllm_mha_graph_metadata(
            req_pool_indices=forward_batch.req_pool_indices[:bs],
            seq_lens=forward_batch.seq_lens[:bs],
            req_to_token=self.req_to_token,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_k=metadata.cu_seqlens_k,
            page_table=metadata.page_table,
            bs=bs,
            seqlen_offset=0,
            max_seq_pages=self.max_num_pages,
            page_size=self.page_size,
        )
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def _check_layer_supported(self, layer: RadixAttention):
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise ValueError(
                "The hpc_ops attention backend does not support sliding window."
            )
        if layer.logit_cap > 0:
            raise ValueError(
                "The hpc_ops attention backend does not support logit cap."
            )
        # The HPC-Ops kernels hard-code the softmax scale to head_dim ** -0.5.
        if abs(layer.scaling - layer.head_dim**-0.5) > 1e-8:
            raise ValueError(
                "The hpc_ops attention backend only supports the default "
                f"softmax scaling head_dim ** -0.5, got {layer.scaling}."
            )

    def _paged_kv_buffers(self, layer: RadixAttention):
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)
        return k_cache, v_cache

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        import hpc

        self._check_layer_supported(layer)

        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        metadata = self.forward_metadata
        k_cache, v_cache = self._paged_kv_buffers(layer)

        o = hpc.attention_with_kvcache_prefill_bf16(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache,
            v_cache,
            metadata.cu_seqlens_q,
            metadata.page_table,
            metadata.cache_seqlens_int32,
            metadata.max_seq_len_q,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        import hpc

        self._check_layer_supported(layer)

        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        metadata = self.forward_metadata
        k_cache, v_cache = self._paged_kv_buffers(layer)

        o = hpc.attention_decode_bf16(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache,
            v_cache,
            metadata.page_table,
            metadata.cache_seqlens_int32,
            mtp=0,
            new_kv_included=True,
            splitk=True,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
