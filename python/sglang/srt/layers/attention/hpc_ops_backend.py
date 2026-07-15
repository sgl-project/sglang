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
- bf16 model dtype; bf16 or fp8_e4m3 KV cache (the FP8 path additionally
  requires the model to run the fused QKNorm+RoPE+quant+StoreKV op via
  ``fused_qk_rope_store_kv_fp8`` — wired for HunYuan V3)
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

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.trtllm_mha_graph_metadata import (
    update_trtllm_mha_graph_metadata,
)
from sglang.srt.layers.attention.triton_ops.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# (num_q_heads // num_kv_heads) values supported by the HPC-Ops MHA kernels.
_SUPPORTED_GQA_GROUP_SIZES = (4, 8)
_SUPPORTED_HEAD_DIM = 128
_REQUIRED_PAGE_SIZE = 64
# The fused QKNorm+RoPE+FP8-quant+StoreKV kernel is specialized per
# (num_q_heads, num_kv_heads); only these shard shapes exist today.
FP8_ROPE_SUPPORTED_HEAD_CONFIGS = ((8, 1), (64, 8))


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

    # --- FP8 pass-through fields ---
    # Written by fused_qk_rope_store_kv_fp8() and consumed by the attention
    # call of the same layer (the fused RoPE op runs right before attention,
    # so the fields always hold the current layer's scales).
    # Dynamic per-token-per-head Q scale.
    # shape: extend [bs, num_q_heads, max_seq_len_q_pad128]; decode [tokens, num_q_heads]
    hpc_q_scale: Optional[torch.Tensor] = None
    # Split-K flag tensor for FP8 decode. shape: [bs, num_kv_heads], int32
    hpc_split_k_flag: Optional[torch.Tensor] = None


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
        if model_runner.kv_cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "The hpc_ops attention backend only supports bf16 or fp8_e4m3 "
                f"KV cache, got {model_runner.kv_cache_dtype}."
            )
        # The FP8 path needs per-token-per-head Q scales that only the fused
        # HPC-Ops QKNorm+RoPE+quant+StoreKV op produces, so it requires the
        # model to call fused_qk_rope_store_kv_fp8() (wired for HunYuan V3).
        self.use_fp8 = model_runner.kv_cache_dtype == torch.float8_e4m3fn
        if self.use_fp8:
            heads = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size,
                model_runner.model_config.get_num_kv_heads(model_runner.tp_size),
            )
            if heads not in FP8_ROPE_SUPPORTED_HEAD_CONFIGS:
                raise ValueError(
                    "The hpc_ops FP8 KV cache path requires per-rank "
                    f"(num_q_heads, num_kv_heads) in "
                    f"{FP8_ROPE_SUPPORTED_HEAD_CONFIGS} (the fused RoPE+quant "
                    f"kernel is specialized per shape), got {heads}. Adjust "
                    "--tp accordingly or use --kv-cache-dtype bfloat16."
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

        # Per-request query prefix sums [0, 1, ..., bs] for the FP8 fused RoPE
        # op in decode mode; a static buffer so it is CUDA-graph friendly.
        self._decode_qo_indptr = torch.arange(
            model_runner.req_to_token_pool.size + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # Fallback per-tensor KV scale for checkpoints without kv scales.
        self._ones_scale = torch.ones(1, dtype=torch.float32, device=self.device)

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

    def _layer_kv_scales(self, layer: RadixAttention):
        """Per-tensor K/V scales as fp32 [1] tensors (ones when absent)."""
        k_scale = (
            layer.k_scale.reshape(1).float()
            if layer.k_scale is not None
            else self._ones_scale
        )
        v_scale = (
            layer.v_scale.reshape(1).float()
            if layer.v_scale is not None
            else self._ones_scale
        )
        return k_scale, v_scale

    def fused_qk_rope_store_kv_fp8(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        qkv: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_norm_weight: Optional[torch.Tensor],
        k_norm_weight: Optional[torch.Tensor],
        qk_norm_policy: int,
    ) -> torch.Tensor:
        """Fused QKNorm + RoPE + FP8 quant + paged-KV write (FP8 mode only).

        Called by the model layer instead of its own norm/rope/KV-write; the
        dynamic per-token-per-head Q scale and the split-K flag are stashed in
        the forward metadata and consumed by the attention call of the same
        layer. Returns the FP8 query of shape [num_tokens, num_q_heads,
        head_dim]. The subsequent ``self.attn(...)`` call must pass
        ``save_kv_cache=False`` (K/V are already written here).

        Under a captured prefill graph (breakable / tc_piecewise) this routes
        through a graph-splitting op — like attention itself — so the Python
        Q-scale hand-off between this op and attention stays alive at replay.

        Note: the HPC-Ops RMSNorm hard-codes eps=1e-6; checkpoints with a
        slightly different rms_norm_eps (e.g. HunYuan V3's 1e-5) accept this
        approximation, matching the reference HPC-Ops integration.
        """
        assert self.use_fp8, "fused_qk_rope_store_kv_fp8 requires fp8 KV cache"

        is_extend = not forward_batch.forward_mode.is_decode_or_idle()
        out_q = torch.empty(
            (qkv.shape[0], layer.tp_q_head_num, layer.head_dim),
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )

        if is_extend and get_tc_piecewise_forward_context() is not None:
            # Captured prefill graph: run through the splitting op (eager at
            # capture AND at replay, keeping the metadata hand-off alive).
            if is_in_breakable_cuda_graph():
                breakable_hpc_ops_fp8_rope_store_kv(
                    qkv,
                    cos_sin_cache,
                    out_q,
                    layer.layer_id,
                    qk_norm_policy,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                )
            else:
                hpc_ops_fp8_rope_store_kv(
                    qkv,
                    cos_sin_cache,
                    out_q,
                    layer.layer_id,
                    qk_norm_policy,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                )
        else:
            self._run_fp8_rope_store_kv(
                layer=layer,
                forward_batch=forward_batch,
                qkv=qkv,
                cos_sin_cache=cos_sin_cache,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                qk_norm_policy=qk_norm_policy,
                is_extend=is_extend,
                out_q=out_q,
            )
        return out_q

    def _run_fp8_rope_store_kv(
        self,
        layer: RadixAttention,
        forward_batch: Optional[ForwardBatch],
        qkv: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_norm_weight: Optional[torch.Tensor],
        k_norm_weight: Optional[torch.Tensor],
        qk_norm_policy: int,
        is_extend: bool,
        out_q: torch.Tensor,
    ) -> None:
        """Invoke the fused kernel and stash the Q scales in the metadata."""
        import hpc

        metadata = self.forward_metadata
        k_cache, v_cache = self._paged_kv_buffers(layer)
        k_scale, v_scale = self._layer_kv_scales(layer)

        if is_extend:
            q_index = metadata.cu_seqlens_q
            max_seqlens = metadata.max_seq_len_q
        else:
            q_index = self._decode_qo_indptr[: forward_batch.batch_size + 1]
            max_seqlens = 1

        # QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR: dynamic per-token-per-head
        # Q quant, static per-tensor K/V quant.
        quant_policy = hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR.value
        _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
            key_cache=k_cache,
            value_cache=v_cache,
            qkv=qkv,
            cos_sin=cos_sin_cache,
            num_seqlen_per_req=metadata.cache_seqlens_int32,
            q_index=q_index,
            kvcache_indices=metadata.page_table,
            is_prefill=is_extend,
            k_scale=k_scale,
            v_scale=v_scale,
            quant_policy=quant_policy,
            max_seqlens=max_seqlens,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            out_q=out_q,
            qk_norm_policy=qk_norm_policy,
        )
        metadata.hpc_q_scale = q_scale
        metadata.hpc_split_k_flag = split_k_flag

    def _take_fp8_scales(self, metadata: HPCOpsMetadata):
        """Pop the per-layer FP8 scales written by the fused RoPE op."""
        q_scale = metadata.hpc_q_scale
        split_k_flag = metadata.hpc_split_k_flag
        if q_scale is None:
            raise RuntimeError(
                "The hpc_ops attention backend with an fp8_e4m3 KV cache "
                "requires the model to run the fused HPC-Ops "
                "QKNorm+RoPE+quant+StoreKV op (fused_qk_rope_store_kv_fp8), "
                "which produces the per-token-per-head Q scales. This is "
                "currently wired for HunYuan V3 only; use "
                "--kv-cache-dtype bfloat16 for other models."
            )
        metadata.hpc_q_scale = None
        metadata.hpc_split_k_flag = None
        return q_scale, split_k_flag

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

        if self.use_fp8:
            q_scale, _ = self._take_fp8_scales(metadata)
            k_scale, v_scale = self._layer_kv_scales(layer)
            o = hpc.attention_with_kvcache_prefill_fp8(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                q_scale,
                k_scale,
                v_scale,
                metadata.cu_seqlens_q,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                metadata.max_seq_len_q,
            )
        else:
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

        if self.use_fp8:
            q_scale, split_k_flag = self._take_fp8_scales(metadata)
            k_scale, v_scale = self._layer_kv_scales(layer)
            o = hpc.attention_decode_fp8(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                q_scale,
                k_scale,
                v_scale,
                mtp=0,
                new_kv_included=True,
                splitk=True,
                split_flag=split_k_flag,
            )
        else:
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


@register_custom_op(mutates_args=["out_q"])
@register_split_op()
def hpc_ops_fp8_rope_store_kv(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_q: torch.Tensor,
    layer_id: int,
    qk_norm_policy: int,
    *,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
) -> None:
    """Graph-splitting wrapper for the fused QKNorm+RoPE+FP8-quant+StoreKV op.

    Like ``unified_attention_with_output``, this runs eagerly between captured
    prefill-graph segments (at capture and at every replay), so the Python
    hand-off of the dynamic Q scales to the following attention op stays
    alive. ``out_q`` is preallocated by the captured segment and mutated in
    place, which is what stitches the surrounding graph segments together.
    """
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layer = context.attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    backend = get_attn_backend()
    backend._run_fp8_rope_store_kv(
        layer=attention_layer,
        forward_batch=forward_batch,
        qkv=qkv[:real_num_tokens],
        cos_sin_cache=cos_sin_cache,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        is_extend=True,
        out_q=out_q[:real_num_tokens],
    )


breakable_hpc_ops_fp8_rope_store_kv = eager_on_graph(True)(hpc_ops_fp8_rope_store_kv)
