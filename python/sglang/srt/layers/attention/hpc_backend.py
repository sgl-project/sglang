"""HPC-ops Attention Backend for SGLang.

Integrates Tencent's hpc-ops paged attention kernels into SGLang's attention
backend framework. Supports:

  * **BF16** KV cache — ``hpc.attention_with_kvcache_prefill_bf16`` / ``hpc.attention_decode_bf16``
  * **FP8** KV cache — ``hpc.attention_with_kvcache_prefill_fp8`` / ``hpc.attention_decode_fp8``

Constraints:
  * SM90+ (Hopper / H20)
  * head_dim == 128
  * page_size should match hpc-ops block_size (recommended 64)
  * KV cache layout must be NHD (not vectorized_5d)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# FP8 E4M3 max representable value
_FP8_E4M3_MAX = 448.0


@dataclass
class HpcForwardMetadata:
    """Metadata required by the HPC attention kernels."""

    # Block table: [bs, max_blocks], int32
    block_ids: torch.Tensor
    # KV cache sequence lengths: [bs], int32
    seqlens_kvcache: torch.Tensor
    # Cumulative Q lengths (prefill only): [bs + 1], int32
    cu_seqlens_q: Optional[torch.Tensor]
    # Max Q length (prefill only)
    max_seqlens_q: Optional[int]
    # Batch size
    bs: int
    # Whether new KV was already written (affects seqlens_kvcache)
    new_kv_included: bool


class HpcAttnBackend(AttentionBackend):
    """Attention backend using Tencent hpc-ops kernels."""

    # HPC backend reads seq_lens on CPU for metadata building
    needs_cpu_seq_lens: bool = True

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        try:
            import hpc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "hpc-ops package is not installed. "
                "Please install it from https://github.com/Tencent/hpc-ops"
            ) from e

        from sglang.srt.configs.model_config import AttentionArch

        if model_runner.model_config.attention_arch == AttentionArch.MLA:
            raise ValueError("HPC attention backend does not support MLA models.")

        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # Model config
        self.num_heads = (
            model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        if self.v_head_dim == -1:
            self.v_head_dim = self.head_dim

        # Check constraints
        if self.head_dim != 128:
            raise ValueError(
                f"HPC attention backend requires head_dim=128, got {self.head_dim}"
            )

        if self.v_head_dim != self.head_dim:
            raise ValueError(
                f"HPC attention backend requires v_head_dim == head_dim, "
                f"got v_head_dim={self.v_head_dim}, head_dim={self.head_dim}"
            )

        major, _ = torch.cuda.get_device_capability(model_runner.gpu_id)
        if major < 9:
            raise ValueError(
                f"HPC attention backend requires SM90+ (Hopper), got SM{major}x"
            )

        # Page size and KV cache dtype
        self.page_size = model_runner.server_args.page_size or 1
        if self.page_size != 64:
            logger.warning(
                f"HPC attention backend is optimized for page_size=64, "
                f"got page_size={self.page_size}. "
                f"Performance may be degraded."
            )

        self.kv_cache_dtype = model_runner.server_args.kv_cache_dtype
        self.use_fp8 = self.kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2")

        self.max_context_len = model_runner.model_config.context_len
        self.max_blocks = (self.max_context_len + self.page_size - 1) // self.page_size

        self.forward_metadata: Optional[HpcForwardMetadata] = None

    def _get_paged_kv_cache(self, layer: RadixAttention):
        """Get KV cache in hpc-ops paged format: [num_blocks, block_size, num_kv_heads, head_dim]."""
        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if k_buffer.ndim == 3:
            # NHD layout: (total_slots, head_num, head_dim)
            kcache = k_buffer.view(-1, self.page_size, self.num_kv_heads, self.head_dim)
            vcache = v_buffer.view(-1, self.page_size, self.num_kv_heads, self.v_head_dim)
        elif k_buffer.ndim == 5:
            raise ValueError(
                "HPC attention backend does not support vectorized_5d KV cache layout. "
                "Please use NHD layout (set SGLANG_KV_CACHE_LAYOUT=nhd or equivalent)."
            )
        else:
            raise ValueError(
                f"Unexpected KV cache buffer ndim: {k_buffer.ndim}"
            )

        return kcache, vcache

    def _build_block_ids(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Build the block table [bs, max_blocks] from SGLang's req_to_token.

        For request *i* and block *j*:
            block_ids[i, j] = req_to_token[req_pool_indices[i], j * page_size] // page_size
        """
        device = seq_lens.device
        max_seq_len = int(seq_lens[:bs].max().item()) if bs > 0 else 1
        max_blocks = (max_seq_len + self.page_size - 1) // self.page_size
        max_blocks = min(max_blocks, self.req_to_token_pool.req_to_token.shape[1] // self.page_size)

        # Index req_to_token with req_pool_indices
        token_table = self.req_to_token_pool.req_to_token[req_pool_indices[:bs]]

        # Get first token of each block: indices 0, page_size, 2*page_size, ...
        block_starts = torch.arange(
            0, max_blocks * self.page_size, self.page_size, device=device, dtype=torch.int32
        )
        # token_table[:, block_starts] -> [bs, max_blocks]
        block_ids = (token_table[:, block_starts] // self.page_size).to(torch.int32)

        return block_ids

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build metadata for the current forward batch (eager path)."""
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        device = seq_lens.device

        block_ids = self._build_block_ids(req_pool_indices, seq_lens, bs)

        if forward_batch.forward_mode.is_decode_or_idle():
            # Decode: KV cache already includes new tokens (we save before attention)
            seqlens_kvcache = seq_lens[:bs].to(torch.int32)
            cu_seqlens_q = None
            max_seqlens_q = None
            new_kv_included = True
        else:
            # Extend / prefill
            extend_seq_lens = forward_batch.extend_seq_lens[:bs]
            cu_seqlens_q = torch.zeros(
                bs + 1, dtype=torch.int32, device=device
            )
            cu_seqlens_q[1:] = torch.cumsum(extend_seq_lens, dim=0).to(torch.int32)

            # After saving KV, total cache length = seq_lens
            seqlens_kvcache = seq_lens[:bs].to(torch.int32)
            max_seqlens_q = (
                int(extend_seq_lens.max().item()) if bs > 0 else 1
            )
            new_kv_included = True

        self.forward_metadata = HpcForwardMetadata(
            block_ids=block_ids,
            seqlens_kvcache=seqlens_kvcache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlens_q=max_seqlens_q,
            bs=bs,
            new_kv_included=new_kv_included,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate static buffers for CUDA graph capture/replay."""
        self.cuda_graph_block_ids = torch.zeros(
            (max_bs, self.max_blocks), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cu_seqlens_q = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_seqlens_kvcache = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        """Fill pre-allocated CUDA graph buffers for the current batch."""
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices

        # Build block_ids into pre-allocated buffer
        block_ids = self._build_block_ids(req_pool_indices, seq_lens, bs)
        max_blocks = block_ids.shape[1]
        self.cuda_graph_block_ids[:bs, :max_blocks] = block_ids
        # Zero out unused blocks
        if max_blocks < self.max_blocks:
            self.cuda_graph_block_ids[:bs, max_blocks:] = 0

        if forward_batch.forward_mode.is_decode_or_idle():
            self.cuda_graph_seqlens_kvcache[:bs] = seq_lens[:bs].to(torch.int32)
            cu_seqlens_q = None
            max_seqlens_q = None
            new_kv_included = True
        else:
            extend_seq_lens = forward_batch.extend_seq_lens[:bs]
            self.cuda_graph_cu_seqlens_q[0] = 0
            self.cuda_graph_cu_seqlens_q[1 : bs + 1] = torch.cumsum(
                extend_seq_lens, dim=0
            ).to(torch.int32)
            cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]

            self.cuda_graph_seqlens_kvcache[:bs] = seq_lens[:bs].to(torch.int32)
            max_seqlens_q = (
                int(extend_seq_lens.max().item()) if bs > 0 else 1
            )
            new_kv_included = True

        self.forward_metadata = HpcForwardMetadata(
            # Use full-width slice (contiguous); the kernel relies on
            # num_seq_kvcache to determine the valid block count per request.
            block_ids=self.cuda_graph_block_ids[:bs],
            seqlens_kvcache=self.cuda_graph_seqlens_kvcache[:bs],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlens_q=max_seqlens_q,
            bs=bs,
            new_kv_included=new_kv_included,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        """Seq lens are padded with 1 (valid single-token decode)."""
        return 1

    def _save_kv_cache(self, k, v, layer, forward_batch, save_kv_cache):
        """Save new K/V tensors into the paged KV cache."""
        if not save_kv_cache or k is None or v is None:
            return

        if layer.k_scale is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Decode attention using hpc-ops kernels."""
        import hpc
        from hpc import QuantType

        md = self.forward_metadata
        bs = md.bs

        # Save new KV to paged cache first
        self._save_kv_cache(k, v, layer, forward_batch, save_kv_cache)

        # Get paged KV cache
        kcache, vcache = self._get_paged_kv_cache(layer)

        # Reshape Q: [num_tokens, num_heads, head_dim]
        # For decode, num_tokens == bs (1 token per request)
        q_3d = q.view(-1, self.num_heads, self.head_dim)

        # Output: [num_tokens, num_heads, v_head_dim]
        o = torch.empty(
            q_3d.shape[0], self.num_heads, self.v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        if self.use_fp8:
            # Compute per-head per-batch Q scale
            q_max = q_3d.abs().amax(dim=-1)  # [bs, num_heads]
            qscale = (q_max / _FP8_E4M3_MAX).clamp(min=1e-12).to(torch.float32)

            # K/V scales are per-tensor
            kscale = torch.tensor(
                [layer.k_scale_float if layer.k_scale_float else 1.0],
                dtype=torch.float32,
                device=self.device,
            )
            vscale = torch.tensor(
                [layer.v_scale_float if layer.v_scale_float else 1.0],
                dtype=torch.float32,
                device=self.device,
            )

            hpc.attention_decode_fp8(
                q=q_3d,
                kcache=kcache,
                vcache=vcache,
                block_ids=md.block_ids,
                num_seq_kvcache=md.seqlens_kvcache,
                qscale=qscale,
                kscale=kscale,
                vscale=vscale,
                mtp=0,
                new_kv_included=md.new_kv_included,
                quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
                splitk=True,
                output=o,
            )
        else:
            hpc.attention_decode_bf16(
                q=q_3d,
                kcache=kcache,
                vcache=vcache,
                block_ids=md.block_ids,
                num_seq_kvcache=md.seqlens_kvcache,
                mtp=0,
                new_kv_included=md.new_kv_included,
                splitk=True,
                output=o,
            )

        # Return [num_tokens, num_heads * head_dim]
        return o.reshape(-1, self.num_heads * self.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Prefill / extend attention using hpc-ops kernels."""
        import hpc
        from hpc import QuantType

        md = self.forward_metadata
        bs = md.bs

        # Save new KV to paged cache first
        self._save_kv_cache(k, v, layer, forward_batch, save_kv_cache)

        # Get paged KV cache
        kcache, vcache = self._get_paged_kv_cache(layer)

        # Reshape Q: [total_seq, num_heads, head_dim]
        q_3d = q.view(-1, self.num_heads, self.head_dim)

        # Output: [total_seq, num_heads, v_head_dim]
        o = torch.empty(
            q_3d.shape[0], self.num_heads, self.v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        if self.use_fp8:
            # Quantize Q to FP8 and compute per-token-per-head scales
            q_max = q_3d.abs().amax(dim=-1)  # [total_seq, num_heads]
            q_scale_flat = (q_max / _FP8_E4M3_MAX).clamp(min=1e-12).to(torch.float32)
            q_fp8 = (q_3d / q_scale_flat.unsqueeze(-1)).to(torch.float8_e4m3fn)

            # Reshape q_scale to [bs, num_heads, max_seqlens_q_pad]
            max_q = md.max_seqlens_q
            max_q_pad = ((max_q + 15) // 16) * 16  # Pad to 16
            qscale = torch.zeros(
                bs, self.num_heads, max_q_pad, dtype=torch.float32, device=self.device
            )
            cu_q = md.cu_seqlens_q
            for i in range(bs):
                start = int(cu_q[i].item())
                end = int(cu_q[i + 1].item())
                seq_i = end - start
                if seq_i > 0:
                    qscale[i, :, :seq_i] = q_scale_flat[start:end].t()

            # K/V scales are per-tensor
            kscale = torch.tensor(
                [layer.k_scale_float if layer.k_scale_float else 1.0],
                dtype=torch.float32,
                device=self.device,
            )
            vscale = torch.tensor(
                [layer.v_scale_float if layer.v_scale_float else 1.0],
                dtype=torch.float32,
                device=self.device,
            )

            hpc.attention_with_kvcache_prefill_fp8(
                q=q_fp8,
                kcache=kcache,
                vcache=vcache,
                qscale=qscale,
                kscale=kscale,
                vscale=vscale,
                cu_seqlens_q=cu_q,
                block_ids=md.block_ids,
                seqlens_kvcache=md.seqlens_kvcache,
                max_seqlens_q=max_q,
                quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
                output=o,
            )
        else:
            hpc.attention_with_kvcache_prefill_bf16(
                q=q_3d,
                kcache=kcache,
                vcache=vcache,
                cu_seqlens_q=md.cu_seqlens_q,
                block_ids=md.block_ids,
                seqlens_kvcache=md.seqlens_kvcache,
                max_seqlens_q=md.max_seqlens_q,
                output=o,
            )

        # Return [num_tokens, num_heads * v_head_dim]
        return o.reshape(-1, self.num_heads * self.v_head_dim)
