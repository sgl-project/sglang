from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils import (
    get_cuda_graph_max_batch_size,
    is_flashinfer_available,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Check if flashinfer with cuDNN support is available
if is_flashinfer_available():
    try:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
        FLASHINFER_CUDNN_AVAILABLE = True
    except ImportError:
        FLASHINFER_CUDNN_AVAILABLE = False
else:
    FLASHINFER_CUDNN_AVAILABLE = False


@dataclass
class DecodeMetadata:
    """Metadata for decode phase."""
    # We'll store the decode wrapper and any other state needed
    pass


@dataclass
class PrefillMetadata:
    """Metadata for prefill/extend phase."""
    use_ragged: bool
    extend_no_prefix: bool


class CuDNAttnBackend(AttentionBackend):
    """cuDNN attention backend for multi-head attention with paged KV cache.

    This backend uses NVIDIA's cuDNN library for fast attention computation,
    particularly optimized for paged attention with KV cache.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__()

        if not FLASHINFER_CUDNN_AVAILABLE:
            raise RuntimeError(
                "cuDNN attention backend requires flashinfer with cuDNN support. "
                "Please ensure flashinfer_python[cu13] is installed with cuDNN 9.15+."
            )

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.max_context_len = model_runner.model_config.context_len
        self.page_size = model_runner.page_size
        self.skip_prefill = skip_prefill

        # Parse constants
        self.num_attention_heads = model_runner.model_config.num_attention_heads
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.model_config.attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.dtype = model_runner.dtype
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # Workspace buffer for cuDNN
        self.workspace_buffer = get_buffer(
            "cudnn_workspace",
            lambda: torch.empty(
                128 * 1024 * 1024,  # 128 MB workspace
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )

        # Buffers for CUDA graph
        self.cuda_graph_kv_indices: Optional[torch.Tensor] = None
        self.cuda_graph_qo_indptr: Optional[torch.Tensor] = None
        self.cuda_graph_kv_indptr: Optional[torch.Tensor] = None
        self.cuda_graph_kv_last_page_len: Optional[torch.Tensor] = None

        # Metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata, None] = None

        # Decode state
        self.decode_wrapper = None
        self.decode_kv_indptr: Optional[torch.Tensor] = None
        self.decode_kv_last_page_len: Optional[torch.Tensor] = None

        logger.info("Initialized CuDNAttnBackend")

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state."""
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.cuda_graph_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.cuda_graph_kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device="cuda"
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Prepare metadata outside CUDA graph."""
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        forward_mode = forward_batch.forward_mode

        if forward_mode.is_decode_or_idle():
            # For decode, we need to build the kv_indptr and kv_indices
            # This is similar to what FlashInfer does
            self._prepare_decode_metadata(
                bs, seq_lens, seq_lens_cpu, seq_lens_sum, in_capture
            )
        elif forward_mode.is_extend() or forward_mode.is_target_verify():
            # For prefill/extend, prepare prefill metadata
            self._prepare_prefill_metadata(
                forward_batch, in_capture
            )

    def _prepare_decode_metadata(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        in_capture: bool,
    ):
        """Prepare metadata for decode phase."""
        # Build kv_indptr from seq_lens
        kv_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=seq_lens.device)
        kv_indptr[1:] = torch.cumsum(seq_lens[:bs], dim=0, dtype=torch.int32)

        if in_capture:
            # Store for CUDA graph capture
            self.cuda_graph_kv_indptr[:bs + 1].copy_(kv_indptr)
            self.cuda_graph_kv_last_page_len[:bs].fill_(1)  # page_size=1 for decode
        else:
            self.decode_kv_indptr = kv_indptr
            self.decode_kv_last_page_len = torch.ones(
                (bs,), dtype=torch.int32, device=seq_lens.device
            )

        self.forward_metadata = DecodeMetadata()

    def _prepare_prefill_metadata(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool,
    ):
        """Prepare metadata for prefill/extend phase."""
        # For prefill with paged KV cache, we need to build indptrs
        # This is a simplified version - full implementation would be more complex
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens

        use_ragged = not in_capture
        extend_no_prefix = not any(
            forward_batch.extend_prefix_lens_cpu
        ) if forward_batch.extend_prefix_lens_cpu is not None else True

        if in_capture:
            # For CUDA graph, we use fixed-size buffers
            self.cuda_graph_qo_indptr[:bs + 1].fill_(0)
            self.cuda_graph_kv_indptr[:bs + 1].fill_(0)

        self.forward_metadata = PrefillMetadata(
            use_ragged=use_ragged,
            extend_no_prefix=extend_no_prefix,
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Prepare metadata inside CUDA graph."""
        # For cuDNN, the metadata preparation is done out of graph
        # This is a no-op for now
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens."""
        return 1

    @debug_kernel_api
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
        """Run forward for extend (prefill with cached prefix)."""
        metadata = self.forward_metadata
        assert isinstance(metadata, PrefillMetadata), "Expected PrefillMetadata for extend"

        # Reshape q, k, v for flashinfer (NHD format)
        # q: [total_tokens, num_heads, head_dim]
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # Get KV cache buffer
        cache_loc = forward_batch.out_cache_loc
        if k is not None and save_kv_cache:
            assert v is not None
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(cache_loc, None),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        # Build indptrs for cuDNN
        # cuDNN expects packed indptrs: [qk_indptr, v_indptr, o_indptr]
        # Each is (batch_size + 1, 1, 1, 1) with element offsets
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens[:bs]
        extend_prefix_lens = forward_batch.extend_prefix_lens[:bs]

        # Total tokens per sequence (prefix + extend)
        total_lens = seq_lens
        # Build qk_indptr (element offsets)
        qk_indptr = torch.zeros((bs + 1, 1, 1, 1), dtype=torch.int32, device=q.device)
        qk_indptr[1:, 0, 0, 0] = torch.cumsum(
            total_lens * layer.tp_q_head_num * layer.head_dim, dim=0, dtype=torch.int32
        )

        # v_indptr (same as qk for now)
        v_indptr = qk_indptr.clone()
        # o_indptr (output)
        o_indptr = qk_indptr.clone()

        # Pack indptrs
        packed_indptrs = torch.cat([qk_indptr.view(-1), v_indptr.view(-1), o_indptr.view(-1)])

        # Sequence lengths in 4D format (B, 1, 1, 1)
        actual_seq_lens_q = total_lens.view(bs, 1, 1, 1)
        actual_seq_lens_kv = total_lens.view(bs, 1, 1, 1)

        # Compute scale
        scale = layer.scaling if layer.scaling is not None else layer.head_dim ** -0.5

        # Determine max seqlen
        max_seqlen = int(total_lens.max().item())

        # Call cuDNN batch prefill with KV cache
        output, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            kv_cache[0],  # k_cache
            kv_cache[1],  # v_cache
            scale,
            self.workspace_buffer,
            max_token_per_sequence=max_seqlen,
            max_sequence_kv=max_seqlen,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            causal=True,
            return_lse=False,
            batch_offsets_q=qk_indptr,
            batch_offsets_k=qk_indptr,
            batch_offsets_v=v_indptr,
            batch_offsets_o=o_indptr,
            is_cuda_graph_compatible=True,
        )

        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
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
        """Run forward for decode phase.

        For decode, cuDNN doesn't have a direct batch decode with paged KV cache
        in flashinfer yet. We'll use a fallback to FlashInfer's decode or
        implement a custom kernel. For now, we raise NotImplementedError
        and will implement it once we confirm the API.
        """
        # TODO: Implement cuDNN decode with paged KV cache
        # Currently flashinfer doesn't expose cudnn_batch_decode_with_kv_cache
        # We can either:
        # 1. Use FlashInfer's regular decode as fallback
        # 2. Implement a custom cuDNN decode kernel
        # 3. Wait for flashinfer to add cuDNN decode support

        raise NotImplementedError(
            "cuDNN decode with paged KV cache is not yet implemented. "
            "FlashInfer currently only provides cudnn_batch_prefill_with_kv_cache. "
            "Please use --attention-backend flashinfer for decode."
        )

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run forward for mixed prefill/decode (NPU only)."""
        raise NotImplementedError("Mixed mode not supported for cuDNN backend")

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self,
        forward_batch: ForwardBatch,
    ):
        """Create forward metadata for breakable CUDA graph capture."""
        raise NotImplementedError("Breakable CUDA graph not supported yet")

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ):
        """Refresh captured metadata for BCG replay."""
        raise NotImplementedError("Breakable CUDA graph not supported yet")

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int]
    ):
        """Update verify buffers after draft."""
        raise NotImplementedError("Speculative decoding not supported yet")