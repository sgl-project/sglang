from __future__ import annotations

"""
Support attention backend for TRTLLM MHA kernels from flashinfer.
The kernel supports sm100 only, with sliding window and attention sink features.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import canonicalize_stride
from sglang.kernels.ops.kvcache.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
from sglang.kernels.ops.kvcache.trtllm_mha_graph_metadata import (
    Q_MODE_NONE,
    Q_MODE_STRIDED,
    update_trtllm_mha_graph_metadata,
)
from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_buffer
from sglang.srt.speculative.ragged_verify import (
    build_ragged_target_verify_geometry,
    resolve_ragged_verify_layout,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
    from sglang.srt.speculative.spec_info import SpecInput

# Constants
# Default workspace size in MB for TRTLLM MHA
# Can be configured via SGLANG_FLASHINFER_WORKSPACE_SIZE environment variable
DEFAULT_WORKSPACE_SIZE_MB = 512

# Reuse this workspace buffer across all TRTLLM MHA wrappers


@dataclass
class TRTLLMMHAMetadata:
    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Cumulative sequence lengths for `query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # Page table for SWA layers (translated from full pool indices to SWA pool indices)
    swa_page_table: torch.Tensor = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: torch.Tensor = None
    is_ragged_verify: bool = False


class TRTLLMHAAttnBackend(FlashInferAttnBackend):
    """TRTLLM MHA attention kernel from flashinfer."""

    # Build the page table on-device from seq_lens (incl. the SWA-translated table
    # via the full->SWA lookup; see _fill_page_table_device), so we never need the
    # seq_lens_cpu D2H sync; opt out of it, matching trtllm_mla / triton.
    needs_cpu_seq_lens: bool = False

    supports_ragged_verify_graph: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
    ):
        # Capture workspace size before super().__init__() to preserve user's
        # SGLANG_FLASHINFER_WORKSPACE_SIZE setting (may be overridden by parent)
        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
        workspace_size_bytes = (
            env_var.get()
            if env_var.is_set()
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        )

        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        config = model_runner.model_config

        # MHA-specific dimensions
        self.max_context_len = model_runner.model_config.context_len
        self.hidden_size = config.hidden_size

        # Runtime parameters
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device

        # Workspace allocation
        self.workspace_size = workspace_size_bytes
        # Allocate buffers
        self.workspace_buffer = get_buffer(
            "trtllm_mha_zero_workspace",
            lambda: torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}

        # Speculative decoding
        # Only support topk <= 1 for now.
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_step_id = speculative_step_id
        self.target_verify_metadata = {}

        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )

        # SWA hybrid models split the KV cache into full and SWA pools with
        # separate index spaces; SWA layers need a translated page_table.
        self._swa_kv_pool: Optional[SWAKVPool] = self._resolve_swa_kv_pool(model_runner)
        # Raw full->swa index mapping tensor for the fused cuda-graph
        # metadata kernel (gather + // page_size happen on device).
        if self._swa_kv_pool is not None:
            self._swa_full_to_swa_mapping = self._swa_kv_pool.full_to_swa_index_mapping
            assert self._swa_full_to_swa_mapping is not None, (
                "SWA pool must register full_to_swa_index_mapping before "
                "TRTLLMHAAttnBackend is constructed"
            )
        else:
            self._swa_full_to_swa_mapping = None

        # Static page-table width (upper bound). The CUDA-graph path builds the
        # page table on-device sized to this constant, so it never reads a runtime
        # max. See _fill_page_table_device.
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size

        # Forward metadata
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None

        # Init backend (XQA or TRTLLM-GEN)
        # We need to specify q_type and out_type for different backend
        # XQA: (q_type must be bf16)
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = bf16, out_type=model_runner.dtype
        # TRTLLM-GEN:
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = fp8, out_type=model_runner.dtype
        self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[SWAKVPool]:
        """Return the SWAKVPool to translate against, or None for non-SWA models.

        EAGLE draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's
        SWA mapping for that draft pool. FROZEN_KV MTP is the exception: its
        draft path reads target KV directly, so it still needs the allocator
        pool when the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, SWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        allocator = model_runner.token_to_kv_pool_allocator
        kvcache = allocator.get_kvcache()
        return kvcache if isinstance(kvcache, SWAKVPool) else None

    def _alloc_swa_page_table(
        self, max_bs: int, max_num_pages: int
    ) -> Optional[torch.Tensor]:
        """Allocate a SWA page_table buffer, or return None for non-SWA models."""
        if self._swa_kv_pool is None:
            return None
        return torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device)

    def _fill_page_table_device(
        self,
        metadata: TRTLLMMHAMetadata,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ):
        """Build the page table on-device from per-request KV lengths (no sync).

        Fills ``metadata.page_table`` (a [bs, max_num_pages] buffer) in place with
        block ids derived from ``cache_seqlens`` (a GPU tensor); for SWA models it
        also fills ``metadata.swa_page_table`` via the full->SWA lookup. The Triton
        kernel self-guards per request on the device-side length, so the grid and
        buffer width use the static ``max_num_pages`` upper bound while the actual
        writes stay bounded by ``cache_seqlens`` — no host-side max / D2H sync.
        """
        has_swa = self._swa_kv_pool is not None
        build_trtllm_mha_page_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            cache_seqlens=cache_seqlens,
            page_table=metadata.page_table,
            page_size=self.page_size,
            swa_page_table=metadata.swa_page_table if has_swa else None,
            full_to_swa=(
                self._swa_kv_pool.full_to_swa_index_mapping if has_swa else None
            ),
        )

    def _get_layer_cache_loc(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return cache locations in the correct index space for the given layer."""
        if self._swa_kv_pool is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return self._swa_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
        return forward_batch.out_cache_loc

    def _bind_swa_page_table(
        self, metadata: TRTLLMMHAMetadata, source: dict, key: str, bs: int
    ):
        """Bind a pre-allocated SWA page_table slice to metadata for CUDA graph."""
        buf = source.get(key)
        if buf is not None:
            metadata.swa_page_table = buf[:bs, :]

    def _get_layer_page_table(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Return the correct page_table for the given layer (SWA or full)."""
        swa_pt = self.forward_metadata.swa_page_table
        if swa_pt is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return swa_pt
        return self.forward_metadata.page_table

    @staticmethod
    def _get_scalar_scale(
        layer: RadixAttention,
        float_attr: str,
        scale_attr: str,
    ) -> float:
        scale = getattr(layer, float_attr, None)
        if scale is None:
            scale = getattr(layer, scale_attr, None)
        if scale is None:
            return 1.0
        if isinstance(scale, torch.Tensor):
            logger.warning_once(
                "Ignoring tensor %s for TRT-LLM MHA FP8 KV cache scale. "
                "Expected %s to be populated with a Python scalar.",
                scale_attr,
                float_attr,
            )
            return 1.0
        scale = float(scale)
        return scale if scale > 0.0 else 1.0

    def _get_bmm_scales(
        self, layer: RadixAttention, q_scale: float | torch.Tensor = 1.0
    ) -> tuple[float | torch.Tensor, float]:
        """Return FlashInfer TRT-LLM MHA BMM scales.

        The FP8 paths store Q/K/V as values divided by their per-tensor scales.
        FlashInfer applies bmm1_scale to QK and bmm2_scale to PV, so FP8 reads
        need Q and K descales in BMM1 and V descale in BMM2. Non-FP8 KV cache
        entries are already in model dtype.
        """
        if self.data_type != torch.float8_e4m3fn:
            return layer.scaling, 1.0

        k_scale = self._get_scalar_scale(layer, "k_scale_float", "k_scale")
        v_scale = self._get_scalar_scale(layer, "v_scale_float", "v_scale")
        return q_scale * k_scale * layer.scaling, v_scale

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MHA."""
        max_num_pages = self.max_num_pages
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
        }

        # SWA write-target buffer; bound as a [:num_tokens] view in
        # _build_cuda_graph_metadata and refilled by the fused metadata kernel.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device=self.device)
            if self.use_sliding_window_kv_pool
            else None
        )

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_cuda_graph_metadata["swa_page_table_draft_decode"] = (
                self._alloc_swa_page_table(max_bs, max_num_pages)
            )

            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                # Static uniform preset (Q_MODE_NONE: the fused kernel never
                # rewrites it). Ragged verify overwrites the [:bs+1] slice
                # eagerly on every capture/replay-prep, and the ragged-verify
                # mode is fixed for the whole server run, so the two never mix.
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            }

    def _build_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info,
        device: torch.device,
    ) -> TRTLLMMHAMetadata:
        """Create TRTLLMMHAMetadata with pre-allocated buffer slice refs, stored in the dict."""
        metadata = TRTLLMMHAMetadata()

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode (topk = 1)
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                    : bs + 1
                ]
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][:bs, :]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table_draft_decode",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = torch.arange(
                    0, bs + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.zeros(
                    bs + 1, dtype=torch.int32, device=device
                )
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():
            # Target Verify (topk = 1)
            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][
                : bs + 1
            ]
            metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                : bs + 1
            ]
            metadata.is_ragged_verify = (
                spec_info is not None and spec_info.ragged_verify_layout is not None
            )
            metadata.max_seq_len_q = (
                self.speculative_num_draft_tokens
                if metadata.is_ragged_verify
                else num_tokens // bs
            )
            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.target_verify_metadata,
                "swa_page_table",
                bs,
            )
            self.target_verify_metadata[bs] = metadata
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = num_tokens // bs
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cu_seqlens_q = self.draft_extend_metadata["cu_seqlens_q"][: bs + 1]
            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][: bs + 1]
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.draft_extend_metadata,
                "swa_page_table",
                bs,
            )
            self.draft_extend_metadata[bs] = metadata

        # Bind the SWA write-target buffer slice (refilled by in-graph metadata).
        if self.use_sliding_window_kv_pool:
            metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[:num_tokens]

        return metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        One fused triton kernel (update_trtllm_mha_graph_metadata) rebuilds
        cache_seqlens, cu_seqlens_k/q, the page table(s), and swa_out_cache_loc.
        The previous aten-op implementation issued ~25 host dispatches per graph
        replay, whose per-rank jitter was paid as spin time inside the first
        all-reduce of every replayed graph.

        The page table is rewritten to the static ``max_num_pages`` width (the
        same upper bound ``_fill_page_table_device`` uses); the kernel
        bounds the actual KV reads by the on-device ``cache_seqlens``, so no
        runtime host max / seq_lens_cpu D2H sync is needed.

        Public entry: :py:meth:`init_forward_metadata_in_graph`.
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]

        metadata = None
        seqlen_offset = 0
        cu_seqlens_q = None
        qlens = None
        q_stride = 0
        q_mode = Q_MODE_NONE
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata = self.decode_cuda_graph_metadata[bs]
                seqlen_offset = self.speculative_step_id + 1
            else:
                # Normal Decode
                metadata = self.decode_cuda_graph_metadata[bs]
        elif forward_mode.is_target_verify():
            # Here we only support topk = 1 for now.
            metadata = self.target_verify_metadata[bs]
            if spec_info is not None and spec_info.ragged_verify_layout is not None:
                # Ragged verify: the per-request k-extension is not a
                # uniform scalar seqlen_offset, so the fused kernel cannot
                # rebuild this metadata. It is written eagerly on every
                # capture/replay-prep in init_forward_metadata_out_graph;
                # record nothing here.
                return
            seqlen_offset = metadata.max_seq_len_q
        elif forward_mode.is_draft_extend_v2():
            metadata = self.draft_extend_metadata[bs]
            # Static per-request query width, fixed by the captured graph shape.
            # Do not inspect replay-time tensors here; this body is recorded into
            # the CUDA graph.
            num_tokens_per_req = metadata.max_seq_len_q
            cu_seqlens_q = metadata.cu_seqlens_q
            q_stride = num_tokens_per_req
            q_mode = Q_MODE_STRIDED
        else:
            raise ValueError(
                "TRTLLM-MHA CUDA graph metadata build got an unsupported forward "
                f"mode: {forward_mode}"
            )

        assert metadata is not None
        # Static upper-bound page-table width (see docstring); the kernel
        # bounds real KV reads by cache_seqlens, so this is a fixed loop
        # bound only — never a host max / seq_lens_cpu D2H sync.
        max_seq_pages = self.max_num_pages
        update_trtllm_mha_graph_metadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token=self.req_to_token,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_k=metadata.cu_seqlens_k,
            page_table=metadata.page_table,
            bs=bs,
            seqlen_offset=seqlen_offset,
            max_seq_pages=max_seq_pages,
            page_size=self.page_size,
            swa_mapping=self._swa_full_to_swa_mapping,
            swa_page_table=metadata.swa_page_table,
            out_cache_loc=out_cache_loc,
            swa_out_cache_loc=metadata.swa_out_cache_loc,
            cu_seqlens_q=cu_seqlens_q,
            qlens=qlens,
            q_stride=q_stride,
            q_mode=q_mode,
        )

        self.forward_metadata = metadata

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:
        """Check if we should use the fused FP8 KV cache write path."""
        return save_kv_cache and k is not None and self.data_type == torch.float8_e4m3fn

    def _fused_fp8_set_kv_buffer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Fused FP8 quantization and KV cache write."""
        cache_loc = self._get_layer_cache_loc(layer, forward_batch)

        # Get K/V cache buffers from token_to_kv_pool
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        fused_fp8_set_kv_buffer(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc,
            k_scale=layer.k_scale,  # May be None
            v_scale=layer.v_scale,  # May be None
            page_size=self.page_size,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if (
            forward_mode.is_target_verify()
            and resolve_ragged_verify_layout(forward_batch) is not None
        ):
            self._assert_ragged_verify_supported()

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._build_cuda_graph_metadata(
                bs, num_tokens, forward_mode, spec_info, forward_batch.seq_lens.device
            )

        if forward_mode.is_decode_or_idle():
            self.forward_metadata = self.decode_cuda_graph_metadata[bs]
        elif forward_mode.is_target_verify():
            self.forward_metadata = self.target_verify_metadata[bs]
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            if ragged_layout is not None:
                self._write_ragged_verify_graph_metadata(
                    self.forward_metadata, forward_batch, ragged_layout, bs
                )
        elif forward_mode.is_draft_extend_v2():
            self.forward_metadata = self.draft_extend_metadata[bs]
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def _assert_ragged_verify_supported(self) -> None:
        if self.is_xqa_impl:
            raise NotImplementedError(
                "Compact ragged verify (variable-length cum_seq_lens_q) "
                "requires the trtllm-gen decode kernel; the xqa impl (sm90 / sm120) "
                "rejects it. Disable SGLANG_RAGGED_VERIFY_MODE for this configuration."
            )

    def _write_ragged_verify_graph_metadata(
        self,
        metadata: TRTLLMMHAMetadata,
        forward_batch: ForwardBatch,
        ragged_layout: RaggedVerifyLayout,
        bs: int,
    ) -> None:
        """Eagerly rebuild the target-verify graph metadata for ragged verify.

        The per-request verify lengths make the k-extension non-uniform, which
        the fused in-graph kernel cannot express (scalar ``seqlen_offset``
        only), so this runs out-of-graph on every capture/replay-prep and
        ``_apply_cuda_graph_metadata`` records nothing for ragged batches.
        """
        seq_lens = forward_batch.seq_lens[:bs]
        req_pool_indices = forward_batch.req_pool_indices[:bs]
        padded_layout = ragged_layout.padded_to_bucket(padded_bs=bs)
        geometry = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=padded_layout
        )
        metadata.cache_seqlens_int32.copy_(geometry.cache_seqlens_int32)
        metadata.cu_seqlens_q.copy_(geometry.cu_seqlens_q)
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
        )
        self._fill_page_table_device(
            metadata, req_pool_indices, metadata.cache_seqlens_int32
        )
        # The fused in-graph kernel also skips ragged batches, so refill the
        # SWA write-target buffer here (out_cache_loc -> SWA locs).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=forward_batch.out_cache_loc,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""

        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata.cache_seqlens_int32 = (
                    seqlens_in_batch + (self.speculative_step_id + 1)
                ).to(torch.int32)
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
        elif forward_batch.forward_mode.is_target_verify():
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            if ragged_layout is not None:
                self._assert_ragged_verify_supported()
                geometry = build_ragged_target_verify_geometry(
                    seq_lens=seqlens_in_batch, layout=ragged_layout
                )
                metadata.cache_seqlens_int32 = geometry.cache_seqlens_int32
                # Device-only layouts carry no host lens; the verify window
                # is a valid varlen upper bound.
                metadata.max_seq_len_q = (
                    geometry.max_seq_len_q
                    if geometry.max_seq_len_q is not None
                    else self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = geometry.cu_seqlens_q
                metadata.cu_seqlens_k = geometry.cu_seqlens_k
                metadata.is_ragged_verify = True
            else:
                tokens_per_req = forward_batch.input_ids.shape[0] // batch_size
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + tokens_per_req
                ).to(torch.int32)
                metadata.max_seq_len_q = tokens_per_req
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * tokens_per_req + 1,
                    tokens_per_req,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )

        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            # Query-side max length, sourced from the host-resident extend lengths
            # (sync-free); for plain prefill these equal the full seq lens.
            metadata.max_seq_len_q = int(max(forward_batch.extend_seq_lens_cpu))
            if (
                forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
            ) or forward_batch.forward_mode.is_draft_extend_v2():
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        has_swa = self._swa_kv_pool is not None
        metadata.page_table = torch.empty(
            (batch_size, self.max_num_pages), dtype=torch.int32, device=device
        )
        metadata.swa_page_table = (
            torch.empty(
                (batch_size, self.max_num_pages), dtype=torch.int32, device=device
            )
            if has_swa
            else None
        )
        self._fill_page_table_device(
            metadata, forward_batch.req_pool_indices, metadata.cache_seqlens_int32
        )

        # int64 scatter index (unlike the int32 read page table above).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

        self.forward_metadata = metadata

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
        """Run forward for decode using TRTLLM MHA kernel."""
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        # For XQA, q_dtype should be bf16. For trtllm-gen,
        # q_dtype should be FP8 when KV is in FP8.
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn and not self.is_xqa_impl:
            q = q.to(torch.float8_e4m3fn)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # shape conversion:
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        kv_cache = (k_cache, v_cache)

        bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        page_table = self._get_layer_page_table(layer, forward_batch)

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            out_dtype=self.q_data_type,  # model_runner.dtype
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn and (
            not self.is_xqa_impl or not forward_batch.forward_mode.is_target_verify()
        ):
            q = q.to(torch.float8_e4m3fn)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        kv_cache = (k_cache, v_cache)

        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)
        bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)

        page_table = self._get_layer_page_table(layer, forward_batch)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            if self.forward_metadata.is_ragged_verify:
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=page_table,
                    seq_lens=self.forward_metadata.cache_seqlens_int32,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=None,
                    max_q_len=self.forward_metadata.max_seq_len_q,
                    cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                )
            else:
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=page_table,
                    seq_lens=self.forward_metadata.cache_seqlens_int32,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=self.forward_metadata.max_seq_len_q,
                )
        else:
            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=self.forward_metadata.cu_seqlens_q.shape[0] - 1,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,  # model_runner.dtype
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):
    """Multi-step TRTLLM MHA attention kernel used by EAGLE."""

    # Per-step backends build the page table on-device (sync-free); mirror that so
    # decide_needs_cpu_seq_lens sees a consistent target + draft value.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # TRTLLM-MHA uses encoder_lens from the original fb for inner dispatch
        # (FlashInfer parent forces encoder_lens=None instead).
        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_in_graph(inner_fb)
