"""SGLang's TileLang attention backend for V100 (SM70).

Prefill reads the paged KV cache as
``[num_pages, page_size, num_kv_heads, head_dim]`` (block-major, normally
page_size=16), giving coalesced block reads on V100. Long D256 prefill first
gathers logical pages into a reusable dense workspace, then runs the native
TileLang dense kernel.

Native prefix handling is via ``prefix_kv_lens`` — no ragged+paged+merge_state
double-kernel, no FlattenKV, no FlashInfer wrapper ``plan()`` CPU overhead.

Decode uses an exact grouped TileLang split-KV kernel for Qwen TP4
H6/Hkv1/D256 FP16/E4M3/E5M2. Other shapes retain SGLang's Triton SM70 split-K
path. No external FlashAttention-V100 package is imported or installed.

The KV cache layout is shared with the Triton decode path (which views the 4D
cache as flat 3D — see ``triton_backend._flatten_paged_kv_cache``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# page_size used by this backend (must match server_args SM70 default).
V100_PAGE_SIZE = 16

_paged_forward = None
_paged_forward_loaded = False
_use_tilelang = False


def _should_skip_triton_prefill(model_runner: "ModelRunner") -> bool:
    """Keep baseline decode lean while allocating metadata needed by spec verify."""
    uses_sm70_fp8_kv = model_runner.kv_cache_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    return not (model_runner.spec_algorithm.is_speculative() or uses_sm70_fp8_kv)


def _get_native_paged_attention_params(
    layer: "RadixAttention", default_causal: bool
) -> tuple[bool, int]:
    """Resolve the per-layer mask used by the native paged extend kernel."""
    causal = default_causal and not (
        layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
    )
    sliding_window_size = (
        int(layer.sliding_window_size)
        if causal
        and layer.sliding_window_size is not None
        and layer.sliding_window_size >= 0
        else -1
    )
    return causal, sliding_window_size


def _is_dflash_draft_native_shape_supported(
    layer: "RadixAttention", kv_cache_dtype: torch.dtype = torch.float16
) -> bool:
    """Return whether the TileLang paged kernel supports this draft shape."""
    return (
        layer.head_dim == 128
        and layer.tp_k_head_num > 0
        # The grouped TileLang verifier supports any integral GQA ratio.  The
        # earlier DFlash checkpoints use GQA-4; Qwen3.8 DSpark is trained with
        # 40 Q / 8 KV heads and therefore uses GQA-5.
        and layer.tp_q_head_num % layer.tp_k_head_num == 0
        # The E4M3 DFlash integration is validated only for the TP4 layout.
        # The kernel is numerically sound with four KV heads in isolation, but
        # the complete TP2 verify pipeline can otherwise admit corrupt tokens.
        and (kv_cache_dtype != torch.float8_e4m3fn or layer.tp_k_head_num == 2)
    )


def _load_paged_forward():
    """Lazy-load SGLang's packaged TileLang paged-forward kernel."""
    global _paged_forward, _paged_forward_loaded, _use_tilelang
    if _paged_forward_loaded:
        return _paged_forward
    _paged_forward_loaded = True

    try:
        import tilelang  # noqa: F401

        from sglang.srt.layers.attention.tilelang_fa_v100 import (
            paged_forward as _tl_paged,
        )
        from sglang.srt.utils.common import get_device_sm, is_cuda

        if is_cuda() and get_device_sm() == 70:
            _paged_forward = _tl_paged
            _use_tilelang = True
            logger.info("SM70 paged prefill: using SGLang's TileLang kernel.")
            return _paged_forward
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "The SM70 attention backend requires SGLang's TileLang kernels "
            f"and tilelang. Loading them failed: {e}"
        ) from e
    raise RuntimeError("TileLang SM70 attention is only available on CUDA SM70.")


def _grouped_decode_requested() -> bool:
    """Enable the page-16 G6/D256 grouped TileLang decode specialization."""
    value = (
        os.environ.get(
            "SGLANG_V100_GROUPED_DECODE",
            os.environ.get("SGLANG_V100_LONG_DECODE_XQA", "1"),
        )
        .strip()
        .lower()
    )
    if value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError(
            f"SGLANG_V100_GROUPED_DECODE must be a boolean value, got {value!r}."
        )
    return value in ("1", "true", "on", "yes")


@dataclass
class FlashAttnV100ExtendMetadata:
    """Per-forward metadata for the paged prefill (extend) path."""

    page_table: torch.Tensor  # [num_seqs, max_pages] int32 — page indices
    seq_lens: torch.Tensor  # [num_seqs] int32 — total KV length per seq
    query_start_loc: torch.Tensor  # [num_seqs+1] int32 — cumsum of query lens
    prefix_kv_lens: torch.Tensor  # [num_seqs] int32 — cached prefix length
    causal: bool
    # SWA layers use a separate KV pool with a different physical page
    # numbering. Keep a second table for that pool.
    swa_page_table: Optional[torch.Tensor] = None
    # A linear speculative block can be expressed as independent decode rows
    # with monotonically increasing visible sequence lengths. These persistent
    # buffers are shared by every full-attention layer and are graph-safe.
    smallq_page_table: Optional[torch.Tensor] = None
    smallq_swa_page_table: Optional[torch.Tensor] = None
    smallq_seq_lens: Optional[torch.Tensor] = None
    smallq_active_num_partitions: Optional[torch.Tensor] = None
    smallq_max_seq_len: int = 0


class FlashAttnV100Backend(AttentionBackend):
    """V100 attention backend: TileLang prefill/grouped decode + Triton fallback."""

    # Decode metadata is rebuilt by the triton backend from cuda-graph buffers;
    # this backend never reads seq_lens_cpu / seq_lens_sum for decode.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
    ):
        super().__init__()
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.model_runner = model_runner
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        if self.page_size != V100_PAGE_SIZE:
            logger.warning(
                f"tilelang_fa_v100 expects page_size={V100_PAGE_SIZE}, "
                f"got page_size={self.page_size}. page_size=1 is supported for "
                "Mamba no_buffer scheduling, but page_size=16 gives coalesced "
                "paged-prefill reads."
            )

        self.max_context_len = model_runner.model_config.context_len
        self._max_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.skip_prefill = skip_prefill
        self._uses_sm70_e4m3_kv = (
            model_runner.kv_cache_dtype == torch.float8_e4m3fn
        )
        self._uses_sm70_e5m2_kv = model_runner.kv_cache_dtype == torch.float8_e5m2
        self._uses_sm70_fp8_kv = self._uses_sm70_e4m3_kv or self._uses_sm70_e5m2_kv
        fp8_prefill_scratch = (
            os.environ.get("SGLANG_V100_FP8_PREFILL_SCRATCH", "1").strip().lower()
        )
        if fp8_prefill_scratch not in (
            "0",
            "false",
            "off",
            "no",
            "1",
            "true",
            "on",
            "yes",
        ):
            raise ValueError(
                "SGLANG_V100_FP8_PREFILL_SCRATCH must be a boolean value, "
                f"got {fp8_prefill_scratch!r}."
            )
        self._fp8_prefill_scratch_enabled = (
            self._uses_sm70_fp8_kv
            and fp8_prefill_scratch in ("1", "true", "on", "yes")
            and model_runner.model_config.head_dim
            == model_runner.model_config.v_head_dim
            and (self._uses_sm70_e4m3_kv or self._uses_sm70_e5m2_kv)
        )
        self._grouped_decode_enabled = _grouped_decode_requested()
        if self._grouped_decode_enabled:
            logger.info(
                "SM70 decode: enabled grouped TileLang G6/D256 split-KV "
                "decoder (no external FlashAttention-V100 dependency)."
            )
        # This backend is self-contained: startup validates TileLang and never
        # probes an externally installed attention package.
        _load_paged_forward()
        if (
            model_runner.spec_algorithm.is_dflash_family()
            and not model_runner.is_draft_worker
        ):
            logger.info("DFLASH target verifier: grouped TileLang block verifier.")

        # Decode is delegated to the Triton backend (GooseLLM SM70 split-K
        # tuning already lives there). Speculative target verification also
        # delegates its extend pass, so it needs Triton's qo/mask indptr
        # buffers; ordinary decoding can retain the lean decode-only setup.
        self._triton = TritonAttnBackend(
            model_runner,
            skip_prefill=_should_skip_triton_prefill(model_runner),
        )

        self.forward_metadata: Optional[FlashAttnV100ExtendMetadata] = None
        # Buffers for piecewise cuda-graph capture of extend.
        self._cg_page_table: Optional[torch.Tensor] = None
        self._cg_swa_page_table: Optional[torch.Tensor] = None
        self._cg_seq_lens: Optional[torch.Tensor] = None
        self._cg_query_start_loc: Optional[torch.Tensor] = None
        self._cg_prefix_kv_lens: Optional[torch.Tensor] = None
        self._cg_strided: Optional[torch.Tensor] = None
        self._cg_smallq_page_table: Optional[torch.Tensor] = None
        self._cg_smallq_swa_page_table: Optional[torch.Tensor] = None
        self._cg_smallq_seq_lens: Optional[torch.Tensor] = None
        self._cg_smallq_active_num_partitions: Optional[torch.Tensor] = None
        self._grouped_decode_page_table = torch.empty(
            1,
            self._max_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self._fp8_prefill_k_scratch: Optional[torch.Tensor] = None
        self._fp8_prefill_v_scratch: Optional[torch.Tensor] = None
        self._fp8_prefill_logical_pages: Optional[torch.Tensor] = None
        self._fp8_prefill_page_capacity = 0
        self._fp8_prefill_head_shape: Optional[tuple[int, int]] = None
        self._fp8_prefill_scratch_logged = False

    def _uses_fp8_prefill_scratch(self, forward_mode) -> bool:
        return (
            self._fp8_prefill_scratch_enabled
            and forward_mode.is_extend()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        )

    def _ensure_fp8_prefill_scratch(
        self,
        total_pages: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_shape = (num_kv_heads, head_dim)
        if self._fp8_prefill_head_shape != head_shape:
            self._fp8_prefill_k_scratch = None
            self._fp8_prefill_v_scratch = None
            self._fp8_prefill_logical_pages = None
            self._fp8_prefill_page_capacity = 0
            self._fp8_prefill_head_shape = head_shape

        if self._fp8_prefill_page_capacity < total_pages:
            capacity = 1 << (max(1, total_pages) - 1).bit_length()
            self._fp8_prefill_k_scratch = torch.empty(
                capacity,
                self.page_size,
                num_kv_heads,
                head_dim,
                dtype=torch.float16,
                device=self.device,
            )
            self._fp8_prefill_v_scratch = torch.empty_like(self._fp8_prefill_k_scratch)
            self._fp8_prefill_logical_pages = torch.arange(
                capacity, dtype=torch.int32, device=self.device
            )
            self._fp8_prefill_page_capacity = capacity

        assert self._fp8_prefill_k_scratch is not None
        assert self._fp8_prefill_v_scratch is not None
        assert self._fp8_prefill_logical_pages is not None
        if not self._fp8_prefill_scratch_logged:
            logger.info(
                "SM70 FP8 KV prefill: dequantizing each active prefix page once "
                "into reusable FP16 scratch before native paged attention."
            )
            self._fp8_prefill_scratch_logged = True
        return (
            self._fp8_prefill_k_scratch[:total_pages],
            self._fp8_prefill_v_scratch[:total_pages],
            self._fp8_prefill_logical_pages[:total_pages],
        )

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------
    def _uses_native_linear_verify(self, forward_mode) -> bool:
        if not forward_mode.is_target_verify():
            return False
        if self.model_runner.is_draft_worker:
            # FP8 sliding-window verify still uses Triton: the grouped
            # TileLang byte-decoding verifier currently represents full-prefix
            # masks, while the ordinary sliding kernel accepts FP16 KV.
            draft_tilelang = (
                os.environ.get("SGLANG_V100_DFLASH_DRAFT_TILELANG", "1").strip().lower()
            )
            if draft_tilelang in ("0", "false", "off", "no"):
                return False
            if draft_tilelang not in ("1", "true", "on", "yes"):
                raise ValueError(
                    "SGLANG_V100_DFLASH_DRAFT_TILELANG must be a boolean value, "
                    f"got {draft_tilelang!r}."
                )
            return (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.page_size == V100_PAGE_SIZE
                and not self._uses_sm70_fp8_kv
            )
        native_verify = (
            os.environ.get("SGLANG_V100_NATIVE_LINEAR_VERIFY", "1").strip().lower()
        )
        if native_verify in ("0", "false", "off", "no"):
            return False
        if native_verify not in ("1", "true", "on", "yes"):
            raise ValueError(
                "SGLANG_V100_NATIVE_LINEAR_VERIFY must be a boolean value, "
                f"got {native_verify!r}."
            )
        if self._uses_sm70_e5m2_kv:
            return (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.page_size == V100_PAGE_SIZE
            )
        if self._uses_sm70_e4m3_kv:
            return (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.page_size == V100_PAGE_SIZE
            )
        spec_algorithm = self.model_runner.spec_algorithm
        return spec_algorithm.is_dflash_family() or (
            spec_algorithm.is_eagle()
            and self.model_runner.server_args.speculative_eagle_topk <= 1
        )

    def _smallq_partition_size(self) -> int:
        """Workspace partition envelope for native linear verification.

        Retained for metadata compatibility with the grouped verifier. Its
        actual partition count is selected on device from sequence length.
        """
        return 128

    def _build_extend_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        causal: bool,
        page_table_buf: Optional[torch.Tensor] = None,
        swa_page_table_buf: Optional[torch.Tensor] = None,
        build_smallq: bool = False,
    ) -> FlashAttnV100ExtendMetadata:
        """Gather page indices for each sequence and pack into a page table.

        Token slot indices live in ``req_to_token[req, pos]``; the first token
        of each 16-token page determines the page index (``slot // page_size``).
        SWA cache slots are translated through the pool's full-to-SWA mapping.
        """
        # Synthetic prefill warmup uses a decode-oriented seq_len fill value
        # while still submitting a full extend block. Prefix + extend is the
        # lower bound required by both the page table and scratch allocation.
        seq_lens = torch.maximum(
            seq_lens.to(torch.int32),
            extend_prefix_lens.to(torch.int32) + extend_seq_lens.to(torch.int32),
        )
        num_seqs = seq_lens.shape[0]
        max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size

        def build_page_table(
            page_table_buffer: Optional[torch.Tensor],
            translate_to_swa: bool,
        ) -> torch.Tensor:
            # Keep a fixed-width table so the TileLang kernel's max_blocks key
            # remains stable across requests.
            if page_table_buffer is None:
                table = torch.zeros(
                    num_seqs,
                    self._max_pages,
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                table = page_table_buffer[:num_seqs, : self._max_pages]
                table.zero_()

            if max_pages > 0:
                strided = torch.arange(
                    0, max_seq_len, self.page_size, device=self.device
                )  # [max_pages] token positions (page starts)
                token_indices = self.req_to_token[
                    req_pool_indices[:, None], strided[None, :]
                ]  # [num_seqs, max_pages]
                if translate_to_swa:
                    token_indices = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            token_indices
                        )
                    )
                table[:, :max_pages] = (token_indices // self.page_size).to(torch.int32)
            return table

        page_table = build_page_table(page_table_buf, translate_to_swa=False)
        has_swa_mapping = (
            hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa")
            and getattr(self.token_to_kv_pool, "full_to_swa_index_mapping", None)
            is not None
        )
        swa_page_table = (
            build_page_table(swa_page_table_buf, translate_to_swa=True)
            if has_swa_mapping
            else None
        )

        query_start_loc = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        query_start_loc[1:] = torch.cumsum(extend_seq_lens.to(torch.int32), dim=0)

        smallq_page_table = None
        smallq_swa_page_table = None
        smallq_seq_lens = None
        smallq_active_num_partitions = None
        if build_smallq:
            total_query_tokens = int(query_start_loc[-1].item())
            query_lens_i32 = extend_seq_lens.to(torch.int32)
            smallq_page_table = torch.repeat_interleave(
                page_table,
                query_lens_i32,
                dim=0,
                output_size=total_query_tokens,
            ).contiguous()
            if swa_page_table is not None:
                smallq_swa_page_table = torch.repeat_interleave(
                    swa_page_table,
                    query_lens_i32,
                    dim=0,
                    output_size=total_query_tokens,
                ).contiguous()
            repeated_prefix_lens = torch.repeat_interleave(
                extend_prefix_lens.to(torch.int32),
                query_lens_i32,
                output_size=total_query_tokens,
            )
            repeated_query_starts = torch.repeat_interleave(
                query_start_loc[:-1],
                query_lens_i32,
                output_size=total_query_tokens,
            )
            token_indices = torch.arange(
                total_query_tokens, dtype=torch.int32, device=self.device
            )
            smallq_seq_lens = (
                repeated_prefix_lens + token_indices - repeated_query_starts + 1
            ).contiguous()
            partition_size = self._smallq_partition_size()
            smallq_active_num_partitions = torch.tensor(
                [max(1, (max_seq_len + partition_size - 1) // partition_size)],
                dtype=torch.int32,
                device=self.device,
            )

        return FlashAttnV100ExtendMetadata(
            page_table=page_table,
            seq_lens=seq_lens.to(torch.int32),
            query_start_loc=query_start_loc,
            prefix_kv_lens=extend_prefix_lens.to(torch.int32),
            causal=causal,
            swa_page_table=swa_page_table,
            smallq_page_table=smallq_page_table,
            smallq_swa_page_table=smallq_swa_page_table,
            smallq_seq_lens=smallq_seq_lens,
            smallq_active_num_partitions=smallq_active_num_partitions,
            smallq_max_seq_len=max_seq_len,
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        mode = forward_batch.forward_mode
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(mode)
                and not self._uses_fp8_prefill_scratch(mode)
            )
            or mode.is_decode_or_idle()
            or mode.is_draft_extend(include_v2=True)
            or (mode.is_target_verify() and not self._uses_native_linear_verify(mode))
        ):
            # Decode / spec paths run on the Triton backend.
            self._triton.init_forward_metadata(forward_batch)
            self.forward_metadata = None
            return

        if self._uses_native_linear_verify(mode):
            prefix_lens = forward_batch.seq_lens
            draft_token_num = int(forward_batch.spec_info.draft_token_num)
            extend_seq_lens = torch.full_like(prefix_lens, draft_token_num)
            seq_lens = prefix_lens + draft_token_num
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_seq_lens = forward_batch.extend_seq_lens
            seq_lens = forward_batch.seq_lens

        causal = True
        md = self._build_extend_metadata(
            forward_batch.req_pool_indices,
            seq_lens,
            extend_seq_lens,
            prefix_lens,
            causal=causal,
            build_smallq=self._uses_native_linear_verify(mode),
        )
        self.forward_metadata = md
        if self._uses_sm70_fp8_kv and (
            self._uses_native_linear_verify(mode)
            # Full aligned E5M2 chunks use the native bridge, but the final
            # partial chunk deliberately falls back to Triton. Build both
            # metadata views so a 32K+1 prompt cannot reach Triton with None.
            or self._uses_sm70_e5m2_kv
        ):
            self._triton.init_forward_metadata(forward_batch)

    # ------------------------------------------------------------------
    # CUDA graph state
    # ------------------------------------------------------------------
    def init_cuda_graph_state(
        self, max_bs: int, max_num_tokens: int, kv_indices_buf=None
    ):
        # Decode cuda-graph state lives in the Triton backend.
        self._triton.init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

        self._cg_page_table = torch.zeros(
            max_bs, self._max_pages, dtype=torch.int32, device=self.device
        )
        if (
            hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa")
            and getattr(self.token_to_kv_pool, "full_to_swa_index_mapping", None)
            is not None
        ):
            self._cg_swa_page_table = torch.zeros(
                max_bs, self._max_pages, dtype=torch.int32, device=self.device
            )
        self._cg_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        self._cg_query_start_loc = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._cg_prefix_kv_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._cg_strided = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )
        self._cg_smallq_page_table = torch.zeros(
            max_num_tokens,
            self._max_pages,
            dtype=torch.int32,
            device=self.device,
        )
        if self._cg_swa_page_table is not None:
            self._cg_smallq_swa_page_table = torch.zeros(
                max_num_tokens,
                self._max_pages,
                dtype=torch.int32,
                device=self.device,
            )
        self._cg_smallq_seq_lens = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._cg_smallq_active_num_partitions = torch.ones(
            1, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_mode)
            )
            or forward_mode.is_decode_or_idle()
            or forward_mode.is_draft_extend(include_v2=True)
            or (
                forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_mode)
            )
        ):
            return self._triton.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        if self._uses_native_linear_verify(forward_mode):
            if self._uses_sm70_fp8_kv:
                self._triton.init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )
            self._set_linear_verify_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                int(spec_info.draft_token_num),
            )
            return
        # Extend capture: metadata buffers are filled at replay time.
        self.forward_metadata = FlashAttnV100ExtendMetadata(
            page_table=self._cg_page_table[:bs],
            seq_lens=self._cg_seq_lens[:bs],
            query_start_loc=self._cg_query_start_loc[: bs + 1],
            prefix_kv_lens=self._cg_prefix_kv_lens[:bs],
            causal=True,
            swa_page_table=(
                self._cg_swa_page_table[:bs]
                if self._cg_swa_page_table is not None
                else None
            ),
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_mode)
            )
            or forward_mode.is_decode_or_idle()
            or forward_mode.is_draft_extend(include_v2=True)
            or (
                forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_mode)
            )
        ):
            return self._triton.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )
        if self._uses_native_linear_verify(forward_mode):
            if self._uses_sm70_fp8_kv:
                self._triton.init_forward_metadata_replay_cuda_graph(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                    seq_lens_cpu,
                )
            self._set_linear_verify_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                int(spec_info.draft_token_num),
            )
            return

        # Extend replay: refresh page table + seq metadata from the new batch.
        seq_lens_b = seq_lens[:bs].to(torch.int32)
        max_len = int(seq_lens_b.max().item()) if bs > 0 else 0
        max_pages = (max_len + self.page_size - 1) // self.page_size
        if max_pages > 0:
            token_indices = self.req_to_token[
                req_pool_indices[:bs, None],
                self._cg_strided[:max_pages][None, :],
            ]
            self._cg_page_table[:bs, :max_pages].copy_(
                (token_indices // self.page_size).to(torch.int32)
            )
            if self._cg_swa_page_table is not None:
                swa_token_indices = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(token_indices)
                )
                self._cg_swa_page_table[:bs, :max_pages].copy_(
                    (swa_token_indices // self.page_size).to(torch.int32)
                )
        if self._cg_swa_page_table is not None:
            self._cg_swa_page_table[:bs, max_pages:].zero_()
        self._cg_seq_lens[:bs].copy_(seq_lens_b)
        # query_start_loc / prefix_kv_lens for extend replay are approximated
        # from seq_lens (pure extend, no prefix) — sufficient for the captured
        # graph shape; mixed-prefix extends fall back to eager metadata.
        self._cg_query_start_loc.zero_()
        self._cg_query_start_loc[1 : bs + 1] = torch.cumsum(seq_lens_b, dim=0)
        self._cg_prefix_kv_lens[:bs].zero_()

    def _set_linear_verify_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        draft_token_num: int,
    ):
        """Refresh fixed buffers for a top-k=1 linear causal verify block."""
        prefix_lens_b = prefix_lens[:bs].to(torch.int32)
        seq_lens_b = prefix_lens_b + draft_token_num
        max_len = int(seq_lens_b.max().item()) if bs > 0 else 0
        max_pages = (max_len + self.page_size - 1) // self.page_size
        if max_pages > 0:
            token_indices = self.req_to_token[
                req_pool_indices[:bs, None],
                self._cg_strided[:max_pages][None, :],
            ]
            self._cg_page_table[:bs, :max_pages].copy_(
                (token_indices // self.page_size).to(torch.int32)
            )
            if self._cg_swa_page_table is not None:
                swa_token_indices = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(token_indices)
                )
                self._cg_swa_page_table[:bs, :max_pages].copy_(
                    (swa_token_indices // self.page_size).to(torch.int32)
                )
        if self._cg_swa_page_table is not None:
            self._cg_swa_page_table[:bs, max_pages:].zero_()
        self._cg_seq_lens[:bs].copy_(seq_lens_b)
        self._cg_query_start_loc[: bs + 1].copy_(
            torch.arange(
                0,
                (bs + 1) * draft_token_num,
                step=draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
        )
        self._cg_prefix_kv_lens[:bs].copy_(prefix_lens_b)
        num_smallq_rows = bs * draft_token_num
        self._cg_smallq_page_table[:num_smallq_rows].copy_(
            torch.repeat_interleave(
                self._cg_page_table[:bs],
                draft_token_num,
                dim=0,
                output_size=num_smallq_rows,
            )
        )
        if self._cg_smallq_swa_page_table is not None:
            self._cg_smallq_swa_page_table[:num_smallq_rows].copy_(
                torch.repeat_interleave(
                    self._cg_swa_page_table[:bs],
                    draft_token_num,
                    dim=0,
                    output_size=num_smallq_rows,
                )
            )
        repeated_prefix_lens = torch.repeat_interleave(
            prefix_lens_b,
            draft_token_num,
            output_size=num_smallq_rows,
        )
        smallq_offsets = (
            torch.arange(
                draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
            .add_(1)
            .repeat(bs)
        )
        self._cg_smallq_seq_lens[:num_smallq_rows].copy_(
            repeated_prefix_lens + smallq_offsets
        )
        partition_size = self._smallq_partition_size()
        self._cg_smallq_active_num_partitions.fill_(
            max(1, (max_len + partition_size - 1) // partition_size)
        )
        self.forward_metadata = FlashAttnV100ExtendMetadata(
            page_table=self._cg_page_table[:bs],
            seq_lens=self._cg_seq_lens[:bs],
            query_start_loc=self._cg_query_start_loc[: bs + 1],
            prefix_kv_lens=self._cg_prefix_kv_lens[:bs],
            causal=True,
            swa_page_table=(
                self._cg_swa_page_table[:bs]
                if self._cg_swa_page_table is not None
                else None
            ),
            smallq_page_table=self._cg_smallq_page_table[:num_smallq_rows],
            smallq_swa_page_table=(
                self._cg_smallq_swa_page_table[:num_smallq_rows]
                if self._cg_smallq_swa_page_table is not None
                else None
            ),
            smallq_seq_lens=self._cg_smallq_seq_lens[:num_smallq_rows],
            smallq_active_num_partitions=self._cg_smallq_active_num_partitions,
            smallq_max_seq_len=max_len,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        # Tree verification uses these Triton buffers. Linear verification does
        # not consume the custom mask, but EAGLE's shared tree builder may still
        # fill it while constructing the acceptance metadata.
        return self._triton.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int]
    ):
        return self._triton.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _can_use_grouped_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        sinks,
    ) -> bool:
        if not self._grouped_decode_enabled or q.shape[0] != 1:
            return False
        if (
            self.page_size != V100_PAGE_SIZE
            or layer.tp_k_head_num not in (1, 2, 4)
            or layer.tp_q_head_num != 6 * layer.tp_k_head_num
            or layer.qk_head_dim != 256
            or layer.v_head_dim != 256
            or q.dtype != torch.float16
            or sinks is not None
            or layer.xai_temperature_len > 0
            or (
                layer.sliding_window_size is not None
                and layer.sliding_window_size >= 0
            )
            or getattr(layer, "logit_cap", None) not in (None, 0, 0.0)
        ):
            return False

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        # Accept both the 4D paged [num_pages, page_size, tp_k, 256] layout and
        # the 3D flat [tokens, tp_k, 256] layout (viewed to 4D on entry).
        same_dtype = v_cache.dtype == k_cache.dtype
        fp_dtype = k_cache.dtype in (
            torch.float16, torch.float8_e4m3fn, torch.float8_e5m2
        )
        layout_ok = (
            (k_cache.ndim == 4 and v_cache.ndim == 4
             and k_cache.shape[1:3] == (V100_PAGE_SIZE, layer.tp_k_head_num)
             and k_cache.shape[3] == 256
             and v_cache.shape[1:3] == (V100_PAGE_SIZE, layer.tp_k_head_num)
             and v_cache.shape[3] == 256)
            or (k_cache.ndim == 3 and v_cache.ndim == 3
                and k_cache.shape[1] == layer.tp_k_head_num
                and k_cache.shape[2] == 256
                and v_cache.shape[1] == layer.tp_k_head_num
                and v_cache.shape[2] == 256)
        )
        return (
            fp_dtype and same_dtype and layout_ok
            and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
        )

    def _forward_grouped_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool,
    ) -> torch.Tensor:
        """Run grouped TileLang decode using Triton's KV ordering metadata."""
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        out = torch.empty_like(q)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        # For B=1, Triton's logical token list is also an inexpensive source
        # for the page table: every 16th slot is a logical page start.
        # Eager metadata is runtime-sized; CUDA-graph metadata is capacity-sized,
        # which intentionally fixes the launch grid while seq_lens controls the
        # number of active partitions on device.
        kv_indices = self._triton.forward_metadata.kv_indices
        token_capacity = min(kv_indices.numel(), self.max_context_len)
        page_starts = kv_indices[:token_capacity:V100_PAGE_SIZE]
        page_table = self._grouped_decode_page_table
        torch.div(
            page_starts,
            V100_PAGE_SIZE,
            rounding_mode="floor",
            out=page_table[0, : page_starts.numel()],
        )

        seq_lens = forward_batch.seq_lens[:1]
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if k_cache.ndim == 3:
            k_cache = k_cache.view(
                -1, self.page_size, k_cache.shape[-2], k_cache.shape[-1]
            )
            v_cache = v_cache.view(
                -1, self.page_size, v_cache.shape[-2], v_cache.shape[-1]
            )
        k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
        from sglang.srt.layers.attention.tilelang_fa_v100 import (
            grouped_decode_forward,
        )

        result = grouped_decode_forward(
            q.view(1, layer.tp_q_head_num, 256),
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            softmax_scale=layer.scaling,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out.copy_(result.reshape_as(out))
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        sinks = kwargs.get("sinks")
        if self._can_use_grouped_decode(q, layer, sinks):
            return self._forward_grouped_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
            )
        return self._triton.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        use_fp8_prefill_scratch = self._uses_fp8_prefill_scratch(
            forward_batch.forward_mode
        )
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_batch.forward_mode)
                and not use_fp8_prefill_scratch
            )
            or (
                forward_batch.forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_batch.forward_mode)
            )
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # Tree verification and unsupported FP8 draft masks use Triton's
            # custom-mask path.
            return self._triton.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self.forward_metadata
        num_tokens = q.shape[0]
        # DFlash's fused QKV projection can return Q as a strided view of the
        # wider projection buffer. TileLang specializes tensor strides, so
        # pack the small extend/verify block before entering the kernel.
        q3 = q.reshape(num_tokens, layer.tp_q_head_num, layer.head_dim).contiguous()

        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # paged_fwd requires 4D [num_blocks, block_size, num_kv_heads, head_dim].
        # SGLang's token pool is physically flat for every allocator page size,
        # so expose its contiguous page groups as the 4D layout expected by the
        # TileLang kernel.  Previously this only added a size-1 dimension and
        # made the supported Mamba extra-buffer/page-16 mode fail at runtime.
        if k_cache.ndim == 3:
            k_cache = k_cache.view(
                -1, self.page_size, k_cache.shape[-2], k_cache.shape[-1]
            )
            v_cache = v_cache.view(
                -1, self.page_size, v_cache.shape[-2], v_cache.shape[-1]
            )

        if (
            self.model_runner.is_draft_worker
            and forward_batch.forward_mode.is_target_verify()
            and k_cache.dtype == torch.float8_e4m3fn
            and not _is_dflash_draft_native_shape_supported(layer, k_cache.dtype)
        ):
            raise RuntimeError(
                "V100 DFlash with E4M3 KV is not correctness-validated for "
                "this tensor-parallel draft layout. Use FP16 KV "
                "(--kv-cache-dtype auto) for TP2, or use TP4."
            )

        prefill_page_table = None
        prefill_block_size = self.page_size
        prefill_k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
        prefill_v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
        logical_dense_kv = False
        if (
            use_fp8_prefill_scratch
            and k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and k_cache.shape == v_cache.shape
            and k_cache.shape[2:] == (layer.tp_k_head_num, layer.head_dim)
        ):
            from sglang.srt.layers.attention.tilelang_fa_v100 import (
                gather_fp8_paged_kv,
            )

            batch_size = int(md.prefix_kv_lens.numel())
            # The workspace/page-table shape is fixed for a batch shape, so
            # TileLang compiles once. Only pages covered by md.seq_lens are
            # touched. New extend tokens are read back from the same quantized
            # cache as every later decode token, preserving cache semantics.
            pages_per_sequence = self._max_pages
            total_pages = batch_size * pages_per_sequence
            scratch_k, scratch_v, logical_pages = self._ensure_fp8_prefill_scratch(
                total_pages,
                layer.tp_k_head_num,
                layer.head_dim,
            )
            gather_fp8_paged_kv(
                k_cache,
                v_cache,
                md.page_table,
                md.seq_lens,
                scratch_k,
                scratch_v,
            )
            k_cache = scratch_k
            v_cache = scratch_v
            prefill_page_table = logical_pages.view(batch_size, pages_per_sequence)
            logical_dense_kv = batch_size == 1

        out = torch.empty(
            num_tokens,
            layer.tp_q_head_num,
            layer.head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        paged_forward = _load_paged_forward()
        causal, sliding_window_size = _get_native_paged_attention_params(
            layer, md.causal
        )
        paged_kwargs = dict(
            out=out,
            block_size=prefill_block_size,
            softmax_scale=layer.scaling,
            causal=causal,
            num_kv_heads=layer.tp_k_head_num,
        )
        paged_kwargs.update(
            sliding_window_size=sliding_window_size,
            linear_verify=self._uses_native_linear_verify(forward_batch.forward_mode),
            k_scale=prefill_k_scale,
            v_scale=prefill_v_scale,
            max_seq_len_hint=getattr(md, "smallq_max_seq_len", None),
            logical_dense_kv=logical_dense_kv,
        )

        page_table = md.page_table
        if prefill_page_table is not None:
            page_table = prefill_page_table
        if sliding_window_size >= 0 and md.swa_page_table is not None:
            page_table = md.swa_page_table

        paged_forward(
            q3,
            k_cache,
            v_cache,
            page_table,
            md.seq_lens,
            md.query_start_loc,
            md.prefix_kv_lens,
            **paged_kwargs,
        )
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.head_dim)

    def support_triton(self):
        return True
