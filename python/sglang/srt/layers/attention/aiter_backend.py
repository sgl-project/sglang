from __future__ import annotations

from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec

"""
end to end attention solution with aiter kernels
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.kernels.ops.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    kv_indices_num_token_blocks,
)
from sglang.kernels.ops.kvcache.aiter_unified_attention import (
    scatter_ragged_to_page_table_kernel,
    scatter_req_to_token_to_page_table_kernel,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dcp import update_local_kv_lens_for_dcp
from sglang.srt.layers.dcp.planner import plan_dcp_decode_metadata
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_fp8_pertensor_func,
        flash_attn_varlen_func,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_ps_metadata_info_v1,
        get_ps_metadata_v1,
        mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd,
        mla_reduce_v1,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.ops.triton.attention.unified_attention import unified_attention

    from sglang.kernels.ops.attention.unified_attention_3d_mtp import (
        asm_verify_attn_enabled,
        reset_verify_attn_plan_cache,
        unified_attention_3d_mtp_decode_func,
        unified_attention_3d_mtp_func,
        unified_attention_3d_mtp_ragged_func,
    )
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.kernels.ops.attention.dcp_kernels import (
    compact_dcp_verify_token_table_to_ragged,
    create_mla_kv_page_table_for_dcp,
)
from sglang.kernels.ops.attention.merge_state import merge_state_triton
from sglang.kernels.ops.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_mla_gluon import (
    log_mla_gluon_capability,
    mla_gluon_decode,
    prefer_mla_gluon_decode,
)
from sglang.srt.layers.attention.aiter_utils import (
    forward_decode_vectorized_5d,
    forward_extend_vectorized_5d,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# Max qlen limitation of the asm ps kernels to handle q with bf16 dtype
_MLA_ASM_PS_MAX_QLEN = 4

# (query head count -> query lengths) at which aiter's scheduler treats the
# count as native, on gfx950 with an fp8 query and KV. None means any length.
# This map is copied from natively_supported in aiter/ops/attention.py. See
# https://github.com/ROCm/aiter/blob/bf37db00749722ae29ac27c101f534da43c586f5/aiter/ops/attention.py#L1669-L1706
_MLA_ASM_CPRR_NATIVE_QLENS = {
    16: None,
    32: frozenset({4}),
    96: frozenset(range(1, 6 + 1)),
    128: None,
}


def _asm_cprr_supports_shape(heads: int, q_len: int) -> bool:
    """Whether aiter schedules this head count natively at this query length."""
    if heads not in _MLA_ASM_CPRR_NATIVE_QLENS:
        return False
    allowed = _MLA_ASM_CPRR_NATIVE_QLENS[heads]
    return allowed is None or q_len in allowed


def _asm_cprr_kernel_heads(gathered_heads: int, q_len: int) -> int:
    """The smallest supported head count at or above the gathered one, or 0."""
    for heads in sorted(_MLA_ASM_CPRR_NATIVE_QLENS):
        if heads >= gathered_heads and _asm_cprr_supports_shape(heads, q_len):
            return heads
    return 0


# (v_head_dim -> query head counts) that aiter's mla_reduce_v1 has an
# instantiation. This map is copied from MLA_REDUCE_ROUTER in
# aiter/csrc/kernels/mla/reduce.cu. See https://github.com/ROCm/aiter/blob/7915f53a4225b3f9cb632a97a23369dbebcf1be0/csrc/kernels/mla/reduce.cu#L923
_MLA_REDUCE_V1_HEADS = {
    64: frozenset({64}),
    128: frozenset({1, 2, 4, 8, 10, 16, 32, 40, 64, 128}),
    512: frozenset({8, 16, 32, 48, 64, 80, 96, 112, 128}),
}


# Persist
# fast_mode=True if _use_mla_ps_kernel else False
# intra_batch_mode=False if _use_mla_ps_kernel else True

# fake non-ps, intra_batch_mode needs to be True for non-ps-mode
fast_mode = False
intra_batch_mode = True if _use_mla_ps_kernel else False


# Token-block parallel KV-index building is enabled only where it pays:
# the speculative-decoding paths (target_verify / draft_extend / draft
# decode) of long-context servers. Everything else keeps the historical
# one-program-per-request launch.
_KV_INDEX_BLOCKS_MIN_CONTEXT = 32768


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class MlaPrefillPsMetadata:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    work_metadata: torch.Tensor
    work_indptr: torch.Tensor
    work_info_set: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    max_q_len: int
    is_causal: bool
    need_lse: bool
    num_partial_tiles: int


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    run_graph: Optional[bool] = True
    custom_mask: Optional[torch.Tensor] = None
    mask_indptr: Optional[torch.Tensor] = None
    max_extend_len: Optional[int] = None
    swa_page_table: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    local_kv_lens: Optional[torch.Tensor] = None
    verify_token_table: Optional[torch.Tensor] = None
    # ASM context-chunk prefill: KV slots to gather and cu_seqlens_k, computed
    # once per batch by AiterAttnBackend._asm_context_prefill_indices.
    asm_ctx_ready: bool = False
    asm_ctx_tok_idx: Optional[torch.Tensor] = None
    asm_ctx_cu_k: Optional[torch.Tensor] = None
    # (page_indptr, page_ids, last_page_len); None means use the token-level table
    paged_kv_view: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    prefill_ps_metadata: Optional[MlaPrefillPsMetadata] = None
    chunked_skip_prefix_ps_metadata: Optional[MlaPrefillPsMetadata] = None
    chunked_prefix_ps_metadatas: Optional[list[Optional[MlaPrefillPsMetadata]]] = None
    # cprr
    global_kv_indptr: Optional[torch.Tensor] = None
    use_asm_cprr_verify: bool = False
    cprr_kernel_heads: int = 0
    cprr_kv_indices: Optional[torch.Tensor] = None
    cprr_kv_indptr: Optional[torch.Tensor] = None
    cprr_empty_rows: Optional[torch.Tensor] = None


def _build_paged_kv_view(
    kv_indices: torch.Tensor, seq_lens_cpu: torch.Tensor, page_size: int
):
    """Token-level flashinfer KV table -> the page-level view aiter's asm
    paged-varlen prefill expects.

    The paged allocator packs each page contiguously, so a token slot's page id
    is slot // page_size. The arithmetic runs on seq_lens_cpu so it costs no
    device sync; only the gather touches the GPU.
    """
    device = kv_indices.device
    seq_lens = seq_lens_cpu.to(torch.int64)
    pages = (seq_lens + page_size - 1) // page_size

    page_indptr = torch.zeros(pages.numel() + 1, dtype=torch.int32)
    page_indptr[1:] = torch.cumsum(pages, dim=0)
    tok_base = torch.zeros(pages.numel() + 1, dtype=torch.int64)
    tok_base[1:] = torch.cumsum(seq_lens, dim=0)

    within = torch.arange(
        int(page_indptr[-1]), dtype=torch.int64
    ) - torch.repeat_interleave(page_indptr[:-1].to(torch.int64), pages)
    gather = torch.repeat_interleave(tok_base[:-1], pages) + within * page_size
    page_ids = (kv_indices[gather.to(device)] // page_size).to(torch.int32)
    last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)
    return page_indptr.to(device), page_ids, last_page_len.to(device)


def _paged_prefill_asm_supports_gqa(num_q_heads: int, num_kv_heads: int) -> bool:
    """aiter's asm paged-varlen guard takes only a power-of-two GQA ratio."""
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return False
    gqa = num_q_heads // num_kv_heads
    return gqa & (gqa - 1) == 0


_AITER_PARTITION_SIZE_ROCM = 256

# Query rows one asm-prefill work tile covers. The scheduler, the partial
# buffers and mla_reduce_v1 must all agree on it.
_PREFILL_TILE_Q = 256


_DCP_VERIFY_TABLE_COLS_PER_BLOCK = 128


# AITER's gfx950 FP8 FMHA ASM kernels only cover these GQA ratios. Other
# ratios (e.g. Qwen3.8-27B 24Q/4KV = 6) must not take the pertensor shortcut.
_AITER_FP8_ASM_GQA_RATIOS = frozenset({1, 2, 4, 8, 16})


def _aiter_fp8_asm_supports_gqa(num_q_heads: int, num_kv_heads: int) -> bool:
    """Whether AITER's FP8 FMHA ASM kernel supports this GQA ratio."""
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return False
    return (num_q_heads // num_kv_heads) in _AITER_FP8_ASM_GQA_RATIOS


# Cross-check the per-batch fast indices against the generic gather (syncs).
_GFX_ASM_CTX_GATHER_CHECK = (
    os.environ.get("SGLANG_GFX_ASM_CTX_GATHER_CHECK", "0") == "1"
)


def _asm_context_prefill_gather_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_slots: int,
    forward_mode=None,
):
    """KV-pool slots to gather for the ASM context-chunk prefill.

    kv_indptr/kv_indices are token-level for every page_size:
    AiterIndicesUpdaterPrefill sets kv_indptr = cumsum(seq_lens) and writes one
    kv_indices entry per token, so token t of sequence i lives in pool slot
    kv_indices[kv_indptr[i] + t]. There is no page arithmetic to apply here.

    Returns (tok_idx, cu_seqlens_k), or None if the metadata disagrees with
    seq_lens (mixed/spec batches) and the caller must use the paged kernel.
    """
    bs = kv_indptr.numel() - 1
    device = kv_indices.device
    kv_indptr = kv_indptr.to(torch.long)
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    # kvlen must not exceed the tokens this batch actually has in kv_indices,
    # otherwise the gather runs off the end of the table.
    seq_lens = torch.minimum(seq_lens, kv_indptr[1:] - kv_indptr[:bs])
    total_k = int(seq_lens.sum().item())
    cu_k = torch.zeros(bs + 1, dtype=torch.long, device=device)
    torch.cumsum(seq_lens, 0, out=cu_k[1:])
    seq_ids = torch.repeat_interleave(torch.arange(bs, device=device), seq_lens)
    pos_in_seq = torch.arange(total_k, device=device) - cu_k[seq_ids]
    kv_slot = kv_indptr[seq_ids] + pos_in_seq
    if total_k and int(kv_slot.max().item()) >= kv_indices.numel():
        logger.warning(
            "[asm-context-prefill] metadata mismatch, falling back:"
            " mode=%s bs=%s kv_slot_max=%s kv_indices=%s seq_lens=%s kv_indptr=%s",
            forward_mode,
            bs,
            int(kv_slot.max().item()),
            kv_indices.numel(),
            seq_lens.tolist(),
            kv_indptr.tolist(),
        )
        return None
    tok_idx = kv_indices[kv_slot].to(torch.long)
    if total_k and int(tok_idx.max().item()) >= num_kv_slots:
        logger.warning(
            "[asm-context-prefill] gather index out of pool, falling back:"
            " mode=%s bs=%s tok_idx_max=%s num_kv_slots=%s",
            forward_mode,
            bs,
            int(tok_idx.max().item()),
            num_kv_slots,
        )
        return None
    return tok_idx, cu_k


# mha_batch_prefill over-reads 128 page ids past the last token (see
# AiterIndicesUpdaterPrefill); 256 keeps that tile inside the buffer.
_MHA_PREFILL_KV_INDEX_PAD = 256


def _pad_mha_prefill_kv_indices(kv_indices: torch.Tensor) -> torch.Tensor:
    """Repeat a live page id into the tail the CK prefill kernel over-reads."""
    if kv_indices.numel() == 0:
        return kv_indices
    pad = kv_indices[:1].expand(_MHA_PREFILL_KV_INDEX_PAD)
    return torch.cat([kv_indices, pad])


class AiterAttnBackend(AttentionBackend):
    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    # declare here to avoid CI failure in test_aiter_fp8_q_unified_attention.py
    use_mla_auto_kv_splits: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = get_schedule().page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.topk = topk
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_index_translator = model_runner.kv_index_translator

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        self.dcp_world_size = get_parallel().attn_dcp_size
        self.mla_decode_backend = envs.SGLANG_AITER_MLA_DECODE_BACKEND.get().lower()
        if self.mla_decode_backend not in ("gluon", "asm"):
            raise ValueError(
                "SGLANG_AITER_MLA_DECODE_BACKEND must be 'gluon' or 'asm', "
                f"got {self.mla_decode_backend!r}"
            )
        self.use_mla_dcp_asm = (
            self.use_mla
            and self.dcp_world_size > 1
            and self.mla_decode_backend == "asm"
        )
        self.mla_verify_backend = envs.SGLANG_AITER_MLA_VERIFY_BACKEND.get().lower()
        if self.mla_verify_backend not in ("gluon", "asm"):
            raise ValueError(
                "SGLANG_AITER_MLA_VERIFY_BACKEND must be 'gluon' or 'asm', "
                f"got {self.mla_verify_backend!r}"
            )
        self._cprr_launch_logged = False
        # fast_mode / intra_batch_mode for the cp verify schedule. Not the
        # module defaults: those size reduce_partial_map at 65536 entries
        # instead of 1276, and mla_decode_fwd then asks for a 64 GiB logits
        # buffer.
        self.cprr_fast_mode = True
        self.cprr_intra_batch_mode = False
        gathered_heads = self.num_head * self.dcp_world_size
        self.cprr_kernel_heads = _asm_cprr_kernel_heads(
            gathered_heads, self.num_draft_tokens or 1
        )

        # Get v_head_dim based on model type
        if self.use_mla:
            # For MLA models, get v_head_dim from model config
            self.v_head_dim = model_runner.model_config.v_head_dim
        elif hasattr(model_runner.token_to_kv_pool, "get_v_head_dim"):
            # For hybrid models (Mamba+attention, GDN, Kimi linear),
            # layer_id=0 may not be a full attention layer
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]

        # The asm fp8 prefill reduces through mla_reduce_v1, which only has
        # instantiations for the head shapes in _MLA_REDUCE_V1_HEADS; anything
        # else is tiled up to one it does carry, or falls back to
        # flash_attn_varlen_func when the table has nothing to reach.
        self.fp8_prefill_num_head = (
            self.check_fp8_prefill_num_head(
                num_head=self.num_head,
                num_kv_head=self.num_kv_head,
                v_head_dim=self.v_head_dim,
            )
            if self.use_mla
            else None
        )
        self.use_fp8_prefill_attn = (
            _use_fp8_prefill_attn and self.fp8_prefill_num_head is not None
        )
        # Padding is only offered at GQA ratio 1, so the kv side takes the same
        # delta and the ratio the PS metadata is built for stays put.
        self.fp8_prefill_num_kv_head = self.num_kv_head + (
            (self.fp8_prefill_num_head or self.num_head) - self.num_head
        )
        if self.use_fp8_prefill_attn and self.fp8_prefill_num_head != self.num_head:
            logger.info(
                f"aiter asm fp8 MLA prefill pads {self.num_head} query heads to "
                f"{self.fp8_prefill_num_head}; mla_reduce_v1 has no "
                f"{self.num_head}-head instantiation at head_dim {self.v_head_dim}."
            )

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        # qo_indptr for the unified-attn decode path (q_len == 1 per request)
        # is always arange(0, bs+1); precompute once to avoid a per-step cumsum.
        self.qo_indptr_unified_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
        )
        self._kv_indices_scratch: Optional[torch.Tensor] = None
        self._arange_buf: Optional[torch.Tensor] = None

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = model_runner.kv_index_translator

        # sliding window attention. Resolve the SWA pool rather than reading it
        # straight off the active pool: a frozen-KV MTP draft worker's active
        # pool is its own draft pool, but its draft path reads target KV, so the
        # SWA mapping must still come from the target allocator. Mirrors
        # TRTLLMHAAttnBackend._resolve_swa_kv_pool.
        self.swa_kv_pool = self._resolve_swa_kv_pool(model_runner)
        self.use_sliding_window_kv_pool = (
            self.swa_kv_pool is not None and self.swa_kv_pool.swa_layer_nums > 0
        )

        # Detect SHUFFLE 5D ("vectorized") KV cache layout. When active
        # we (a) skip the launch_reshape_and_cache_flash shortcut and always go
        # through `set_kv_buffer` (which dispatches to the 5D Triton writer),
        # and (b) route the decode attention through pa_decode_gluon (see the
        # corresponding branch in forward_decode), since unified_attention's
        # 4D `.view(-1, page, H, D)` cannot be applied to a 5D pool.
        def _pool_is_vec5d(pool):
            if isinstance(pool, SWAKVPool):
                return getattr(pool.full_kv_pool, "kv_cache_layout", "nhd") == (
                    "vectorized_5d"
                )
            return getattr(pool, "kv_cache_layout", "nhd") == "vectorized_5d"

        self.kv_cache_is_vectorized_5d = _pool_is_vec5d(model_runner.token_to_kv_pool)

        if self.use_sliding_window_kv_pool:
            self.use_triton_unified_attention = True
        else:
            self.use_triton_unified_attention = get_bool_env_var(
                "SGLANG_USE_AITER_UNIFIED_ATTN"
            )

        # When topk == 1 the EAGLE draft chain is linear, so target_verify's
        # mask reduces to pure causal and can go through unified_attention
        # instead of the legacy triton extend_attention_fwd. Gated on non-MLA
        # (MLA has its own verify path) and env var for opt-out.
        self._use_unified_verify = (
            self.use_triton_unified_attention
            and not self.use_mla
            and self.topk == 1
            and get_bool_env_var("SGLANG_AITER_UNIFIED_VERIFY", "1")
        )
        # Draft-extend only (prefill/target verify keep their kernels). Read the
        # eagle topk from the spec config: the registry builds this backend with the
        # default topk=1, so self.topk is 1 even when the server runs topk > 1.
        # topk > 1 stays on the CK path.
        draft_extend_topk = get_spec().speculative_eagle_topk
        self._use_unified_draft_extend = (
            not self.use_mla
            and envs.SGLANG_AITER_UNIFIED_DRAFT_EXTEND.get()
            and (draft_extend_topk is None or int(draft_extend_topk) <= 1)
        )
        if self._use_unified_draft_extend:
            logger.info("Aiter draft extend uses unified_attention (topk<=1).")

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not (self.use_mla or self.use_triton_unified_attention):
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            _mla_low_head_repeat = (4, 8)
            _mla_low_head_zero_pad = (12,)
            _valid_heads = (
                self.num_head in _mla_low_head_repeat
                or self.num_head in _mla_low_head_zero_pad
                or (self.num_head % 16 == 0 and 16 <= self.num_head <= 128)
            )
            may_run_mla_decode = self.may_run_mla_decode_kernel(
                decode_attention_backend=model_runner.decode_attention_backend_str,
                speculative_algorithm=get_spec().speculative_algorithm,
                speculative_attention_mode=get_spec().speculative_attention_mode,
            )
            # _mla_decode_fwd_with_head_pad brings any count below 16 up to it,
            # by repetition when it divides 16 and by tiling otherwise.
            _pad_heads_to_16 = self.num_head < 16
            assert (
                self.dcp_world_size > 1
                or _valid_heads
                or _pad_heads_to_16
                or not may_run_mla_decode
            ), (
                f"Aiter MLA supports num_head of 4, 8, 12, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value, or run decode on "
                "another backend (--decode-attention-backend)."
            )

            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            if self.num_head in _mla_low_head_repeat:
                self.head_pad_mode = "repeat"
                self.head_repeat_factor = 16 // self.num_head
            elif self.num_head in _mla_low_head_zero_pad:
                self.head_pad_mode = "zero"
                self.head_repeat_factor = 1
            else:
                self.head_pad_mode = "none"
                self.head_repeat_factor = 1

            _gathered_num_head = self.num_head * self.dcp_world_size
            self.mla_kernel_num_head_padded = (
                16 if _gathered_num_head < 16 else _gathered_num_head
            )

            self.attn_dp_enabled = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # fake-nps (fast_mode False, intra_batch_mode True) picks aiter's
            # v1_0 scheduler, which folds these counts to 16 while the gfx950
            # fp8 launch does not. Both sides must fold alike or the kernel
            # faults. At 16 neither folds, so fake-nps is correct there.
            if self.mla_kernel_num_head_padded in (32, 64, 128):
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or tp8 with --attn-dp-size 8)
            # Native 16-head persist is slow on TP8; keep disabled unless zero-pad
            # (e.g. Kimi K3 h12 -> qh16) where persist ASM is the fast path.
            if (
                (self.mla_kernel_num_head_padded == 16 and self.head_pad_mode != "zero")
                or self.mla_kernel_num_head_padded == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False
            if self.head_pad_mode == "zero" and self.kv_cache_dtype == fp8_dtype:
                # Disable ps only when gluon kernel is selected to avoid falling
                # back to incorrect aiter kernel
                if (
                    prefer_mla_gluon_decode(
                        head_pad_mode=self.head_pad_mode,
                        num_head=self.num_head,
                        kv_cache_dtype=self.kv_cache_dtype,
                    )
                    and not self._asm_ps_supports_decode_and_verify()
                ):
                    _use_mla_ps_kernel = False
                    fast_mode = False
                    intra_batch_mode = False
                log_mla_gluon_capability(logger)

            use_any_mla_persist = _use_mla_ps_kernel or self.use_mla_dcp_asm
            self.max_split_per_batch = 32 if use_any_mla_persist else None

            if self.num_draft_tokens is None and use_any_mla_persist:
                self.max_split_per_batch = 64

            # When the env var is on, aiter plans the KV splits. The default
            # scheduler gives every request ceil(num_cu / batch_size) splits.
            # It ignores the KV length. That clamp is inside the kernel, so no
            # host value lifts it. The v1_2 scheduler sizes the count from the
            # workload. A max_split_per_batch of -1 selects its auto branch.
            #
            # Under DCP the cp verify route already runs the v1_2 scheduler:
            # make_mla_decode_meta_data_buffer takes cprr_fast_mode and
            # cprr_intra_batch_mode instead. There only the -1 has an effect.
            self.use_mla_auto_kv_splits = (
                use_any_mla_persist and envs.SGLANG_AITER_MLA_AUTO_KV_SPLITS.get()
            )
            if self.use_mla_auto_kv_splits:
                logger.info(
                    "aiter MLA: aiter plans the KV splits "
                    "(SGLANG_AITER_MLA_AUTO_KV_SPLITS=1)"
                )
                fast_mode = True
                intra_batch_mode = False
                self.max_split_per_batch = -1

            self.fix_max_split_per_batch = self.max_split_per_batch

    def pad_heads(self, x: torch.Tensor, padded: int) -> torch.Tensor:
        num_head = x.shape[1]
        reps = -(-padded // num_head)  # ceil(padded / num_head)
        return x.repeat(1, reps, 1)[:, :padded, :].contiguous()

    def check_fp8_prefill_num_head(
        self, *, num_head: int, num_kv_head: int, v_head_dim: int
    ) -> Optional[int]:
        """Check _MLA_REDUCE_V1_HEADS to get head count to run the asm fp8
        mla prefill, return None if it is invalid so we will fall it back to
        aiter fa implementation.
        """
        supported = _MLA_REDUCE_V1_HEADS.get(v_head_dim, frozenset())
        if num_head in supported:
            return num_head
        if num_head != num_kv_head:
            return None
        larger = [h for h in supported if h > num_head]
        return min(larger) if larger else None

    def may_run_mla_decode_kernel(
        self,
        *,
        decode_attention_backend: Optional[str],
        speculative_algorithm: Optional[str],
        speculative_attention_mode: str,
    ) -> bool:
        """Decode whether aiter backend will invoke mla_decode_fwd"""
        if decode_attention_backend == "aiter":
            return True
        return (
            speculative_algorithm is not None
            and speculative_attention_mode == "prefill"
        )

    def _get_aiter_paged_ragged_kv_cache_dtype(self) -> str:
        """``kv_cache_dtype`` string for ``paged_attention_ragged`` (aiter ``pa/pa_ragged.py``).

        **Behavior change:** we no longer upcast FP8 KV to the activations dtype for this decode path.
        Paged K/V stay in native FP8 storage; we pass ``\"fp8_e4m3\"`` so the kernel dequants on read
        (``k_scale`` / ``v_scale``) instead of widening the cache to bf16/fp16 for ``\"auto\"``.

        **Context (short):** aiter accepts ``auto`` / ``fp8`` / ``fp8_e4m3`` only (not ``fp8_e5m2``).
        On HIP, ``configure_kv_cache_dtype`` maps CLI ``fp8_e5m2`` and ``fp8_e4m3`` to ``fp8_dtype``;
        return ``\"fp8_e4m3\"`` when ``self.kv_cache_dtype == fp8_dtype``, else ``\"auto\"``.
        """
        if self.kv_cache_dtype != fp8_dtype:
            return "auto"
        return "fp8_e4m3"

    def _use_mla_decode_persist_metadata(self) -> bool:
        """Persist MLA metadata for non-DCP PS decode, or DCP ASM decode.

        Matches main's ``_use_mla_ps_kernel and dcp_world_size <= 1`` when ASM
        is off. DCP ASM needs the same buffers with fast_mode scheduling.
        """
        return (_use_mla_ps_kernel and self.dcp_world_size <= 1) or (
            self.use_mla_dcp_asm and self.dcp_world_size > 1
        )

    def _mla_decode_metadata_modes(self) -> tuple[bool, bool]:
        if self.use_mla_dcp_asm and self.dcp_world_size > 1:
            # Match the AITER/vLLM DCP path: persistent scheduling with the
            # gathered-head tensor folded internally to qh16.
            return True, False
        return fast_mode, intra_batch_mode

    def make_mla_decode_meta_data_buffer(
        self,
        max_seqlen_qo,
        batch_size,
        *,
        metadata_fast_mode: Optional[bool] = None,
        metadata_intra_batch_mode: Optional[bool] = None,
        nhead_override: Optional[int] = None,
        max_split_per_batch: Optional[int] = None,
    ):
        """Allocate the PS work-schedule buffers, sized for nhead_override."""
        # Under DCP this is the gathered head count (num_head * dcp_world_size);
        # equals num_head_padded when DCP is off.
        nhead = (
            self.mla_kernel_num_head_padded
            if nhead_override is None
            else nhead_override
        )
        dtype = self.kv_cache_dtype

        # The auto count is -1, which this min() would turn into a real cap.
        if self.attn_dp_enabled and not self.use_mla_auto_kv_splits:
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        metadata_fast_mode = (
            fast_mode if metadata_fast_mode is None else metadata_fast_mode
        )
        metadata_intra_batch_mode = (
            intra_batch_mode
            if metadata_intra_batch_mode is None
            else metadata_intra_batch_mode
        )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            nhead,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=metadata_fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=metadata_intra_batch_mode,
            **(
                {}
                if max_split_per_batch is None
                else {"max_split_per_batch": max_split_per_batch}
            ),
        )

        # aiter implementation
        # the tensor's meaning please refer aiter/ops/attention.py
        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
        is_cp_round_robin=False,
        nhead_override=None,
    ):
        nhead_kv = 1
        page_size = self.page_size
        dtype = self.kv_cache_dtype
        nhead = (
            self.mla_kernel_num_head_padded
            if nhead_override is None
            else nhead_override
        )

        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            nhead // nhead_kv,
            nhead_kv,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(page_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            is_cp_round_robin=is_cp_round_robin,
            dtype_q=dtype,
            dtype_kv=dtype,
        )

    @property
    def _prefill_qlen_granularity(self) -> int:
        """Query rows per scheduling unit: one tile's worth, divided across the
        query heads the kernel processes together."""
        return _PREFILL_TILE_Q // (
            self.fp8_prefill_num_head // self.fp8_prefill_num_kv_head
        )

    def make_mla_prefill_ps_meta_data_buffer(
        self, batch_size: int, max_qlen: int, qlen_granularity: int
    ):
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_ps_metadata_info_v1(
            batch_size=batch_size,
            num_head_k=self.fp8_prefill_num_kv_head,
            max_qlen=max_qlen,
            qlen_granularity=qlen_granularity,
        )

        device = self.device
        work_metadata_ptrs = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

        return (
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_prefill_ps_meta_data(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        work_metadata: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
        is_causal: bool = True,
        need_lse: bool = False,
    ):
        gqa_ratio = self.fp8_prefill_num_head // self.fp8_prefill_num_kv_head
        num_heads_k = self.fp8_prefill_num_kv_head
        qhead_granularity = gqa_ratio
        qlen_granularity = self._prefill_qlen_granularity
        kvlen_granularity = 128
        block_size = 1

        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = kv_indptr.to("cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_heads_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
            need_lse=need_lse,
        )

    # for page size > 1 useful conversion function
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def _build_unified_page_table_from_spec(
        self,
        spec_info,
        bs: int,
        dest_buf: Optional[torch.Tensor] = None,
        swa_dest_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert ragged (token-level) kv_indices from spec_info into a 2D
        block-level page_table of shape (bs, max_num_blocks_per_seq).
        unified_attention expects max_seqlen_k = page_table.shape[1] *
        page_size to be a captured constant, so rows are sized to the
        backend-level max_num_blocks_per_seq regardless of seqused_k.
        """
        kv_indptr = spec_info.kv_indptr
        kv_flat = spec_info.kv_indices
        page_size = self.page_size
        max_blocks = (self.max_context_len + page_size - 1) // page_size

        swa_slot_mapping = None
        swa_page_table = None

        if dest_buf is not None:
            # The scatter kernel fills [0, num_blocks) and loads past that use
            # other=0, so the tail is 0-filled. Under graph replay rows > bs
            # are stale but unified_attention only walks rows [0, bs).
            page_table = dest_buf
        else:
            page_table = torch.zeros(
                bs, max_blocks, dtype=torch.int32, device=self.device
            )

        if self.use_sliding_window_kv_pool:
            swa_slot_mapping = self.swa_kv_pool.full_to_swa_index_mapping.long()

            if swa_dest_buf is not None:
                swa_page_table = swa_dest_buf
            else:
                swa_page_table = torch.zeros(
                    bs, max_blocks, dtype=torch.int32, device=self.device
                )

        BLOCK_SIZE = 1024
        grid = (bs, triton.cdiv(max(max_blocks, 1), BLOCK_SIZE))
        scatter_ragged_to_page_table_kernel[grid](
            kv_flat,
            kv_indptr,
            page_table,
            page_table.stride(0),
            swa_page_table,
            swa_slot_mapping,
            PAGE_SIZE=page_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SWA=(swa_slot_mapping is not None),
        )

        return page_table, swa_page_table

    def _build_verify_unified_metadata(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        draft_num: int,
        page_table_dest: Optional[torch.Tensor] = None,
        swa_page_table_dest: Optional[torch.Tensor] = None,
    ):
        """Build the 2D block page_table + qo_indptr for EAGLE target_verify
        through unified_attention. Assumes the new draft K/V have already been
        written by set_kv_buffer, so req_to_token[rp, :seq_lens[i]+draft_num]
        covers both the prefix and the freshly committed draft tokens. Returns
        (page_table, qo_indptr, max_q_len=draft_num).
        """
        device = seq_lens.device
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * draft_num,
            step=draft_num,
            dtype=torch.int32,
            device=device,
        )

        page_size = self.page_size
        max_blocks = (self.max_context_len + page_size - 1) // page_size

        swa_slot_mapping = None
        swa_page_table = None

        if page_table_dest is not None:
            page_table = page_table_dest
        else:
            page_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)

        if self.use_sliding_window_kv_pool:
            swa_slot_mapping = self.swa_kv_pool.full_to_swa_index_mapping.long()

            if swa_page_table_dest is not None:
                swa_page_table = swa_page_table_dest
            else:
                swa_page_table = torch.zeros(
                    bs, max_blocks, dtype=torch.int32, device=device
                )

        BLOCK_SIZE = 1024
        grid = (bs, triton.cdiv(max(max_blocks, 1), BLOCK_SIZE))
        scatter_req_to_token_to_page_table_kernel[grid](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_table,
            self.req_to_token.stride(0),
            page_table.stride(0),
            swa_page_table,
            swa_slot_mapping,
            DRAFT_NUM=draft_num,
            PAGE_SIZE=page_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SWA=(swa_slot_mapping is not None),
        )

        return page_table, qo_indptr, draft_num, swa_page_table

    def _build_extend_unified_page_table(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        max_kv_len: int,
    ):
        """Build the 2D block page_table (+ SWA translation) that
        unified_attention needs for a plain extend/prefill batch. Mirrors the
        target_verify builder with draft_num=0; rows are sized to the batch's own
        longest sequence since extend is never graph-captured."""
        device = seq_lens.device
        page_size = self.page_size
        max_blocks = max((max_kv_len + page_size - 1) // page_size, 1)

        page_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)

        swa_slot_mapping = None
        swa_page_table = None
        if self.use_sliding_window_kv_pool:
            swa_slot_mapping = self.swa_kv_pool.full_to_swa_index_mapping.long()
            swa_page_table = torch.zeros(
                bs, max_blocks, dtype=torch.int32, device=device
            )

        BLOCK_SIZE = 1024
        grid = (bs, triton.cdiv(max_blocks, BLOCK_SIZE))
        scatter_req_to_token_to_page_table_kernel[grid](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_table,
            self.req_to_token.stride(0),
            page_table.stride(0),
            swa_page_table,
            swa_slot_mapping,
            DRAFT_NUM=0,
            PAGE_SIZE=page_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SWA=(swa_slot_mapping is not None),
        )
        return page_table, swa_page_table

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        bs0: int,
        window_size,
        sinks,
        k_descale,
        v_descale,
    ):
        """Prefill/extend through aiter's Triton ``unified_attention``. The CK
        ``mha_batch_prefill_func`` hard-asserts head_dim <= 256, which rules out
        Gemma-4's 512-wide full-attention layers. unified_attention pads the head
        dim to the next power of two and reads the same paged KV the decode path
        reads, so one kernel serves prefill and decode."""
        bs = forward_batch.batch_size
        max_kv_len = int(forward_batch.seq_lens_cpu.max().item())
        page_table, swa_page_table = self._build_extend_unified_page_table(
            bs, forward_batch.seq_lens, forward_batch.req_pool_indices, max_kv_len
        )

        # Build cu_seqlens_q from this batch's extend lengths. The standard
        # prefill metadata path leaves self.qo_indptr unset (qo_indptr=None in
        # ForwardMetadata), so relying on it corrupts multi-sequence batches
        # (only bs=1 happens to work).
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_q[1:] = torch.cumsum(
            forward_batch.extend_seq_lens.to(torch.int32), dim=0
        )

        # unified_attention uses (left, right) window = (window-1, 0), NOT the
        # CK convention (window, -1). Match the decode path (`de_window`).
        uni_window = (-1, -1)
        pt = page_table
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            uni_window = (layer.sliding_window_size - 1, 0)
            if swa_page_table is not None:
                pt = swa_page_table

        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        q_u = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o = q_u.new_empty(
            (q_u.shape[0], layer.tp_q_head_num, layer.v_head_dim),
            dtype=self.input_dtype,
        )
        unified_attention(
            q=q_u,
            k=k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim),
            v=v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
            out=o,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=forward_batch.seq_lens,
            max_seqlen_q=self.forward_metadata.max_q_len,
            max_seqlen_k=pt.shape[1] * self.page_size,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=uni_window,
            block_table=pt,
            softcap=layer.logit_cap,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _resolve_v2_num_draft_tokens(
        self,
        extend_seq_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[list[int]] = None,
    ) -> int:
        """Resolve fixed per-request extend length for DRAFT_EXTEND_V2."""
        num_draft_tokens = self.num_draft_tokens
        if num_draft_tokens is None:
            if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
                # Avoid list scans in hot path when tensor lengths are already available.
                num_draft_tokens = int(extend_seq_lens[0].item())
            elif extend_seq_lens_cpu:
                num_draft_tokens = max(extend_seq_lens_cpu)
            else:
                raise ValueError(
                    "DRAFT_EXTEND_V2 requires speculative_num_draft_tokens or "
                    "non-empty extend_seq_lens/extend_seq_lens_cpu."
                )

        num_draft_tokens = int(num_draft_tokens)
        if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
            if not torch.all(extend_seq_lens == num_draft_tokens):
                raise ValueError(
                    "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                    f"extend_seq_lens={extend_seq_lens}, expected all == {num_draft_tokens}."
                )
        if extend_seq_lens_cpu and any(
            x != num_draft_tokens for x in extend_seq_lens_cpu
        ):
            raise ValueError(
                "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                f"{extend_seq_lens_cpu}, expected all == {num_draft_tokens}."
            )
        return num_draft_tokens

    def _get_kv_indices_scratch(
        self, required_tokens: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._kv_indices_scratch is None
            or self._kv_indices_scratch.device != device
            or self._kv_indices_scratch.numel() < required_tokens
        ):
            self._kv_indices_scratch = torch.empty(
                required_tokens, dtype=torch.int32, device=device
            )
        return self._kv_indices_scratch[:required_tokens]

    def _get_arange(self, length: int) -> torch.Tensor:
        """arange(length) grown to the next power of two."""
        if self._arange_buf is None or self._arange_buf.numel() < length:
            self._arange_buf = torch.arange(
                1 << max(length - 1, 0).bit_length(),
                device=self.device,
                dtype=torch.int32,
            )
        return self._arange_buf[:length]

    def _asm_context_prefill_indices(
        self, forward_batch: ForwardBatch, bs: int, num_kv_slots: int
    ):
        """KV slots and cu_seqlens_k for the ASM context-chunk prefill, computed
        once per batch and shared by every full-attention layer.

        For a plain extend batch AiterIndicesUpdaterPrefill lays kv_indices out
        token by token with kv_indptr = cumsum(seq_lens), so the slots to gather
        are the first sum(seq_lens) entries and cu_seqlens_k is kv_indptr itself;
        both follow from host-side lengths without a device sync. Anything else
        (spec batches, missing host lengths, or a short table) takes the generic
        gather, which validates the metadata on the device.
        """
        fm = self.forward_metadata
        if fm.asm_ctx_ready:
            return fm.asm_ctx_tok_idx, fm.asm_ctx_cu_k
        fm.asm_ctx_ready = True
        total_k = 0
        if (
            forward_batch.spec_info is None
            and forward_batch.forward_mode.is_extend()
            and forward_batch.seq_lens_cpu is not None
        ):
            total_k = int(forward_batch.seq_lens_cpu[:bs].sum())
        if 0 < total_k <= fm.kv_indices.numel():
            tok_idx = fm.kv_indices[:total_k]
            cu_k = fm.kv_indptr[: bs + 1]
            if cu_k.dtype != torch.int32:
                cu_k = cu_k.to(torch.int32)
            if _GFX_ASM_CTX_GATHER_CHECK:
                ref = _asm_context_prefill_gather_indices(
                    fm.kv_indptr[: bs + 1],
                    fm.kv_indices,
                    forward_batch.seq_lens[:bs],
                    num_kv_slots,
                    forward_batch.forward_mode,
                )
                assert (
                    ref is not None
                    and torch.equal(ref[0], tok_idx.to(torch.long))
                    and torch.equal(ref[1].to(torch.int32), cu_k)
                ), (
                    "asm context prefill: fast gather indices differ from the generic gather"
                )
                logger.info(
                    "[asm-context-prefill] fast gather indices verified: bs=%d total_k=%d",
                    bs,
                    total_k,
                )
        else:
            gathered = _asm_context_prefill_gather_indices(
                fm.kv_indptr[: bs + 1],
                fm.kv_indices,
                forward_batch.seq_lens[:bs],
                num_kv_slots,
                forward_batch.forward_mode,
            )
            if gathered is None:
                return None, None
            tok_idx, cu_k = gathered
            cu_k = cu_k.to(torch.int32)
        fm.asm_ctx_tok_idx, fm.asm_ctx_cu_k = tok_idx, cu_k
        return tok_idx, cu_k

    def _set_uniform_qo_indptr(
        self, bs: int, tokens_per_req: int, device: torch.device
    ) -> torch.Tensor:
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * tokens_per_req + 1,
            step=tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        return qo_indptr

    def _ensure_spec_v2_topk_supported(self):
        if self.topk > 1:
            raise NotImplementedError(
                "AiterAttnBackend SPEC_V2 path currently supports topk <= 1 only. "
                f"Got topk={self.topk}."
            )

    def _asm_ps_supports_decode_and_verify(self) -> bool:
        """Whether the asm PS MLA kernels support both decode and target verify."""
        verify_q_len = self.num_draft_tokens or 1
        if self.dcp_world_size > 1:
            if _asm_cprr_supports_shape(self.cprr_kernel_heads, verify_q_len):
                return True
            logger.info(
                "aiter MLA decode and verify stay on the Gluon kernel: the cp "
                "verify kernel supports no %d-token window at %d gathered heads.",
                verify_q_len,
                self.num_head * self.dcp_world_size,
            )
            return False
        if self._asm_ps_supports_qlen(verify_q_len):
            return True
        logger.info(
            "aiter MLA decode and verify stay on the Gluon kernel: target "
            "verify is %d tokens wide and the asm kernels support at most %d at "
            "%d heads. A DSPARK block size of %d or lower would fit.",
            verify_q_len,
            _MLA_ASM_PS_MAX_QLEN,
            self.mla_kernel_num_head_padded,
            _MLA_ASM_PS_MAX_QLEN - 1,
        )
        return False

    def _asm_ps_supports_qlen(self, q_len: int) -> bool:
        """Whether the asm PS dispatch has a kernel for this query length."""
        return q_len <= _MLA_ASM_PS_MAX_QLEN or (
            self.kv_cache_dtype == fp8_dtype
            and self.mla_kernel_num_head_padded == 16
            and is_gfx95_supported()
        )

    def _asm_cprr_supports_verify_shape(self, q_len: Optional[int]) -> bool:
        """Whether this topology has an asm cp kernel for DCP target verify."""
        return (
            _use_mla_ps_kernel
            and self.mla_verify_backend == "asm"
            and self.dcp_world_size > 1
            and self.kv_cache_dtype == fp8_dtype
            and q_len is not None
            and self.cprr_kernel_heads > 0
            and _asm_cprr_supports_shape(self.cprr_kernel_heads, q_len)
        )

    def _mla_decode_fwd_with_head_pad(
        self,
        q: torch.Tensor,
        k_buffer_flat: torch.Tensor,
        layer,
        **kwargs,
    ):
        """Wrap mla_decode_fwd with head-dimension padding for num_head < 16.

        repeat (4/8): tile q heads to 16, slice back to num_head.
        zero (12): pad four zero-valued dummy heads to 16 (vLLM #50371 style).
        q / o must already be shaped (..., num_head, head_dim).
        """
        num_head = layer.tp_q_head_num
        if (
            _use_mla_ps_kernel
            and self.dcp_world_size <= 1
            and self._asm_ps_supports_qlen(kwargs.get("max_seqlen_q") or 1)
        ):
            # The asm kernel reads KV row 0 for masked positions and weights it
            # by p = 0. A NaN there spreads to every request, because
            # 0 * NaN = NaN. Row 0 is where CUDA-graph padding tokens write
            # their KV, and that KV can be NaN. Gluon, the kernel this path
            # replaces, does not read row 0.
            k_buffer_flat[0].zero_()
        if (
            self.kv_cache_dtype == fp8_dtype
            and self.mla_kernel_num_head_padded == 16
            and is_gfx95_supported()
            and (kwargs.get("max_seqlen_q") or 1) > _MLA_ASM_PS_MAX_QLEN
            and q.dtype != fp8_dtype
        ):
            q_flat = q.reshape(q.shape[0], -1).to(fp8_dtype)
            kwargs["q_scale"] = torch.ones((), dtype=torch.float32, device=q.device)
            q = q_flat.view(q.shape)
        if self.head_pad_mode == "repeat" or (
            self.head_pad_mode == "none" and self.num_head_padded != self.num_head
        ):
            q_in = self.pad_heads(q, self.num_head_padded)
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, : self.num_head, :]
        if self.head_pad_mode == "zero":
            q_in = q.new_zeros(
                (q.shape[0], self.num_head_padded, q.shape[-1]),
                dtype=q.dtype,
            )
            q_in[:, :num_head, :] = q
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, :num_head, :]
        o = q.new_empty(
            (q.shape[0], num_head, layer.v_head_dim),
            dtype=self.input_dtype,
        )
        mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
        return o

    def _zero_pad_mla_q_heads(
        self, q: torch.Tensor, layer: RadixAttention
    ) -> torch.Tensor:
        """Zero-pad q heads num_head -> num_head_padded (12 -> 16) for the
        aiter MLA prefill kernels. Input/return are 3-D (T, H, D); the extra
        heads compute garbage outputs that are sliced away after the kernel."""
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        q_pad = q3.new_zeros((q3.shape[0], self.num_head_padded, q3.shape[-1]))
        q_pad[:, : layer.tp_q_head_num, :] = q3
        return q_pad

    def _resolve_fp8_kv_scale_float(self, layer: RadixAttention, k_descale) -> float:
        cached = getattr(layer, "_aiter_kv_scale_float", None)
        if cached is not None:
            return cached
        if k_descale is None:
            val = 1.0
        elif isinstance(k_descale, torch.Tensor):
            val = float(k_descale.item())
        else:
            val = float(k_descale)
        layer._aiter_kv_scale_float = val
        return val

    def _resolve_mla_gluon_min_kv_seq_len(self, forward_batch: ForwardBatch) -> int:
        try:
            if torch.cuda.is_current_stream_capturing():
                return int(self.max_context_len)
        except Exception:
            pass
        if forward_batch.seq_lens_cpu is not None:
            return int(forward_batch.seq_lens_cpu.max())
        seq_lens = forward_batch.seq_lens
        if seq_lens is None or seq_lens.numel() == 0:
            return 1
        return int(seq_lens.max().item())

    def _kernel_num_kv_splits(self) -> Optional[int]:
        """The split count for a decode kernel call, None to let aiter pick.

        The metadata call takes an int and gets -1 for the auto branch. The
        kernel takes None instead. mla.py re-decides persistent mode only on
        None. It would read -1 as a real count.
        """
        if self.use_mla_auto_kv_splits:
            return None
        return self.forward_metadata.num_kv_splits

    def _forward_mla_decode(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_descale,
    ):
        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        max_q_len = self.forward_metadata.max_q_len or 1

        if prefer_mla_gluon_decode(
            head_pad_mode=getattr(self, "head_pad_mode", "none"),
            num_head=getattr(self, "num_head", layer.tp_q_head_num),
            kv_cache_dtype=self.kv_cache_dtype,
        ) and not (
            _use_mla_ps_kernel
            and self.mla_decode_backend == "asm"
            and self._asm_ps_supports_qlen(max_q_len)
        ):
            return mla_gluon_decode(
                q=q,
                k_buffer=k_buffer,
                layer=layer,
                kv_indices=self.forward_metadata.kv_indices,
                kv_indptr=self.forward_metadata.kv_indptr,
                sm_scale=layer.scaling,
                kv_scale=self._resolve_fp8_kv_scale_float(layer, k_descale),
                min_kv_seq_len=self._resolve_mla_gluon_min_kv_seq_len(forward_batch),
                qlen=max_q_len,
            )

        work_metadata = self.forward_metadata.work_metadata
        work_indptr = self.forward_metadata.work_indptr
        work_info_set = self.forward_metadata.work_info_set
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map
        num_kv_splits = self._kernel_num_kv_splits()

        return self._mla_decode_fwd_with_head_pad(
            q,
            k_buffer.view(-1, 1, 1, layer.qk_head_dim),
            layer,
            qo_indptr=self.forward_metadata.qo_indptr,
            kv_indptr=self.forward_metadata.kv_indptr,
            kv_indices=self.forward_metadata.kv_indices,
            kv_last_page_lens=self.forward_metadata.kv_last_page_len,
            max_seqlen_q=max_q_len,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=k_descale,
            kv_scale=k_descale,
            intra_batch_mode=intra_batch_mode,
            num_kv_splits=num_kv_splits,
        )

    def _dcp_max_local_kv_len(self, max_global_kv_len: int) -> int:
        """Widest shard one rank can hold for a global kv length."""
        w = max(self.dcp_world_size, 1)
        return (max_global_kv_len + w - 1) // w

    def _dcp_graph_max_local_kv_len(self) -> int:
        """Static upper bound on this rank's shard under a cuda graph."""
        max_global_kv_len = self.max_context_len
        if self._asm_cprr_supports_verify_shape(self.num_draft_tokens):
            max_global_kv_len += self.num_draft_tokens
        return self._dcp_max_local_kv_len(max_global_kv_len)

    def _forward_decode_dcp(self, q, k_buffer, layer, k_descale):
        """Attend this rank's KV shard for decode -> (out, natural-log lse)."""
        fm = self.forward_metadata
        bs = fm.kv_indptr.shape[0] - 1
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp

        if self.use_mla_dcp_asm:
            if fm.work_metadata is None:
                raise RuntimeError(
                    "AITER MLA DCP ASM selected without persistent metadata"
                )

            q_mla = q.view(bs, num_heads, layer.qk_head_dim)
            q_scale = k_descale
            if q_mla.dtype != fp8_dtype:
                q_mla = (
                    q_mla.reshape(bs, -1)
                    .to(fp8_dtype)
                    .view(bs, num_heads, layer.qk_head_dim)
                )
                q_scale = torch.ones((), dtype=torch.float32, device=q.device)

            out = torch.empty(
                (bs, num_heads, layer.v_head_dim),
                dtype=self.input_dtype,
                device=q.device,
            )
            _, lse = mla_decode_fwd(
                q_mla,
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                out,
                fm.qo_indptr,
                fm.kv_indptr[: bs + 1],
                fm.kv_indices,
                fm.kv_last_page_len,
                fm.max_q_len or 1,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                num_kv_splits=self._kernel_num_kv_splits(),
                work_meta_data=fm.work_metadata,
                work_indptr=fm.work_indptr,
                work_info_set=fm.work_info_set,
                reduce_indptr=fm.reduce_indptr,
                reduce_final_map=fm.reduce_final_map,
                reduce_partial_map=fm.reduce_partial_map,
                q_scale=q_scale,
                kv_scale=k_descale,
                intra_batch_mode=False,
                return_lse=True,
            )
            if lse is None:
                raise RuntimeError(
                    "aiter mla_decode_fwd(return_lse=True) returned no LSE"
                )
            return out, lse.view(bs, num_heads)

        out, lse = mla_gluon_decode(
            q=q.view(bs, num_heads, layer.qk_head_dim),
            k_buffer=k_buffer,
            layer=layer,
            kv_indices=fm.kv_indices,
            kv_indptr=fm.kv_indptr[: bs + 1],
            sm_scale=layer.scaling,
            kv_scale=self._resolve_fp8_kv_scale_float(layer, k_descale),
            min_kv_seq_len=1,
            return_lse=True,
        )
        return out, lse.view(bs, num_heads)

    def _forward_verify_asm_cprr(self, q, layer, k_descale, n_rows):
        """DCP target verify on the asm cp round-robin kernel: one masked pass."""
        fm = self.forward_metadata
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp
        kernel_heads = fm.cprr_kernel_heads or num_heads
        if not self._cprr_launch_logged:
            self._cprr_launch_logged = True
            logger.info(
                "aiter DCP cp verify: gathered=%d kernel_heads=%d%s "
                "intra_batch=%s max_q_len=%d reduce_partial_map=%d",
                num_heads,
                kernel_heads,
                " (padded)" if kernel_heads != num_heads else " (native)",
                self.cprr_intra_batch_mode,
                fm.max_q_len,
                fm.reduce_partial_map.numel(),
            )
        # torch.full, not torch.as_tensor: as_tensor copies from host memory,
        # which a cuda graph capture cannot record.
        kv_scale = torch.full(
            (),
            self._resolve_fp8_kv_scale_float(layer, k_descale),
            dtype=torch.float32,
            device=q.device,
        )
        kv_indices, kv_indptr = fm.cprr_kv_indices, fm.cprr_kv_indptr
        q_in = (
            q.view(n_rows, -1).to(fp8_dtype).view(n_rows, num_heads, layer.qk_head_dim)
        )
        q_scale = torch.ones((), dtype=torch.float32, device=q.device)
        if kernel_heads != num_heads:
            q_in = self.pad_heads(q_in, kernel_heads)
        out = q.new_empty(
            (n_rows, kernel_heads, layer.v_head_dim), dtype=self.input_dtype
        )
        _, lse = mla_decode_fwd(
            q_in,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, 1, 1, layer.qk_head_dim
            ),
            out,
            fm.qo_indptr,
            kv_indptr,
            kv_indices,
            fm.kv_last_page_len,
            fm.max_q_len,
            sm_scale=layer.scaling,
            work_meta_data=fm.work_metadata,
            work_indptr=fm.work_indptr,
            work_info_set=fm.work_info_set,
            reduce_indptr=fm.reduce_indptr,
            reduce_final_map=fm.reduce_final_map,
            reduce_partial_map=fm.reduce_partial_map,
            q_scale=q_scale,
            kv_scale=kv_scale,
            intra_batch_mode=self.cprr_intra_batch_mode,
            num_kv_splits=self._kernel_num_kv_splits(),
            return_lse=True,
            g_kv_indptr=fm.global_kv_indptr,
            cp_world_size=self.dcp_world_size,
            cp_rank=get_parallel().attn_dcp_rank,
            causal=True,
        )
        lse = lse.view(n_rows, kernel_heads)
        if kernel_heads != num_heads:
            out = out[:, :num_heads, :]
            lse = lse[:, :num_heads]
        return out, lse

    def _build_dcp_verify_ragged_indices(
        self,
        verify_token_table: torch.Tensor,
        local_kv_lens: torch.Tensor,
        bs: int,
        q_len: int,
        out_indices: Optional[torch.Tensor] = None,
        out_indptr: Optional[torch.Tensor] = None,
        out_empty: Optional[torch.Tensor] = None,
    ):
        """This rank's verify shard as ragged (kv_indices, kv_indptr, empty_rows)."""
        # Pass out_* to use the address-stable buffers of the cuda graph.
        # Omit them to allocate new tensors.
        # The .contiguous() call is necessary. The slice has stride q_len, and
        # .to() returns an int32 tensor unchanged, so without the copy the
        # kernel reads lens[req] instead of lens[req * q_len]. The two agree
        # only at batch size 1.
        lens = local_kv_lens[: bs * q_len : q_len].to(torch.int32).contiguous()
        # The kernel faults on a zero-length shard, so give every shard a slot.
        clamped = lens.clamp_min(1)
        if out_indptr is None:
            kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=lens.device)
        else:
            kv_indptr = out_indptr[: bs + 1]
            # Not kv_indptr[0] = 0: that copies from host memory, which a
            # cuda graph capture cannot record.
            kv_indptr[:1].zero_()
        torch.cumsum(clamped, dim=0, out=kv_indptr[1:])

        if out_indices is None:
            kv_indices = verify_token_table.new_empty(
                (int(clamped.sum().item()),), dtype=torch.int32
            )
        else:
            kv_indices = out_indices
        compact_dcp_verify_token_table_to_ragged[(bs,)](
            verify_token_table,
            lens,
            kv_indptr,
            kv_indices,
            table_stride=verify_token_table.shape[1],
            Q_LEN=q_len,
            BLOCK_SIZE=_DCP_VERIFY_TABLE_COLS_PER_BLOCK,
        )
        empty_rows = local_kv_lens[: bs * q_len].eq(0)
        if out_empty is not None:
            empty_rows = out_empty[: bs * q_len].copy_(empty_rows)
        return kv_indices, kv_indptr, empty_rows

    def _forward_verify_dcp(self, q, k_window, layer, k_descale):
        """DCP target verify -> (out, natural-log lse) for the cross-rank merge."""
        fm = self.forward_metadata
        q_len = fm.max_q_len
        num_heads = layer.tp_q_head_num  # gathered heads = num_local_heads * dcp
        seqused_k = fm.local_kv_lens
        n_rows = seqused_k.shape[0]
        bs = n_rows // q_len

        if fm.use_asm_cprr_verify:
            out_a, lse_a = self._forward_verify_asm_cprr(q, layer, k_descale, n_rows)
            empty = fm.cprr_empty_rows
            out_a.masked_fill_(empty[:, None, None], 0.0)
            lse_a.masked_fill_(empty[:, None], -1e30)
            return out_a, lse_a

        out_a, lse_a = mla_gluon_decode(
            q=q.view(n_rows, num_heads, layer.qk_head_dim),
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            layer=layer,
            kv_indices=fm.verify_token_table,
            kv_indptr=seqused_k,
            sm_scale=layer.scaling,
            kv_scale=self._resolve_fp8_kv_scale_float(layer, k_descale),
            min_kv_seq_len=1,
            return_lse=True,
            use_2d_view=True,
        )
        lse_a = lse_a.view(n_rows, num_heads)

        # The verify window arrives as `k_window`, computed this forward and
        # identical on every rank, so only one rank attends it; the others
        # return their prefix partial for the cross-rank merge.
        if get_parallel().attn_dcp_rank != 0:
            return out_a, lse_a

        # The window latent is request-major and contiguous, so it IS the pool:
        # row i of request b lives at b * q_len + i. mla_gluon's MTP mask at
        # seq_len == qlen is exactly the dense causal window this needs.
        out_b, lse_b = mla_gluon_decode(
            q=q.view(n_rows, num_heads, layer.qk_head_dim),
            k_buffer=k_window,
            layer=layer,
            kv_indices=torch.arange(n_rows, dtype=torch.int32, device=q.device),
            kv_indptr=torch.arange(bs + 1, dtype=torch.int32, device=q.device) * q_len,
            sm_scale=layer.scaling,
            min_kv_seq_len=1,
            qlen=q_len,
            return_lse=True,
        )
        return merge_state_triton(out_a, lse_a, out_b, lse_b.view(n_rows, num_heads))

    def mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        out, _ = self._mla_fp8_prefill_attn_ps(
            q, k, v, layer, self.forward_metadata.prefill_ps_metadata
        )
        return out

    def _mla_fp8_prefill_attn_ps(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        ps: MlaPrefillPsMetadata,
    ):
        """Run the asm prefill over one PS metadata, returning (out, lse)."""
        total_q = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim
        # mla_reduce_v1 dispatches on the head count, so a model it has no
        # instantiation for runs on the next one up, with the query heads tiled
        # to fill and the extra output columns sliced back off.
        head_pad = self.fp8_prefill_num_head - nhead
        if head_pad:
            q = self.pad_heads(q, self.fp8_prefill_num_head)
            k = self.pad_heads(k, self.fp8_prefill_num_head)
            v = self.pad_heads(v, self.fp8_prefill_num_head)
            nhead = self.fp8_prefill_num_head

        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        partial_q = max(ps.num_partial_tiles, 1) * _PREFILL_TILE_Q
        logits = torch.empty(
            (partial_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (partial_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = (
            torch.empty((total_q, nhead), dtype=torch.float32, device=q.device)
            if ps.need_lse
            else None
        )
        output = q.new_empty(
            (total_q, nhead, v_head_dim),
            dtype=self.input_dtype,
        )

        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            ps.qo_indptr,
            ps.kv_indptr,
            ps.kv_indices,
            ps.work_indptr,
            ps.work_info_set,
            ps.max_q_len,
            layer.scaling,
            ps.is_causal,
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        mla_reduce_v1(
            logits,
            attn_lse,
            ps.reduce_indptr,
            ps.reduce_final_map,
            ps.reduce_partial_map,
            _PREFILL_TILE_Q,
            # Prefill PS metadata has no split cap; 0 keeps AITER's default reduce sizing.
            0,
            output,
            final_lse,
        )
        if head_pad:
            output = output[:, : layer.tp_q_head_num, :]
            if final_lse is not None:
                # Slicing the padded head axis leaves the kernel's stride
                # behind, and merge_state reads its inputs as contiguous.
                output = output.contiguous()
                final_lse = final_lse[:, : layer.tp_q_head_num].contiguous()
        return output, final_lse

    def _kv_index_blocks(self, bs: int) -> int:
        if self.max_context_len < _KV_INDEX_BLOCKS_MIN_CONTEXT:
            return 1
        return kv_indices_num_token_blocks(self.req_to_token.shape[1], bs)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        reset_verify_attn_plan_cache()
        seq_lens_cpu = (
            forward_batch.seq_lens.cpu() if in_capture else forward_batch.seq_lens_cpu
        )
        verify_tokens_per_req = (
            forward_batch.input_ids.shape[0] // forward_batch.batch_size
            if forward_batch.forward_mode.is_target_verify()
            else None
        )
        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=None if in_capture else forward_batch.seq_lens_sum,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
            verify_tokens_per_req=verify_tokens_per_req,
        )

        # Refill the SWA write-target buffer from the live out_cache_loc and
        # bind it onto the metadata before replay (_apply rebuilds it each call).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            if in_capture:
                self.cuda_graph_swa_out_cache_loc[:n].zero_()
            else:
                self.cuda_graph_swa_out_cache_loc[:n].copy_(
                    self.swa_kv_pool.translate_loc_from_full_to_swa(
                        forward_batch.out_cache_loc
                    )
                )
            self.forward_metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                :n
            ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for aiter attention backend."""
        reset_verify_attn_plan_cache()

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        max_kv_len = None

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        swa_page_table = None
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )
        # Absent under the sync-free path (needs_cpu_seq_lens=False); unified
        # verify sizes from its page table instead.
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = forward_batch.seq_lens_cpu.max().item()

        # dcp metadata
        local_kv_lens = None
        verify_token_table = None
        global_kv_indptr = None
        use_asm_cprr_verify = False
        cprr_kv_indices = cprr_kv_indptr = cprr_empty_rows = None
        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or forward_batch.forward_mode.is_idle():
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                if not self.use_triton_unified_attention:
                    kv_indices = self._get_kv_indices_scratch(
                        forward_batch.seq_lens_sum, forward_batch.seq_lens.device
                    )
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )

                    if (
                        self.use_mla
                        and self.dcp_world_size > 1
                        and not forward_batch.forward_mode.is_idle()
                    ):
                        kv_lens = forward_batch.seq_lens[:bs].to(torch.int32).clone()
                        self._plan_dcp_decode_metadata(
                            kv_indptr,
                            kv_indices,
                            kv_lens,
                            forward_batch.seq_lens_cpu,
                            bs,
                        )
                else:
                    max_q_len = 1
                    page_size = self.page_size
                    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
                    kv_indices = torch.zeros(
                        bs, max_kv_len, dtype=torch.int32, device=self.device
                    )

                    create_flashmla_kv_indices_triton[
                        (bs, get_num_kv_index_blocks_flashmla(max_kv_len, 1))
                    ](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        max_kv_len,
                        1,
                    )

                    if self.use_sliding_window_kv_pool:
                        # AITER attention kernels require int32 page indices;
                        # full_to_swa_index_mapping is stored as int64.
                        swa_page_table = (
                            self.swa_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            ).to(torch.int32)
                        )

                        kv_indices = self._transform_table_1_to_real(kv_indices)
                        swa_page_table = self._transform_table_1_to_real(swa_page_table)
                    elif self.page_size > 1:
                        kv_indices = self._transform_table_1_to_real(kv_indices)

                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]

            else:
                if self.use_triton_unified_attention and not self.use_mla:
                    bs = spec_info.kv_indptr.shape[0] - 1
                    kv_indices, swa_page_table = (
                        self._build_unified_page_table_from_spec(spec_info, bs)
                    )
                    max_q_len = 1
                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]
                    kv_indptr = None
                else:
                    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                    bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1

                if self._use_mla_decode_persist_metadata():
                    metadata_fast_mode, metadata_intra_batch_mode = (
                        self._mla_decode_metadata_modes()
                    )
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(
                        max_q_len,
                        bs,
                        metadata_fast_mode=metadata_fast_mode,
                        metadata_intra_batch_mode=metadata_intra_batch_mode,
                    )

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=metadata_fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=metadata_intra_batch_mode,
                    )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                run_graph=False,
                swa_page_table=swa_page_table,
                swa_out_cache_loc=swa_out_cache_loc,
            )

        elif forward_batch.forward_mode.is_draft_extend_v2():
            # EAGLE V2: DRAFT_EXTEND_V2 mode - extend draft KV cache with all predicted tokens
            self._ensure_spec_v2_topk_supported()
            if self.use_mla:
                device = forward_batch.seq_lens.device
                num_draft_tokens = self._resolve_v2_num_draft_tokens()
                qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)

                kv_indices = self._get_kv_indices_scratch(
                    forward_batch.seq_lens_sum, device
                )

                num_token_blocks = self._kv_index_blocks(bs)
                create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                    TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
                )

                if _use_mla_ps_kernel:
                    max_seqlen_qo = num_draft_tokens
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    num_draft_tokens,
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                kv_indices, kv_indptr, qo_indptr, _ = (
                    forward_batch.spec_info.generate_attn_arg_prefill(
                        req_pool_indices=forward_batch.req_pool_indices,
                        paged_kernel_lens=forward_batch.seq_lens,
                        paged_kernel_lens_sum=forward_batch.seq_lens_sum,
                        translator=self.kv_index_translator,
                        plan=forward_batch.kv_loc_plan,
                    )
                )
                # CK fallback: publish this step's short qo_indptr into the
                # shared buffer (prompt extend left it at the prompt length),
                # set max_kv_len, and pad the page ids the kernel over-reads.
                kv_indices = _pad_mha_prefill_kv_indices(kv_indices)
                n_qo = qo_indptr.shape[0]
                self.qo_indptr[:n_qo].copy_(qo_indptr)
                qo_indptr = self.qo_indptr[:n_qo]
                max_kv_len = int(forward_batch.seq_lens_cpu.max().item())
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    forward_batch.spec_info.num_tokens_per_req,
                    max_kv_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                device = forward_batch.seq_lens.device
                use_asm_cprr_verify = self._asm_cprr_supports_verify_shape(draft_num)
                if self.dcp_world_size > 1 and not use_asm_cprr_verify:
                    kv_lens = forward_batch.seq_lens.to(torch.int32).clone()
                    kv_lens_sum = forward_batch.seq_lens_sum
                else:
                    kv_lens = forward_batch.seq_lens + draft_num
                    kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs

                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indices = self._get_kv_indices_scratch(
                    kv_lens_sum,
                    device,
                )
                num_token_blocks = self._kv_index_blocks(bs)
                create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                    TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
                )

                if self.dcp_world_size > 1:
                    global_kv_indptr = kv_indptr.clone()
                    self._plan_dcp_decode_metadata(
                        kv_indptr,
                        kv_indices,
                        kv_lens,
                        None,
                        bs,
                    )
                    (
                        verify_token_table,
                        local_kv_lens,
                    ) = self._build_dcp_verify_token_table(
                        kv_indptr,
                        forward_batch.req_pool_indices,
                        bs,
                        draft_num,
                        self._dcp_max_local_kv_len(
                            max_kv_len + (draft_num if use_asm_cprr_verify else 0)
                        ),
                    )

                if use_asm_cprr_verify:
                    (
                        cprr_kv_indices,
                        cprr_kv_indptr,
                        cprr_empty_rows,
                    ) = self._build_dcp_verify_ragged_indices(
                        verify_token_table, local_kv_lens, bs, draft_num
                    )
                if _use_mla_ps_kernel and (
                    self.dcp_world_size <= 1 or use_asm_cprr_verify
                ):
                    max_seqlen_qo = draft_num
                    is_cp_round_robin = self.dcp_world_size > 1
                    metadata_heads = (
                        self.cprr_kernel_heads if use_asm_cprr_verify else None
                    )
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(
                        max_seqlen_qo,
                        bs,
                        metadata_fast_mode=(
                            self.cprr_fast_mode if use_asm_cprr_verify else None
                        ),
                        metadata_intra_batch_mode=(
                            self.cprr_intra_batch_mode if use_asm_cprr_verify else None
                        ),
                        nhead_override=metadata_heads,
                        max_split_per_batch=(
                            self.max_split_per_batch if use_asm_cprr_verify else None
                        ),
                    )

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        cprr_kv_indptr if use_asm_cprr_verify else kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=(
                            self.cprr_fast_mode if use_asm_cprr_verify else fast_mode
                        ),
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=(
                            self.cprr_intra_batch_mode
                            if use_asm_cprr_verify
                            else intra_batch_mode
                        ),
                        is_cp_round_robin=is_cp_round_robin,
                        nhead_override=metadata_heads,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                    local_kv_lens=local_kv_lens,
                    verify_token_table=verify_token_table,
                    global_kv_indptr=global_kv_indptr,
                    use_asm_cprr_verify=use_asm_cprr_verify,
                    cprr_kernel_heads=self.cprr_kernel_heads
                    if use_asm_cprr_verify
                    else 0,
                    cprr_kv_indices=cprr_kv_indices,
                    cprr_kv_indptr=cprr_kv_indptr,
                    cprr_empty_rows=cprr_empty_rows,
                )
            else:
                draft_num = forward_batch.input_ids.shape[0] // bs
                bs = len(forward_batch.req_pool_indices)

                if self._use_unified_verify:
                    page_table, qo_indptr, max_q_len, swa_page_table = (
                        self._build_verify_unified_metadata(
                            bs,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            draft_num,
                        )
                    )
                    max_kv_len = page_table.shape[1] * self.page_size
                    self.forward_metadata = ForwardMetadata(
                        None,  # kv_indptr unused in unified-verify path
                        page_table,  # 2D block page_table stored in kv_indices
                        qo_indptr,
                        None,
                        max_q_len,
                        max_kv_len,
                        max_extend_len=max_q_len,
                        swa_page_table=swa_page_table,
                        swa_out_cache_loc=swa_out_cache_loc,
                    )
                else:
                    qo_indptr = torch.arange(
                        0,
                        (1 + bs) * draft_num,
                        step=draft_num,
                        dtype=torch.int32,
                        device=self.device,
                    )

                    kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]

                    kv_indices = torch.empty(
                        kv_indptr[-1], dtype=torch.int64, device=self.device
                    )
                    num_token_blocks = self._kv_index_blocks(bs)
                    create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
                    )

                    custom_mask = spec_info.custom_mask
                    seq_mask_len = draft_num * (forward_batch.seq_lens + draft_num)
                    mask_indptr = self.mask_indptr
                    mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                    mask_indptr = mask_indptr[: bs + 1]

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        None,
                        draft_num,
                        None,
                        custom_mask=custom_mask,
                        mask_indptr=mask_indptr,
                        max_extend_len=draft_num,
                    )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    spec_info=None,
                    plan=forward_batch.kv_loc_plan,
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr
                kv_indptr = self.mla_indices_updater_prefill.kv_indptr

                prefill_ps_metadata = None
                if self.use_fp8_prefill_attn:
                    prefill_ps_metadata = self._build_prefill_ps_metadata(
                        qo_indptr=qo_indptr,
                        kv_indptr=kv_indptr,
                        kv_lens_cpu=forward_batch.seq_lens_cpu,
                        num_kv_tokens=forward_batch.seq_lens_sum,
                        max_q_len=max_q_len,
                        # The keys are the whole sequence, and the extend
                        # tokens sit at its tail.
                        is_causal=True,
                        need_lse=False,
                        # Keep the static bound this path has always sized its
                        # partial buffers with, rather than the emitted count.
                        exact_partial_count=False,
                    )

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    prefill_ps_metadata=prefill_ps_metadata,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                    plan=forward_batch.kv_loc_plan,
                )

                if self.use_sliding_window_kv_pool:
                    # AITER attention kernels (e.g. mha_batch_prefill_func)
                    # require int32 page indices; full_to_swa_index_mapping is
                    # stored as int64.
                    swa_page_table = self.swa_kv_pool.translate_loc_from_full_to_swa(
                        self.indices_updater_prefill.kv_indices
                    ).to(torch.int32)

                # Once per batch, not per layer: forward_extend only consumes it.
                # The arch test is not redundant with the flag: the asm guard is
                # gfx95-only, and elsewhere there is no kernel for this shape at
                # all, so building a view we must not pass is wasted work.
                paged_kv_view = None
                if (
                    envs.SGLANG_AITER_PAGED_PREFILL_ASM.get()
                    and is_gfx95_supported()
                    and self.page_size == 64
                    and not self.kv_cache_is_vectorized_5d
                    and not self.use_sliding_window_kv_pool
                    and not self.use_triton_unified_attention
                ):
                    paged_kv_view = _build_paged_kv_view(
                        self.indices_updater_prefill.kv_indices,
                        forward_batch.seq_lens_cpu,
                        self.page_size,
                    )

                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                    swa_out_cache_loc=swa_out_cache_loc,
                    paged_kv_view=paged_kv_view,
                )

    def _plan_dcp_decode_metadata(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_lens_gpu: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        bs: int,
        static_local_kv_lens_cpu: Optional[torch.Tensor] = None,
    ):
        """Localize kv_indptr / kv_indices to this rank's DCP shard, in place."""
        if static_local_kv_lens_cpu is not None:
            # The planner reads `kv_len_arr_cpu` only for (max, sum), never for
            # the lengths it writes back, so a static upper bound yields the same
            # metadata without the GPU->CPU sync.
            total_local_len = plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=True,
                fast_decode_kwargs={"kv_len_arr_cpu": static_local_kv_lens_cpu},
                bs=bs,
            )
        elif seq_lens_cpu is not None:
            kv_len_arr_cpu = seq_lens_cpu[:bs].to(torch.int32).clone()
            update_local_kv_lens_for_dcp(kv_len_arr_cpu)
            total_local_len = plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=True,
                fast_decode_kwargs={"kv_len_arr_cpu": kv_len_arr_cpu},
                bs=bs,
            )
        else:
            total_local_len = plan_dcp_decode_metadata(
                kv_lens_gpu,
                kv_indptr,
                kv_indices,
                init_metadata_replay=False,
                fast_decode_kwargs={},
                bs=bs,
            )

        # The planner leaves the compacted ids WIDENED (see its docstring), and
        # mla_gluon indexes the pool directly, so collapse them once per forward
        # -- the same contract flashinfer_mla_backend.py follows.
        translator = self.kv_index_translator
        if total_local_len > 0 and translator.needs_read_translate:
            valid = kv_indices[:total_local_len]
            valid.copy_(translator.translate_dcp_read_ids(valid))

    def _build_dcp_local_kv_lens(
        self,
        kv_indptr: torch.Tensor,
        bs: int,
        out_lens: Optional[torch.Tensor] = None,
    ):
        """This rank's shard length per request, in TOKENS (mla_gluon's
        ``cache_seqlens``). ``kv_indptr`` must already be localized.
        """
        lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int32)
        if out_lens is None:
            return lens
        out_lens.copy_(lens)
        return out_lens

    def _build_dcp_verify_token_table(
        self,
        kv_indptr: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
        q_len: int,
        max_local_kv_len: int,
        out: Optional[torch.Tensor] = None,
        out_lens: Optional[torch.Tensor] = None,
    ):
        """Token table + shard lengths for the prefix attention of DCP verify.

        One row per query token, one column per TOKEN (mla_gluon fixes
        PAGE_SIZE at 1). Rows of a request repeat that request's shard.
        """
        local_kv_lens = self._build_dcp_local_kv_lens(kv_indptr, bs)
        n_rows = bs * q_len
        if out is None:
            # The row stride below is a Triton constexpr, so quantize the eager
            # width: every distinct value costs a JIT specialization.
            out = local_kv_lens.new_empty(
                (
                    n_rows,
                    triton.cdiv(max_local_kv_len, _DCP_VERIFY_TABLE_COLS_PER_BLOCK)
                    * _DCP_VERIFY_TABLE_COLS_PER_BLOCK,
                )
            )
        num_cols = out.shape[1]

        # Write each request's row 0 in place: the row stride handed to the
        # kernel spans that request's whole block of q_len rows.
        translator = self.kv_index_translator
        v2p = translator.full_v2p_table
        create_mla_kv_page_table_for_dcp[
            (bs, triton.cdiv(num_cols, _DCP_VERIFY_TABLE_COLS_PER_BLOCK))
        ](
            self.req_to_token,
            req_pool_indices,
            local_kv_lens,
            out,
            v2p,
            self.req_to_token.stride(0),
            q_len * num_cols,
            PHYSICAL_PAGE_SIZE=1,
            DCP_SIZE=self.dcp_world_size,
            DCP_RANK=get_parallel().attn_dcp_rank,
            PAGES_PER_BLOCK=_DCP_VERIFY_TABLE_COLS_PER_BLOCK,
            HAS_V2P=v2p is not None,
        )
        rows = out.view(bs, q_len, num_cols)
        if q_len > 1:
            # Source is row 0, destination rows 1.., so the copy never overlaps.
            rows[:, 1:, :].copy_(rows[:, :1, :].expand(bs, q_len - 1, num_cols))

        if out_lens is None:
            out_lens = local_kv_lens.new_empty((n_rows,))
        out_lens.view(bs, q_len).copy_(local_kv_lens.unsqueeze(1).expand(bs, q_len))
        return out, out_lens

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # PR #20978 pads max_bs beyond pool_size for higher cuda-graph
        # coverage. Reallocate indptr buffers so they fit the padded max_bs.
        # See: https://github.com/sgl-project/sglang/pull/20978
        if max_bs + 1 > self.kv_indptr.shape[0]:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=self.device
            )
            if hasattr(self, "qo_indptr_"):
                self.qo_indptr_ = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=self.device
                )

        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )
        if self.use_mla and self.dcp_world_size > 1:
            if self.num_draft_tokens:
                # Target-verify flattens the window into single-token rows, so it
                # needs max_bs * num_draft_tokens.
                n_verify_rows = max_bs * self.num_draft_tokens
                self.cuda_graph_verify_local_kv_lens = torch.zeros(
                    (n_verify_rows,), dtype=torch.int32, device=self.device
                )
                self.cuda_graph_verify_token_table = torch.zeros(
                    (n_verify_rows, self._dcp_graph_max_local_kv_len()),
                    dtype=torch.int32,
                    device=self.device,
                )
                # Static per-rank shard bound, one entry per request. Sizes the
                # verify plan without a sync; see _plan_dcp_decode_metadata.
                self.cuda_graph_dcp_static_local_kv_lens = torch.full(
                    (max_bs,),
                    self._dcp_graph_max_local_kv_len(),
                    dtype=torch.int32,
                    device="cpu",
                )
                self.cuda_graph_global_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=self.device
                )
                self.cuda_graph_cprr_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=self.device
                )
                self.cuda_graph_cprr_kv_indices = torch.zeros(
                    (max_bs * (self._dcp_graph_max_local_kv_len() + 1),),
                    dtype=torch.int32,
                    device=self.device,
                )
                self.cuda_graph_cprr_empty_rows = torch.zeros(
                    (n_verify_rows,), dtype=torch.bool, device=self.device
                )
        if kv_indices_buf is None:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            # Non-unified AITER CUDA graph paths fill this buffer with flat
            # token-level kv_indices via create_flashinfer_kv_indices_triton
            # (kv_indptr = cumsum(seq_lens)).  Even when the allocator is
            # page-based, these writes are per-token, so page-sized allocation
            # would under-allocate by page_size when page_size > 1.
            # TODO(aiter, page_size>1): root fix is to make page_size>1
            # actually engage the attention kernel (`forward_decode` still
            # calls paged_attention_ragged with view(-1, 1, ...) and
            # block_size=1). That requires a per-page indices kernel + all
            # metadata sites + paged_attention_ragged call site + FP8 KV
            # coordination, after which this allocation can revert to
            # per-page (gated on use_mla).
            # Reserve draft slack: MLA target_verify writes seq_len +
            # num_draft_tokens per row; without it a near-full sequence
            # overflows the buffer. Mirrors dsa / flashmla.
            draft_slack = self.num_draft_tokens or 0
            buffer_numel = max_bs * (
                max_num_blocks_per_seq * self.page_size + draft_slack
            )
            self.cuda_graph_kv_indices = torch.zeros(
                (buffer_numel,),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if self.use_triton_unified_attention:
            # Keep a distinct page-table buffer for unified attention.  Sharing
            # cuda_graph_kv_indices with non-unified token indices makes
            # page-table width ambiguous after the token buffer is expanded.
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        # if self.use_mla and (_use_mla_ps_kernel or self.kv_cache_dtype == fp8_dtype):
        if self.use_mla and (_use_mla_ps_kernel or self.use_mla_dcp_asm):
            # for persistent mla_decode_fwd
            max_seqlen_qo = (
                1 if self.num_draft_tokens is None else self.num_draft_tokens
            )
            metadata_fast_mode, metadata_intra_batch_mode = (
                (True, False) if self.use_mla_dcp_asm else (fast_mode, intra_batch_mode)
            )

            graph_use_asm_cprr_verify = self._asm_cprr_supports_verify_shape(
                self.num_draft_tokens
            )
            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(
                max_seqlen_qo,
                max_bs,
                metadata_fast_mode=(
                    self.cprr_fast_mode
                    if graph_use_asm_cprr_verify
                    else metadata_fast_mode
                ),
                metadata_intra_batch_mode=(
                    self.cprr_intra_batch_mode
                    if graph_use_asm_cprr_verify
                    else metadata_intra_batch_mode
                ),
                nhead_override=(
                    self.cprr_kernel_heads if graph_use_asm_cprr_verify else None
                ),
                max_split_per_batch=(
                    self.max_split_per_batch if graph_use_asm_cprr_verify else None
                ),
            )
            logger.info(
                "aiter DCP cp verify graph buffers: use_asm_cprr_verify=%s num_draft_tokens=%s "
                "max_seqlen_qo=%d max_bs=%d nhead=%s intra=%s split_cap=%s "
                "reduce_partial_map=%d",
                graph_use_asm_cprr_verify,
                self.num_draft_tokens,
                max_seqlen_qo,
                max_bs,
                self.cprr_kernel_heads if graph_use_asm_cprr_verify else None,
                self.cprr_intra_batch_mode
                if graph_use_asm_cprr_verify
                else metadata_intra_batch_mode,
                self.max_split_per_batch if graph_use_asm_cprr_verify else None,
                self.reduce_partial_map.numel(),
            )

        else:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None

            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.use_sliding_window_kv_pool:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_swa_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )
            # SWA write-target buffer; refilled and bound onto forward_metadata
            # in init_forward_metadata_out_graph before each replay.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        verify_tokens_per_req: Optional[int],
    ):
        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        # DCP metadata which will be populated for MLA decode when dcp enabled
        local_kv_lens = None
        verify_token_table = None
        global_kv_indptr = None
        cprr_kv_indices = cprr_kv_indptr = cprr_empty_rows = None
        use_asm_cprr_verify = False

        swa_page_table = None
        # Unified verify never reads this host max (replay sizes from
        # max_context_len); torch.max(seq_lens).item() would sync every replay.
        max_kv_len = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else None
            if forward_mode.is_target_verify() and self._use_unified_verify
            else torch.max(seq_lens).item()
        )

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None or (
                self.use_triton_unified_attention and not self.use_mla
            ):
                max_num_blocks_per_seq = (
                    self.max_context_len + self.page_size - 1
                ) // self.page_size

                if not self.use_triton_unified_attention:
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )

                    if (
                        self.use_mla
                        and self.dcp_world_size > 1
                        and not forward_mode.is_idle()
                    ):
                        kv_lens = seq_lens[:bs].to(torch.int32).clone()
                        self._plan_dcp_decode_metadata(
                            kv_indptr,
                            kv_indices,
                            kv_lens,
                            seq_lens_cpu,
                            bs,
                        )
                else:
                    max_q_len = 1
                    kv_indices = self.cuda_graph_page_table

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table

                    if spec_info is not None:
                        self._build_unified_page_table_from_spec(
                            spec_info,
                            bs,
                            dest_buf=kv_indices,
                            swa_dest_buf=swa_page_table,
                        )
                    else:
                        page_indices = self.req_to_token[
                            req_pool_indices[:bs], :max_kv_len
                        ]

                        if self.use_sliding_window_kv_pool:
                            # AITER attention kernels require int32 page indices;
                            # full_to_swa_index_mapping is stored as int64.
                            swa_page_indices = (
                                self.swa_kv_pool.translate_loc_from_full_to_swa(
                                    page_indices
                                ).to(torch.int32)
                            )

                            page_indices = self._transform_table_1_to_real(page_indices)
                            swa_page_indices = self._transform_table_1_to_real(
                                swa_page_indices
                            )

                            new_rows = swa_page_indices.shape[0]
                            new_cols = swa_page_indices.shape[1]

                            kv_indices[:new_rows, :new_cols].copy_(page_indices)
                            swa_page_table = self.cuda_graph_swa_page_table
                            swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                        elif self.page_size > 1:
                            page_indices = self._transform_table_1_to_real(page_indices)
                            new_rows = page_indices.shape[0]
                            new_cols = page_indices.shape[1]
                            kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr_unified_decode[: bs + 1]

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if self._use_mla_decode_persist_metadata():
                    metadata_fast_mode, metadata_intra_batch_mode = (
                        self._mla_decode_metadata_modes()
                    )
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=metadata_fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=metadata_intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            assert verify_tokens_per_req is not None
            # MLA uses a fixed draft length (num_draft_tokens); the non-MLA
            # unified path derives it per batch from input_ids.
            tokens_per_req = (
                self.num_draft_tokens if self.use_mla else verify_tokens_per_req
            )
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * tokens_per_req,
                step=tokens_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                use_asm_cprr_verify = self._asm_cprr_supports_verify_shape(
                    self.num_draft_tokens
                )
                kv_lens = (
                    seq_lens
                    if (self.dcp_world_size > 1 and not use_asm_cprr_verify)
                    else seq_lens + self.num_draft_tokens
                )
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            # seq_lens_sum is None at capture (dummy seq_lens); only check on replay.
            if seq_lens_sum is not None:
                kv_indices_used = seq_lens_sum + (
                    self.num_draft_tokens * bs if self.use_mla else 0
                )
                assert_buffer_fits(
                    kv_indices_used,
                    kv_indices.numel(),
                    "aiter target_verify kv_indices",
                    bs=bs,
                    seq_lens_sum=seq_lens_sum,
                )
            num_token_blocks = self._kv_index_blocks(bs)
            create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
                TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]

            if self.use_mla and self.dcp_world_size > 1:
                global_kv_indptr = self.cuda_graph_global_kv_indptr[: bs + 1]
                global_kv_indptr.copy_(kv_indptr[: bs + 1])
                self._plan_dcp_decode_metadata(
                    kv_indptr,
                    kv_indices,
                    seq_lens[:bs].to(torch.int32).clone(),
                    None,
                    bs,
                    static_local_kv_lens_cpu=self.cuda_graph_dcp_static_local_kv_lens[
                        :bs
                    ],
                )
                n_rows = bs * self.num_draft_tokens
                (
                    verify_token_table,
                    local_kv_lens,
                ) = self._build_dcp_verify_token_table(
                    kv_indptr,
                    req_pool_indices,
                    bs,
                    self.num_draft_tokens,
                    self._dcp_graph_max_local_kv_len(),
                    out=self.cuda_graph_verify_token_table[:n_rows],
                    out_lens=self.cuda_graph_verify_local_kv_lens[:n_rows],
                )
                if use_asm_cprr_verify:
                    (
                        cprr_kv_indices,
                        cprr_kv_indptr,
                        cprr_empty_rows,
                    ) = self._build_dcp_verify_ragged_indices(
                        verify_token_table,
                        local_kv_lens,
                        bs,
                        self.num_draft_tokens,
                        out_indices=self.cuda_graph_cprr_kv_indices,
                        out_indptr=self.cuda_graph_cprr_kv_indptr,
                        out_empty=self.cuda_graph_cprr_empty_rows,
                    )

            if self.use_mla:
                max_q_len = self.num_draft_tokens
                if _use_mla_ps_kernel and (
                    self.dcp_world_size <= 1 or use_asm_cprr_verify
                ):
                    num_kv_splits = self.max_split_per_batch
                    is_cp_round_robin = self.dcp_world_size > 1

                    self.make_mla_meta_data(
                        qo_indptr,
                        cprr_kv_indptr if use_asm_cprr_verify else kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=(
                            self.cprr_fast_mode if use_asm_cprr_verify else fast_mode
                        ),
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=(
                            self.cprr_intra_batch_mode
                            if use_asm_cprr_verify
                            else intra_batch_mode
                        ),
                        is_cp_round_robin=is_cp_round_robin,
                        nhead_override=(
                            self.cprr_kernel_heads if use_asm_cprr_verify else None
                        ),
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    local_kv_lens=local_kv_lens,
                    verify_token_table=verify_token_table,
                    global_kv_indptr=global_kv_indptr,
                    use_asm_cprr_verify=use_asm_cprr_verify,
                    cprr_kernel_heads=self.cprr_kernel_heads
                    if use_asm_cprr_verify
                    else 0,
                    cprr_kv_indices=cprr_kv_indices,
                    cprr_kv_indptr=cprr_kv_indptr,
                    cprr_empty_rows=cprr_empty_rows,
                )
            else:
                max_q_len = verify_tokens_per_req
                if self._use_unified_verify:
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    page_table = self.cuda_graph_page_table[:bs]

                    swa_page_table = None

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = self.cuda_graph_swa_page_table.view(
                            -1, max_num_blocks_per_seq
                        )[:bs]

                    _page_table, _qo_indptr, _max_q_len, _swa_page_table = (
                        self._build_verify_unified_metadata(
                            bs,
                            seq_lens,
                            req_pool_indices,
                            verify_tokens_per_req,
                            page_table_dest=page_table,
                            swa_page_table_dest=swa_page_table,
                        )
                    )

                    max_kv_len_unified = max_num_blocks_per_seq * self.page_size
                    self.forward_metadata = ForwardMetadata(
                        None,
                        _page_table,
                        _qo_indptr,
                        kv_last_page_len,
                        _max_q_len,
                        max_kv_len_unified,
                        max_extend_len=_max_q_len,
                        swa_page_table=_swa_page_table,
                    )
                else:
                    custom_mask = self.cuda_graph_custom_mask
                    custom_mask[: spec_info.custom_mask.shape[0]] = (
                        spec_info.custom_mask
                    )
                    seq_mask_len = max_q_len * (seq_lens + max_q_len)
                    mask_indptr = self.mask_indptr[: bs + 1]
                    mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        max_kv_len,
                        custom_mask=custom_mask,
                        mask_indptr=mask_indptr,
                        max_extend_len=max_q_len,
                    )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            seq_lens = seq_lens[:bs]
            num_tokens_per_req = self._resolve_v2_num_draft_tokens()
            extend_lens = torch.full(
                (bs,), num_tokens_per_req, dtype=torch.int32, device=seq_lens.device
            )

            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            num_token_blocks = self._kv_index_blocks(bs)
            create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
                TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
            )

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_req

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1 if self.num_draft_tokens is None else self.num_draft_tokens

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # AITER verify path does not require post-draft buffer patching currently.
        # This override prevents overlap-plan stream mode from failing with the
        # base class NotImplementedError.
        pass

    def _use_fused_fp8_kv_write(self, layer: RadixAttention) -> bool:
        # Fused write reuses K's num_heads/head_dim for V, so it needs FP8,
        # non-MLA, non-SWA, and matching K/V head count + head_dim.
        return (
            self.kv_cache_dtype == fp8_dtype
            and not self.use_mla
            and not self.use_sliding_window_kv_pool
            and layer.tp_k_head_num == layer.tp_v_head_num
            and layer.qk_head_dim == layer.v_head_dim
        )

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ) -> None:
        """Build the chunked-prefix route's PS metadata, once per forward.

        Its attentions each attend a subset of the sequence, while
        init_forward_metadata's PS metadata covers the whole sequence.
        """
        # The one-shot core calls this hook too, with num_prefix_chunks 0.
        if not forward_batch.num_prefix_chunks or not self.use_fp8_prefill_attn:
            return

        # Both passes query the extend tokens, so they share qo_indptr.
        extend_lens_cpu = torch.tensor(
            forward_batch.extend_seq_lens_cpu, dtype=torch.int32
        )
        qo_indptr_cpu = torch.zeros(len(extend_lens_cpu) + 1, dtype=torch.int32)
        qo_indptr_cpu[1:] = torch.cumsum(extend_lens_cpu, dim=0)
        qo_indptr = self.forward_metadata.qo_indptr
        max_q_len = self.forward_metadata.max_q_len

        # disable_flashinfer_ragged asks for no ragged pass. The skip-prefix
        # pass is this backend's ragged one, so leave it on the varlen path.
        if not disable_flashinfer_ragged:
            self.forward_metadata.chunked_skip_prefix_ps_metadata = (
                self._build_prefill_ps_metadata(
                    qo_indptr=qo_indptr,
                    qo_indptr_cpu=qo_indptr_cpu,
                    kv_indptr=qo_indptr,
                    kv_indptr_cpu=qo_indptr_cpu,
                    kv_lens_cpu=extend_lens_cpu,
                    num_kv_tokens=int(qo_indptr_cpu[-1]),
                    max_q_len=max_q_len,
                    is_causal=True,
                    need_lse=True,
                )
            )

        metadatas: list[Optional[MlaPrefillPsMetadata]] = []
        for i in range(forward_batch.num_prefix_chunks):
            chunk_lens_cpu = forward_batch.prefix_chunk_seq_lens_cpu[i].to(torch.int32)
            # An empty kv range has no PS metadata shape.
            # _forward_extend_prefix_chunk falls back to varlen there.
            if forward_batch.prefix_chunk_has_zero_kv[i]:
                metadatas.append(None)
                continue
            chunk_indptr_cpu = torch.zeros(len(chunk_lens_cpu) + 1, dtype=torch.int32)
            chunk_indptr_cpu[1:] = torch.cumsum(chunk_lens_cpu, dim=0)
            metadatas.append(
                self._build_prefill_ps_metadata(
                    qo_indptr=qo_indptr,
                    qo_indptr_cpu=qo_indptr_cpu,
                    kv_indptr=forward_batch.prefix_chunk_cu_seq_lens[i],
                    kv_indptr_cpu=chunk_indptr_cpu,
                    kv_lens_cpu=chunk_lens_cpu,
                    num_kv_tokens=forward_batch.prefix_chunk_num_tokens[i],
                    max_q_len=max_q_len,
                    # Every key in a prefix chunk precedes every extend token.
                    is_causal=False,
                    need_lse=True,
                )
            )
        self.forward_metadata.chunked_prefix_ps_metadatas = metadatas

    def _build_prefill_ps_metadata(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        num_kv_tokens: int,
        max_q_len: int,
        is_causal: bool,
        need_lse: bool,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_indptr_cpu: Optional[torch.Tensor] = None,
        exact_partial_count: bool = True,
    ) -> MlaPrefillPsMetadata:
        """Plan one asm-prefill PS metadata over the key set the caller names."""
        (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        ) = self.make_mla_prefill_ps_meta_data_buffer(
            len(kv_lens_cpu), max_q_len, self._prefill_qlen_granularity
        )
        self.make_mla_prefill_ps_meta_data(
            qo_indptr if qo_indptr_cpu is None else qo_indptr_cpu,
            kv_indptr if kv_indptr_cpu is None else kv_indptr_cpu,
            kv_lens_cpu,
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            is_causal=is_causal,
            need_lse=need_lse,
        )
        if exact_partial_count:
            # One D2H sync per PS metadata, once per forward rather than per
            # layer. get_ps_metadata_v1 already pays several.
            num_partial_tiles = int(reduce_indptr[-1].item())
            if need_lse:
                assert num_partial_tiles > 0, (
                    "the scheduler emitted no partial tiles, so mla_reduce_v1 "
                    "would write no LSE for this partition"
                )
            if num_partial_tiles:
                max_partial_row = int(reduce_partial_map[:num_partial_tiles].max())
                assert (
                    max_partial_row + _PREFILL_TILE_Q
                    <= num_partial_tiles * _PREFILL_TILE_Q
                ), (
                    f"reduce_partial_map reaches row {max_partial_row} and the "
                    f"tile is {_PREFILL_TILE_Q} rows, but the partial buffers "
                    f"hold only {num_partial_tiles * _PREFILL_TILE_Q} rows"
                )
        else:
            num_partial_tiles = reduce_partial_map.size(0)
        return MlaPrefillPsMetadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            # The k/v handed to the kernel are contiguous and in key order, so
            # the page table is the identity.
            kv_indices=self._get_arange(num_kv_tokens),
            work_metadata=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            max_q_len=max_q_len,
            is_causal=is_causal,
            need_lse=need_lse,
            num_partial_tiles=num_partial_tiles,
        )

    def _forward_extend_prefix_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        idx = forward_batch.prefix_chunk_idx
        ps_metadatas = self.forward_metadata.chunked_prefix_ps_metadatas
        if ps_metadatas is not None and ps_metadatas[idx] is not None:
            return self._mla_fp8_prefill_attn_ps(q, k, v, layer, ps_metadatas[idx])

        output, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            self.forward_metadata.qo_indptr,
            forward_batch.prefix_chunk_cu_seq_lens[idx],
            self.forward_metadata.max_q_len,
            forward_batch.prefix_chunk_max_seq_lens[idx],
            softmax_scale=layer.scaling,
            causal=False,
            return_lse=True,
        )[:2]
        # the merge_state_triton needs lse layout [tokens, heads].
        # See https://github.com/sgl-project/sglang/blob/98fce73d5bd0a25afe7d68443d314190b1c47e64/python/sglang/kernels/ops/attention/merge_state.py#L13-L15
        return output, lse.transpose(0, 1).contiguous()

    def _forward_extend_skip_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        ps = self.forward_metadata.chunked_skip_prefix_ps_metadata
        if ps is not None:
            return self._mla_fp8_prefill_attn_ps(q, k, v, layer, ps)

        qo_indptr = self.forward_metadata.qo_indptr
        max_q_len = self.forward_metadata.max_q_len
        output, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            qo_indptr,
            qo_indptr,
            max_q_len,
            max_q_len,
            softmax_scale=layer.scaling,
            causal=True,
            return_lse=True,
        )[:2]
        # the merge_state_triton needs lse layout [tokens, heads].
        # See https://github.com/sgl-project/sglang/blob/98fce73d5bd0a25afe7d68443d314190b1c47e64/python/sglang/kernels/ops/attention/merge_state.py#L13-L15
        return output, lse.transpose(0, 1).contiguous()

    @staticmethod
    def _reject_target_verify_cross_layer_kv(k, v):
        """Reject cross-layer KV sharing on the legacy ragged target-verify path.

        Cross-layer KV sharing (e.g. Gemma4) passes ``k=v=None`` so the kernel
        reads K/V from the pool. The legacy ``extend_attention_fwd`` path takes
        ragged K/V and has no pool-reading fallback, so it would raise an opaque
        ``AttributeError`` on ``.contiguous``. Fail loudly instead. Inert when
        real K/V is passed.
        """
        if k is None or v is None:
            raise ValueError(
                "aiter target_verify does not support cross-layer KV "
                "sharing (k/v are None). Use the unified verify path "
                "(speculative_eagle_topk=1 and SGLANG_AITER_UNIFIED_VERIFY=1)."
            )

    @staticmethod
    def _resolve_swa_kv_pool(model_runner):
        """Return the SWAKVPool to translate against, or None for non-SWA models.

        EAGLE draft workers share the target allocator for token bookkeeping but
        own a separate draft KV pool, so the target allocator's SWA mapping must
        not be used for them. FROZEN_KV MTP is the exception: its draft path reads
        target KV directly, so it still needs the allocator pool when the active
        pool is not itself an SWAKVPool. Mirrors
        ``TRTLLMHAAttnBackend._resolve_swa_kv_pool``.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, SWAKVPool):
            return active_pool
        if getattr(model_runner, "is_draft_worker", False):
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None
        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, SWAKVPool) else None

    @staticmethod
    def _reject_paged_decode_sliding_window(layer):
        """Reject sliding-window layers on the aiter paged-decode path.

        ``paged_attention_ragged`` takes no sliding-window argument, so a
        sliding-window layer routed to it would attend over the full context and
        silently return wrong results. Raise instead of silently dropping the
        window. Layers with ``sliding_window_size`` unset or -1 are unaffected.
        """
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise ValueError(
                "aiter paged decode cannot honor sliding-window "
                f"attention (layer {layer.layer_id} has "
                f"sliding_window_size={layer.sliding_window_size}). "
                "Enable the unified attention path "
                "(SGLANG_USE_AITER_UNIFIED_ATTN=1) or select a "
                "different attention backend."
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        self.logits_soft_cap = layer.logit_cap

        if forward_batch.attn_attend_prefix_cache:
            return self._forward_extend_prefix_chunk(q, k, v, layer, forward_batch)

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 5D pool cannot be reshaped to the 4D paged view used by
                # launch_reshape_and_cache_flash; always route through
                # set_kv_buffer which dispatches to the SHUFFLE 5D writer.
                if self.kv_cache_is_vectorized_5d:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc.for_layer(
                            forward_batch,
                            layer,
                            swa_loc=self.forward_metadata.swa_out_cache_loc,
                        ),
                        k,
                        v,
                        k_descale,
                        v_descale,
                    )
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                elif (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):
                    token_to_kv_pool = self.token_to_kv_pool
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    slot_mapping_swa = self.swa_kv_pool.full_to_swa_index_mapping

                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        (
                            slot_mapping_swa.long()
                            if layer.sliding_window_size > 0
                            else None
                        ),
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                elif self.use_mla:
                    if self.dcp_world_size > 1:
                        kv_lora_rank = v.shape[-1]
                        self.token_to_kv_pool.set_mla_kv_buffer(
                            layer,
                            KVWriteLoc.for_layer(forward_batch, layer),
                            k[..., :kv_lora_rank],
                            k[..., kv_lora_rank:],
                        )
                    else:
                        self.token_to_kv_pool.set_kv_buffer(
                            layer,
                            KVWriteLoc.for_layer(forward_batch, layer),
                            k,
                            v,
                        )
                elif self._use_fused_fp8_kv_write(layer):
                    # FP8: fuse bf16->fp8 cast + paged write in one kernel.
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                else:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc.for_layer(
                            forward_batch,
                            layer,
                            swa_loc=self.forward_metadata.swa_out_cache_loc,
                        ),
                        k,
                        v,
                        k_descale,
                        v_descale,
                    )

        if self.use_mla:
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank

            if (
                forward_batch.forward_mode.is_target_verify()
                and self.dcp_world_size > 1
            ):
                return self._forward_verify_dcp(q, k, layer, k_descale)

            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mha_return_lse:
                    return self._forward_extend_skip_prefix(q, k, v, layer)
                if self.dcp_world_size > 1:
                    if self.use_fp8_prefill_attn:
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    return flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        forward_batch.attn_dcp_metadata.dcp_kv_indptr,
                        max_q_len,
                        max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if self.use_fp8_prefill_attn:
                        output = self.mla_fp8_prefill_attn(
                            q,
                            k,
                            v,
                            layer,
                        )
                    else:
                        output = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            qo_indptr,
                            max_q_len,
                            max_q_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                    return output
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )

                    if self.kv_cache_dtype == fp8_dtype:
                        dtype = q.dtype

                        kvc = kvc.to(dtype)
                        k_pe = k_pe.to(dtype)

                    if (
                        self.use_fp8_prefill_attn
                        and layer.kv_b_proj.weight.dtype == torch.uint8
                    ):
                        # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
                        # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
                        # directly in FP8, avoiding a separate elementwise cast
                        k, v = layer.kv_b_proj(
                            (
                                kvc.squeeze(1),
                                k_pe.expand(-1, layer.tp_k_head_num, -1),
                                qk_nope_head_dim,
                                layer.v_head_dim,
                                fp8_dtype,
                            )
                        )[0]
                    else:
                        kv = layer.kv_b_proj(kvc.contiguous())[0]

                        kv = kv.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        k, v = torch.split(
                            kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        k = torch.cat(
                            [
                                k,
                                torch.broadcast_to(
                                    k_pe,
                                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                                ),
                            ],
                            dim=-1,
                        )

                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    if self.use_fp8_prefill_attn:
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    else:
                        return flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            max_q_len,
                            max_kv_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )

                else:
                    if self.head_pad_mode == "zero":
                        # 12 heads/rank (Kimi-K3 TP8): zero-pad q to 16 heads,
                        # run the aiter MLA prefill kernel, slice 12 back.
                        q_in = self._zero_pad_mla_q_heads(q, layer)
                        o = q.new_empty(
                            (q.shape[0], self.num_head_padded, layer.v_head_dim)
                        )
                    else:
                        q_in = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        if layer.qk_head_dim != layer.v_head_dim:
                            o = q.new_empty(
                                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                            )
                        else:
                            o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q_in,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        (
                            o.view(-1, self.num_head_padded, layer.v_head_dim)
                            if self.head_pad_mode == "zero"
                            else o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                        ),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    if self.head_pad_mode == "zero":
                        return (
                            o[:, : layer.tp_q_head_num, :]
                            .contiguous()
                            .view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                if prefer_mla_gluon_decode(
                    head_pad_mode=getattr(self, "head_pad_mode", "none"),
                    num_head=getattr(self, "num_head", layer.tp_q_head_num),
                    kv_cache_dtype=self.kv_cache_dtype,
                ) and not (
                    _use_mla_ps_kernel
                    and self.mla_verify_backend == "asm"
                    and self._asm_ps_supports_qlen(self.forward_metadata.max_q_len or 1)
                ):
                    return mla_gluon_decode(
                        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        k_buffer=K_Buffer,
                        layer=layer,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        sm_scale=layer.scaling,
                        kv_scale=self._resolve_fp8_kv_scale_float(layer, k_descale),
                        min_kv_seq_len=self._resolve_mla_gluon_min_kv_seq_len(
                            forward_batch
                        ),
                        qlen=self.forward_metadata.max_q_len or 1,
                    )

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                o = self._mla_decode_fwd_with_head_pad(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    layer,
                    qo_indptr=self.forward_metadata.qo_indptr,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=k_descale,
                    kv_scale=k_descale,
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                return o
            elif forward_batch.forward_mode.is_draft_extend_v2():
                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                if self.forward_metadata.run_graph is not True:
                    bs, q_pad, q_mask = pad_sequence_with_mask(
                        q.view(q.shape[0], -1),
                        qo_indptr[:-1],
                        forward_batch.extend_seq_lens,
                        self.forward_metadata.max_q_len,
                    )
                    o = self._mla_decode_fwd_with_head_pad(
                        q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )

                    total_valid_q = int(qo_indptr[-1].item())
                    return o[:total_valid_q]
                else:
                    o = self._mla_decode_fwd_with_head_pad(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            if forward_batch.forward_mode.is_target_verify():
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

                # target_verify goes through unified_attention when topk == 1
                # (the linear draft chain gives a pure causal mask). MLA and
                # draft_extend still use the legacy extend_attention_fwd path.
                if (
                    self._use_unified_verify
                    and forward_batch.forward_mode.is_target_verify()
                ):
                    k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    page_table = self.forward_metadata.kv_indices
                    max_kv_len = page_table.shape[1] * self.page_size

                    window_size = (-1, -1)

                    if (
                        layer.sliding_window_size is not None
                        and layer.sliding_window_size > -1
                    ):
                        window_size = (layer.sliding_window_size - 1, 0)
                        if self.forward_metadata.swa_page_table is not None:
                            page_table = self.forward_metadata.swa_page_table

                    q_unified = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    k_unified = k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    )
                    v_unified = v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    )
                    # Shape gate, not a model gate: the kernel is tuned for a 16:1
                    # GQA ratio (block_m=32 -> block_q=2 packs two draft tokens per
                    # tile), and head_dim 256 is the only validated size. The kv-head
                    # count itself is free (kv_head_idx = program_id(1)), but K and V
                    # must share it -- the kernel has a single kv-head grid dim.
                    # Qwen3.5-397B-A17B at TP1/TP2 is the only config known to
                    # match today; any model with the same shapes qualifies.
                    # TODO(yichiche): relax the head_dim gate once other sizes are
                    # measured -- the wrapper passes HEAD_SIZE_PADDED unpadded, so
                    # only powers of 2 work (128/256 OK, 192 is not).
                    num_queries_per_kv = layer.tp_q_head_num // layer.tp_k_head_num
                    use_unified_attention_3d_mtp = (
                        is_gfx95_supported()
                        and 1 < self.forward_metadata.max_q_len <= 4
                        and max_kv_len > 512
                        and (
                            num_queries_per_kv == 16
                            or (
                                num_queries_per_kv == 8
                                # GQA 8 is validated for the 4-token verify
                                # block; shorter draft configs keep the old path
                                and self.forward_metadata.max_q_len == 4
                                and asm_verify_attn_enabled()
                            )
                        )
                        and layer.tp_k_head_num == layer.tp_v_head_num
                        and layer.qk_head_dim == 256
                        and layer.v_head_dim == 256
                        and self.page_size == 16
                        and q_unified.dtype == torch.bfloat16
                        and k_unified.dtype == fp8_dtype
                        and window_size == (-1, -1)
                        and not layer.logit_cap
                        and sinks is None
                    )
                    if use_unified_attention_3d_mtp:
                        unified_attention_3d_mtp_func(
                            q=q_unified,
                            k=k_unified,
                            v=v_unified,
                            out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                            cu_seqlens_q=self.forward_metadata.qo_indptr,
                            seqused_k=(
                                forward_batch.seq_lens + self.forward_metadata.max_q_len
                            ),
                            max_seqlen_q=self.forward_metadata.max_q_len,
                            max_seqlen_k=max_kv_len,
                            softmax_scale=layer.scaling,
                            block_table=page_table,
                            k_descale=k_descale,
                            v_descale=v_descale,
                        )
                        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

                    # GQA-packing fix: do NOT expand the single KV head to
                    # tp_q_head_num. Passing K/V with the true kv-head count (exactly
                    # like forward_decode) lets unified_attention derive
                    # num_queries_per_kv = tp_q_head_num (the GQA group) and pack all Q
                    # heads against one KV load. The old stride-0 .expand() made the
                    # wrapper see num_kv_heads=tp_q_head_num -> num_queries_per_kv=1 ->
                    # full MHA tiling (~7x more KV traffic at long context; trace
                    # signature num_query_heads_16/num_queries_per_kv_1). GQA head
                    # mapping here is identical to the proven decode path.

                    # The seq_lens + draft_num add has to run INSIDE the graph
                    # region; a host-side pre-add would allocate a new tensor
                    # each replay and break the captured pointer.
                    unified_attention(
                        q=q_unified,
                        k=k_unified,
                        v=v_unified,
                        out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        cu_seqlens_q=self.forward_metadata.qo_indptr,
                        seqused_k=(
                            forward_batch.seq_lens + self.forward_metadata.max_q_len
                        ),
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        max_seqlen_k=max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                        window_size=window_size,
                        block_table=page_table,
                        softcap=layer.logit_cap,
                        q_descale=None,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        sinks=sinks,
                    )
                    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

                self._reject_target_verify_cross_layer_kv(k, v)

                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.custom_mask,
                    True,  # causal
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    1.0,  # k_scale
                    1.0,  # v_scale
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            # draft_extend (EAGLE-v2 KV catch-up): short Q, long paged KV. Route
            # it to the #30105 unified_attention path. Gating on
            # _use_unified_draft_extend (not _use_unified_verify) lets the
            # default aiter backend take it; topk>1 and the kill switch fall
            # through to the padded CK path below.
            if (
                self._use_unified_draft_extend
                and forward_batch.forward_mode.is_draft_extend_v2()
            ):
                bs = forward_batch.batch_size
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                page_table, swa_page_table = self._build_unified_page_table_from_spec(
                    self.forward_metadata, bs
                )
                pt = page_table
                de_window = (-1, -1)
                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    de_window = (layer.sliding_window_size - 1, 0)
                    if swa_page_table is not None:
                        pt = swa_page_table
                kv_indptr = self.forward_metadata.kv_indptr
                # seqused_k MUST be int64 (kv_indptr is int32, so the diff is
                # int32 and has to be widened). unified_attention derives the
                # per-tile KV addresses from this dtype: with an int32
                # seqused_k the whole K/V offset chain stays 32-bit and wraps
                # once a per-layer KV buffer reaches 2 GiB, silently returning
                # NaN. The verify (seq_lens + max_q_len) and decode
                # (seq_lens) call sites pass int64 for the same reason.
                seqused_k = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int64)
                # draft_extend has ragged short Q (accepted tokens, 1..4); the
                # asm verify kernel serves it via in-kernel tail alignment.
                if (
                    is_gfx95_supported()
                    and de_window == (-1, -1)
                    and not layer.logit_cap
                    and sinks is None
                    and self.page_size == 16
                    and layer.qk_head_dim == 256
                    and layer.v_head_dim == 256
                    and layer.tp_k_head_num == layer.tp_v_head_num
                    and 1 <= self.forward_metadata.max_q_len <= 4
                    and unified_attention_3d_mtp_ragged_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        k=k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v=v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        cu_seqlens_q=self.forward_metadata.qo_indptr,
                        seqused_k=seqused_k,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        max_seqlen_k=pt.shape[1] * self.page_size,
                        softmax_scale=layer.scaling,
                        block_table=pt,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )
                ):
                    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=seqused_k,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=pt.shape[1] * self.page_size,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=de_window,
                    block_table=pt,
                    softcap=layer.logit_cap,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            bs0 = forward_batch.batch_size + 1
            q_descale = None

            window_size = (-1, -1)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)

            # Whether the paged asm prefill kernel can serve this layer on this
            # batch. Both the gather branch below and the paged branch at the
            # end of this method key off it, so the two stay mutually exclusive
            # by construction: one flag, read twice, cannot drift out of sync
            # the way two copies of the same condition would.
            paged_asm_available = (
                self.forward_metadata.paged_kv_view is not None
                and _paged_prefill_asm_supports_gqa(
                    layer.tp_q_head_num, layer.tp_k_head_num
                )
            )

            if (
                envs.SGLANG_AITER_ASM_PREFILL_HD128.get()
                and is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and not layer.is_cross_attention
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == layer.v_head_dim == 128
                and layer.tp_k_head_num == layer.tp_v_head_num
                and self.kv_cache_dtype == fp8_dtype
                and q.dtype == torch.bfloat16
                and _aiter_fp8_asm_supports_gqa(
                    layer.tp_q_head_num, layer.tp_k_head_num
                )
                and not self.kv_cache_is_vectorized_5d
                and self.forward_metadata.max_kv_len is not None
            ):
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                tok_idx, cu_k = self._asm_context_prefill_indices(
                    forward_batch, forward_batch.batch_size, k_cache.shape[0]
                )
                if tok_idx is not None:
                    # Read the already quantized cache for first and later
                    # chunks alike. Raw K/V may come from different producers;
                    # casting them here would skip or repeat their KV scaling.
                    k_gather = (
                        k_cache.view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(fp8_dtype)
                    )
                    v_gather = (
                        v_cache.view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(fp8_dtype)
                    )
                    o = flash_attn_varlen_fp8_pertensor_func(
                        q.contiguous().view(-1, layer.tp_q_head_num, 128).to(fp8_dtype),
                        k_gather,
                        v_gather,
                        self.k_scale.reshape(1),  # Q is cast at unit scale.
                        k_descale.reshape(1),
                        v_descale.reshape(1),
                        self.qo_indptr[:bs0],
                        cu_k,
                        self.forward_metadata.max_q_len,
                        int(self.forward_metadata.max_kv_len),
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o.to(self.input_dtype).view(-1, layer.tp_q_head_num * 128)

            # Context-chunk prefill (extend batches WITH a prefix) via the
            # gfx950 ASM fp8 varlen fmha. The ck_tile paged batch_prefill runs
            # at ~15% FP8 MFU at these shapes while the ASM kernel is ~3.5x
            # faster; gathering the paged fp8 KV into a contiguous varlen
            # buffer costs only ~20 us per layer at 70k context. The no-prefix
            # first chunk already takes the ASM branch below.
            # This applies to Qwen3.5 full-attention layers only currently,
            # Other configurations fall through to the attention paths below.
            #
            # Yields to the paged asm kernel when there is one, since that reads
            # the cache in place and does the same attention with no gather at
            # all. This is not a blanket disable: on any batch or layer that
            # kernel cannot serve, the flag is False and this branch runs
            # exactly as it does today.
            if (
                is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 256
                and layer.v_head_dim == 256
                and self.kv_cache_dtype == fp8_dtype
                and _aiter_fp8_asm_supports_gqa(
                    layer.tp_q_head_num, layer.tp_k_head_num
                )
                and not self.kv_cache_is_vectorized_5d
                and self.forward_metadata.max_kv_len is not None
                and not paged_asm_available
            ):
                bs = forward_batch.batch_size
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                tok_idx, cu_k = self._asm_context_prefill_indices(
                    forward_batch, bs, k_cache.shape[0]
                )
                if tok_idx is not None:
                    hk = layer.tp_k_head_num * layer.qk_head_dim
                    hv = layer.tp_v_head_num * layer.v_head_dim
                    # uint8 view: index_select is not implemented for fp8.
                    k_gather = (
                        k_cache.view(-1, hk)
                        .view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(k_cache.dtype)
                        .view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    )
                    v_gather = (
                        v_cache.view(-1, hv)
                        .view(torch.uint8)
                        .index_select(0, tok_idx)
                        .view(v_cache.dtype)
                        .view(-1, layer.tp_v_head_num, layer.v_head_dim)
                    )
                    fp8_q_descale = (
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    )
                    o = flash_attn_varlen_fp8_pertensor_func(
                        q.contiguous()
                        .view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        .to(fp8_dtype),
                        k_gather,
                        v_gather,
                        fp8_q_descale.reshape(1),
                        k_descale.reshape(1),
                        v_descale.reshape(1),
                        self.qo_indptr[:bs0],
                        cu_k,
                        self.forward_metadata.max_q_len,
                        int(self.forward_metadata.max_kv_len),
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    if o.dtype != self.input_dtype:
                        o = o.to(self.input_dtype)
                    return o.view(-1, layer.tp_q_head_num * layer.qk_head_dim)

            if (
                is_gfx95_supported()
                and forward_batch.forward_mode.is_extend()
                and forward_batch.extend_prefix_lens_cpu is not None
                and not any(forward_batch.extend_prefix_lens_cpu)
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 256
                and layer.v_head_dim == 256
                and self.kv_cache_dtype == fp8_dtype
                and _aiter_fp8_asm_supports_gqa(
                    layer.tp_q_head_num, layer.tp_k_head_num
                )
            ):
                q_c = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                k_c = k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim)
                v_c = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
                fp8_q_descale = (
                    layer.k_scale if layer.k_scale is not None else self.k_scale
                )
                o = flash_attn_varlen_fp8_pertensor_func(
                    q_c.to(fp8_dtype),
                    k_c.to(fp8_dtype),
                    v_c.to(fp8_dtype),
                    fp8_q_descale.reshape(1),
                    k_descale.reshape(1),
                    v_descale.reshape(1),
                    self.qo_indptr[:bs0],
                    self.qo_indptr[:bs0],
                    self.forward_metadata.max_q_len,
                    self.forward_metadata.max_q_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
                if o.dtype != self.input_dtype:
                    o = o.to(self.input_dtype)
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if self.kv_cache_is_vectorized_5d:
                return forward_extend_vectorized_5d(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    bs0,
                    window_size,
                    sinks,
                )

            if self.use_triton_unified_attention:
                # unified_attention has no head_dim cap; route extend through it
                # so Gemma-4's 512-wide full-attention layers don't hit the CK
                # `head dimension at most 256` assert.
                return self._forward_extend_unified(
                    q,
                    layer,
                    forward_batch,
                    bs0,
                    window_size,
                    sinks,
                    k_descale,
                    v_descale,
                )

            # NHD path — original aiter paged batch_prefill.
            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                q = q.to(fp8_dtype)
                q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale

            k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            page_table = self.forward_metadata.kv_indices
            if (
                layer.sliding_window_size is not None
                and layer.sliding_window_size > -1
                and self.forward_metadata.swa_page_table is not None
            ):
                page_table = self.forward_metadata.swa_page_table

            extra_kwargs = {}
            attn_out = getattr(forward_batch, "_attn_output", None)
            if attn_out is not None and q.dtype != fp8_dtype:
                extra_kwargs["out"] = attn_out.view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

            kv_indptr_arg = self.forward_metadata.kv_indptr[:bs0]
            if (
                paged_asm_available
                and page_table is self.forward_metadata.kv_indices
                and window_size == (-1, -1)
                and sinks is None
                and self.logits_soft_cap == 0.0
                and layer.qk_head_dim == 256
                and layer.v_head_dim == 256
                and self.kv_cache_dtype == fp8_dtype
            ):
                # These must match aiter's asm guard: there is no CK arm for 4D
                # LINEAR page-64 fp8 hd256, so a shape the guard rejects raises
                # rather than falling back. The arch half of the guard is
                # checked where paged_kv_view is built.
                page_indptr, page_ids, last_page_len = (
                    self.forward_metadata.paged_kv_view
                )
                kv_indptr_arg = page_indptr[:bs0]
                page_table = page_ids
                k_cache = k_cache.view(-1, self.page_size, *k_cache.shape[-2:])
                v_cache = v_cache.view(-1, self.page_size, *v_cache.shape[-2:])
                extra_kwargs["kv_last_page_lens"] = last_page_len[: bs0 - 1]

            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                kv_indptr_arg,
                page_table,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
                window_size=window_size,
                sink_ptr=sinks,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                **extra_kwargs,
            )

            # The fp8bf16 aiter prefill kernel returns bf16 even when the
            # model computes in fp16. Cast back so the attention output keeps
            # the same dtype as the rest of the model activations.
            if o.dtype != self.input_dtype:
                o = o.to(self.input_dtype)

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale

        if save_kv_cache:
            # SHUFFLE 5D pool path — see forward_extend for rationale.
            if self.kv_cache_is_vectorized_5d:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_batch(
                        forward_batch,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    k_descale,
                    v_descale,
                )
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            elif self.use_triton_unified_attention and self.use_sliding_window_kv_pool:
                token_to_kv_pool = self.token_to_kv_pool
                k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    slot_mapping_swa.long() if layer.sliding_window_size > 0 else None,
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            elif self.use_mla:
                # MLA pool has its own set_kv_buffer (no scale args).
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_batch(forward_batch),
                    k,
                    v,
                )
            elif self._use_fused_fp8_kv_write(layer):
                # FP8: fuse bf16->fp8 cast + paged write in one kernel.
                token_to_kv_pool = self.token_to_kv_pool
                k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_batch(
                        forward_batch,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    k_descale,
                    v_descale,
                )

        if self.use_mla:
            if self.dcp_world_size > 1 and not forward_batch.forward_mode.is_idle():
                k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                return self._forward_decode_dcp(q, k_buffer, layer, k_descale)

            o = self._forward_mla_decode(q, layer, forward_batch, k_descale)
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            if layer.qk_head_dim != layer.v_head_dim:
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=self.input_dtype,
                )
            else:
                o = torch.empty_like(q, dtype=self.input_dtype)

            if self.kv_cache_is_vectorized_5d:
                # SHUFFLE 5D pool: pa_decode_gluon for full + SWA layers
                # (see :func:`aiter_utils.forward_decode_vectorized_5d`
                # for the dispatch rationale).
                forward_decode_vectorized_5d(
                    self, q, layer, forward_batch, k_cache, v_cache, o, sinks
                )
            elif self.use_triton_unified_attention:
                bs = forward_batch.batch_size
                window_size = (-1, -1)
                page_table = self.forward_metadata.kv_indices

                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    window_size = (layer.sliding_window_size - 1, 0)
                    if self.forward_metadata.swa_page_table is not None:
                        page_table = self.forward_metadata.swa_page_table

                max_kv_len = page_table.shape[1] * self.page_size
                # q_len==1 decode (incl. EAGLE draft decode steps) via the asm
                # verify kernel (in-kernel tail alignment; static shapes, so
                # the path is cuda-graph capture safe). Must run BEFORE the
                # scaled_fp8_quant below: the asm kernel takes bf16 Q.
                if (
                    is_gfx95_supported()
                    and self.forward_metadata.max_q_len == 1
                    and window_size == (-1, -1)
                    and sinks is None
                    and self.page_size == 16
                    and layer.qk_head_dim == 256
                    and layer.v_head_dim == 256
                    and layer.tp_k_head_num == layer.tp_v_head_num
                    and unified_attention_3d_mtp_decode_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        k=k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v=v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        seqused_k=forward_batch.seq_lens,
                        max_seqlen_k=max_kv_len,
                        softmax_scale=layer.scaling,
                        block_table=page_table,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )
                ):
                    return o
                q_descale = None
                if self.kv_cache_dtype == fp8_dtype:
                    q_descale = (
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    )
                    q, _ = scaled_fp8_quant(q, q_descale)

                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=forward_batch.seq_lens,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=window_size,
                    block_table=page_table,
                    softcap=0,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
            else:
                self._reject_paged_decode_sliding_window(layer)
                # Drop FP8 KV upcast: keep paged cache in native FP8 and use ``fp8_e4m3`` for
                # in-kernel dequant in ``paged_attention_ragged``. (HIP maps CLI e5m2/e4m3 to
                # ``fp8_dtype``; aiter has no ``fp8_e5m2`` string.)
                aiter_kv_str = self._get_aiter_paged_ragged_kv_cache_dtype()

                paged_attention_ragged(
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.workspace_buffer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
                    self.scale,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.kv_last_page_len,
                    1,
                    self.max_num_partitions,
                    None,
                    aiter_kv_str,
                    "NHD",
                    self.logits_soft_cap,
                    self.k_scale,
                    self.v_scale,
                    None,
                    _AITER_PARTITION_SIZE_ROCM,
                )

        return o


class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_index_translator = model_runner.kv_index_translator
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        *,
        plan: KVLocPlan,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        *,
        plan: KVLocPlan,
    ):
        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )

            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            extend_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices=req_pool_indices,
                    paged_kernel_lens=paged_kernel_lens,
                    paged_kernel_lens_sum=paged_kernel_lens_sum,
                    translator=self.kv_index_translator,
                    plan=plan,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_index_translator = model_runner.kv_index_translator
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
        *,
        plan: KVLocPlan,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
        *,
        plan: KVLocPlan,
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices=req_pool_indices,
                    paged_kernel_lens=kv_lens,
                    paged_kernel_lens_sum=kv_lens_sum,
                    translator=self.kv_index_translator,
                    plan=plan,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    topk=topk,
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = get_schedule().page_size
        self.kv_index_translator = model_runner.kv_index_translator

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        num_token_blocks = (
            kv_indices_num_token_blocks(
                self.pool_len, self.speculative_num_steps * num_seqs * self.topk
            )
            if self.max_context_len >= _KV_INDEX_BLOCKS_MIN_CONTEXT
            else 1
        )
        src = self.kv_index_translator.read_source(
            forward_batch.kv_loc_plan,
            req_pool_indices=forward_batch.req_pool_indices,
            bs=num_seqs,
        )
        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps * num_token_blocks, num_seqs, self.topk)
        ](
            src.row_ids,
            src.ids,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            src.row_stride,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
            # A single token block is the historical launch; NUM_STEPS=0 keeps
            # its 128-wide program instead of the token-block specialization.
            NUM_STEPS=self.speculative_num_steps if num_token_blocks > 1 else 0,
            ENTRY_PAGE_SIZE=src.entry_page_size,
            v2p=src.v2p,
            TRANSLATE=src.v2p is not None,
        )

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)
