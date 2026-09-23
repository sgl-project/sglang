from __future__ import annotations

import enum
import functools
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
    can_use_swapab_attention,
)
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    cast_q_fp8_for_q8kv8_prefill,
    dequantize_k_cache_paged,
    fp8_dtype,
    gather_dequant_requant_fp8_paged,
    q8kv8_padded_num_heads,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    fill_all_compressed_indices,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.kernels.ops.attention.dsv4.online_c128_mtp import OnlineC128MTPController
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildCausalSwaPageIndices,
    BuildPageTablePositions,
    ExpandPrefillCausally,
    late_layer_tail_layout,
)
from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    BuildBlockSeqLensCausal,
    BuildDsparkSwaPageIndices,
    ComputeDsparkWindowGather,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMasks,
    CandidateMetadata,
    IndexerInputs,
    PrefillIndexerInputs,
    cut_request_masks,
    expand_index_page_table,
    make_candidate_indexer,
    mask_topk_scores,
    published_masks,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import plain_prefill_topk
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    _rope_fq4,
    token_req_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    C4IndexerBackendMixin,
    deep_gemm_fp4_paged_mqa_logits,
    topk_transform_paged_from_metadata,
)
from sglang.srt.layers.attention.dsv4.metadata import (
    _LARGE_INDEXER_QUERY_THRESHOLD,
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
    SparsePrefillWorkspace,
    use_dsv4_q8kv8_sparse_prefill,
)
from sglang.srt.layers.attention.verify_mask import (
    VerifyMask,
    maybe_create_verify_mask,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    interleave_rows_per_request,
)
from sglang.srt.layers.cp.utils import (
    cp_materialize_global_token_order,
    is_cp_active,
)
from sglang.srt.layers.dp_attention import (
    get_local_dp_buffer_len,
    set_local_dp_buffer_len,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
    get_spec,
)
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    compute_target_verify_graph_key,
    compute_uniform_extend_lengths,
    read_ragged_verify_mode,
    resolve_ragged_verify_layout,
)
from sglang.srt.utils import ceil_align, is_cuda, is_xpu

if TYPE_CHECKING:
    from sgl_kernel.flash_mla import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_is_cuda = is_cuda()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
DEFAULT_INDEX_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


@functools.lru_cache(maxsize=None)
def _is_sm100_or_newer() -> bool:
    # DeepGEMM's fp8_fp4 mqa-logits kernels need SM100+; Hopper takes the torch indexer.
    return torch.cuda.get_device_capability()[0] >= 10


def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    # IDLE is a real per-DP-rank mode. Do not let a stale _original_forward_mode
    # from a reused/padded ForwardBatch turn an empty rank into TARGET_VERIFY.
    if forward_batch.forward_mode.is_idle():
        return forward_batch.forward_mode
    if forward_batch.forward_mode == ForwardMode.EXTEND:
        return forward_batch.forward_mode
    return (
        getattr(forward_batch, "_original_forward_mode", None)
        or forward_batch.forward_mode
    )


def _get_target_verify_bs(forward_batch: ForwardBatch) -> int:
    actual_forward_mode = getattr(
        forward_batch, "actual_forward_mode", forward_batch.forward_mode
    )
    if actual_forward_mode.is_idle():
        return 0

    spec_info = getattr(forward_batch, "spec_info", None)
    draft_token_num = getattr(spec_info, "draft_token_num", 0)
    draft_token = getattr(spec_info, "draft_token", None)
    if draft_token is None:
        return forward_batch.batch_size
    if draft_token_num <= 0:
        return 0
    draft_count = len(draft_token)
    if draft_count % draft_token_num != 0:
        return 0
    return draft_count // draft_token_num


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    if get_platform().is_sm120 or _is_xpu:
        return None
    import sgl_kernel.flash_mla as flash_mla

    return flash_mla.get_mla_metadata()[0]


# FlashMLA's head64 sm100 decode scheduling constants; not exported, so they hold
# only for that shape and go stale silently if FlashMLA retunes it.
_FLASHMLA_SCHED_BLOCK_SIZE_N = 64
_FLASHMLA_SCHED_FIXED_OVERHEAD = 5


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _fast_flashmla_sched_shape(q: torch.Tensor) -> bool:
    return q.is_cuda and get_platform().is_blackwell and q.shape[-2] == 64


def _maybe_precompute_flashmla_sched_meta(
    flashmla_metadata,
    *,
    q: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
) -> None:
    """Fill the split-KV schedule buffers so `sparse_decode_fwd` skips its own
    `<<<1, 32>>>` scheduling kernel on the decode critical path;
    `flashmla_sched_meta` produces the same schedule bit for bit."""
    if flashmla_metadata is None:
        return
    if getattr(flashmla_metadata, "tile_scheduler_metadata", None) is not None:
        return
    if not _fast_flashmla_sched_shape(q):
        return
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )

    b, s_q = q.shape[0], q.shape[1]
    num_sm_parts = max(_num_sms(q.device.index) // s_q, 1)
    meta = torch.empty((num_sm_parts, META_INTS), dtype=torch.int32, device=q.device)
    num_splits = torch.empty((b + 1,), dtype=torch.int32, device=q.device)
    flashmla_sched_meta(
        meta,
        num_splits,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        block_size_n=_FLASHMLA_SCHED_BLOCK_SIZE_N,
        fixed_overhead_num_blocks=_FLASHMLA_SCHED_FIXED_OVERHEAD,
        topk=indices.shape[-1],
        extra_topk=0 if extra_indices is None else extra_indices.shape[-1],
    )
    flashmla_metadata.tile_scheduler_metadata = meta
    flashmla_metadata.num_splits = num_splits


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30


def _every_request_fits() -> bool:
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
    )

    # Captured only for batches where every request fits the candidate budget, so
    # the plain top-k is the whole selection.
    return get_capture_attention_variant() in (
        "candidate_all",
        "candidate_c2_all",
        "candidate_unfiltered",
    )


@functools.cache
def _has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


def _low_ratio_source_projections(layer, x, q_lora, positions, bufs):
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    real = (
        get_tc_piecewise_forward_context().forward_batch.global_num_token_non_padded_cpu
    )
    if real is None:
        real = x.shape[0]

    # These GEMMs pick their algorithm by M, so at the bucket size the live rows
    # differ from eager; everything downstream is row-independent.
    def put(name, value):
        buf = bufs[name]
        buf[:real].copy_(value)
        buf[real:].zero_()

    if real == 0:
        # An idle DP-attention rank replays on fabricated rows with no live
        # token; a zero-row GEMM is a launch error, so only zero the buffers.
        for buf in bufs.values():
            buf.zero_()
        return

    if layer.compressor is not None:
        kv, score = layer.compressor.project(x[:real])
        put("kv", kv)
        if score is not None:
            put("score", score)
    if layer.indexer is not None:
        indexer = layer.indexer
        put("q", indexer.queries(q_lora[:real], layer.freqs_cis[positions[:real]]))
        put("w", indexer.head_weights(x[:real]))


def _bcg_low_ratio_source_projections(*args):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        eager_on_graph,
    )

    global _bcg_low_ratio_source_projections_fn
    if _bcg_low_ratio_source_projections_fn is None:
        _bcg_low_ratio_source_projections_fn = eager_on_graph(True)(
            _low_ratio_source_projections
        )
    return _bcg_low_ratio_source_projections_fn(*args)


_bcg_low_ratio_source_projections_fn = None


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        values = values.tolist()
    return [int(v) for v in values]


def _low_ratio_compression_metadata(
    compress_ratio: int, seq_lens_casual: torch.Tensor, raw_out_loc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_write_tokens = raw_out_loc.shape[0]
    completes_group = seq_lens_casual[:num_write_tokens] % compress_ratio == 0
    out_loc = torch.where(
        completes_group, raw_out_loc.to(torch.int64) // compress_ratio, -1
    )
    topk_lengths_clamp1 = (seq_lens_casual // compress_ratio).clamp_min(1)
    return out_loc, topk_lengths_clamp1.to(torch.int32)


def _low_ratio_sparse_buffers(
    topk_lengths_clamp1: torch.Tensor, topk: int, is_prefill: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Not the extra_topk_length the kernel reads: that one comes from positions.
    sparse_topk_lengths = torch.clamp(topk_lengths_clamp1, max=topk)
    page_indices = _pad_last_dim(
        torch.full(
            (topk_lengths_clamp1.size(0), topk),
            -1,
            dtype=torch.int32,
            device=topk_lengths_clamp1.device,
        )
    )
    raw_indices = torch.empty_like(page_indices) if is_prefill else None
    return sparse_topk_lengths, page_indices, raw_indices


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


def _copy_or_replace(dst, src):
    if dst is not None and src is not None:
        dst.copy_(src)
        return dst
    return src


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    index_topk: int
    # Sorted compress ratios present in this stage; absent ratios keep no
    # buffers or schedules.
    present_ratios: Tuple[int, ...]
    request_window_layout: Optional[object] = None
    # Shared by all layer stores; locations are in SWA space.
    swa_out_cache_loc: Optional[torch.Tensor] = None
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)
    c4_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    # The (1, 2) subset of present_ratios: one latent per ratio tokens, at slot
    # raw_out_loc // ratio of the c1 / c2 pool, attended through the extra cache.
    low_ratios: Tuple[int, ...] = ()
    c1_out_loc: Optional[torch.Tensor] = None
    c1_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c1_sparse_topk_lengths: Optional[torch.Tensor] = field(init=False, default=None)
    c1_sparse_page_indices: Optional[torch.Tensor] = field(init=False, default=None)
    c1_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)
    c2_out_loc: Optional[torch.Tensor] = None
    c2_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c2_sparse_topk_lengths: Optional[torch.Tensor] = field(init=False, default=None)
    c2_sparse_page_indices: Optional[torch.Tensor] = field(init=False, default=None)
    c2_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)

    # Combined decode tables. Only the c4 tail and lens vary by layer.
    trtllm_swa_lens: Optional[torch.Tensor] = None
    trtllm_c4_indices: Optional[torch.Tensor] = None
    trtllm_c4_lens: Optional[torch.Tensor] = None
    trtllm_c128_indices: Optional[torch.Tensor] = None
    trtllm_c128_lens: Optional[torch.Tensor] = None
    # Lazy eager-prefill caches: qmeta and per-ratio combined tables.
    trtllm_prefill_qmeta: Optional[tuple] = None
    trtllm_prefill_swa_lens: Optional[torch.Tensor] = None
    trtllm_prefill_c4_indices: Optional[torch.Tensor] = None
    trtllm_prefill_c128: Optional[tuple] = None

    c0_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c1_flashmla_metadata: Optional[FlashMLASchedMeta] = field(
        init=False, default=None, repr=False
    )
    c2_flashmla_metadata: Optional[FlashMLASchedMeta] = field(
        init=False, default=None, repr=False
    )
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    @property
    def has_c4(self) -> bool:
        return 4 in self.present_ratios

    @property
    def has_c128(self) -> bool:
        return 128 in self.present_ratios

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 1, 2, 4, 128]):
        if compress_ratio == 0:
            return self.c0_flashmla_metadata
        elif compress_ratio == 1:
            return self.c1_flashmla_metadata
        elif compress_ratio == 2:
            return self.c2_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    # Per-ratio extra-cache metadata is stored as flat fields; these accessors
    # unify the read and write paths over the ratio.

    def sparse_page_indices(self, compress_ratio: int) -> torch.Tensor:
        """Slots into the ratio's extra cache, -1 padded: the indexer's top-k for
        c1 / c2 / c4, every compressed block up to the position for c128."""
        if compress_ratio == 1:
            return self.c1_sparse_page_indices
        elif compress_ratio == 2:
            return self.c2_sparse_page_indices
        elif compress_ratio == 4:
            return self.c4_sparse_page_indices
        elif compress_ratio == 128:
            return self.c128_page_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_topk_lengths(self, compress_ratio: int) -> torch.Tensor:
        if compress_ratio == 1:
            return self.c1_sparse_topk_lengths
        elif compress_ratio == 2:
            return self.c2_sparse_topk_lengths
        elif compress_ratio == 4:
            return self.c4_sparse_topk_lengths
        elif compress_ratio == 128:
            return self.c128_topk_lengths_clamp1
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_raw_indices(self, compress_ratio: int) -> Optional[torch.Tensor]:
        """The top-k as request-local compressed positions, for the sparse
        prefill workspace; allocated for prefill metadata only. Only the indexer
        ratios have one (c128 remaps its page indices instead)."""
        if compress_ratio == 1:
            return self.c1_sparse_raw_indices
        elif compress_ratio == 2:
            return self.c2_sparse_raw_indices
        elif compress_ratio == 4:
            return self.c4_sparse_raw_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def set_sparse_topk(
        self,
        compress_ratio: int,
        *,
        page_indices: torch.Tensor,
        topk_lengths: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Writer counterpart of the accessors above."""
        if compress_ratio == 1:
            self.c1_sparse_page_indices = page_indices
            self.c1_sparse_topk_lengths = topk_lengths
            if raw_indices is not None:
                self.c1_sparse_raw_indices = raw_indices
        elif compress_ratio == 2:
            self.c2_sparse_page_indices = page_indices
            self.c2_sparse_topk_lengths = topk_lengths
            if raw_indices is not None:
                self.c2_sparse_raw_indices = raw_indices
        elif compress_ratio == 4:
            self.c4_sparse_page_indices = page_indices
            self.c4_sparse_topk_lengths = topk_lengths
            if raw_indices is not None:
                self.c4_sparse_raw_indices = raw_indices
        elif compress_ratio == 128:
            assert raw_indices is None, "c128 has no raw top-k"
            self.c128_page_indices = page_indices
            self.c128_topk_lengths_clamp1 = topk_lengths
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def init_trtllm_sparse_buffers(self) -> None:
        """Decode tables of 128 SWA columns then compressed KV, -1 for an invalid
        index, lens counting all 128 SWA slots; only the c4 tail is per layer."""

        num_tokens = self.seq_lens_casual.shape[0]
        assert self.swa_page_indices.shape == (num_tokens, SWA_WINDOW)

        # VarSeq reads rows to the 64-token tile boundary. Back every live view
        # with an aligned parent whose extra rows contain inert values.
        n_pad = (num_tokens + 63) // 64 * 64

        def _tile_padded(fill, src=None, width=None):
            shape = (n_pad,) if width is None else (n_pad, width)
            buf = torch.full(shape, fill, **self.cuda_int32_kwargs)
            if src is not None:
                buf[:num_tokens].copy_(src)
            return buf[:num_tokens]

        if n_pad != num_tokens:
            self.seq_lens_casual = _tile_padded(1, self.seq_lens_casual)
            self.swa_page_indices = _tile_padded(
                -1, self.swa_page_indices, width=SWA_WINDOW
            )
        self.trtllm_swa_lens = _tile_padded(SWA_WINDOW)
        if self.c4_sparse_page_indices is not None:
            w4 = self.c4_sparse_page_indices.shape[-1]
            assert w4 % 4 == 0, f"{w4=}"
            # Unwritten c4 rows must remain inert until the per-layer fill.
            self.trtllm_c4_indices = _tile_padded(-1, width=SWA_WINDOW + w4)
            self.trtllm_c4_indices[:, :SWA_WINDOW].copy_(self.swa_page_indices)
            self.trtllm_c4_lens = _tile_padded(SWA_WINDOW)
        if self.c128_page_indices is not None:
            w128 = self.c128_page_indices.shape[-1]
            assert w128 % 4 == 0, f"{w128=}"
            self.trtllm_c128_indices = _tile_padded(-1, width=SWA_WINDOW + w128)
            self.trtllm_c128_indices[:, :SWA_WINDOW].copy_(self.swa_page_indices)
            self.trtllm_c128_indices[:, SWA_WINDOW:].copy_(self.c128_page_indices)
            self.trtllm_c128_lens = _tile_padded(
                SWA_WINDOW,
                (self.c128_topk_lengths_clamp1 + SWA_WINDOW).to(torch.int32),
            )

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "index_topk",
                "page_size",
                "cuda_int32_kwargs",
                "present_ratios",
                "low_ratios",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
                "c1_out_loc",
                "c1_topk_lengths_clamp1",
                "c1_sparse_topk_lengths",
                "c1_sparse_page_indices",
                "c1_sparse_raw_indices",
                "c2_out_loc",
                "c2_topk_lengths_clamp1",
                "c2_sparse_topk_lengths",
                "c2_sparse_page_indices",
                "c2_sparse_raw_indices",
                "request_window_layout",
                "trtllm_swa_lens",
                "trtllm_c4_indices",
                "trtllm_c4_lens",
                "trtllm_c128_indices",
                "trtllm_c128_lens",
            ],
            assign_fields=[
                # Recomputed by the recorded init_forward_metadata_in_graph op
                # each forward; not copied across replays.
                "swa_out_cache_loc",
                "c0_flashmla_metadata",
                "c1_flashmla_metadata",
                "c2_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
                # Eager-only lazy caches are assigned, not content-copied.
                "trtllm_prefill_qmeta",
                "trtllm_prefill_swa_lens",
                "trtllm_prefill_c4_indices",
                "trtllm_prefill_c128",
            ],
        )

    def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> None:
        assert self.index_topk == other.index_topk
        assert self.page_size == other.page_size
        assert self.cuda_int32_kwargs == other.cuda_int32_kwargs
        assert self.present_ratios == other.present_ratios
        assert self.low_ratios == other.low_ratios

        tensor_copy_fields = [
            "raw_out_loc",
            "seq_lens_casual",
            "positions_casual",
            "c4_out_loc",
            "c128_out_loc",
            "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1",
            "c4_sparse_topk_lengths",
            "c1_out_loc",
            "c1_topk_lengths_clamp1",
            "c1_sparse_topk_lengths",
            "c2_out_loc",
            "c2_topk_lengths_clamp1",
            "c2_sparse_topk_lengths",
            # Preserve graph-captured table addresses; refill c4 per layer.
            "trtllm_swa_lens",
            "trtllm_c4_indices",
            "trtllm_c4_lens",
            "trtllm_c128_indices",
            "trtllm_c128_lens",
        ]
        reference_assign_fields = [
            "page_table",
            "swa_page_indices",
            "swa_topk_lengths",
            "c128_page_indices",
            "c128_topk_lengths_clamp1",
            "c0_flashmla_metadata",
            "c1_flashmla_metadata",
            "c2_flashmla_metadata",
            "c4_flashmla_metadata",
            "c128_flashmla_metadata",
            # Reset eager-only caches so a replay cannot reuse another shape.
            "trtllm_prefill_qmeta",
            "trtllm_prefill_swa_lens",
            "trtllm_prefill_c4_indices",
            "trtllm_prefill_c128",
        ]
        # Keep graph-captured tensor objects alive for fields that captured
        # kernels read by address; overwrite only their contents.
        for field_name in tensor_copy_fields:
            src_val = getattr(other, field_name)
            dst_val = getattr(self, field_name)
            if src_val is None and dst_val is None:
                continue
            assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
            dst_val.copy_(src_val)

        # These fields are safe to replace because captured kernels only need
        # the current per-replay objects, or the field is produced inside the
        # captured graph before the attention graph break consumes it.
        for field_name in reference_assign_fields:
            setattr(self, field_name, getattr(other, field_name))

    def init_compression_metadata(
        self, num_tokens: Optional[int] = None, low_ratio_buffers=None
    ) -> None:
        assert self.page_table.dim() == 2
        # CP pads causal metadata for per-rank partitioning, while cache-write
        # locations remain one-per-logical-token. num_tokens tracks that unpadded
        # length; legacy paths use the metadata length.
        if num_tokens is None:
            num_tokens = self.seq_lens_casual.shape[0]
        assert self.raw_out_loc.shape[0] == num_tokens, (
            f"{self.raw_out_loc.shape=}, {num_tokens=}"
        )

        if self.has_c4 or self.has_c128:
            # One kernel produces both ratios; compute_page_indices=False only
            # drops the [T, max_c128_len] table, which is c128-only.
            (
                c4_out_loc,
                _,
                c4_topk_lengths_raw,
                c4_topk_lengths_clamp1,
                c128_out_loc,
                _,
                _,
                c128_topk_lengths_clamp1,
                c128_page_indices,
            ) = _init_compression_metadata_triton(
                self.seq_lens_casual,
                self.positions_casual,
                self.raw_out_loc,
                self.page_table,
                self.page_size,
                compute_page_indices=self.has_c128,
            )
            if self.has_c4:
                self.c4_out_loc = c4_out_loc
                self.c4_topk_lengths_raw = c4_topk_lengths_raw
                self.c4_topk_lengths_clamp1 = c4_topk_lengths_clamp1
            if self.has_c128:
                self.c128_out_loc = c128_out_loc
                self.c128_topk_lengths_clamp1 = c128_topk_lengths_clamp1
                self.c128_page_indices = _pad_last_dim(c128_page_indices)

        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

        if low_ratio_buffers is not None:
            self.c1_out_loc, self.c1_topk_lengths_clamp1 = low_ratio_buffers[:2]
            self.c2_out_loc, self.c2_topk_lengths_clamp1 = low_ratio_buffers[4:6]
            return
        if 1 in self.low_ratios:
            self.c1_out_loc, self.c1_topk_lengths_clamp1 = (
                _low_ratio_compression_metadata(
                    1, self.seq_lens_casual, self.raw_out_loc
                )
            )
        if 2 in self.low_ratios:
            self.c2_out_loc, self.c2_topk_lengths_clamp1 = (
                _low_ratio_compression_metadata(
                    2, self.seq_lens_casual, self.raw_out_loc
                )
            )

    # Cache-write locations stay in global logical order and are intentionally
    # excluded from CP reindexing.
    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
    ]
    # Same treatment, None for models without that compress ratio.
    _CP_REINDEX_OPTIONAL_FIELDS = [
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
        "c1_topk_lengths_clamp1",
        "c2_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "swa_out_cache_loc",
        "c4_out_loc",
        "c128_out_loc",
        "c1_out_loc",
        "c2_out_loc",
    ]

    def apply_cp_reindex(
        self,
        num_tokens: Optional[int] = None,
        local_index: Optional[torch.Tensor] = None,
    ) -> None:
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        pre_global_len = self.seq_lens_casual.shape[0]
        if local_index is not None:
            idx = local_index
            expected_local_len = local_index.shape[0]
        else:
            idx = slice(cp_rank, None, cp_size)
            assert pre_global_len % cp_size == 0, (
                f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
                "CP round-robin requires padding to ensure divisibility."
            )
            expected_local_len = pre_global_len // cp_size
        if num_tokens is None:
            num_tokens = pre_global_len
        for field_name in self._CP_REINDEX_FIELDS + self._CP_REINDEX_OPTIONAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                assert field_name in self._CP_REINDEX_OPTIONAL_FIELDS, (
                    f"CP reindex: {field_name} is None"
                )
                continue
            assert isinstance(val, torch.Tensor), (
                f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            )
            val = val[idx].contiguous()
            setattr(self, field_name, val)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == num_tokens, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= num_tokens={num_tokens} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self, is_prefill: bool = False, low_ratio_buffers=None):
        assert self.index_topk in (512, 1024), (
            f"unexpected index_topk={self.index_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        if self.has_c4:
            assert self.c4_topk_lengths_clamp1 is not None
            self.c4_sparse_topk_lengths = torch.clamp(
                self.c4_topk_lengths_clamp1, max=self.index_topk
            )
            self.c4_sparse_page_indices = torch.full(
                (self.c4_topk_lengths_clamp1.size(0), self.index_topk),
                -1,
                dtype=torch.int32,
                device=self.c4_topk_lengths_clamp1.device,
            )
            self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
            if is_prefill:
                self.c4_sparse_raw_indices = torch.empty_like(
                    self.c4_sparse_page_indices
                )
        else:
            self.c4_sparse_topk_lengths = None
            self.c4_sparse_page_indices = None
            self.c4_sparse_raw_indices = None
        self.c0_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata() if self.has_c4 else None
        self.c128_flashmla_metadata = (
            _create_flashmla_metadata() if self.has_c128 else None
        )
        if low_ratio_buffers is not None:
            assert not is_prefill and self.low_ratios == (1, 2)
            self.c1_sparse_topk_lengths, self.c1_sparse_page_indices = (
                low_ratio_buffers[2:4]
            )
            self.c2_sparse_topk_lengths, self.c2_sparse_page_indices = (
                low_ratio_buffers[6:8]
            )
            self.c1_flashmla_metadata = _create_flashmla_metadata()
            self.c2_flashmla_metadata = _create_flashmla_metadata()
            return
        if 1 in self.low_ratios:
            (
                self.c1_sparse_topk_lengths,
                self.c1_sparse_page_indices,
                self.c1_sparse_raw_indices,
            ) = _low_ratio_sparse_buffers(
                self.c1_topk_lengths_clamp1, self.index_topk, is_prefill
            )
            self.c1_flashmla_metadata = _create_flashmla_metadata()
        if 2 in self.low_ratios:
            (
                self.c2_sparse_topk_lengths,
                self.c2_sparse_page_indices,
                self.c2_sparse_raw_indices,
            ) = _low_ratio_sparse_buffers(
                self.c2_topk_lengths_clamp1, self.index_topk, is_prefill
            )
            self.c2_flashmla_metadata = _create_flashmla_metadata()


class LateLayerTail(msgspec.Struct, frozen=True):
    """Rows the layers after the last kv_source layer run over under decoder SWA
    bounded replay: the last tail tokens of each request in the extend."""

    token_indices: torch.Tensor
    positions: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: List[int]
    swa_out_cache_loc: torch.Tensor
    # Set when the tail is the extend's last rows (one request): a view, not a gather.
    contiguous_start: Optional[int] = None
    # prefill CP: this rank's tail rows padded to the largest share; cp_metadata is that layout
    pad_rows: int = 0
    cp_metadata: Optional[InterleaveContextParallelMetadata] = None
    local_lens_cpu: Optional[List[int]] = None
    req_global: Optional[torch.Tensor] = None
    pos_global: Optional[torch.Tensor] = None

    def rows(self, t: torch.Tensor) -> torch.Tensor:
        rows = self.real_rows(t)
        if self.pad_rows:
            rows = torch.cat([rows, rows.new_zeros((self.pad_rows, *rows.shape[1:]))])
        return rows

    def real_rows(self, t: torch.Tensor) -> torch.Tensor:
        return _tail_rows(
            t, token_indices=self.token_indices, contiguous_start=self.contiguous_start
        )


def _tail_rows(
    t: torch.Tensor, *, token_indices: torch.Tensor, contiguous_start: Optional[int]
) -> torch.Tensor:
    if contiguous_start is not None:
        return t[contiguous_start:]
    return t[token_indices]


# Rows per logits chunk for the ratio-1/2 indexer inside the prefill CUDA graph;
# its width is the graph's max_seq_len, and longer contexts replay eagerly.
_PREFILL_GRAPH_INDEXER_ROW_CHUNK = 2048


def _prefill_graph_max_seq_len() -> Optional[int]:
    from sglang.srt.runtime_context import get_exec

    return get_exec().graph.cuda_graph_config.prefill.max_seq_len


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    # Low-ratio paged indexer metadata; ratio 4 uses indexer_metadata above.
    c1_indexer_metadata: Optional[PagedIndexerMetadata] = None
    c2_indexer_metadata: Optional[PagedIndexerMetadata] = None

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    # Shared by all low-ratio source layers; graph replay refreshes them live.
    low_ratio_req_indices: Optional[torch.Tensor] = None
    low_ratio_pos_i64: Optional[torch.Tensor] = None

    # Per-step scratch for TP-padded query heads, zeroed by the first user.
    # Later layers overwrite real heads and preserve the zero padding.
    q_pad_buffer: Optional[torch.Tensor] = None

    # What the candidate-source layer published for the index-source layers after
    # it, in the implementation's own type; never copied from the host.
    candidate_metadata: Optional[CandidateMetadata] = None

    # Built at the runner's prefill WAR boundary when the fast path is on,
    # otherwise lazily by ``_forward_prefill_sparse``.
    sparse_prefill_cache: Optional[SparsePrefillChunkCache] = None
    prefill_shared_reads_snapshotted: bool = False

    # Set only on the metadata built for the late layers under bounded SWA replay.
    late_layer_tail: Optional[LateLayerTail] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c1_indexer_metadata, src=other.c1_indexer_metadata)
        maybe_copy_inplace(self.c2_indexer_metadata, src=other.c2_indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )
        self.sparse_prefill_cache = None
        self.prefill_shared_reads_snapshotted = False

    def refresh_for_breakable_cuda_graph_replay_(self, static_metadata: DSV4Metadata):
        self.core_attn_metadata.refresh_for_breakable_cuda_graph_replay_(
            static_metadata.core_attn_metadata
        )
        maybe_copy_inplace(self.indexer_metadata, src=static_metadata.indexer_metadata)
        maybe_copy_inplace(
            self.c1_indexer_metadata, src=static_metadata.c1_indexer_metadata
        )
        maybe_copy_inplace(
            self.c2_indexer_metadata, src=static_metadata.c2_indexer_metadata
        )
        maybe_copy_inplace(
            self.low_ratio_req_indices, src=static_metadata.low_ratio_req_indices
        )
        maybe_copy_inplace(
            self.low_ratio_pos_i64, src=static_metadata.low_ratio_pos_i64
        )
        maybe_copy_inplace(
            self.c4_compress_metadata, src=static_metadata.c4_compress_metadata
        )
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            # Online c128 prefill metadata may carry Python-side planner state,
            # so assign the freshly built per-replay object.
            self.c128_compress_metadata = static_metadata.c128_compress_metadata
        else:
            maybe_copy_inplace(
                self.c128_compress_metadata,
                src=static_metadata.c128_compress_metadata,
            )
        self.sparse_prefill_cache = None
        self.prefill_shared_reads_snapshotted = False


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None
    seq_lens_cpu: Optional[List[int]] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    extend_start_loc: Optional[torch.Tensor] = None
    verify_lens: Optional[torch.Tensor] = None
    total_verify_tokens: int = 0

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens
        self.seq_lens_cpu = other.seq_lens_cpu
        self.c128_compress_metadata = _copy_or_replace(
            self.c128_compress_metadata, other.c128_compress_metadata
        )

        self.extend_start_loc = other.extend_start_loc
        self.verify_lens = other.verify_lens
        self.total_verify_tokens = other.total_verify_tokens


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend_v2():
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    use_captured_forward_metadata_for_breakable_cuda_graph: bool = True
    supports_prefill_cuda_graph_max_context_size: bool = True
    supports_ragged_verify_graph: bool = True
    needs_cpu_seq_lens: bool = False
    trtllm_attn: bool = False

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        # Breakable-graph verify rereads shared state across segments.
        # DSPARK verify replays one full (non-breakable) graph that honors the
        # out-graph/in-graph init contract, so the base IN_REPLAY bound holds.
        if fm.is_target_verify():
            if self.model_runner.spec_algorithm.is_dspark():
                return SharedReadEnds.IN_REPLAY
            return SharedReadEnds.POST_REPLAY
        metadata = self.forward_metadata
        if (
            fm == ForwardMode.EXTEND
            and isinstance(metadata, DSV4Metadata)
            and metadata.prefill_shared_reads_snapshotted
        ):
            return SharedReadEnds.PRE_REPLAY
        return super().shared_read_ends(fm)

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.model_runner = model_runner
        self.encoder_replay = False
        self.device = torch.device(model_runner.device)
        self.max_context_len = model_runner.model_config.context_len
        head_dim = model_runner.model_config.head_dim
        assert head_dim == 512, (
            "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        )
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # Nothing is built for a compress ratio outside the pool's set.
        self.present_ratios: Tuple[int, ...] = self.token_to_kv_pool.present_ratios
        self.low_ratios: Tuple[int, ...] = tuple(
            ratio for ratio in (1, 2) if ratio in self.present_ratios
        )
        self.has_c4: bool = 4 in self.present_ratios
        self.has_c128: bool = 128 in self.present_ratios
        # Two-level low-ratio indexer (dsv4/candidate_indexer.py).
        cfg = model_runner.model_config.hf_text_config
        self.is_dsv41: bool = getattr(cfg, "model_type", None) == "deepseek_v41"
        self.candidate_indexer = make_candidate_indexer(
            getattr(cfg, "candidate_topk_blocks", 0),
            getattr(cfg, "candidate_block_size", 0),
        )
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.index_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", DEFAULT_INDEX_TOPK
        )

        kernel = get_exec().kernel
        self.enable_deepseek_v4_fp4_indexer = kernel.enable_deepseek_v4_fp4_indexer
        self.enable_decoder_swa_bounded_replay: bool = (
            get_exec().features.enable_decoder_swa_bounded_replay
        )
        # The model switches onto this metadata in enter_late_layer_tail.
        self.tail_forward_metadata: Optional[DSV4Metadata] = None
        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend.resolve(model_runner)
        self.dsv4_prefill_backend = getattr(kernel, "dsv4_prefill_backend", "auto")
        if use_dsv4_q8kv8_sparse_prefill(self.dsv4_prefill_backend):
            if not get_platform().is_sm90:
                raise ValueError(
                    "DeepSeek-V4 flashmla_sparse_q8 prefill requires SM90 CUDA GPUs."
                )
            if self.head_dim_v != 512:
                raise ValueError(
                    "DeepSeek-V4 flashmla_sparse_q8 prefill requires d_v=512, "
                    f"got {self.head_dim_v}."
                )
        self._q8kv8_qpad_buf = None
        self._q8kv8_attn_sink_pad = None
        self._q8kv8_identity_scale = None
        self.topk = get_spec().speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = get_spec().speculative_num_draft_tokens
        if self.speculative_num_draft_tokens is not None:
            # Persistent target-verify metadata buffers. Allocated here (not
            # lazily) so they are ordinary tensors: the first touch of a lazy
            # buffer would inherit the caller's context, and a creation inside
            # an inference_mode forward would forbid the in-place updates the
            # graph-capture path performs outside inference mode.
            num_reqs = self.req_to_token.shape[0]
            self.extend_seq_lens_buffer = torch.full(
                (num_reqs,),
                self.speculative_num_draft_tokens,
                **self.cuda_int32_kwargs,
            )
            self.extend_start_loc_buffer = torch.zeros(
                num_reqs, **self.cuda_int32_kwargs
            )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self.online_c128_mtp = OnlineC128MTPController(self)
        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)
        spec_alg = model_runner.spec_algorithm
        self.needs_cpu_seq_lens = not spec_alg.is_dspark() and (
            not _is_cuda or self.online_c128_mtp.enabled()
        )

        self.is_dspark_draft = model_runner.is_draft_worker and spec_alg.is_dspark()
        self.is_draft_runner = model_runner.is_draft_worker
        self._verify_mask = None
        self.cuda_graph_swa_out_cache_loc: Optional[torch.Tensor] = None

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def _resolve_verify_layout(
        self,
        forward_batch: ForwardBatch,
        bs: int,
    ) -> Optional[RaggedVerifyLayout]:
        layout = resolve_ragged_verify_layout(forward_batch)
        if layout is None:
            return None
        if read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT:
            return None
        if get_parallel().attn_cp_size > 1:
            raise NotImplementedError(
                "DSV4 ragged verify does not support context parallel (CP); "
                "set SGLANG_RAGGED_VERIFY_MODE off for CP runs."
            )
        if self.online_c128_mtp.enabled():
            raise NotImplementedError(
                "DSV4 ragged verify does not support online c128 MTP; "
                "set SGLANG_RAGGED_VERIFY_MODE off or disable online compress."
            )
        # Layout invariants (verify_lens >= 1, total == sum) are enforced in
        # RaggedVerifyLayout.__post_init__; don't re-check the device tensor
        # here -- that would D2H-sync the host-free verify prep path.
        layout = layout.padded_to_bucket(padded_bs=bs)
        return layout

    def _target_verify_graph_key(
        self,
        bs: int,
        ragged_layout: Optional[RaggedVerifyLayout],
    ) -> Tuple[int, int]:
        return compute_target_verify_graph_key(
            bs=bs,
            num_draft_tokens=self.speculative_num_draft_tokens,
            ragged_layout=ragged_layout,
        )

    def _make_target_verify_c128_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]],
        extend_seq_lens: torch.Tensor,
        use_prefill_cuda_graph: bool,
        online_c128_state_slot_offset: int,
    ) -> Optional[FusedCompressMetadata]:
        if not self.has_c128 or not self.online_c128_mtp.enabled():
            return None

        assert seq_lens_cpu is not None
        num_draft_tokens = self.speculative_num_draft_tokens
        seq_lens_cpu = [int(x) + num_draft_tokens for x in seq_lens_cpu]
        extend_lens_cpu = [num_draft_tokens] * len(seq_lens_cpu)
        return create_paged_compressor_data(
            compress_ratio=128,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens + self.speculative_num_draft_tokens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_seq_lens,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            online_state_slot_offset=online_c128_state_slot_offset,
        )

    def init_forward_metadata_indexer(
        self,
        core_attn_metadata: DSV4AttnMetadata,
        *,
        compress_ratio: int = 4,
        use_prefill_cuda_graph: bool = False,
    ):
        page_table = core_attn_metadata.page_table
        index_page_size = 0
        if compress_ratio == 4:
            c_seq_lens = core_attn_metadata.c4_topk_lengths_raw
        elif compress_ratio in (1, 2):
            c_seq_lens = (
                core_attn_metadata.c1_topk_lengths_clamp1
                if compress_ratio == 1
                else core_attn_metadata.c2_topk_lengths_clamp1
            )
            # The low-ratio indexer-K pool pages at 64 slots, not page_size //
            # ratio, so the kernel needs a block table at that granularity.
            index_page_size = self.token_to_kv_pool.get_index_k_page_size(
                compress_ratio
            )
            page_table = expand_index_page_table(
                page_table,
                full_page_size=self.page_size,
                compress_ratio=compress_ratio,
                index_page_size=index_page_size,
            )
        else:
            raise ValueError(f"Unsupported indexer {compress_ratio = }")
        return PagedIndexerMetadata(
            page_size=self.page_size,
            compressed_page_size=index_page_size or self.page_size // compress_ratio,
            page_table=page_table,
            compressed_seq_lens=c_seq_lens,
            use_topk_v2=self.dsa_topk_backend.should_use_topk_v2() and not _is_xpu,
            # The SM120 FP4 kernel schedules split_kv=128, while the generic
            # JIT metadata planner encodes split_kv=256.
            force_deep_gemm_metadata=(
                self.enable_deepseek_v4_fp4_indexer and get_platform().is_sm120
            ),
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            compress_ratio=compress_ratio,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        return DSV4RawDecodeMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        extend_start_loc: Optional[torch.Tensor] = None,
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
        dspark_block_size: Optional[int] = None,
        forward_batch: Optional[ForwardBatch] = None,
        swa_replay_start: Optional[torch.Tensor] = None,
        cp_metadata: Optional[InterleaveContextParallelMetadata] = None,
        dspark_swa_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> DSV4Metadata:
        padded_num_tokens = out_cache_loc.shape[0]
        cp_active = forward_batch is not None and is_cp_active(forward_batch)
        if cp_active:
            if cp_metadata is None:
                cp_metadata = forward_batch.attn_cp_metadata
            assert cp_metadata is not None
            padded_num_tokens = sum(cp_metadata.per_rank_actual_token)
            if (
                swa_replay_start is not None
                and swa_replay_start.shape[0] < padded_num_tokens
            ):
                swa_replay_start = torch.nn.functional.pad(
                    swa_replay_start,
                    (0, padded_num_tokens - swa_replay_start.shape[0]),
                )

        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=padded_num_tokens,
            seq_lens_tensor=seq_lens,
            extend_seq_lens_tensor=extend_seq_lens,
            extend_start_loc=extend_start_loc,
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
            dspark_block_size=dspark_block_size,
            dspark_swa_buffers=dspark_swa_buffers,
            num_tokens=num_tokens if cp_active else None,
            swa_replay_start=swa_replay_start,
            num_groups=len(extend_seq_lens_cpu),
        )
        if cp_active:
            core_attn_metadata.apply_cp_reindex(
                num_tokens=num_tokens, local_index=cp_metadata.local_index
            )
            core_attn_metadata.init_flashmla_related(is_prefill=True)
        indexer_metadata = (
            self.init_forward_metadata_indexer(
                core_attn_metadata,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
            if need_compress and self.has_c4
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:

            def create(compress_ratio: Literal[4, 128]):
                # Online c128 uses a different planner that cannot be created in
                # prefill cuda-graph mode. Keep c4 graph-friendly while matching
                # c128's existing online path.
                use_graph_plan = use_prefill_cuda_graph and not (
                    compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
                )
                if use_graph_plan:
                    return create_paged_compressor_data(
                        compress_ratio=compress_ratio,
                        is_prefill=True,
                        token_to_kv_pool=self.token_to_kv_pool,
                        req_to_token=self.req_to_token,
                        req_pool_indices=req_pool_indices,
                        seq_lens=seq_lens,
                        seq_lens_cpu=None,
                        extend_lens=extend_seq_lens,
                        extend_lens_cpu=None,
                        use_prefill_cuda_graph=True,
                        num_q_tokens=out_cache_loc.shape[0],
                        online_state_slot_offset=online_c128_state_slot_offset,
                    )
                return create_paged_compressor_data(
                    compress_ratio=compress_ratio,
                    is_prefill=True,
                    token_to_kv_pool=self.token_to_kv_pool,
                    req_to_token=self.req_to_token,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    extend_lens=extend_seq_lens,
                    extend_lens_cpu=extend_seq_lens_cpu,
                    use_prefill_cuda_graph=use_graph_plan,
                    online_state_slot_offset=online_c128_state_slot_offset,
                )

        metadata = DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4) if self.has_c4 else None,
            c128_compress_metadata=(
                create(compress_ratio=128) if self.has_c128 else None
            ),
        )
        if use_prefill_cuda_graph and self.low_ratio_prefill_graph:
            low = core_attn_metadata.low_ratios
            metadata.c1_indexer_metadata = (
                self._low_ratio_prefill_indexer_metadata(core_attn_metadata, 1)
                if 1 in low
                else None
            )
            metadata.c2_indexer_metadata = (
                self._low_ratio_prefill_indexer_metadata(core_attn_metadata, 2)
                if 2 in low
                else None
            )
            metadata.low_ratio_req_indices = req_pool_indices_repeated.to(torch.int64)
            metadata.low_ratio_pos_i64 = core_attn_metadata.positions_casual.to(
                torch.int64
            )
        return metadata

    def _low_ratio_prefill_indexer_metadata(
        self, core: DSV4AttnMetadata, compress_ratio: int
    ) -> PagedIndexerMetadata:
        num_pages = core.page_table.shape[1]
        max_seq_len = _prefill_graph_max_seq_len()
        if max_seq_len is not None:
            num_pages = min(max_seq_len // self.page_size, num_pages)
        index_page_size = self.token_to_kv_pool.get_index_k_page_size(compress_ratio)
        page_table = expand_index_page_table(
            core.page_table[:, :num_pages],
            full_page_size=self.page_size,
            compress_ratio=compress_ratio,
            index_page_size=index_page_size,
        )
        # Unclamped: a token with no completed group scores nothing, as in eager.
        c_seq_lens = (core.seq_lens_casual // compress_ratio).to(torch.int32)
        row_chunk = _PREFILL_GRAPH_INDEXER_ROW_CHUNK
        return PagedIndexerMetadata(
            page_size=self.page_size,
            compressed_page_size=index_page_size,
            page_table=page_table,
            compressed_seq_lens=c_seq_lens,
            use_topk_v2=False,
            use_prefill_cuda_graph=True,
            compress_ratio=compress_ratio,
            row_chunk=row_chunk if row_chunk < c_seq_lens.shape[0] else 0,
        )

    @property
    def low_ratio_prefill_graph(self) -> bool:
        return (
            bool(self.low_ratios) and _has_dense_fp4_indexer() and _is_sm100_or_newer()
        )

    def can_run_prefill_cuda_graph(self, forward_batch: ForwardBatch) -> bool:
        max_seq_len = _prefill_graph_max_seq_len()
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if max_seq_len is None or seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
            return True
        return int(seq_lens_cpu.max().item()) <= max_seq_len

    def _build_late_layer_tail_metadata(
        self, forward_batch: ForwardBatch
    ) -> DSV4Metadata:
        # Each request contributes only its last SWA_WINDOW extend tokens, with the
        # window floored at the tail start: window KV before it is never written here.
        extend_lens_cpu = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_cpu is not None and seq_lens_cpu is not None
        device = forward_batch.out_cache_loc.device
        token_indices, tail_lens_cpu, swa_replay_start = late_layer_tail_layout(
            extend_lens_cpu=extend_lens_cpu,
            seq_lens_cpu=seq_lens_cpu.tolist(),
            tail_len=SWA_WINDOW,
            device=device,
        )
        contiguous_start = (
            extend_lens_cpu[0] - tail_lens_cpu[0] if len(extend_lens_cpu) == 1 else None
        )
        out_cache_loc = _tail_rows(
            forward_batch.out_cache_loc,
            token_indices=token_indices,
            contiguous_start=contiguous_start,
        )
        tail_lens = torch.tensor(tail_lens_cpu, dtype=torch.int32, device=device)
        cp_tail = (
            self._late_layer_tail_cp_layout(forward_batch, token_indices, tail_lens)
            if is_cp_active(forward_batch)
            else None
        )

        metadata = self.init_forward_metadata_prefill(
            max_seq_len=int(seq_lens_cpu.max().item()),
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            seq_lens_cpu=seq_lens_cpu.tolist(),
            out_cache_loc=out_cache_loc,
            num_tokens=sum(tail_lens_cpu),
            extend_seq_lens=tail_lens,
            extend_seq_lens_cpu=tail_lens_cpu,
            extend_start_loc=torch.cumsum(tail_lens, dim=0) - tail_lens,
            swa_replay_start=swa_replay_start,
            forward_batch=forward_batch if cp_tail is not None else None,
            cp_metadata=cp_tail["cp_metadata"] if cp_tail is not None else None,
        )
        swa_out_cache_loc = (
            metadata.core_attn_metadata.request_window_layout.write_loc
            if self.token_to_kv_pool.request_window is not None
            else self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
                torch.int32
            )
        )
        metadata.core_attn_metadata.swa_out_cache_loc = swa_out_cache_loc
        metadata.low_ratio_req_indices = torch.repeat_interleave(
            forward_batch.req_pool_indices.to(torch.int64), tail_lens.to(torch.int64)
        )
        positions = _tail_rows(
            forward_batch.positions,
            token_indices=token_indices,
            contiguous_start=contiguous_start,
        )
        metadata.low_ratio_pos_i64 = positions.to(torch.int64)
        if cp_tail is None:
            # Without CP, tail rows index the full extend on this rank.
            metadata.late_layer_tail = LateLayerTail(
                token_indices=token_indices,
                positions=positions,
                extend_seq_lens=tail_lens,
                extend_seq_lens_cpu=tail_lens_cpu,
                swa_out_cache_loc=swa_out_cache_loc,
                contiguous_start=contiguous_start,
            )
        else:
            # With CP, select from this rank's extend and pad for collectives.
            metadata.late_layer_tail = LateLayerTail(
                token_indices=cp_tail["local_token_indices"],
                positions=cp_tail["local_positions"],
                extend_seq_lens=tail_lens,
                extend_seq_lens_cpu=tail_lens_cpu,
                swa_out_cache_loc=swa_out_cache_loc,
                pad_rows=cp_tail["pad_rows"],
                cp_metadata=cp_tail["cp_metadata"],
                local_lens_cpu=cp_tail["local_lens_cpu"],
                req_global=metadata.low_ratio_req_indices,
                pos_global=metadata.low_ratio_pos_i64,
            )
        return metadata

    def _late_layer_tail_cp_layout(
        self,
        forward_batch: ForwardBatch,
        token_indices: torch.Tensor,
        tail_lens: torch.Tensor,
    ) -> dict:
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        device = token_indices.device
        total = token_indices.shape[0]
        owner_rank = token_indices % cp_size
        counts = torch.bincount(owner_rank, minlength=cp_size).tolist()
        max_local = max(counts)
        order = torch.argsort(owner_rank, stable=True)
        rank_starts = torch.tensor(
            [sum(counts[:r]) for r in range(cp_size)],
            dtype=torch.int64,
            device=device,
        )
        slot = torch.empty_like(owner_rank)
        slot[order] = (
            torch.arange(total, device=device) - rank_starts[owner_rank[order]]
        )
        gather_index = owner_rank * max_local + slot

        local_tail_rows = (owner_rank == cp_rank).nonzero().squeeze(1)
        pad_rows = max_local - counts[cp_rank]
        # Give each rank distinct padding rows in the compact tail metadata.
        pad_start = total + sum(max_local - c for c in counts[:cp_rank])
        local_metadata_rows = torch.cat(
            [
                local_tail_rows,
                torch.arange(pad_start, pad_start + pad_rows, device=device),
            ]
        )
        tail_request_ids = torch.repeat_interleave(
            torch.arange(forward_batch.batch_size, device=device),
            tail_lens.to(torch.int64),
            output_size=total,
        )
        local_positions = torch.cat(
            [
                forward_batch.positions[token_indices[local_tail_rows]],
                forward_batch.positions.new_zeros(pad_rows),
            ]
        )
        cp_metadata = InterleaveContextParallelMetadata(
            per_rank_actual_token=[max_local] * cp_size,
            max_rank_len=[max_local] * cp_size,
            total_seq_lens=total,
            bs=forward_batch.batch_size,
            per_rank_logical_token=counts,
            gather_index=gather_index,
            local_index=local_metadata_rows,
        )
        return dict(
            cp_metadata=cp_metadata,
            local_token_indices=(token_indices[local_tail_rows] - cp_rank) // cp_size,
            local_positions=local_positions,
            local_lens_cpu=torch.bincount(
                tail_request_ids[local_tail_rows], minlength=forward_batch.batch_size
            ).tolist(),
            pad_rows=pad_rows,
        )

    def enter_late_layer_tail(self, forward_batch: ForwardBatch) -> tuple:
        """Switch the late layers onto the tail; the return value goes back to
        exit_late_layer_tail."""
        tail_metadata = self.tail_forward_metadata
        assert tail_metadata is not None, "no tail metadata for this forward"
        saved = (
            self.forward_metadata,
            forward_batch.attn_cp_metadata,
            get_local_dp_buffer_len(),
        )
        tail = tail_metadata.late_layer_tail
        # The layers before the switch published top-k into the full metadata's
        # buffers; carry the tail rows into the tail metadata's (padding stays -1).
        full_core = saved[0].core_attn_metadata
        tail_core = tail_metadata.core_attn_metadata
        for ratio in tail_core.low_ratios:
            for full_buf, tail_buf in (
                (
                    full_core.sparse_page_indices(ratio),
                    tail_core.sparse_page_indices(ratio),
                ),
                (
                    full_core.sparse_topk_lengths(ratio),
                    tail_core.sparse_topk_lengths(ratio),
                ),
                (
                    full_core.sparse_raw_indices(ratio),
                    tail_core.sparse_raw_indices(ratio),
                ),
            ):
                if full_buf is None or tail_buf is None:
                    continue
                rows = tail.real_rows(full_buf)
                tail_buf[: rows.shape[0]].copy_(rows)
        self.forward_metadata = tail_metadata
        if self.token_to_kv_pool.request_window is not None:
            self.token_to_kv_pool.request_window.activate(
                tail_core.request_window_layout
            )
        if tail.cp_metadata is not None:
            forward_batch.attn_cp_metadata = tail.cp_metadata
            set_local_dp_buffer_len(sum(tail.cp_metadata.per_rank_actual_token))
        return saved

    def exit_late_layer_tail(self, saved: tuple, forward_batch: ForwardBatch) -> None:
        (
            self.forward_metadata,
            forward_batch.attn_cp_metadata,
            local_dp_buffer_len,
        ) = saved
        set_local_dp_buffer_len(local_dp_buffer_len)

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
        ragged_layout: Optional[RaggedVerifyLayout] = None,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        assert out_cache_loc is not None
        bs = len(seq_lens)
        if self.needs_cpu_seq_lens:
            assert seq_lens_cpu is not None
            seq_lens_cpu_list = seq_lens_cpu.tolist()
        else:
            seq_lens_cpu_list = None
        if ragged_layout is None:
            self.extend_seq_lens_buffer[:bs].fill_(self.speculative_num_draft_tokens)
            extend_seq_lens = self.extend_seq_lens_buffer[:bs]
            extend_start_loc = None
            verify_lens = None
            total_verify_tokens = self.speculative_num_draft_tokens * bs
        else:
            self.extend_seq_lens_buffer[:bs].copy_(ragged_layout.verify_lens)
            self.extend_start_loc_buffer[:bs].copy_(ragged_layout.extend_start_loc)
            extend_seq_lens = self.extend_seq_lens_buffer[:bs]
            extend_start_loc = self.extend_start_loc_buffer[:bs]
            verify_lens = self.extend_seq_lens_buffer[:bs]
            total_verify_tokens = ragged_layout.graph_num_tokens

        return DSV4RawVerifyMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            extend_seq_lens=extend_seq_lens,
            seq_lens_cpu=seq_lens_cpu_list,
            c128_compress_metadata=self._make_target_verify_c128_metadata(
                req_pool_indices,
                seq_lens,
                seq_lens_cpu_list,
                extend_seq_lens,
                use_prefill_cuda_graph,
                online_c128_state_slot_offset,
            ),
            extend_start_loc=extend_start_loc,
            verify_lens=verify_lens,
            total_verify_tokens=total_verify_tokens,
        )

    def init_forward_metadata_dspark_draft_block(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        block_size: int,
        dspark_swa_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> DSV4Metadata:
        if seq_lens_cpu is None:
            seq_lens_cpu_list = seq_lens.tolist()
        else:
            seq_lens_cpu_list = [int(x) for x in seq_lens_cpu.tolist()]
        lengths = compute_uniform_extend_lengths(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu_list,
            extend_len=block_size,
        )
        extend_seq_lens = self._move_to_device(lengths.extend_seq_lens_cpu)
        return self.init_forward_metadata_prefill(
            # DSpark draft blocks are SWA-only, like draft extend. Their full
            # context page table is unused; retain only its 2-D placeholder.
            max_seq_len=self.page_size,
            req_pool_indices=req_pool_indices,
            seq_lens=lengths.seq_lens_extended,
            seq_lens_cpu=lengths.seq_lens_cpu_extended,
            out_cache_loc=out_cache_loc,
            num_tokens=lengths.num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=lengths.extend_seq_lens_cpu,
            extend_start_loc=lengths.extend_start_loc,
            need_compress=False,
            use_prefill_cuda_graph=False,
            dspark_block_size=block_size,
            dspark_swa_buffers=dspark_swa_buffers,
        )

    def make_forward_metadata_from_raw_verify(
        self,
        raw_metadata: DSV4RawVerifyMetadata,
        online_c128_state_slot_offset: int = 0,
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens
        assert extend_seq_lens is not None

        is_ragged = raw_metadata.verify_lens is not None
        if is_ragged:
            seq_lens = seq_lens + extend_seq_lens
            num_q_tokens = raw_metadata.total_verify_tokens
            assert num_q_tokens > 0, "ragged verify raw metadata is stale/empty"
            seq_lens_casual, req_pool_indices_repeated = (
                self._expand_prefill_casually_vectorized(
                    num_tokens=num_q_tokens,
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                    extend_start_loc=raw_metadata.extend_start_loc,
                    req_pool_indices=req_pool_indices,
                    padded_num_tokens=out_cache_loc.shape[0],
                )
            )
        else:
            seq_lens = seq_lens + self.speculative_num_draft_tokens
            num_q_tokens = num_draft_tokens * bs
            seq_lens_casual, req_pool_indices_repeated = (
                self.expand_extend_with_same_length(
                    bs=bs,
                    qo_len=num_draft_tokens,
                    seq_lens=seq_lens,
                    req_pool_indices=req_pool_indices,
                )
            )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
            num_groups=bs,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self.has_c4
            else None
        )
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_q_tokens,
            online_state_slot_offset=online_c128_state_slot_offset,
        )
        c128_compress_metadata = raw_metadata.c128_compress_metadata
        if c128_compress_metadata is None and self.has_c128:
            c128_compress_metadata = create(compress_ratio=128)
        low = core_attn_metadata.low_ratios
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c1_indexer_metadata=(
                self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=1)
                if 1 in low
                else None
            ),
            c2_indexer_metadata=(
                self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=2)
                if 2 in low
                else None
            ),
            c4_compress_metadata=create(compress_ratio=4) if self.has_c4 else None,
            c128_compress_metadata=c128_compress_metadata,
        )

    def make_forward_metadata_from_raw_decode(
        self,
        raw_metadata: DSV4RawDecodeMetadata,
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self.has_c4
            else None
        )

        low = core_attn_metadata.low_ratios
        c1_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=1)
            if 1 in low
            else None
        )
        c2_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=2)
            if 2 in low
            else None
        )

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c1_indexer_metadata=c1_indexer_metadata,
            c2_indexer_metadata=c2_indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4) if self.has_c4 else None,
            c128_compress_metadata=(
                create(compress_ratio=128) if self.has_c128 else None
            ),
        )

    def init_forward_metadata_draft_extend(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_tokens_per_req: int,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        num_tokens = num_tokens_per_req * batch_size
        swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(out_cache_loc)
        if swa_out_cache_loc is None and out_cache_loc is not None:
            # Eager-only miss (no graph state / oversized batch): translate once
            # per step instead of per layer at store time.
            if self.token_to_kv_pool.request_window is None:
                swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        out_cache_loc
                    ).to(torch.int32)
                )
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs=batch_size,
                qo_len=num_tokens_per_req,
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
            )
        )
        if self.trtllm_attn:
            # DP padding can expand a length-one request into nonpositive
            # per-token lens; trtllm-gen requires them to remain at least one.
            seq_lens_casual = seq_lens_casual.clamp(min=1)
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            # Draft extend is SWA-only. Keep the required 2-D page-table
            # placeholder narrow instead of materializing the full context.
            max_seq_len=self.page_size,
            out_loc=out_cache_loc,
            need_compress=False,
            is_prefill=True,
            num_groups=batch_size,
        )
        if swa_out_cache_loc is not None:
            # Captures store_cache's cached path instead of a per-layer
            # in-graph mapping translate.
            core_attn_metadata.swa_out_cache_loc = swa_out_cache_loc
        return DSV4Metadata(
            core_attn_metadata=core_attn_metadata,
            indexer_metadata=None,
        )

    def _fill_cuda_graph_swa_out_cache_loc(
        self, out_cache_loc: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        # None (buffer absent / too small) is an eager-only miss: capture and
        # replay always fit the pre-sized buffer.
        buf = self.cuda_graph_swa_out_cache_loc
        if (
            buf is None
            or out_cache_loc is None
            or out_cache_loc.shape[0] > buf.shape[0]
        ):
            return None
        n = out_cache_loc.shape[0]
        buf[n:].zero_()
        buf[:n].copy_(
            self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
                torch.int32
            )
        )
        return buf[:n]

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            skip_low_ratio_indexer,
        )

        # Upgrade Raw->Full so compress + core_attn + indexer materialization is
        # recorded inside the cuda graph; already Full when PREP_IN_CUDA_GRAPH=0.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
                online_c128_state_slot_offset=self.online_c128_mtp.state_slot_offset(),
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata):
            core = metadata.core_metadata
            for ratio in core.low_ratios:
                if skip_low_ratio_indexer(ratio):
                    # Share the full-position indices across layers of this ratio.
                    fill_all_compressed_indices(
                        core.page_table,
                        core.sparse_topk_lengths(ratio),
                        core.sparse_page_indices(ratio),
                        compress_ratio=ratio,
                        page_size=core.page_size,
                        raw_indices=core.sparse_raw_indices(ratio),
                    )

        # Recorded inside the cuda graph, so replay re-reads the live out_cache_loc
        # buffer (spec-v2 and DP padding rebind it). flash_mla needs int32 indices.
        if (
            isinstance(metadata, DSV4Metadata)
            and forward_batch.out_cache_loc is not None
        ):
            out_cache_loc = forward_batch.out_cache_loc
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                and self.topk > 0
                and self.speculative_num_steps > 1
            ):
                # Multi-step draft decode shares one out_cache_loc buffer across
                # steps; mirror the eager init's per-step slice.
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            if self.token_to_kv_pool.request_window is None:
                metadata.core_attn_metadata.swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        out_cache_loc
                    ).to(torch.int32)
                )

            # Refresh low-ratio source metadata from the live decode inputs.
            if (
                metadata.core_metadata.low_ratios
                and forward_batch.forward_mode.is_decode()
            ):
                metadata.low_ratio_req_indices = token_req_indices(forward_batch)
                metadata.low_ratio_pos_i64 = forward_batch.positions.to(torch.int64)

            if (
                self.is_dspark_draft
                and forward_batch.forward_mode.is_target_verify()
                and self.token_to_kv_pool.request_window is None
            ):
                block_size = int(forward_batch.spec_info.draft_token_num)
                seq_lens_casual = self._dspark_seq_lens_casual(
                    seq_lens=forward_batch.seq_lens, block_size=block_size
                )
                req_pool_indices_repeated = (
                    forward_batch.req_pool_indices.repeat_interleave(block_size)
                )
                (
                    swa_page_indices,
                    swa_topk_lengths,
                ) = self.get_dspark_swa_page_indices(
                    seq_lens_casual=seq_lens_casual,
                    req_pool_indices_repeated=req_pool_indices_repeated,
                    out_loc=out_cache_loc,
                    block_size=block_size,
                )
                metadata.core_attn_metadata.swa_page_indices = swa_page_indices
                metadata.core_attn_metadata.swa_topk_lengths = swa_topk_lengths

    def _dspark_seq_lens_casual(
        self, *, seq_lens: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        return BuildBlockSeqLensCausal.execute(
            seq_lens=seq_lens,
            block_size=block_size,
            device=self.cuda_int32_kwargs["device"],
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ) -> None:
        bucket = _GraphBucket.of(forward_batch.forward_mode)
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        uses_cpu_seq_lens = self.needs_cpu_seq_lens or self.is_dspark_draft

        if in_capture:
            # Captured graph does no real cache writes, so synthesize a dummy
            # out_cache_loc per bucket (replay supplies the real value).
            assert req_pool_indices.size(0) == bs
            assert seq_lens.size(0) == bs
            num_tokens = forward_batch.positions.numel()
            if bucket == _GraphBucket.DECODE_OR_IDLE:
                out_cache_loc = torch.zeros_like(seq_lens)
            elif bucket == _GraphBucket.TARGET_VERIFY:
                out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            else:
                out_cache_loc = None
            actual_forward_mode = forward_batch.forward_mode
            seq_lens_sum = int(seq_lens.sum().item())
            seq_lens_cpu = seq_lens.cpu() if uses_cpu_seq_lens else None
        else:
            out_cache_loc = forward_batch.out_cache_loc
            actual_forward_mode = getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            )
            seq_lens_sum = forward_batch.seq_lens_sum
            seq_lens_cpu = forward_batch.seq_lens_cpu if uses_cpu_seq_lens else None

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            if uses_cpu_seq_lens:
                seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
            actual_max_seq_len = seq_lens_cpu.max().item()
            assert actual_max_seq_len <= chosen_max_seq_len

        graph_key = bs
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            self.online_c128_mtp.prepare_forward(
                actual_forward_mode,
                req_pool_indices,
                seq_lens,
            )
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY and self.is_dspark_draft:
            block_size = self.speculative_num_draft_tokens - 1
            num_tokens_block = block_size * bs
            assert out_cache_loc is not None
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens_block - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            self.online_c128_mtp.prepare_forward(
                actual_forward_mode,
                req_pool_indices,
                seq_lens,
            )
            dspark_swa_buffers = None
            captured_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket].get(
                bs
            )
            if not in_capture and captured_metadata is not None:
                # Reuse only storage: the draft graph rebuilds both tensors from
                # live inputs before attention. copy_ onto itself is a no-op.
                core = captured_metadata.core_attn_metadata
                dspark_swa_buffers = (core.swa_page_indices, core.swa_topk_lengths)
            temp_metadata = self.init_forward_metadata_dspark_draft_block(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc_padded,
                block_size=block_size,
                dspark_swa_buffers=dspark_swa_buffers,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            verify_bs = _get_target_verify_bs(forward_batch)
            ragged_layout = self._resolve_verify_layout(forward_batch, bs=bs)
            graph_key, num_tokens_v = self._target_verify_graph_key(
                bs=bs, ragged_layout=ragged_layout
            )
            if self.online_c128_mtp.enabled() and verify_bs == 0:
                self.online_c128_mtp.clear()
                self.forward_metadata = self.cuda_graph_metadata_of_bucket_and_bs[
                    bucket
                ][graph_key]
                return
            assert out_cache_loc is not None
            assert num_tokens_v >= len(out_cache_loc), (
                f"ragged verify token-keyed graph requires the decode cuda-graph "
                f"runner to supply out_cache_loc sized to graph_num_tokens "
                f"({num_tokens_v}), got {len(out_cache_loc)}; the decode graph "
                "runner does not yet route token-keyed ragged captures."
            )
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens_v - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            online_c128_state_slot_offset = self.online_c128_mtp.prepare_forward(
                actual_forward_mode,
                req_pool_indices,
                seq_lens,
                verify_bs=verify_bs,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
                ragged_layout=ragged_layout,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            self.online_c128_mtp.prepare_forward(
                actual_forward_mode,
                req_pool_indices,
                seq_lens,
            )
            num_tokens_per_req = self.draft_extend_num_tokens_per_req
            if out_cache_loc is not None:
                # Pad the real write locations to the captured token count so
                # raw_out_loc reflects the actual replay out_cache_loc.
                out_cache_loc = torch.nn.functional.pad(
                    out_cache_loc,
                    pad=(0, num_tokens_per_req * bs - len(out_cache_loc)),
                    mode="constant",
                    value=0,
                )
            temp_metadata = self.init_forward_metadata_draft_extend(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_tokens_per_req=num_tokens_per_req,
                out_cache_loc=out_cache_loc,
            )
        else:
            self.online_c128_mtp.clear()
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=graph_key, temp_metadata=temp_metadata, bucket=bucket
        )

        if in_capture:
            # Preserve _current_capture_raw for on_after_cuda_graph_warmup
            metadata = self.forward_metadata
            self._current_capture_raw = (
                metadata
                if isinstance(
                    metadata,
                    (DSV4RawDecodeMetadata, DSV4RawVerifyMetadata),
                )
                else None
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        logical_forward_mode = _get_logical_forward_mode(forward_batch)
        if self.mtp_enabled and logical_forward_mode.is_idle():
            self.online_c128_mtp.clear()
            return

        self.encoder_replay = forward_batch.encoder_swa_replay
        self.forward_metadata = self._build_forward_metadata(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)
        self.tail_forward_metadata = (
            self._build_late_layer_tail_metadata(forward_batch)
            if self.enable_decoder_swa_bounded_replay
            and forward_batch.forward_mode.is_extend_without_speculative()
            else None
        )

        if self.token_to_kv_pool.request_window is not None:
            self.token_to_kv_pool.request_window.activate(
                self.forward_metadata.core_attn_metadata.request_window_layout
            )

    def prepare_prefill_shared_read_snapshot(
        self, forward_batch: ForwardBatch, *, num_qo_tokens: int
    ) -> None:
        # Sparse prefill otherwise reads req_to_token/full_to_swa lazily in its
        # first layer. DFLASH/DSPARK have no later prefill draft-extend reader;
        # CP shards the query layout that this global snapshot assumes.
        metadata = self.forward_metadata
        if self.token_to_kv_pool.request_window is not None:
            return
        if isinstance(metadata, DSV4Metadata):
            metadata.prefill_shared_reads_snapshotted = False
        snapshot_shared_prefill_reads = (
            envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.get()
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and self.model_runner.spec_algorithm.is_dflash_family()
            and not is_cp_active(forward_batch)
        )
        if not snapshot_shared_prefill_reads:
            return

        assert isinstance(metadata, DSV4Metadata)
        # The tail never takes the sparse path, so it carries no chunk cache.
        use_sparse_prefill = (
            not get_platform().is_sm120
            and metadata.late_layer_tail is None
            and (
                num_qo_tokens > _LARGE_INDEXER_QUERY_THRESHOLD
                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
            )
        )
        if use_sparse_prefill:
            metadata.sparse_prefill_cache = self._build_sparse_prefill_chunk_cache(
                forward_batch, metadata.core_attn_metadata, num_qo_tokens=num_qo_tokens
            )
        # Marked for dense prefill too: that path reads only core_attn_metadata,
        # which init_forward_metadata already snapshotted.
        metadata.prefill_shared_reads_snapshotted = True

    def _build_sparse_prefill_chunk_cache(
        self,
        forward_batch: ForwardBatch,
        core_attn_metadata: DSV4AttnMetadata,
        *,
        num_qo_tokens: int,
    ) -> SparsePrefillChunkCache:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert seq_lens_cpu is not None
        # The chunk cache gathers the W-1 positions before the chunk; under the
        # tail those are late-layer window slots this prefill never wrote.
        assert self.forward_metadata.late_layer_tail is None
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        assert extend_seq_lens_cpu is not None
        seq_lens_cpu_list = seq_lens_cpu.tolist()
        total_swa = sum(
            min(int(seq_len), int(extend_len) + SWA_WINDOW - 1)
            for seq_len, extend_len in zip(
                seq_lens_cpu_list, extend_seq_lens_cpu, strict=True
            )
        )
        if is_cp_active(forward_batch):
            query_lens = torch.tensor(
                interleave_rows_per_request(
                    _as_int_list(extend_seq_lens_cpu),
                    get_parallel().attn_cp_rank,
                    get_parallel().attn_cp_size,
                ),
                dtype=torch.int32,
                device=extend_seq_lens.device,
            )
        else:
            query_lens = extend_seq_lens.to(torch.int32)
        # padding rows are never combined
        query_pos = core_attn_metadata.seq_lens_casual[:num_qo_tokens] - 1
        if query_pos.shape[0] < num_qo_tokens:
            query_pos = _pad_tensor_to_size(query_pos, num_qo_tokens, value=0)
        return SparsePrefillChunkCache.build(
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            extend_seq_lens=extend_seq_lens.to(torch.int32),
            query_lens=query_lens,
            query_pos=query_pos,
            req_pool_indices=forward_batch.req_pool_indices.to(torch.int32),
            req_to_token=self.req_to_token,
            full_to_swa=self.token_to_kv_pool.full_to_swa_index_mapping,
            swa_window_size=SWA_WINDOW,
            swa_page_size=self.token_to_kv_pool.swa_kv_pool.page_size,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max(seq_lens_cpu_list),
            total_swa=total_swa,
        )

    def _build_forward_metadata(
        self,
        forward_batch: ForwardBatch,
        *,
        max_seq_len_override: Optional[int] = None,
        use_prefill_cuda_graph: bool = False,
    ):
        logical_forward_mode = _get_logical_forward_mode(forward_batch)
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        # Regular prefill batches already carry scheduler-maintained CPU lengths.
        # Keep using those when present; needs_cpu_seq_lens only controls whether
        # speculative overlap must publish a new GPU-to-CPU mirror each step.
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert self.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        if max_seq_len_override is None:
            max_seq_len_override = forward_batch.max_seq_len_override
        if max_seq_len_override is not None:
            max_seq_len = max_seq_len_override
            if seq_lens_cpu is not None and len(seq_lens_cpu) > 0:
                actual_max_seq_len = int(seq_lens_cpu.max().item())
                if actual_max_seq_len > max_seq_len:
                    raise ValueError(
                        "Prefill CUDA graph max context size is smaller than the "
                        f"live context: {max_seq_len=} < {actual_max_seq_len=}"
                    )
        elif seq_lens_cpu is not None:
            max_seq_len = int(seq_lens_cpu.max().item())
        else:
            max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        verify_bs = _get_target_verify_bs(forward_batch)
        online_c128_state_slot_offset = self.online_c128_mtp.prepare_forward(
            logical_forward_mode,
            req_pool_indices,
            seq_lens,
            verify_bs=verify_bs,
        )

        if logical_forward_mode.is_decode_or_idle():
            # DSv4 bakes this step's KV write target (c4/c128) into metadata,
            # so slice the shared multi-step out_cache_loc now, not at forward time.
            out_cache_loc = forward_batch.out_cache_loc
            if self.topk > 0 and self.speculative_num_steps > 1:
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )
        elif self.is_dspark_draft and logical_forward_mode.is_target_verify():
            block_size = int(forward_batch.spec_info.draft_token_num)
            metadata = self.init_forward_metadata_dspark_draft_block(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
                block_size=block_size,
            )
        elif logical_forward_mode.is_target_verify():
            ragged_layout = self._resolve_verify_layout(forward_batch, bs=len(seq_lens))
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
                ragged_layout=ragged_layout,
            )
        elif logical_forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.speculative_num_draft_tokens
            assert num_tokens_per_req > 0
            metadata = self.init_forward_metadata_draft_extend(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_tokens_per_req=num_tokens_per_req,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif logical_forward_mode.is_prefill():
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                extend_start_loc=forward_batch.extend_start_loc,
                need_compress=True,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
                forward_batch=forward_batch,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        return metadata

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: ForwardBatch
    ):
        max_seq_len = forward_batch.max_seq_len_override or self.MAX_SEQ_LEN_FOR_CAPTURE
        self.forward_metadata = self._build_forward_metadata(
            forward_batch,
            max_seq_len_override=max_seq_len,
            use_prefill_cuda_graph=True,
        )
        if self.low_ratio_prefill_graph and forward_batch.forward_mode.is_extend():
            for ratio in self.low_ratios:
                self._source_projection_buffers(
                    forward_batch.out_cache_loc.shape[0], ratio
                )
        return self.forward_metadata

    def _source_projection_buffers(self, num_tokens: int, ratio: int) -> dict:
        cfg = self.model_runner.model_config.hf_text_config
        heads, dim = int(cfg.index_n_heads), int(cfg.index_head_dim)
        sets = getattr(self, "_source_proj_bufs", None) or {}
        have = sets.get(ratio)
        if have is None or have[0]["q"].shape[0] < num_tokens:
            latent = self.model_runner.model_config.head_dim
            zeros = lambda *shape, dtype: torch.zeros(
                *shape, dtype=dtype, device=self.device
            )
            bufs = {
                "q": zeros(num_tokens, heads, dim, dtype=torch.bfloat16),
                "w": zeros(num_tokens, heads, dtype=torch.bfloat16),
                "kv": zeros(
                    num_tokens,
                    latent,
                    dtype=torch.bfloat16 if ratio == 1 else torch.float32,
                ),
            }
            if ratio == 2:
                bufs["score"] = zeros(num_tokens, latent, dtype=torch.float32)
            # Keep the previous allocations alive: captured graphs still hold them.
            sets[ratio] = [bufs] + (have or [])
            self._source_proj_bufs = sets
        bufs = sets[ratio][0]
        return {name: buf[:num_tokens] for name, buf in bufs.items()}

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        # Build graph-compatible metadata against the padded static batch. The
        # batch still carries live seq/extend lens, so the online c128 prefill
        # plan remains batch-specific without constructing a second metadata set.
        metadata_batch = (
            static_forward_batch if static_forward_batch is not None else forward_batch
        )
        max_seq_len = (
            metadata_batch.max_seq_len_override or self.MAX_SEQ_LEN_FOR_CAPTURE
        )
        static_metadata = self._build_forward_metadata(
            metadata_batch,
            max_seq_len_override=max_seq_len,
            use_prefill_cuda_graph=True,
        )
        assert isinstance(capture_metadata, DSV4Metadata)
        capture_metadata.refresh_for_breakable_cuda_graph_replay_(static_metadata)
        self.forward_metadata = capture_metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[
                    DSV4Metadata,
                    DSV4RawDecodeMetadata,
                    DSV4RawVerifyMetadata,
                ],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_req = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )
        if self.is_draft_runner:
            # Draft-extend SWA write-target buffer; bound as a [:num_tokens]
            # view and refilled outside the graph each step.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                max_num_tokens, dtype=torch.int32, device=self.device
            )
        # Verify metadata never extracts the mask. No skip_prefill notion here.
        self._verify_mask = maybe_create_verify_mask(
            is_draft_runner=self.is_draft_runner,
            skip_prefill=False,
            max_bs=max_bs,
            max_context_len=self.max_context_len,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            is_read=False,
        )

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self._verify_mask

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        bucket_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket]
        chosen_metadata = bucket_metadata.get(bs)
        if chosen_metadata is None:
            bucket_metadata[bs] = temp_metadata
            self.forward_metadata = temp_metadata
            return
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c0_flashmla_metadata = _create_flashmla_metadata()
            if core.has_c4:
                core.c4_flashmla_metadata = _create_flashmla_metadata()
            if core.has_c128:
                core.c128_flashmla_metadata = _create_flashmla_metadata()
            if 1 in core.low_ratios:
                core.c1_flashmla_metadata = _create_flashmla_metadata()
            if 2 in core.low_ratios:
                core.c2_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    # ---- DeepSeek V4.1 ratio 1/2 compressor and indexer, torch bring-up path ----

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch: ForwardBatch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Runs on every ratio 1/2 layer before its attention."""
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.encoder_swa_replay:
            run_compressor = False
        if dsa_use_prefill_cp(forward_batch) and forward_batch.forward_mode.is_extend():
            self._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                run_compressor=run_compressor,
                run_indexer=run_indexer,
            )
            return
        meta = self.forward_metadata
        hoisted_req = getattr(meta, "low_ratio_req_indices", None)
        hoisted_pos = getattr(meta, "low_ratio_pos_i64", None)
        if (
            hoisted_req is not None
            and hoisted_pos is not None
            and hoisted_pos.shape[0] == positions.shape[0]
        ):
            # Bucket-sized under the prefill graph; an eager break sees the
            # live rows only and falls through.
            req, pos = hoisted_req, hoisted_pos
        else:
            req = token_req_indices(forward_batch, num_tokens=positions.shape[0])
            # Every consumer takes int32 or int64 positions; keep the caller's.
            pos = positions
        if (
            forward_batch.forward_mode.is_extend()
            and self._low_ratio_in_prefill_graph()
        ):
            bufs = self._source_projection_buffers(x.shape[0], layer.compress_ratio)
            _bcg_low_ratio_source_projections(layer, x, q_lora, pos, bufs)
            if run_compressor and layer.compressor is not None:
                self._low_ratio_compress_torch(
                    layer, x, req, pos, projected=(bufs["kv"], bufs.get("score"))
                )
            if run_indexer and layer.indexer is not None:
                self._low_ratio_index_topk_prefill_graph(
                    layer, pos, bufs["q"], bufs["w"]
                )
            return
        if run_compressor and layer.compressor is not None:
            self._low_ratio_compress(layer, x, req, pos, forward_batch)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk(layer, x, q_lora, req, pos, forward_batch)

    def _forward_low_ratio_sources_cp(
        self, *, layer, x, q_lora, positions, forward_batch, run_compressor, run_indexer
    ) -> None:
        # Every rank writes the whole prompt's compressed state, scoring its own rows.
        cp_meta = forward_batch.attn_cp_metadata
        total = int(cp_meta.total_seq_lens)
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens_cpu = tail.local_lens_cpu
            req_global, pos_global = tail.req_global, tail.pos_global
        else:
            q_lens_cpu = interleave_rows_per_request(
                _as_int_list(forward_batch.extend_seq_lens_cpu),
                get_parallel().attn_cp_rank,
                get_parallel().attn_cp_size,
            )
            req_global = token_req_indices(forward_batch, num_tokens=total)
            pos_global = forward_batch.positions[:total].to(torch.int64)
        num_local = sum(q_lens_cpu)
        if run_compressor and layer.compressor is not None:
            x_global = cp_materialize_global_token_order(
                x.contiguous(), forward_batch, torch.cuda.current_stream()
            )[:total]
            self._low_ratio_compress_torch(layer, x_global, req_global, pos_global)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk_dense(
                layer,
                x[:num_local],
                q_lora[:num_local],
                positions[:num_local].to(torch.int64),
                forward_batch,
                torch.tensor(q_lens_cpu, dtype=torch.int32, device=x.device),
                q_lens_cpu,
            )

    def _low_ratio_compress(self, layer, x, req, pos, forward_batch) -> None:
        if forward_batch.forward_mode.is_decode():
            self._low_ratio_compress_decode(layer, x, req, pos)
        elif (
            forward_batch.forward_mode.is_target_verify()
            and not self.is_dspark_draft
            and layer.compress_ratio in (1, 2)
            and layer.compressor.use_fused_compress
            and read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT
            and self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 1
            and x.shape[0]
            == forward_batch.batch_size * self.speculative_num_draft_tokens
        ):
            # Static verify is request-major with consecutive positions; compact
            # verify has variable block lengths and keeps the general path.
            self._low_ratio_compress_fused(
                layer, x, req, pos, draft_len=self.speculative_num_draft_tokens
            )
        else:
            self._low_ratio_compress_torch(
                layer,
                x,
                req,
                pos,
                fuse_index_store=(
                    forward_batch.forward_mode.is_target_verify()
                    and layer.compressor.use_fused_compress
                ),
            )

    def _low_ratio_in_prefill_graph(self) -> bool:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        return self.low_ratio_prefill_graph and is_in_breakable_cuda_graph()

    def _low_ratio_compress_decode(self, layer, x, req, pos) -> None:
        # Projection layout and fused-write support are fixed together at load time.
        if layer.compressor.use_fused_compress:
            self._low_ratio_compress_fused(layer, x, req, pos)
            return
        if layer.compress_ratio == 1:
            core = self.forward_metadata.core_metadata
            kv, _ = layer.compressor.project(x)
            slots = torch.where(
                core.c1_out_loc >= 0, core.c1_out_loc, torch.zeros_like(core.c1_out_loc)
            )
            self._low_ratio_write_group(
                layer,
                kv,
                slots,
                pos,
                fuse_index_store=(
                    x.is_cuda
                    and torch.version.cuda is not None
                    and _is_sm100_or_newer()
                ),
            )
            return
        if not (x.is_cuda and torch.version.cuda):
            self._low_ratio_compress_torch(layer, x, req, pos)
            return

        from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool

        core = self.forward_metadata.core_metadata
        state = self.token_to_kv_pool.get_attention_compress_states(layer.layer_id)
        kv, score = layer.compressor.project(x)
        pooled, group_pos, slots = c2_decode_pool(
            kv,
            score,
            pos,
            core.raw_out_loc,
            core.c2_out_loc,
            req,
            state.kv_score_buffer.kv,
            state.kv_score_buffer.score,
            state.kv_score_buffer.shape[0] - 1,
            ring_size=state.ring_size,
        )
        self._low_ratio_write_group(
            layer,
            pooled,
            slots,
            group_pos,
            fuse_index_store=_is_sm100_or_newer(),
        )

    def _low_ratio_compress_fused(self, layer, x, req, pos, *, draft_len=1) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_k_norm_rope_pack_store,
        )
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
            c1_decode_norm_rope_store,
            c2_decode_norm_rope_store,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        compressor = layer.compressor
        layer_id = layer.layer_id
        # Contiguous complex64 freqs_cis gives a real/imag-interleaved view without copying.
        freqs_cis = torch.view_as_real(layer.freqs_cis).flatten(-2)
        kv_cache = pool.get_extra_key_buffer(layer_id)
        page_size = pool.get_extra_key_page_size(layer_id)
        # The pool's page format: V4, or the V4.1 fp8 / fp4 layouts.
        kv_layout = pool.get_extra_key_layout(layer_id)
        assert kv_cache is not None

        if layer.compress_ratio == 1:
            # At ratio 1, c1_out_loc equals the int64 raw_out_loc supplied by the scheduler.
            latent = c1_decode_norm_rope_store(
                compressor.wkv(x),
                compressor.norm.weight.data,
                pos,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
                layout=kv_layout,
            )
            out_loc = core.c1_out_loc
        else:
            # CompressStatePool stores each request's pending pairs in a position ring.
            # KVAndScore rows use | kv | score |, addressed as req * ring_size + pos % ring_size.
            state = pool.get_attention_compress_states(layer_id)
            latent = c2_decode_norm_rope_store(
                compressor.project_fused(x),
                state.kv_score_buffer.kv_score,
                compressor.norm.weight.data,
                pos,
                req,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
                ring_size=state.ring_size,
                draft_len=draft_len,
                layout=kv_layout,
            )
            out_loc = core.c2_out_loc

        indexer = layer.indexer
        if indexer is not None and indexer.owns_k:
            # out_loc is -1 for an incomplete group and 0 for padding;
            # the kernel suppresses both stores.
            assert out_loc is not None
            if pool.low_ratio_index_k_is_split(layer_id):
                # ROCm keeps the index-K payload and scales in two buffers
                from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
                    index_k_norm_rope_pack_store_split,
                )

                index_k_norm_rope_pack_store_split(
                    indexer.forward_wk(latent),
                    indexer.k_norm.weight.data,
                    indexer.k_norm.eps,
                    freqs_cis,
                    pos,
                    out_loc,
                    pool.get_index_k_fp4_payload_buffer(layer_id),
                    pool.get_index_k_fp4_scale_buffer(layer_id),
                    ratio=layer.compress_ratio,
                )
                return
            index_k_norm_rope_pack_store(
                indexer.forward_wk(latent),
                indexer.k_norm.weight.data,
                indexer.k_norm.eps,
                freqs_cis,
                pos,
                out_loc,
                pool.get_index_k_with_scale_buffer(layer_id),
                ratio=layer.compress_ratio,
            )

    def _low_ratio_compress_torch(
        self, layer, x, req, pos, projected=None, *, fuse_index_store=False
    ) -> None:
        core = self.forward_metadata.core_metadata
        num_tokens = pos.shape[0]
        kv, score = projected if projected is not None else layer.compressor.project(x)
        if not num_tokens:
            return
        if layer.compress_ratio == 1:
            self._low_ratio_write_group(
                layer,
                kv,
                core.c1_out_loc[:num_tokens],
                pos,
                fuse_index_store=fuse_index_store,
            )
            return

        partner_kv, partner_score = self._low_ratio_pair_partners(
            layer_id=layer.layer_id,
            kv=kv,
            score=score,
            req=req,
            pos=pos,
            pad=core.raw_out_loc[:num_tokens] == 0,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        group_pos = torch.where(pos % 2 == 1, pos - 1, pos)
        out_loc = core.c2_out_loc[:num_tokens]
        slots = torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc))
        self._low_ratio_write_group(
            layer, pooled, slots, group_pos, fuse_index_store=fuse_index_store
        )

    def _low_ratio_pair_partners(
        self, *, layer_id, kv, score, req, pos, pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.token_to_kv_pool.get_attention_compress_states(layer_id)
        ring = state.ring_size
        num_tokens = pos.shape[0]
        if not num_tokens:
            return kv, score

        read_pos = (pos - 1).masked_fill(pad, -1)
        carried = state.get_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, read_pos)
        )
        in_batch = torch.zeros_like(pad)
        in_batch[1:] = (
            (req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1) & ~pad[1:] & ~pad[:-1]
        )
        partner_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried.kv)
        partner_score = torch.where(
            in_batch[:, None], torch.roll(score, 1, 0), carried.score
        )

        # All reads above precede every write, and keeping only a request's last
        # ring_size rows leaves each write a distinct live slot, even when a
        # prefill chunk is longer than the ring.
        keep = ~pad
        if num_tokens > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        state.set_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, write_pos),
            KVAndScore.from_kv_score(kv=kv, score=score),
        )
        return partner_kv, partner_score

    def _low_ratio_write_group(
        self,
        layer,
        pooled,
        slots,
        group_pos,
        *,
        fuse_index_store=False,
    ) -> None:
        pool = self.token_to_kv_pool
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        # Index keys come from the pre-RoPE latent, so publish them first. Stored
        # as fp4 (per-32 ue8m0, no hadamard), matching the reference indexer.
        if layer.indexer is not None and layer.indexer.owns_k:
            if (
                fuse_index_store
                and latent.is_cuda
                and torch.version.cuda is not None
                and latent.dtype == torch.bfloat16
                and layer.indexer.index_head_dim == 128
            ):
                from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
                    index_k_rope_pack,
                )

                indexer = layer.indexer
                k = indexer.k_norm(indexer.forward_wk(latent))
                index_k_rope_pack(
                    k,
                    freqs,
                    indexer.rope_head_dim,
                    cache=pool.get_index_k_with_scale_buffer(layer.layer_id),
                    loc=slots,
                )
            else:
                pool.set_index_k_fp4(
                    layer_id=layer.layer_id,
                    loc=slots,
                    cache_k=layer.indexer.index_keys(latent, freqs),
                )
        if pool.get_extra_key_layout(layer.layer_id) is KVLayout.V41_FP4:
            # The fp4 cache stores e2m1 codes: the kernel rotates the tail and
            # rounds once, with no fake quantization in between.
            pool.set_extra_key_buffer_fused(
                layer_id=layer.layer_id, loc=slots, cache_k=latent, freqs_cis=freqs
            )
            return
        # The fp8 FlashMLA caches requantize the FP4/E4M3 latent into their layout.
        latent = _rope_fq4(latent, freqs, layer.rope_head_dim, compressed_kv=True)
        pool.set_extra_key_buffer_fused(
            layer_id=layer.layer_id, loc=slots, cache_k=latent
        )

    def _low_ratio_index_topk(self, layer, x, q_lora, req, pos, forward_batch) -> None:
        is_decode_or_verify = (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        if is_decode_or_verify:
            if _is_sm100_or_newer():
                # DeepGEMM pairs verify rows by request id; decode has one row each.
                req_ids = None if forward_batch.forward_mode.is_decode() else req
                self._low_ratio_index_topk_decode(layer, x, q_lora, pos, req_ids)
            else:
                self._low_ratio_index_topk_sm90_decode(layer, x, q_lora, req, pos)
        elif (
            self._use_dense_fp4_prefill_indexer(forward_batch) and _is_sm100_or_newer()
        ):
            self._low_ratio_index_topk_extend(layer, x, q_lora, pos, forward_batch)
        else:
            self._low_ratio_index_topk_torch(layer, x, q_lora, req, pos)

    @staticmethod
    def _use_dense_fp4_prefill_indexer(forward_batch) -> bool:
        return (
            not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
            and _has_dense_fp4_indexer()
            and forward_batch.forward_mode.is_extend()
            and forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

    def _low_ratio_index_topk_extend(
        self, layer, x, q_lora, pos, forward_batch
    ) -> None:
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens, q_lens_cpu = tail.extend_seq_lens, tail.extend_seq_lens_cpu
        else:
            q_lens = forward_batch.extend_seq_lens
            q_lens_cpu = _as_int_list(forward_batch.extend_seq_lens_cpu)
        assert q_lens_cpu is not None
        self._low_ratio_index_topk_dense(
            layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
        )

    def _low_ratio_index_topk_dense(
        self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)

        seq_lens_cpu = _as_int_list(forward_batch.seq_lens_cpu)
        assert seq_lens_cpu is not None
        device = pos.device
        # Visible compressed positions per request at its newest token; the
        # per-token count (pos + 1) // ratio bounds each row below.
        lc_per_req = [s // ratio for s in seq_lens_cpu]
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
        slot_chunks, starts, start = [], [], 0
        for r, lc in enumerate(lc_per_req):
            starts.append(start)
            if lc == 0:
                continue
            j = torch.arange(lc, device=device)
            slot_chunks.append(
                self.req_to_token[req_pool_indices[r], j * ratio].to(torch.int64)
                // ratio
            )
            start += lc
        num_tokens = pos.shape[0]
        if not slot_chunks or num_tokens == 0:
            if indexer.is_candidate_source:
                # no row to publish for; the consumers return here as well
                self.forward_metadata.candidate_metadata = None
            return
        k_slots = torch.cat(slot_chunks)
        k_fp4, k_sf = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
        num_heads = q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
        q_sf = q_sf.view(num_tokens, num_heads)
        weights = indexer.head_weights(x).float()
        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        inputs = self._prefill_indexer_inputs(
            layer,
            (q_fp4, q_sf),
            (k_fp4, k_sf),
            weights,
            compress_lens,
            ks,
            lc_per_req,
            list(q_lens_cpu),
        )
        topk = indexer.index_topk
        selected = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
        if indexer.is_candidate_source:
            self._publish_prefill(
                self.candidate_indexer.publish_prefill(inputs, selected)
            )
        elif indexer.uses_candidates:
            self.candidate_indexer.select_prefill(
                self.forward_metadata.candidate_metadata, inputs, selected
            )
        else:
            plain_prefill_topk(inputs, selected)
        # ascending positions, padding last: the layout the consumers expect
        unselected = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
        chosen = selected != unselected
        page_indices[:num_tokens, :topk] = torch.where(
            chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
        ).to(torch.int32)
        if raw_indices is not None:
            raw_indices[:num_tokens, :topk] = torch.where(
                chosen, selected - ks[:, None], -1
            )

    def _prefill_indexer_inputs(
        self, layer, q, kv, weights, compress_lens, ks, lc_per_req, q_lens_cpu
    ) -> PrefillIndexerInputs:
        pool = self.token_to_kv_pool
        ratio = layer.compress_ratio
        num_tokens = q[0].shape[0]
        index_page_size = pool.get_index_k_page_size(ratio)
        k_cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
        assert k_cache.dim() == 2
        return PrefillIndexerInputs(
            q_fp4=q[0],
            q_sf=q[1],
            weights=weights,
            compress_lens=compress_lens,
            request_starts=ks,
            lens_per_request=lc_per_req,
            rows_per_request=q_lens_cpu,
            kv=kv,
            k_cache=k_cache.view(k_cache.shape[0], index_page_size, 1, 68),
            page_size=index_page_size,
            kv_page_table=self.forward_metadata.core_metadata.page_table[:num_tokens],
            kv_page_size=self.page_size,
            compress_ratio=ratio,
        )

    def _tail_lens_to_publish(self) -> Optional[List[int]]:
        """Rows per request of the late-layer tail, when the layers after the
        switch will consume what is published now."""
        tail_metadata = self.tail_forward_metadata
        if tail_metadata is None or tail_metadata is self.forward_metadata:
            return None
        tail = tail_metadata.late_layer_tail
        return (
            tail.local_lens_cpu
            if tail.cp_metadata is not None
            else tail.extend_seq_lens_cpu
        )

    def _publish_prefill(self, published: CandidateMetadata) -> None:
        self.forward_metadata.candidate_metadata = published
        tail_lens = self._tail_lens_to_publish()
        if tail_lens is not None:
            self.tail_forward_metadata.candidate_metadata = (
                self.candidate_indexer.prefill_tail(published, tail_lens)
            )

    def _publish_prefill_masks(self, masks: CandidateMasks) -> None:
        """The torch prefill path's inline masks (see the TODO on CandidateMasks)."""
        self.forward_metadata.candidate_metadata = masks
        tail_lens = self._tail_lens_to_publish()
        if tail_lens is not None:
            self.tail_forward_metadata.candidate_metadata = cut_request_masks(
                masks, tail_lens
            )

    def _low_ratio_index_topk_prefill_graph(self, layer, pos, q, w) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None, f"no prefill graph indexer metadata for {ratio = }"
        assert indexer.n_local_heads == indexer.n_heads
        width = metadata.max_compressed_seq_len
        if indexer.uses_candidates or indexer.is_candidate_source:
            # Every reachable block is a candidate inside the window, so the
            # two-level selection collapses to the plain top-k below.
            assert (
                width <= indexer.candidate_topk_blocks * indexer.candidate_block_size
            ), f"prefill graph indexer width {width} exceeds the candidate window"

        num_tokens, num_heads = q.shape[0], q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, 1, num_heads, 64)
        q_sf = q_sf.view(num_tokens, 1, num_heads)
        weights = w.float()

        k_cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
        assert k_cache.dim() == 2
        page_size = metadata.compressed_page_size
        k_cache = k_cache.view(k_cache.shape[0], page_size, 1, 68)

        lens = metadata.compressed_seq_lens
        page_table = metadata.page_table
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        topk = min(indexer.index_topk, width)
        columns = torch.arange(width, device=lens.device)
        for rows, plan in metadata.row_chunks():
            logits = deep_gemm_fp4_paged_mqa_logits(
                (q_fp4[rows], q_sf[rows]),
                k_cache,
                weights[rows],
                lens[rows],
                page_table[rows],
                plan,
                width,
            )
            lens_c = lens[rows].unsqueeze(-1)
            # Columns past a row's length hold garbage.
            s = logits.masked_fill(columns[None, :] >= lens_c, -torch.inf)
            idx = s.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            reach = idx < lens_c
            slots = page_table[rows].gather(-1, idx // page_size) * page_size + (
                idx % page_size
            )
            page_indices[rows, :topk] = torch.where(reach, slots, -1).to(torch.int32)
            if raw_indices is not None:
                raw_indices[rows, :topk] = torch.where(reach, idx, -1).to(torch.int32)

    def _low_ratio_index_topk_decode(self, layer, x, q_lora, pos, req=None) -> None:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            skip_low_ratio_indexer,
        )

        if skip_low_ratio_indexer(layer.compress_ratio):
            # The compressor still writes index K for later, longer contexts.
            return

        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None, f"no decode indexer metadata for {ratio = }"

        # fp4 query as (payload, scale), kernel layout [bs, 1, n_heads, dim]. The
        # kernel sums head scores locally, so the indexer heads must be replicated.
        assert indexer.n_local_heads == indexer.n_heads
        if (
            x.is_cuda
            and torch.version.cuda is not None
            and x.dtype == torch.bfloat16
            and indexer.index_head_dim == 128
        ):
            from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
                index_q_rope_pack_weights,
            )

            q, _ = indexer.wq_b(q_lora)
            q = q.view(q.shape[0], indexer.n_local_heads, indexer.index_head_dim)
            # The fused pack also computes head_weights(x).float(), same rounding.
            q_fp4, q_sf, weights = index_q_rope_pack_weights(
                q,
                torch.view_as_real(layer.freqs_cis).flatten(-2),
                pos,
                indexer.head_weights_raw(x),  # [bs, n_local] bf16, n32k5120
                indexer.head_weight_scale,
            )
        else:
            q = indexer.queries(q_lora, layer.freqs_cis[pos])
            q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
            weights = indexer.head_weights(x).float()  # [bs, n_local]
        bs = q.shape[0]
        q_fp4 = q_fp4.view(bs, 1, indexer.n_local_heads, 64)
        q_sf = q_sf.view(bs, 1, indexer.n_local_heads)

        k_cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
        assert k_cache.dim() == 2
        # Index pool page (64 slots); metadata.page_table is expanded to match.
        page_size = metadata.compressed_page_size
        k_cache = k_cache.view(
            k_cache.shape[0], page_size, 1, 68
        )  # fp4: 64 payload + 4 scale

        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        inputs = IndexerInputs(
            q_fp4,
            q_sf,
            k_cache,
            weights,
            metadata,
            request_ids=req,  # one per query row; verify rows of a request share one
        )
        candidate_layer = not _every_request_fits()
        # use special selection for candidate layers
        if indexer.uses_candidates and candidate_layer:
            return self.candidate_indexer.select_decode(
                self.forward_metadata.candidate_metadata,
                inputs,
                page_indices,
                raw_indices,
            )
        if indexer.is_candidate_source and candidate_layer:
            self.forward_metadata.candidate_metadata = (
                self.candidate_indexer.publish_decode(inputs, page_indices, raw_indices)
            )
            return
        if isinstance(metadata.deep_gemm_metadata, list):
            topk_plans = metadata.topk_metadata_chunks
            assert not metadata.use_topk_v2 or topk_plans is not None
            for chunk_idx, (rows, plan) in enumerate(metadata.row_chunks()):
                logits = deep_gemm_fp4_paged_mqa_logits(
                    (q_fp4[rows], q_sf[rows]),
                    k_cache,
                    weights[rows],
                    metadata.compressed_seq_lens[rows],
                    metadata.page_table[rows],
                    plan,
                    metadata.max_compressed_seq_len,
                )
                # TODO(dark): add bf16 topk
                topk_transform_paged_from_metadata(
                    logits,
                    metadata,
                    page_indices,
                    raw_indices,
                    rows=rows,
                    topk_metadata=(
                        topk_plans[chunk_idx] if topk_plans is not None else None
                    ),
                )
        else:
            logits = deep_gemm_fp4_paged_mqa_logits(
                (q_fp4, q_sf),
                k_cache,
                weights,
                metadata.compressed_seq_lens,
                metadata.page_table,
                metadata.deep_gemm_metadata,
                metadata.max_compressed_seq_len,
            )
            # TODO(dark): add bf16 topk
            topk_transform_paged_from_metadata(
                logits, metadata, page_indices, raw_indices
            )

    # TODO(candidate): Hopper decode still publishes / consumes masks inline (torch
    # top-k); move into the candidate indexer with the prefill paths.
    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        bs = req.shape[0]
        assert pos.shape[0] == bs, (
            f"decode expects one token per request, {pos.shape=} {bs=}"
        )
        if bs == 0:
            return
        lens = (pos + 1) // ratio
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None
        # V4 reserves the replay bound in metadata; visibility stays on device.
        # A capture-time length read would both synchronize and truncate replay.
        lmax = min(metadata.max_compressed_seq_len, self.req_to_token.shape[1] // ratio)
        if lmax == 0:
            return
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        j = torch.arange(lmax, device=pos.device)
        valid = j[None, :] < lens[:, None]
        slots = (
            self.req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64)
            // ratio
        )
        slots = slots.masked_fill(~valid, 0)
        table = pool.get_index_k_with_scale_buffer(layer.layer_id)
        s = fp4_index_logits_decode(
            q, weights, slots, lens, table, table.shape[1] // 68
        )
        if indexer.is_candidate_source:
            mask = select_candidate_blocks(
                s,
                lens[:, None],
                topk_blocks=indexer.candidate_topk_blocks,
                block_size=indexer.candidate_block_size,
            )
            self.forward_metadata.candidate_metadata = CandidateMasks(mask=mask)
        elif indexer.uses_candidates:
            # Published this step by the candidate-source layer's decode pass above.
            consume = published_masks(self.forward_metadata.candidate_metadata).mask
            assert torch.is_tensor(consume) and consume.shape[0] == bs, (
                "candidate mask missing for decode"
            )
            s = s.masked_fill(~consume[:, :lmax], -torch.inf)
        k = min(indexer.index_topk, lmax)
        idx = s.topk(k, dim=-1, sorted=False).indices
        if indexer.uses_candidates and not indexer.is_candidate_source:
            idx = mask_topk_scores(s, idx)
            idx = idx.masked_fill(idx < 0, lmax)
        idx = idx.sort(dim=-1).values
        reach = idx < lens[:, None]
        page_indices[:bs, :k] = torch.where(
            reach, slots.gather(1, idx.clamp_max(lmax - 1)), -1
        ).to(torch.int32)
        if raw_indices is not None:
            raw_indices[:bs, :k] = torch.where(reach, idx, -1).to(torch.int32)

    # TODO(candidate): torch prefill still publishes / consumes masks inline; same
    # move as above.
    def _low_ratio_index_topk_torch(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        # Attention scans sparse_topk_lengths slots and skips -1 entries.
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        # A compressed position is visible once the query has passed its last token.
        compress_lens = (pos + 1) // ratio
        topk = indexer.index_topk
        publish = [] if indexer.is_candidate_source else None
        consume = (
            published_masks(self.forward_metadata.candidate_metadata).request_masks
            if indexer.uses_candidates
            else None
        )
        for b, r in enumerate(torch.unique_consecutive(req).tolist()):
            tok = (req == r).nonzero().squeeze(1)
            lens = compress_lens[tok]
            lc = int(lens.max().item())
            if lc == 0:
                # Consumers address masks by request position, including empty requests.
                if publish is not None:
                    publish.append(
                        torch.zeros(0, 0, dtype=torch.bool, device=pos.device)
                    )
                continue
            j = torch.arange(lc, device=pos.device)
            slots_j = self.req_to_token[r, j * ratio].to(torch.int64) // ratio
            # Dequantize only this request's visible K rows; the table is pool-sized.
            index_k = pool.get_low_ratio_index_k_dequant(layer.layer_id, slots_j)
            k = min(topk, lc)
            # Every step below is per query row; chunk rows so the [rows, heads, lc]
            # bf16 scores stay under the budget (16 GiB at once for a 16k-token prompt).
            rows_per_chunk = max(
                1,
                _TORCH_INDEXER_SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2),
            )
            masks = [] if publish is not None else None
            for start in range(0, tok.numel(), rows_per_chunk):
                rows = slice(start, start + rows_per_chunk)
                tok_c, lens_c = tok[rows], lens[rows]
                s = indexer.scores(q[tok_c], index_k, weights[tok_c])
                s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
                if masks is not None:
                    masks.append(
                        select_candidate_blocks(
                            s,
                            lens_c[:, None],
                            topk_blocks=indexer.candidate_topk_blocks,
                            block_size=indexer.candidate_block_size,
                        )
                    )
                elif consume is not None:
                    s = s.masked_fill(~consume[b][rows], -torch.inf)
                idx = s.topk(k, dim=-1, sorted=False).indices
                if consume is not None and masks is None:
                    idx = mask_topk_scores(s, idx)
                    idx = idx.masked_fill(idx < 0, lc)
                idx = idx.sort(dim=-1).values
                reach = idx < lens_c[:, None]
                page_indices[tok_c, :k] = torch.where(
                    reach, slots_j[idx.clamp_max(lc - 1)], -1
                ).to(torch.int32)
                if raw_indices is not None:
                    raw_indices[tok_c, :k] = torch.where(reach, idx, -1).to(torch.int32)
            if masks is not None:
                publish.append(torch.cat(masks) if len(masks) > 1 else masks[0])
        if publish is not None:
            self._publish_prefill_masks(CandidateMasks(request_masks=publish))

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Idle always re-translates at store time: its metadata may be stale, and
        translating the zero-padded out_cache_loc writes to the dummy slot."""
        metadata = self.forward_metadata
        if self.token_to_kv_pool.request_window is not None:
            layout = metadata.core_attn_metadata.request_window_layout
            self.token_to_kv_pool.request_window.activate(layout)
            return layout.write_loc
        if isinstance(metadata, DSV4Metadata) and metadata.late_layer_tail is not None:
            # The tail's q rows are a subset of the extend, so the full
            # out_cache_loc below would be the wrong length; the tail owns its own.
            return metadata.late_layer_tail.swa_out_cache_loc
        out_cache_loc = forward_batch.out_cache_loc
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        cached = core.swa_out_cache_loc if core is not None else None
        if (
            cached is not None
            and not forward_batch.forward_mode.is_idle()
            and cached.shape[0] == out_cache_loc.shape[0]
        ):
            return cached
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
            torch.int32
        )

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        swa_loc = self.get_swa_out_cache_loc(forward_batch)
        self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
            layer_id=layer_id,
            swa_loc=swa_loc,
            cache_k=swa_k,
        )

    def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        result = self._forward_attention(q, k, v, layer, forward_batch, *args, **kwargs)
        window = self.token_to_kv_pool.request_window
        if (
            window is not None
            and not self.is_dspark_draft
            and not forward_batch.forward_mode.is_idle()
        ):
            window.commit(self.token_to_kv_pool._swa_local_layer_id(layer.layer_id))
        return result

    def _forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio != 0:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)
                extra_topk_lengths = core_attn_metadata.sparse_topk_lengths(
                    compress_ratio
                )

            swa_kv_page_size = token_to_kv_pool.swa_kv_pool.page_size
            assert swa_k_cache.ndim == 2
            # The kernel detects each cache's format from the last dim of this
            # view: 584 (V4), 528 (V4.1 fp8) or 288 (V4.1 fp4, extra cache only).
            k_cache_total_dim = token_to_kv_pool.get_swa_key_bytes_per_token()
            swa_k_cache = swa_k_cache[:, : swa_kv_page_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_kv_page_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
                extra_total_dim = token_to_kv_pool.get_extra_key_bytes_per_token(
                    layer_id
                )
                extra_k_cache = extra_k_cache[
                    :, : extra_page_size * extra_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    extra_page_size,
                    1,
                    extra_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            def match_num_queries(x, value):
                if x is None or x.shape[0] == q.shape[0]:
                    return x
                if x.shape[0] > q.shape[0]:
                    return x[: q.shape[0]]
                return _pad_tensor_to_size(x, q.shape[0], value=value)

            swa_page_indices = match_num_queries(swa_page_indices, value=0)
            swa_topk_lengths = match_num_queries(swa_topk_lengths, value=1)
            extra_indices = match_num_queries(extra_indices, value=-1)
            extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)

            if self.trtllm_attn:
                # The uniform-FP8 pool is readable only by trtllm-gen.
                return self._forward_trtllm(
                    q=q,
                    layer=layer,
                    compress_ratio=compress_ratio,
                    core_attn_metadata=core_attn_metadata,
                    forward_batch=forward_batch,
                    attn_sink=attn_sink,
                    swa_page_indices=swa_page_indices,
                    extra_indices=extra_indices,
                    extra_topk_lengths=extra_topk_lengths,
                )

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert swa_page_indices.shape[-1] % 64 == 0, (
                f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            )
            if extra_indices is not None:
                assert extra_indices.shape[-1] % 64 == 0, (
                    f"{extra_indices.shape=}'s last dimension is not aligned to 64"
                )

            # sparse_prefill_fwd does not support SM120. The tail stays dense: its
            # window floor lives in swa_page_indices, which the chunk cache ignores.
            if (
                forward_batch.forward_mode.is_extend_without_speculative()
                and not get_platform().is_sm120
                and self.forward_metadata.late_layer_tail is None
                and token_to_kv_pool.request_window is None
                and (
                    q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD
                    or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
                )
            ):
                if use_dsv4_q8kv8_sparse_prefill(self.dsv4_prefill_backend):
                    return self._forward_prefill_sparse_q8kv8(
                        q=q,
                        layer_id=layer_id,
                        compress_ratio=compress_ratio,
                        forward_batch=forward_batch,
                        token_to_kv_pool=token_to_kv_pool,
                        core_attn_metadata=core_attn_metadata,
                        attn_sink=attn_sink,
                    )
                return self._forward_prefill_sparse(
                    q=q,
                    layer_id=layer_id,
                    compress_ratio=compress_ratio,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    core_attn_metadata=core_attn_metadata,
                    attn_sink=attn_sink,
                )

            if (
                self.is_dsv41
                and get_platform().is_sm100
                and can_use_swapab_attention(
                    q,
                    swa_k_cache,
                    extra_k_cache,
                    layer.tp_q_head_num,
                    self.head_dim_v,
                    self.softmax_scale,
                )
                and (
                    forward_batch.forward_mode.is_decode()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend_v2()
                )
            ):
                from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
                    swapab_attention,
                )

                return swapab_attention(
                    q[..., :16, :],
                    swa_k_cache,
                    swa_page_indices,
                    swa_topk_lengths,
                    attn_sink,
                    extra_k_cache,
                    extra_indices,
                    extra_topk_lengths,
                )

            if get_platform().is_sm120:
                from sglang.kernels.ops.attention.flash_mla_sm120 import (
                    SM120_DECODE_MAX_TOKENS,
                    flash_mla_with_kvcache_sm120,
                )

                # The pad to 64 heads only serves the decode kernel's h_q
                # specialization; the prefill kernel takes arbitrary h_q, so
                # drop it instead of attending on garbage heads (4x the work
                # at attn-TP 4).
                real_heads = layer.tp_q_head_num
                if q.shape[0] > SM120_DECODE_MAX_TOKENS:
                    if q.shape[-2] > real_heads:
                        q = q[..., :real_heads, :].contiguous()
                    if attn_sink is not None and attn_sink.shape[0] > real_heads:
                        attn_sink = attn_sink[:real_heads]

                o = flash_mla_with_kvcache_sm120(
                    q=q,
                    k_cache=swa_k_cache,
                    head_dim_v=self.head_dim_v,
                    softmax_scale=self.softmax_scale,
                    indices=swa_page_indices,
                    topk_length=swa_topk_lengths,
                    attn_sink=attn_sink,
                    extra_k_cache=extra_k_cache,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=extra_topk_lengths,
                )[0]
            else:
                if _is_xpu:
                    from sgl_kernel import flash_mla_with_kvcache
                else:
                    from sgl_kernel.flash_mla import flash_mla_with_kvcache

                if self.is_dsv41:
                    _maybe_precompute_flashmla_sched_meta(
                        flashmla_metadata,
                        q=q,
                        indices=swa_page_indices,
                        topk_length=swa_topk_lengths,
                        extra_indices=extra_indices,
                        extra_topk_length=extra_topk_lengths,
                    )
                o = flash_mla_with_kvcache(
                    q=q,
                    k_cache=swa_k_cache,
                    head_dim_v=self.head_dim_v,
                    block_table=None,
                    cache_seqlens=None,
                    tile_scheduler_metadata=flashmla_metadata,
                    softmax_scale=self.softmax_scale,
                    is_fp8_kvcache=True,
                    indices=swa_page_indices,
                    topk_length=swa_topk_lengths,
                    attn_sink=attn_sink,
                    extra_k_cache=extra_k_cache,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=extra_topk_lengths,
                )[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def _forward_prefill_sparse(
        self,
        q: torch.Tensor,
        layer_id: int,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        """Unified prefill via flash_mla_sparse_fwd. Replaces the
        flash_mla_with_kvcache call on the extend path. Per request,
        positionally gathers the SWA window (always) and the compressed
        cache (c4/c128) into a flat bf16 workspace, then lets
        flash_mla_sparse_fwd consume the workspace via per-query rebased
        indices. Chunk-invariant scaffolding lives in
        ``self.forward_metadata.sparse_prefill_cache``.
        """
        if _is_xpu:
            from sgl_kernel import flash_mla_sparse_fwd
        else:
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # q is (b, 1, h_q, d_qk); flash_mla_sparse_fwd takes (s_q, h_q, d_qk).
        q_flat = q.squeeze(1)

        cache = self.forward_metadata.sparse_prefill_cache
        if cache is None:
            cache = self._build_sparse_prefill_chunk_cache(
                forward_batch, core_attn_metadata, num_qo_tokens=q_flat.shape[0]
            )
            self.forward_metadata.sparse_prefill_cache = cache

        # Resolve the workspace + indices for this ratio, then dequant
        # SWA + compressed regions directly into the workspace (no torch.cat).
        compressed_slice = None
        extra_k_cache = None
        extra_page_size = None
        flat_token_ids = None
        if compress_ratio == 0:
            workspace = self.sparse_prefill_workspace.get(cache.swa_token_ids.shape[0])
            combined_indices = cache.c0_combined_indices
            combined_lens = cache.c0_combined_lens
            swa_slice = workspace
        else:
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            flat_token_ids, combined_indices, combined_lens = cache.layer_inputs(
                compress_ratio, core_attn_metadata, extra_page_size
            )
            n_compressed = flat_token_ids.shape[0]
            workspace = self.sparse_prefill_workspace.get(
                n_compressed + cache.swa_token_ids.shape[0]
            )
            compressed_slice = workspace[:n_compressed]
            swa_slice = workspace[n_compressed:]

        if compressed_slice is not None:
            dequantize_k_cache_paged(
                extra_k_cache,
                flat_token_ids,
                page_size=extra_page_size,
                out=compressed_slice,
                layout=token_to_kv_pool.get_extra_key_layout(layer_id),
            )
        dequantize_k_cache_paged(
            token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            cache.swa_token_ids,
            page_size=cache.swa_page_size,
            out=swa_slice,
            layout=token_to_kv_pool.get_swa_key_layout(),
        )
        kv = workspace

        o, _, _ = flash_mla_sparse_fwd(
            q=q_flat,
            kv=kv,
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            d_v=self.head_dim_v,
            attn_sink=attn_sink,
            topk_length=combined_lens,
        )
        return o

    def _prepare_q8kv8_q_and_sink(
        self,
        q: torch.Tensor,
        attn_sink: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad TP-local heads to the SM90 kernel's 64-head CTA granularity."""
        num_tokens, num_heads, head_dim = q.shape
        padded_heads = q8kv8_padded_num_heads(num_heads)

        qpad = getattr(self, "_q8kv8_qpad_buf", None)
        if (
            qpad is None
            or qpad.shape[0] < num_tokens
            or qpad.shape[1] != padded_heads
            or qpad.shape[2] != head_dim
            or qpad.device != q.device
        ):
            qpad = torch.empty(
                (num_tokens, padded_heads, head_dim),
                dtype=fp8_dtype,
                device=q.device,
            )
            self._q8kv8_qpad_buf = qpad

        qpad = qpad[:num_tokens]

        q_fp8, _ = cast_q_fp8_for_q8kv8_prefill(
            q,
            padded_num_heads=padded_heads,
            out=qpad,
        )

        sink_pad = getattr(self, "_q8kv8_attn_sink_pad", None)
        if (
            sink_pad is None
            or sink_pad.shape != (padded_heads,)
            or sink_pad.device != q.device
        ):
            sink_pad = torch.zeros(padded_heads, dtype=torch.float32, device=q.device)
            self._q8kv8_attn_sink_pad = sink_pad

        sink_pad[:num_heads].copy_(attn_sink.reshape(-1)[:num_heads])
        if padded_heads > num_heads:
            sink_pad[num_heads:].zero_()

        scale = getattr(self, "_q8kv8_identity_scale", None)
        if scale is None or scale.device != q.device:
            scale = torch.ones((), dtype=torch.float32, device=q.device)
            self._q8kv8_identity_scale = scale

        return q_fp8, sink_pad, scale, num_heads

    def _forward_prefill_sparse_q8kv8(
        self,
        q: torch.Tensor,
        layer_id: int,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        """Experimental DeepSeek-V4 sparse prefill path using Q8KV8 kernels.

        This mirrors ``_forward_prefill_sparse``'s cache/index construction, but
        writes the gathered KV workspace as FP8 and calls the SM90 Q8KV8 sparse
        prefill kernel. The path is selected by ``--dsv4-prefill-backend
        flashmla_sparse_q8``; ``SGLANG_DSV4_Q8KV8_PREFILL`` remains as a debug
        override for focused runtime validation.
        """

        from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import (
            sparse_mla_q8kv8_prefill_fwd,
        )

        q_flat = q.squeeze(1)
        if q_flat.ndim != 3:
            raise ValueError(
                f"Q8KV8 sparse prefill expects 3D Q after squeeze, got {q_flat.shape}"
            )

        if attn_sink.numel() != q_flat.shape[1]:
            raise ValueError(
                f"attn_sink has {attn_sink.numel()} heads but Q has "
                f"{q_flat.shape[1]} local heads"
            )

        q_fp8, attn_sink_pad, identity_scale, active_heads = (
            self._prepare_q8kv8_q_and_sink(q_flat, attn_sink)
        )

        if not getattr(self, "_q8kv8_sparse_prefill_log_emitted", False):
            logger.info(
                "DSV4_Q8KV8_SPARSE_PREFILL_HIT layer_id=%s "
                "compress_ratio=%s q_shape=%s padded_heads=%s d_v=%s",
                layer_id,
                compress_ratio,
                tuple(q_flat.shape),
                q_fp8.shape[1],
                self.head_dim_v,
            )
            self._q8kv8_sparse_prefill_log_emitted = True

        cache = self.forward_metadata.sparse_prefill_cache
        if cache is None:
            cache = self._build_sparse_prefill_chunk_cache(
                forward_batch, core_attn_metadata, num_qo_tokens=q_flat.shape[0]
            )
            self.forward_metadata.sparse_prefill_cache = cache

        compressed_slice = None
        extra_k_cache = None
        extra_page_size = None
        flat_token_ids = None

        if compress_ratio == 0:
            workspace = self.sparse_prefill_workspace.get(
                cache.swa_token_ids.shape[0] + 1,
                dtype=fp8_dtype,
            )
            combined_indices = cache.c0_combined_indices
            combined_lens = cache.c0_combined_lens
            swa_slice = workspace
        else:
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            flat_token_ids, combined_indices, combined_lens = cache.layer_inputs(
                compress_ratio, core_attn_metadata, extra_page_size
            )

            n_compressed = flat_token_ids.shape[0]
            workspace = self.sparse_prefill_workspace.get(
                n_compressed + cache.swa_token_ids.shape[0] + 1,
                dtype=fp8_dtype,
            )
            compressed_slice = workspace[:n_compressed]
            swa_slice = workspace[n_compressed:]

        # The Q8KV8 gather reads the 584-byte V4 layout only (its kernel is SM90).
        assert token_to_kv_pool.get_swa_key_layout() is KVLayout.V4
        if compressed_slice is not None:
            gather_dequant_requant_fp8_paged(
                extra_k_cache,
                flat_token_ids,
                page_size=extra_page_size,
                out=compressed_slice,
            )

        gather_dequant_requant_fp8_paged(
            token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            cache.swa_token_ids,
            page_size=cache.swa_page_size,
            extra_rows=1,
            out=swa_slice,
        )

        sentinel_row = workspace.shape[0] - 1
        q8_indices = torch.where(
            combined_indices < 0,
            torch.full_like(combined_indices, sentinel_row),
            combined_indices,
        )

        o, _, _ = sparse_mla_q8kv8_prefill_fwd(
            q=q_fp8,
            kv=workspace,
            indices=q8_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            q_scale=identity_scale,
            kv_scale=identity_scale,
            d_v=self.head_dim_v,
            attn_sink=attn_sink_pad,
            topk_length=combined_lens,
        )

        return o[:, :active_heads]

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
        seq_lens_tensor: Optional[torch.Tensor] = None,
        extend_seq_lens_tensor: Optional[torch.Tensor] = None,
        extend_start_loc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert seq_lens_tensor is not None and extend_seq_lens_tensor is not None
        result = ExpandPrefillCausally.execute(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_tensor,
            extend_seq_lens=extend_seq_lens_tensor,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens,
            extend_seq_lens_cpu=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )
        return result.seq_lens_casual, result.req_pool_indices_repeated

    def _expand_prefill_casually_vectorized(
        self,
        num_tokens: int,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = ExpandPrefillCausally.execute(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=None,
            extend_seq_lens_cpu=None,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )
        return result.seq_lens_casual, result.req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        *,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
        dspark_block_size: Optional[int] = None,
        num_tokens: Optional[int] = None,
        swa_replay_start: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        dspark_swa_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        small_metadata = (
            not is_prefill
            and seq_lens_casual.is_cuda
            and 0 < seq_lens_casual.numel() <= 384
            and out_loc.numel() == seq_lens_casual.numel()
            and self.low_ratios == (1, 2)
            and set(self.present_ratios) == {1, 2}
            and get_parallel().attn_cp_size == 1
            and self.token_to_kv_pool.request_window is None
        )
        build_pages = BuildPageTablePositions.execute
        if small_metadata:
            from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
                build_low_ratio_metadata,
            )
            from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
                build_page_table_positions_small,
            )

            build_pages = build_page_table_positions_small
        prep = build_pages(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=self.page_size,
            swa_window=SWA_WINDOW,
        )
        seq_lens_casual = prep.seq_lens_casual

        raw_positions = prep.positions_casual
        request_layout = None
        if self.token_to_kv_pool.request_window is not None:
            from sglang.srt.mem_cache.dsv41_request_window import window_layout

            if self.encoder_replay:
                starts = torch.ones_like(raw_positions, dtype=torch.bool)
                starts[1:] = (
                    req_pool_indices_repeated[1:] != req_pool_indices_repeated[:-1]
                )
                offset = torch.arange(
                    raw_positions.numel(), device=raw_positions.device
                )
                group_first = torch.cummax(torch.where(starts, offset, 0), dim=0).values
                swa_replay_start = raw_positions - (offset - group_first)
            request_layout = window_layout(
                req_pool_indices_repeated,
                raw_positions,
                capacity=self.token_to_kv_pool.request_window.capacity,
                floor=swa_replay_start,
                num_groups=num_groups,
            )
            swa_page_indices = _pad_last_dim(request_layout.indices)
            swa_topk_lengths = request_layout.lengths
        elif dspark_block_size is not None:
            assert (
                self.is_dspark_draft
                and dspark_block_size == self.speculative_num_draft_tokens - 1
            ), (
                f"dspark_block_size={dspark_block_size} must equal gamma = "
                f"speculative_num_draft_tokens-1={self.speculative_num_draft_tokens - 1} "
                f"and is only valid on the DSpark draft backend "
                f"(is_dspark_draft={self.is_dspark_draft})."
            )
            assert swa_replay_start is None, (
                "swa_replay_start is not wired for the DSpark draft window"
            )
            if dspark_swa_buffers is None:
                swa_page_indices, swa_topk_lengths = self.get_dspark_swa_page_indices(
                    seq_lens_casual=seq_lens_casual,
                    req_pool_indices_repeated=req_pool_indices_repeated,
                    out_loc=out_loc,
                    block_size=dspark_block_size,
                )
            else:
                swa_page_indices, swa_topk_lengths = dspark_swa_buffers
        else:
            swa_page_indices = BuildCausalSwaPageIndices.execute(
                req_to_token=self.req_to_token,
                full_to_swa_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
                req_pool_indices_repeated=req_pool_indices_repeated,
                seq_lens_casual=seq_lens_casual,
                swa_window=SWA_WINDOW,
                page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
                swa_replay_start=swa_replay_start,
            )
            swa_topk_lengths = prep.swa_topk_lengths
            if swa_replay_start is not None:
                # Slots below the floor are -1; the valid count shrinks to match.
                floored = raw_positions - swa_replay_start.to(raw_positions.dtype) + 1
                swa_topk_lengths = torch.minimum(
                    swa_topk_lengths, floored.clamp_min(0).to(swa_topk_lengths.dtype)
                )

        page_table = prep.page_table

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            index_topk=self.index_topk,
            present_ratios=self.present_ratios,
            low_ratios=self.low_ratios,
            request_window_layout=request_layout,
            swa_out_cache_loc=(
                request_layout.write_loc if request_layout is not None else None
            ),
        )

        if need_compress:
            low_ratio_buffers = (
                build_low_ratio_metadata(seq_lens_casual, out_loc, self.index_topk)
                if small_metadata
                else None
            )
            core_attn_metadata.init_compression_metadata(num_tokens, low_ratio_buffers)
            core_attn_metadata.init_flashmla_related(
                is_prefill=is_prefill, low_ratio_buffers=low_ratio_buffers
            )
            if self.trtllm_attn:
                core_attn_metadata.init_trtllm_sparse_buffers()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c4_sparse_raw_indices = None
            core_attn_metadata.c0_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
            if self.trtllm_attn:
                core_attn_metadata.init_trtllm_sparse_buffers()
        return core_attn_metadata

    def get_dspark_swa_page_indices(
        self,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        out_loc: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gather = ComputeDsparkWindowGather.execute(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=SWA_WINDOW,
        )

        swa_page_indices, swa_topk_lengths = BuildDsparkSwaPageIndices.execute(
            req_to_token=self.req_to_token,
            full_to_swa_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
            req_pool_indices_per_request=gather.req_pool_indices_per_request,
            offsets=gather.offsets,
            invalid=gather.invalid,
            out_loc=out_loc[: gather.num_q],
            context_lens=gather.context_lens,
            block_size=block_size,
            swa_window=SWA_WINDOW,
            page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
        )
        return swa_page_indices, swa_topk_lengths


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(self._make_step_backend(model_runner, i))

    def _make_step_backend(
        self, model_runner: ModelRunner, step_id: int
    ) -> DeepseekV4AttnBackend:
        return DeepseekV4AttnBackend(
            model_runner,
            speculative_step_id=step_id,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from types import SimpleNamespace

        inner_fb = SimpleNamespace(
            batch_size=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            # Propagate the real runtime mode so inner backends can detect IDLE
            # and apply their idle substitution.
            actual_forward_mode=getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            ),
            input_ids=getattr(forward_batch, "input_ids", None),
            positions=getattr(forward_batch, "positions", None),
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            spec_info=forward_batch.spec_info,
        )
        if in_capture:
            for i in range(self.speculative_num_steps):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
        else:
            if self.speculative_num_steps == 1:
                return
            self.attn_backends[0].init_forward_metadata_out_graph(inner_fb)
            temp_metadata = self.attn_backends[0].forward_metadata
            for i in range(1, self.speculative_num_steps - 1):
                self.attn_backends[i].replay_cuda_graph_metadata_from(
                    bs=forward_batch.batch_size,
                    temp_metadata=temp_metadata,
                    bucket=_GraphBucket.DECODE_OR_IDLE,
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: ForwardBatch
    ):
        ret = []
        for i in range(self.speculative_num_steps - 1):
            ret.append(
                self.attn_backends[
                    i
                ].init_forward_metadata_for_breakable_cuda_graph_capture(forward_batch)
            )
        return ret

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        assert len(capture_metadata) == self.speculative_num_steps - 1
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[
                i
            ].prepare_forward_metadata_for_breakable_cuda_graph_replay(
                capture_metadata[i],
                forward_batch,
                static_forward_batch=static_forward_batch,
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
