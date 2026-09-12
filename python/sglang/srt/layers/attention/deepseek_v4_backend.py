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

from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    cast_q_fp8_for_q8kv8_prefill,
    dequantize_k_cache_paged,
    fp8_dtype,
    gather_dequant_requant_fp8_paged,
    q8kv8_padded_num_heads,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    fill_all_compressed_indices,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.kernels.ops.attention.dsv4.online_c128_mtp import OnlineC128MTPController
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import fp4_index_logits_decode
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
    CandidateMetadata,
    IndexerInputs,
    make_candidate_indexer,
)
from sglang.srt.layers.attention.dsv4.candidate_torch import (
    CandidateMasks,
    mask_topk_scores,
    published_masks,
)
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    _rope_fq4,
    token_req_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    C4IndexerBackendMixin,
    fp4_paged_mqa_logits,
    fp32_jit_paged_topk,
    select_candidate_blocks,
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
    is_cp_v2_active,
)
from sglang.srt.layers.dp_attention import (
    get_local_dp_buffer_len,
    set_local_dp_buffer_len,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
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
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


@functools.lru_cache(maxsize=None)
def _is_sm100_or_newer() -> bool:
    """The DeepGEMM fp8_fp4 mqa-logits kernels need SM100/SM120; Hopper takes the torch indexer."""
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


def _expand_index_page_table(
    page_table: torch.Tensor,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
) -> torch.Tensor:
    """Expand the FULL page table into the block table of a low-ratio indexer-K
    pool, which pages at `index_page_size` slots rather than a FULL page.

    The kernel resolves compressed slot j through
    page_table[b, j // index_page_size] * index_page_size + j % index_page_size,
    which with this expansion is the c1/c2 KV pool slot of the same position.
    [bs, n_full_pages] -> [bs, n_full_pages * blocks_per_page], int32.
    """
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, (
        f"{full_page_size = } / {compress_ratio = } must be a multiple of "
        f"{index_page_size = }"
    )
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    bs, n = page_table.shape
    base = page_table.to(torch.int64) * blocks_per_page
    offsets = torch.arange(blocks_per_page, device=page_table.device, dtype=torch.int64)
    expanded = base.unsqueeze(-1) + offsets  # [bs, n, blocks_per_page]
    return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30
_DENSE_INDEXER_LOGITS_BUDGET_BYTES = 1 << 31


def _every_request_fits() -> bool:
    """The captured decode variants where every request's positions fit the
    candidate block budget (decode_cuda_graph_runner): the plain top-k is the
    whole selection, and the candidate layers behave like any index source."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_dsa_variant,
    )

    return get_capture_dsa_variant() in (
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


def _dense_fp4_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    kv_fp4: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    from deep_gemm import fp8_fp4_mqa_logits as fn

    # q (int8 [T, H, 64], int32 [T, H]) x kv (int8 [L, 64], int32 [L]) -> fp32
    # [T, max_seqlen_k]; row t's column j is k[ks_t + j], valid for j < ke_t - ks_t
    # (the rest is garbage: the kernel rejects clean_logits).
    return fn(q_fp4, kv_fp4, weights, ks, ke, False, max_seqlen_k)


def _low_ratio_source_projections(layer, x, q_lora, positions, bufs):
    """Projections of a ratio-1/2 source layer, run as an eager break on the
    live rows into static buffers: the compressor's kv / score and the indexer's
    query and head weights. These GEMMs pick their algorithm by M, so at the
    bucket size their rows differ from eager; everything downstream is
    row-independent. Padded rows are zeroed."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    real = get_tc_piecewise_forward_context().forward_batch.num_token_non_padded_cpu
    if real is None:
        real = x.shape[0]

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
    """Ratio 1/2 counterpart of the triton c4/c128 metadata; a completed group's
    latent lives at raw_out_loc // ratio, -1 for a token that completes none."""
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
    """Fixed-shape top-k buffers the index_source layers fill in place. The
    extra_topk_length the kernel reads comes from positions, not from the indexer."""
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
    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
    # once per iteration in make_core_attn_metadata and read by the store path.
    request_window_layout: Optional[object] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # Sorted compress ratios present in this stage; absent ratios keep no buffers
    # or schedules. The default (4, 128) supports V4 metadata constructed by hand.
    present_ratios: Tuple[int, ...] = (4, 128)
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)
    c4_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    # The (1, 2) subset of present_ratios (DeepSeek V4.1): one latent per ratio
    # tokens at slot raw_out_loc // ratio of the c1 / c2 pool, attended through
    # the FlashMLA extra cache like c4.
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

    def sparse_page_indices(self, compress_ratio: Literal[1, 2, 4]) -> torch.Tensor:
        """Top-k slots into the ratio's extra cache, -1 padded; the indexer fills them."""
        if compress_ratio == 1:
            return self.c1_sparse_page_indices
        elif compress_ratio == 2:
            return self.c2_sparse_page_indices
        elif compress_ratio == 4:
            return self.c4_sparse_page_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_raw_indices(
        self, compress_ratio: Literal[1, 2, 4]
    ) -> Optional[torch.Tensor]:
        """The same top-k as request-local compressed positions, for the sparse
        prefill workspace; allocated for prefill metadata only."""
        if compress_ratio == 1:
            return self.c1_sparse_raw_indices
        elif compress_ratio == 2:
            return self.c2_sparse_raw_indices
        elif compress_ratio == 4:
            return self.c4_sparse_raw_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_topk_lengths(self, compress_ratio: Literal[1, 2, 4]) -> torch.Tensor:
        if compress_ratio == 1:
            return self.c1_sparse_topk_lengths
        elif compress_ratio == 2:
            return self.c2_sparse_topk_lengths
        elif compress_ratio == 4:
            return self.c4_sparse_topk_lengths
        raise ValueError(f"invalid {compress_ratio=}")

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

    def init_compression_metadata(self, num_tokens: Optional[int] = None) -> None:
        assert self.page_table.dim() == 2
        # CP-v2 pads causal metadata for per-rank partitioning, while cache-write
        # locations remain one-per-logical-token. num_tokens tracks that unpadded
        # length; legacy paths use the metadata length.
        if num_tokens is None:
            num_tokens = self.seq_lens_casual.shape[0]
        assert self.raw_out_loc.shape[0] == num_tokens, (
            f"{self.raw_out_loc.shape=}, {num_tokens=}"
        )

        has_c4 = 4 in self.present_ratios
        has_c128 = 128 in self.present_ratios
        if has_c4 or has_c128:
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
                compute_page_indices=has_c128,
            )
            if has_c4:
                self.c4_out_loc = c4_out_loc
                self.c4_topk_lengths_raw = c4_topk_lengths_raw
                self.c4_topk_lengths_clamp1 = c4_topk_lengths_clamp1
            if has_c128:
                self.c128_out_loc = c128_out_loc
                self.c128_topk_lengths_clamp1 = c128_topk_lengths_clamp1
                self.c128_page_indices = _pad_last_dim(c128_page_indices)

        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

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
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(val, torch.Tensor), (
                f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            )
            setattr(self, field_name, val[idx].contiguous())
        for field_name in self._CP_REINDEX_OPTIONAL_FIELDS:
            val = getattr(self, field_name)
            if val is not None:
                setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS + self._CP_REINDEX_OPTIONAL_FIELDS:
            val = getattr(self, field_name)
            if val is None:
                continue
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

    def init_flashmla_related(self, is_prefill: bool = False):
        assert self.index_topk in (512, 1024), (
            f"unexpected index_topk={self.index_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        has_c4 = 4 in self.present_ratios
        has_c128 = 128 in self.present_ratios
        if has_c4:
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
        self.c4_flashmla_metadata = _create_flashmla_metadata() if has_c4 else None
        self.c128_flashmla_metadata = _create_flashmla_metadata() if has_c128 else None
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
    """The token subset the layers after the last kv_source layer run over under
    decoder SWA bounded replay: the last tail tokens of every request in the
    extend. Consumers that would otherwise read the extend layout off the
    forward_batch read these instead."""

    token_indices: torch.Tensor
    positions: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: List[int]
    swa_out_cache_loc: torch.Tensor
    # Set when the tail is the extend's last rows (one request): row selection
    # is then a view, not a gather.
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


# Prefill CUDA graph: the ratio-1/2 indexer scores through the paged kernel on
# a static logits width of cuda_graph_config[prefill].max_seq_len positions
# (batches with a longer context replay eagerly), in chunks of this many rows.
_PREFILL_GRAPH_INDEXER_ROW_CHUNK = 2048


def _prefill_graph_max_seq_len() -> Optional[int]:
    from sglang.srt.runtime_context import get_exec

    return get_exec().graph.cuda_graph_config.prefill.max_seq_len


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    # Low-ratio paged indexer metadata for decode and prefill graph replay.
    # Ratio 4 uses indexer_metadata above.
    c1_indexer_metadata: Optional[PagedIndexerMetadata] = None
    c2_indexer_metadata: Optional[PagedIndexerMetadata] = None

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    # Shared by all low-ratio source layers in this step;
    # graph replay refreshes these tensors from live request metadata.
    low_ratio_req_indices: Optional[torch.Tensor] = None
    low_ratio_pos_i64: Optional[torch.Tensor] = None

    # Per-step scratch for TP-padded query heads, zeroed by the first user.
    # Later layers overwrite real heads and preserve the zero padding.
    q_pad_buffer: Optional[torch.Tensor] = None

    # Two-level low-ratio indexer (dsv4/candidate_indexer.py): what the
    # candidate-source layer published for the index-source layers after it, in
    # the chosen implementation's own type. Written once per forward by that
    # layer, read by those layers, never copied from the host.
    candidate_metadata: Optional[CandidateMetadata] = None

    # Built at the runner's prefill WAR boundary when the fast path is on,
    # otherwise lazily by ``_forward_prefill_sparse``.
    sparse_prefill_cache: Optional[SparsePrefillChunkCache] = None
    prefill_shared_reads_snapshotted: bool = False

    # Set only on the metadata built for the late layers under decoder SWA
    # bounded replay; None everywhere else.
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
    supports_ragged_verify_graph: bool = True
    needs_cpu_seq_lens: bool = False

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
        # The distinct ratios this stage has, sorted -- (4, 128) for V4, (1, 2)
        # for V4.1 -- not the per-layer hf_config.compress_ratios list. Nothing
        # is built for a ratio outside this set.
        self.present_ratios: Tuple[int, ...] = tuple(
            sorted(self.token_to_kv_pool.kv_pools)
        )
        self.low_ratios: Tuple[int, ...] = tuple(
            ratio for ratio in (1, 2) if ratio in self.present_ratios
        )
        self.has_c4: bool = 4 in self.present_ratios
        self.has_c128: bool = 128 in self.present_ratios
        # Two-level low-ratio indexer, decided here for the model's block budget
        # (dsv4/candidate_indexer.py).
        cfg = model_runner.model_config.hf_text_config
        self.candidate_indexer = make_candidate_indexer(
            getattr(cfg, "candidate_topk_blocks", 0),
            getattr(cfg, "candidate_block_size", 0),
        )
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.index_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )

        self.enable_deepseek_v4_fp4_indexer: bool = (
            model_runner.server_args.enable_deepseek_v4_fp4_indexer
        )
        self.enable_decoder_swa_bounded_replay: bool = (
            model_runner.server_args.enable_decoder_swa_bounded_replay
        )
        # Built with the regular prefill metadata; the model switches onto it
        # after the last kv_source layer (enter_late_layer_tail).
        self.tail_forward_metadata: Optional[DSV4Metadata] = None
        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend.resolve(model_runner)
        self.dsv4_prefill_backend: str = getattr(
            model_runner.server_args, "dsv4_prefill_backend", "auto"
        )
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
            page_table = _expand_index_page_table(
                page_table,
                full_page_size=self.page_size,
                compress_ratio=compress_ratio,
                index_page_size=index_page_size,
            )
        else:
            raise ValueError(f"Unsupported indexer {compress_ratio = }")
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=page_table,
            c4_seq_lens=c_seq_lens,
            use_topk_v2=self.dsa_topk_backend.should_use_topk_v2() and not _is_xpu,
            # The SM120 FP4 kernel schedules split_kv=128, while the generic
            # JIT metadata planner encodes split_kv=256.
            force_deep_gemm_metadata=(
                self.enable_deepseek_v4_fp4_indexer and get_platform().is_sm120
            ),
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            compress_ratio=compress_ratio,
            index_page_size=index_page_size,
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
        cp_v2_active = forward_batch is not None and is_cp_v2_active(forward_batch)
        if cp_v2_active:
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
            num_tokens=num_tokens if cp_v2_active else None,
            swa_replay_start=swa_replay_start,
            num_groups=len(extend_seq_lens_cpu),
        )
        if cp_v2_active:
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
        """Per-token paged indexer metadata for the prefill CUDA graph, capped at
        the graph's max_seq_len so the logits width is static and bounded."""
        num_pages = core.page_table.shape[1]
        max_seq_len = _prefill_graph_max_seq_len()
        if max_seq_len is not None:
            num_pages = min(max_seq_len // self.page_size, num_pages)
        index_page_size = self.token_to_kv_pool.get_index_k_page_size(compress_ratio)
        page_table = _expand_index_page_table(
            core.page_table[:, :num_pages],
            full_page_size=self.page_size,
            compress_ratio=compress_ratio,
            index_page_size=index_page_size,
        )
        # Unclamped: a token with no completed group scores nothing (-1 rows),
        # matching the eager extend indexer.
        c_seq_lens = (core.seq_lens_casual // compress_ratio).to(torch.int32)
        row_chunk = _PREFILL_GRAPH_INDEXER_ROW_CHUNK
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=page_table,
            c4_seq_lens=c_seq_lens,
            use_topk_v2=False,
            use_prefill_cuda_graph=True,
            compress_ratio=compress_ratio,
            index_page_size=index_page_size,
            row_chunk=row_chunk if row_chunk < c_seq_lens.shape[0] else 0,
        )

    @property
    def low_ratio_prefill_graph(self) -> bool:
        """The ratio-1/2 sources can run inside the prefill CUDA graph."""
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
        """Metadata for the layers after the last kv_source layer under decoder
        SWA bounded replay: each request contributes only its last SWA_WINDOW
        extend tokens, and their window is floored at the tail start because
        window KV before it is never written at those layers."""
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
            if is_cp_v2_active(forward_batch)
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
        """Keep bounded-replay tail rows on their original CP ranks, with padding."""
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
        """Switch the late layers onto the tail; hand the return value back to
        exit_late_layer_tail. The candidate-source layer published its masks over
        the full extend, so each request's mask is cut to its tail rows."""
        tail_metadata = self.tail_forward_metadata
        assert tail_metadata is not None, "no tail metadata for this forward"
        saved = (
            self.forward_metadata,
            forward_batch.attn_cp_metadata,
            get_local_dp_buffer_len(),
        )
        tail = tail_metadata.late_layer_tail
        tail_lens_cpu = (
            tail.local_lens_cpu
            if tail.cp_metadata is not None
            else tail.extend_seq_lens_cpu
        )
        # TODO(candidate): goes away once the source publishes its tail rows straight
        # onto the tail metadata (publish_prefill); until then cut the full masks.
        full_masks = self.forward_metadata.candidate_metadata
        if isinstance(full_masks, CandidateMasks) and full_masks.request_masks:
            tail_metadata.candidate_metadata = CandidateMasks(
                request_masks=[
                    mask[mask.shape[0] - t :]
                    for mask, t in zip(full_masks.request_masks, tail_lens_cpu)
                ]
            )
        # The last index-source layer before the switch published its top-k into
        # the full metadata's buffers; the consumer layers after the switch read
        # the tail metadata's, so carry the tail rows over (padding stays -1).
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
            max_seq_len=max_seq_len,
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

        # Upgrade Raw->Full so the c4/c128 compress + core_attn + indexer
        # materialization is recorded inside the cuda graph; a no-op (Full
        # already) when PREP_IN_CUDA_GRAPH=0.
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

        # Compute the SWA KV-store write target once per forward and cache it on
        # the metadata for every layer's store. This is recorded inside the cuda
        # graph, so replay re-reads the live out_cache_loc buffer (spec-v2 and DP
        # padding rebind out_cache_loc after out-graph metadata prep). flash_mla
        # kernels require int32 indices.
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
        # CP-v2 shards the query layout that this global snapshot assumes.
        metadata = self.forward_metadata
        if self.token_to_kv_pool.request_window is not None:
            return
        if isinstance(metadata, DSV4Metadata):
            metadata.prefill_shared_reads_snapshotted = False
        snapshot_shared_prefill_reads = (
            envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.get()
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and self.model_runner.spec_algorithm.is_dflash_family()
            and not is_cp_v2_active(forward_batch)
        )
        if not snapshot_shared_prefill_reads:
            return

        assert isinstance(metadata, DSV4Metadata)
        # The tail never takes the sparse path (see the dispatch in forward_extend),
        # so its metadata carries no chunk cache.
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
        if is_cp_v2_active(forward_batch):
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
        # ``swa_window_size`` on the pool is its storage page size, not the
        # model's SWA window, so pass both explicitly.
        return SparsePrefillChunkCache.build(
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            extend_seq_lens=extend_seq_lens.to(torch.int32),
            query_lens=query_lens,
            query_pos=query_pos,
            req_pool_indices=forward_batch.req_pool_indices.to(torch.int32),
            req_to_token=self.req_to_token,
            full_to_swa=self.token_to_kv_pool.full_to_swa_index_mapping,
            swa_window_size=SWA_WINDOW,
            swa_page_size=self.token_to_kv_pool.swa_window_size,
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
            max_seq_len_override = getattr(forward_batch, "max_seq_len_override", None)
        if max_seq_len_override is not None:
            max_seq_len = max_seq_len_override
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
        self.forward_metadata = self._build_forward_metadata(
            forward_batch,
            max_seq_len_override=self.MAX_SEQ_LEN_FOR_CAPTURE,
            use_prefill_cuda_graph=True,
        )
        if self.low_ratio_prefill_graph and forward_batch.forward_mode.is_extend():
            for ratio in self.low_ratios:
                self._source_projection_buffers(
                    forward_batch.out_cache_loc.shape[0], ratio
                )
        return self.forward_metadata

    def _source_projection_buffers(self, num_tokens: int, ratio: int) -> dict:
        """Reuse static source-projection buffers across prefill graph buckets.

        Growing a buffer set must keep previous allocations alive for captured graphs.
        """
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
        static_metadata = self._build_forward_metadata(
            static_forward_batch if static_forward_batch is not None else forward_batch,
            max_seq_len_override=self.MAX_SEQ_LEN_FOR_CAPTURE,
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
            if 4 in core.present_ratios:
                core.c4_flashmla_metadata = _create_flashmla_metadata()
            if 128 in core.present_ratios:
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
        """Runs on every ratio 1/2 layer before its attention; the metadata and
        latents it writes are what the layers after it attend through."""
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
            pos = positions.to(torch.int64)
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
        """Every rank writes the whole prompt's compressed state and scores its own rows."""
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
            and layer.compress_ratio == 2
            and layer.compressor.use_fused_compress
            and read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT
            and self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 1
            and x.shape[0]
            == forward_batch.batch_size * self.speculative_num_draft_tokens
        ):
            # Static verify is request-major with consecutive positions. Compact
            # verify has variable block lengths and must retain the general path.
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
        # Projection layout and fused-write support are fixed together at load time;
        # HIP and pre-Blackwell use split projections.
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

        from sglang.kernels.ops.attention.dsv4.pair_pool_decode import pair_pool_decode

        core = self.forward_metadata.core_metadata
        state = self.token_to_kv_pool.get_attention_compress_states(layer.layer_id)
        kv, score = layer.compressor.project(x)
        pooled, group_pos, slots = pair_pool_decode(
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
        """Fused compressor write, index-key projection, then fused index-key write.
        Both write kernels consume metadata dtypes directly and suppress padded stores.
        """
        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.c2 import (
            c2_decode_norm_rope_store,
            c2_verify_norm_rope_store,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_rope import (
            index_k_norm_rope_pack_store,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        compressor = layer.compressor
        layer_id = layer.layer_id
        # Contiguous complex64 freqs_cis gives a real/imag-interleaved view without copying.
        freqs_cis = torch.view_as_real(layer.freqs_cis).flatten(-2)
        kv_cache = pool.get_extra_key_buffer(layer_id)
        page_size = pool.get_extra_key_page_size(layer_id)
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
            )
            out_loc = core.c1_out_loc
        else:
            # CompressStatePool stores each request's pending pairs in a position ring.
            # KVAndScore rows use | kv | score |, addressed as req * ring_size + pos % ring_size.
            state = pool.get_attention_compress_states(layer_id)
            c2_compress = (
                c2_verify_norm_rope_store
                if draft_len > 1
                else c2_decode_norm_rope_store
            )
            verify_args = {"draft_len": draft_len} if draft_len > 1 else {}
            latent = c2_compress(
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
                **verify_args,
            )
            out_loc = core.c2_out_loc

        indexer = layer.indexer
        if indexer is not None and indexer.owns_k:
            # out_loc is -1 for an incomplete group and 0 for padding;
            # the kernel suppresses both stores.
            assert out_loc is not None
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
        """Read the preceding token, then publish this batch's trailing ring window.

        Each request occupies consecutive rows and positions. Keeping only its
        last ring_size rows gives every write a distinct live slot, including
        when a prefill chunk is longer than the ring. Reads precede all writes
        so wrapping cannot overwrite a cross-chunk partner.
        """
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
                from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
                    rope_fake_quant_pack_indexer,
                )

                indexer = layer.indexer
                k = indexer.k_norm(indexer.forward_wk(latent))
                rope_fake_quant_pack_indexer(
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
        # The FlashMLA cache requantizes the FP4/E4M3 latent into its FP8 layout.
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
                # verify rows of one request share a request id (DeepGEMM pairs
                # them); decode has one row per request, nothing to pair
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
        """Dense fp4 indexer over rows laid out request after request, `q_lens[b]` each."""
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
        empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
        num_tokens = pos.shape[0]
        # TODO: move this to candidate indexer
        if not slot_chunks or num_tokens == 0:
            if indexer.is_candidate_source:
                self.forward_metadata.candidate_metadata = CandidateMasks(
                    request_masks=[empty_mask for _ in lc_per_req]
                )
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
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

        publish = [] if indexer.is_candidate_source else None
        consume = (
            published_masks(self.forward_metadata.candidate_metadata).request_masks
            if indexer.uses_candidates and publish is None
            else None
        )
        publish_rows_per_req = None
        tail_metadata = self.tail_forward_metadata
        if (
            publish is not None
            and tail_metadata is not None
            and tail_metadata.late_layer_tail is not None
            and getattr(tail_metadata.late_layer_tail, "cp_metadata", None) is None
        ):
            publish_rows_per_req = tail_metadata.late_layer_tail.extend_seq_lens_cpu
            assert len(publish_rows_per_req) == len(q_lens_cpu)

        tok_start = 0
        for b, (lc, t_len) in enumerate(zip(lc_per_req, q_lens_cpu)):
            rows_b = slice(tok_start, tok_start + t_len)
            tok_start += t_len
            if lc == 0 or t_len == 0:
                if publish is not None:
                    publish.append(empty_mask)
                if t_len:
                    selected[rows_b].fill_(-1)
                continue
            lc_aligned = ceil_align(lc, 4)
            rows_per_chunk = max(
                1, _DENSE_INDEXER_LOGITS_BUDGET_BYTES // (lc_aligned * 4)
            )
            mask_b = None
            first_pub = 0
            if publish is not None:
                pub_rows = (
                    t_len
                    if publish_rows_per_req is None
                    else min(int(publish_rows_per_req[b]), t_len)
                )
                first_pub = t_len - pub_rows
                mask_b = torch.empty((pub_rows, lc), dtype=torch.bool, device=device)
            j = torch.arange(lc, device=device)
            for c0 in range(0, t_len, rows_per_chunk):
                c1 = min(c0 + rows_per_chunk, t_len)
                rows = slice(rows_b.start + c0, rows_b.start + c1)
                lens = compress_lens[rows]
                logits = _dense_fp4_mqa_logits(
                    (q_fp4[rows], q_sf[rows]),
                    (k_fp4, k_sf),
                    weights[rows],
                    ks[rows],
                    ks[rows] + lens,
                    lc_aligned,
                )
                scores = logits[:, :lc]
                if publish is not None:
                    scores.masked_fill_(j[None, :] >= lens[:, None], -torch.inf)
                    p0 = max(c0, first_pub)
                    if p0 < c1:
                        mask_b[p0 - first_pub : c1 - first_pub] = (
                            select_candidate_blocks(
                                scores[p0 - c0 :],
                                lens[p0 - c0 :, None],
                                topk_blocks=indexer.candidate_topk_blocks,
                                block_size=indexer.candidate_block_size,
                            )
                        )
                elif consume is not None:
                    scores.masked_fill_(~consume[b][c0:c1], -torch.inf)
                topk_transform_ragged_v2(
                    logits, lens, out_offsets=ks[rows], out_indices=selected[rows]
                )
                if consume is not None:
                    selected[rows] = mask_topk_scores(logits, selected[rows], ks[rows])
                del logits, scores
            if publish is not None:
                publish.append(mask_b)
        if publish is not None:
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=publish
            )
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

    def _low_ratio_index_topk_prefill_graph(self, layer, pos, q, w) -> None:
        """q/w come from live rows; metadata fixes the graph's context width."""
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
        width = metadata.max_c4_seq_len
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
        page_size = metadata.c4_page_size
        k_cache = k_cache.view(k_cache.shape[0], page_size, 1, 68)

        lens = metadata.c4_seq_lens
        page_table = metadata.page_table
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        topk = min(indexer.index_topk, width)
        columns = torch.arange(width, device=lens.device)
        for rows, plan in metadata.row_chunks():
            logits = fp4_paged_mqa_logits(
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
            from sglang.kernels.ops.attention.dsv4.fp4_rope import (
                index_q_rope_pack_weights,
            )

            q, _ = indexer.wq_b(q_lora)
            q = q.view(q.shape[0], indexer.n_local_heads, indexer.index_head_dim)
            # The fused query pack also computes head_weights(x).float(), preserving
            # the fp32 multiply and bf16 rounding. freqs_cis stays a real/imag view.
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
        page_size = metadata.c4_page_size
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
        logits = fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            weights,
            metadata.c4_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_c4_seq_len,
        )
        # TODO(dark): add bf16 topk
        fp32_jit_paged_topk(logits, metadata, page_indices, raw_indices)

    # TODO(candidate): Hopper decode still publishes / consumes masks inline (torch
    # top-k); move into the candidate indexer with the prefill paths.
    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
        """Hopper decode indexer: one token per request, every request scored at
        once against its visible compressed positions straight off the fp4 page
        table (Triton), then the same candidate / top-k contract as the DeepGEMM
        path: level-one candidate blocks where the layer publishes or consumes them,
        and -1 padded slots with the valid prefix first."""
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
        lmax = min(metadata.max_c4_seq_len, self.req_to_token.shape[1] // ratio)
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
            # Dequantize only this request's visible K rows; the full table is
            # pool-sized.
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
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=publish
            )

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Resolve the SWA KV-store write target for the current forward.

        Prefer the value cached by the metadata init: in-graph for
        decode/verify, the hoisted cuda_graph_swa_out_cache_loc buffer for
        draft-extend. Translate at store time when nothing matching is cached
        (paths that skip the init, or a batch re-padded after init). Idle
        always falls back: its metadata may be stale, and
        translating the zero-padded out_cache_loc writes to the dummy slot.
        """
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
            if compress_ratio in (1, 2, 4):
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)
                extra_topk_lengths = core_attn_metadata.sparse_topk_lengths(
                    compress_ratio
                )
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = (
                token_to_kv_pool.qk_nope_head_dim
                + token_to_kv_pool.qk_rope_head_dim * 2
                + 8
            )
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
                extra_k_cache = extra_k_cache[
                    :, : extra_page_size * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    extra_page_size,
                    1,
                    k_cache_total_dim,
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

            # sparse_prefill_fwd does not support SM120. The late-layer tail stays
            # on the dense path: its window floor lives in swa_page_indices, which
            # the sparse chunk cache does not read.
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
            if compress_ratio == 128:
                assert core_attn_metadata.c128_page_indices is not None
                cache.ensure_c128(core_attn_metadata.c128_page_indices)
                flat_token_ids = cache.c128_flat_token_ids
                combined_indices = cache.c128_combined_indices
                combined_lens = cache.c128_combined_lens
            else:
                raw_indices = core_attn_metadata.sparse_raw_indices(compress_ratio)
                assert raw_indices is not None, (
                    f"sparse-prefill c{compress_ratio} path requires the raw "
                    "top-k indices (allocated in init_flashmla_related when "
                    "is_prefill=True)"
                )
                gather = cache.ensure_compressed(
                    compress_ratio, core_attn_metadata.page_table, extra_page_size
                )
                flat_token_ids = gather.flat_token_ids
                combined_indices, combined_lens = cache.combine_compressed(
                    compress_ratio, raw_indices[: cache.num_qo_tokens]
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
            )
        dequantize_k_cache_paged(
            token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            cache.swa_token_ids,
            page_size=cache.swa_page_size,
            out=swa_slice,
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
                forward_batch, num_qo_tokens=q_flat.shape[0]
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

            if compress_ratio == 128:
                assert core_attn_metadata.c128_page_indices is not None
                cache.ensure_c128(core_attn_metadata.c128_page_indices)
                flat_token_ids = cache.c128_flat_token_ids
                combined_indices = cache.c128_combined_indices
                combined_lens = cache.c128_combined_lens
            else:
                raw_indices = core_attn_metadata.sparse_raw_indices(compress_ratio)
                assert raw_indices is not None, (
                    f"Q8KV8 sparse-prefill c{compress_ratio} path requires the raw "
                    "top-k indices (allocated in init_flashmla_related when "
                    "is_prefill=True)"
                )
                gather = cache.ensure_compressed(
                    compress_ratio, core_attn_metadata.page_table, extra_page_size
                )
                flat_token_ids = gather.flat_token_ids
                combined_indices, combined_lens = cache.combine_compressed(
                    compress_ratio, raw_indices[: cache.num_qo_tokens]
                )

            n_compressed = flat_token_ids.shape[0]
            workspace = self.sparse_prefill_workspace.get(
                n_compressed + cache.swa_token_ids.shape[0] + 1,
                dtype=fp8_dtype,
            )
            compressed_slice = workspace[:n_compressed]
            swa_slice = workspace[n_compressed:]

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

        prep = BuildPageTablePositions.execute(
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
            core_attn_metadata.init_compression_metadata(num_tokens)
            core_attn_metadata.init_flashmla_related(is_prefill=is_prefill)
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c4_sparse_raw_indices = None
            core_attn_metadata.c0_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
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
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
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
