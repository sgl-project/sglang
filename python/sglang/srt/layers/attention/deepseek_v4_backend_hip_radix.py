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

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sgl_kernel.flash_mla import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    from sglang.srt.utils import is_hip

    if is_hip():
        return None
    import sgl_kernel.flash_mla as flash_mla

    return flash_mla.get_mla_metadata()[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class UnifiedKvMetadata:
    """
    unified-kv per-forward metadata
    """

    # SWA ring write target (req_slot*ring + pos%ring)
    swa_loc: Optional[torch.Tensor] = None

    # ragged decode index streams
    swa_indices: Optional[torch.Tensor] = None
    swa_indptr: Optional[torch.Tensor] = None
    hca_indices: Optional[torch.Tensor] = None
    hca_indptr: Optional[torch.Tensor] = None
    csa_indices: Optional[torch.Tensor] = None
    csa_indptr: Optional[torch.Tensor] = None

    # prefill/extend per-token mapping
    pf_state_slot: Optional[torch.Tensor] = None
    pf_chunk_start: Optional[torch.Tensor] = None
    pf_cu_q: Optional[torch.Tensor] = None
    pf_final_pos: Optional[torch.Tensor] = None

    # SWA-page-offset compressed-store locations (= c*_out_loc + unified_swa_pages),
    # precomputed once per step to drop the per-layer int add in the store path.
    c4_out_loc: Optional[torch.Tensor] = None
    c128_out_loc: Optional[torch.Tensor] = None

    def copy_(self, other: UnifiedKvMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[],
            copy_fields=[
                "swa_indices",
                "swa_indptr",
                "hca_indices",
                "hca_indptr",
                "csa_indices",
                "csa_indptr",
                "pf_state_slot",
                "pf_chunk_start",
                "pf_cu_q",
                "pf_final_pos",
                "c4_out_loc",
                "c128_out_loc",
            ],
            # swa_loc is recomputed each forward (recorded inside cuda graphs),
            # so it is rebound rather than copied across replays.
            assign_fields=["swa_loc"],
        )


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

    c4_sparse_topk: int
    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
    # once per iteration in make_core_attn_metadata and read by the store path.
    swa_out_cache_loc: Optional[torch.Tensor] = None
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_topk_lengths_raw: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)
    c4_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c128_topk_lengths_raw: Optional[torch.Tensor] = None

    # unified-kv metadata
    unified: Optional[UnifiedKvMetadata] = None

    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 4, 128]):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
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
                "c128_topk_lengths_raw",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_topk_lengths_raw",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
                "unified",
            ],
            assign_fields=[
                # Recomputed by the recorded init_forward_metadata_in_graph op
                # each forward; not copied across replays.
                "swa_out_cache_loc",
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compression_metadata(self, unified_swa_pages: int = 0):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_raw,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

        if unified_swa_pages:
            if self.unified is None:
                self.unified = UnifiedKvMetadata()
            self.unified.c4_out_loc = self.c4_out_loc + unified_swa_pages
            self.unified.c128_out_loc = self.c128_out_loc + unified_swa_pages

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
        "c128_topk_lengths_raw",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "swa_out_cache_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self, is_prefill: bool = False):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        assert self.c4_topk_lengths_raw is not None
        self.c4_sparse_topk_lengths_raw = torch.clamp(
            self.c4_topk_lengths_raw, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
        if is_prefill:
            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens


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


class DeepseekV4HipRadixBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    # DSV4 TBO runs ONLY in eager prefill (prefill cuda-graph is disabled);
    # decode/target-verify graphs are non-TBO (primary backend only). So the TBO
    # child backends must not be driven through cuda-graph capture/replay — doing
    # so rebuilds this backend's compressor/indexer metadata per replay step on
    # both children and leaks ROCm HSA resources (HSA_STATUS_ERROR_OUT_OF_RESOURCES).
    # TboAttnBackend reads this to skip children in the *_graph paths only.
    tbo_supports_cuda_graph = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
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
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )
        self.enable_deepseek_v4_fp4_indexer: bool = (
            model_runner.server_args.enable_deepseek_v4_fp4_indexer
        )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
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

        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        self._attach_unified_kv_decode_streams(core_attn_metadata, req_pool_indices)

        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

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
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
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
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        self._attach_unified_kv_prefill_meta(
            core_attn_metadata, req_pool_indices, seq_lens, extend_seq_lens
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        extend_seq_lens: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
        seq_lens_cpu: Optional[List[int]] = None,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        # HIP path: build target-verify metadata eagerly even when
        # SGLANG_PREP_IN_CUDA_GRAPH is enabled. The raw/lazy-upgrade route can
        # hit planner invariants during graph capture for DSV4+EAGLE.
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.tolist()
        return self.init_forward_metadata_target_verify_old(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens
        if extend_seq_lens is None or extend_seq_lens.numel() != bs:
            extend_seq_lens = torch.full_like(seq_lens, num_draft_tokens)
        else:
            extend_seq_lens = extend_seq_lens.to(
                device=seq_lens.device, dtype=seq_lens.dtype
            )
        extend_seq_lens = torch.minimum(extend_seq_lens, seq_lens).clamp_min_(1)

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)
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
            num_q_tokens=num_draft_tokens * bs,
        )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
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
        self._attach_unified_kv_decode_streams(core_attn_metadata, req_pool_indices)
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

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
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        # Upgrade Raw->Full so the c4/c128 compress + core_attn + indexer
        # materialization is recorded inside the cuda graph; a no-op (Full
        # already) when PREP_IN_CUDA_GRAPH=0.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

        # Compute the SWA KV-store write target once per forward and cache it on
        # the metadata for every layer's store. This is recorded inside the cuda
        # graph, so replay re-reads the live out_cache_loc buffer (spec-v2 and DP
        # padding rebind out_cache_loc after out-graph metadata prep). flash_mla
        # kernels require int32 indices.
        metadata = self.forward_metadata
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
            metadata.core_attn_metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
                    torch.int32
                )
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

        if in_capture:
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
            seq_lens_cpu = seq_lens.cpu()
        else:
            out_cache_loc = forward_batch.out_cache_loc
            actual_forward_mode = getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            )
            seq_lens_sum = forward_batch.seq_lens_sum
            seq_lens_cpu = forward_batch.seq_lens_cpu

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
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
        elif bucket == _GraphBucket.TARGET_VERIFY:
            if resolve_ragged_verify_layout(forward_batch) is not None:
                raise NotImplementedError(
                    "DSV4 ragged verify is not supported on the HIP backend "
                    "(DeepseekV4HipRadixBackend) cuda-graph path; disable "
                    "SGLANG_RAGGED_VERIFY_MODE or use a CUDA device."
                )
            assert out_cache_loc is not None
            num_tokens_v = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens_v - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
                # CPU mirror already available here (== seq_lens, no D2H);
                # pass it so target_verify skips the per-iter seq_lens.tolist() sync.
                seq_lens_cpu=seq_lens_cpu.tolist(),
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            if out_cache_loc is not None:
                # Pad the real write locations to the captured token count so
                # raw_out_loc reflects the actual replay out_cache_loc.
                out_cache_loc = torch.nn.functional.pad(
                    out_cache_loc,
                    pad=(0, num_tokens_per_bs * bs - len(out_cache_loc)),
                    mode="constant",
                    value=0,
                )
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

        if in_capture:
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
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert self.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
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
        elif forward_batch.forward_mode.is_target_verify():
            if resolve_ragged_verify_layout(forward_batch) is not None:
                raise NotImplementedError(
                    "DSV4 ragged verify is not supported on the HIP backend "
                    "(DeepseekV4HipRadixBackend); disable SGLANG_RAGGED_VERIFY_MODE "
                    "or use a CUDA device."
                )
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
                extend_seq_lens=forward_batch.extend_seq_lens,
                seq_lens_cpu=(
                    seq_lens_cpu.tolist() if seq_lens_cpu is not None else None
                ),
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend_v2()
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata
        self.init_forward_metadata_in_graph(forward_batch)

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
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

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
        if bs not in self.cuda_graph_metadata_of_bucket_and_bs[bucket]:
            # First call (from capture): store the new metadata directly.
            self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = temp_metadata
            self.forward_metadata = temp_metadata
            return
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
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
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def _attach_unified_kv_decode_streams(
        self, core: DSV4AttnMetadata, req_pool_indices: torch.Tensor
    ) -> None:
        """build the ragged decode index streams once per forward"""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if not is_unified_kv_triton():
            return
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        pool = self.token_to_kv_pool
        N = core.positions_casual.shape[0]
        if core.unified is None:
            core.unified = UnifiedKvMetadata()
        (
            core.unified.swa_indices,
            core.unified.swa_indptr,
            core.unified.hca_indices,
            core.unified.hca_indptr,
            core.unified.csa_indices,
            core.unified.csa_indptr,
        ) = runtime.build_decode_streams(
            state_slot=req_pool_indices[:N],
            positions=core.positions_casual,
            swa_len=core.swa_topk_lengths,
            hca_len=core.c128_topk_lengths_raw,
            csa_len=core.c4_sparse_topk_lengths_raw,
            hca_page_indices=core.c128_page_indices,
            csa_width=core.c4_sparse_page_indices.shape[1],
            win=pool.unified_swa_window,
            ring_stride=pool.unified_swa_ring_size,
            swa_pages=pool.unified_swa_pages,
        )
        # SWA ring write target, same value for every layer this forward.
        # Decode: N tokens == N reqs, positions already aligned (no repeat).
        req_slot = req_pool_indices[:N].to(torch.int64)
        core.unified.swa_loc = (
            req_slot * pool.unified_swa_ring_size
            + core.positions_casual.to(torch.int64) % pool.unified_swa_ring_size
        ).to(torch.int32)

    def _attach_unified_kv_prefill_meta(
        self,
        core: DSV4AttnMetadata,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if not is_unified_kv_triton():
            return
        device = req_pool_indices.device
        bs = req_pool_indices.shape[0]
        seq_lens = seq_lens.to(torch.int64)
        extend_seq_lens = extend_seq_lens.to(torch.int64)
        # token -> req index (length L = sum(extend_seq_lens))
        bid = torch.repeat_interleave(
            torch.arange(bs, device=device, dtype=torch.int64), extend_seq_lens
        )
        if core.unified is None:
            core.unified = UnifiedKvMetadata()
        core.unified.pf_state_slot = req_pool_indices[bid]
        core.unified.pf_chunk_start = (seq_lens - extend_seq_lens)[bid]
        cu_q_per_req = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
        core.unified.pf_cu_q = cu_q_per_req[bid]
        core.unified.pf_final_pos = (seq_lens - 1)[bid]

    def _forward_unified_kv(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        attn_sink: torch.Tensor,
        core_attn_metadata: DSV4AttnMetadata,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """unified_kv paged-attention path over the bf16 unified_kv"""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        unified = pool.get_unified_kv(layer_id)
        win = pool.unified_swa_window
        ring_stride = pool.unified_swa_ring_size
        swa_pages = pool.unified_swa_pages

        if q.ndim == 4:
            q = q.squeeze(1)
        device = q.device
        positions = forward_batch.positions.to(torch.int64)
        T = q.shape[0]
        positions = positions[:T]

        c128_pi = getattr(core_attn_metadata, "c128_page_indices", None)
        c4_pi = getattr(core_attn_metadata, "c4_sparse_page_indices", None)

        # decode
        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        if is_decode:
            state_slot = forward_batch.req_pool_indices[:T]
            if save_kv_cache:
                runtime.store_swa_into_unified(
                    kv=kv,
                    state_slot=state_slot,
                    positions=positions,
                    unified_kv=unified,
                    win=win,
                    ring_stride=ring_stride,
                    final_pos=positions,
                )
            unified_metadata = core_attn_metadata.unified
            if compress_ratio == 0:
                kv_indices = unified_metadata.swa_indices
                kv_indptr = unified_metadata.swa_indptr
            elif compress_ratio == 128:
                kv_indices = unified_metadata.hca_indices
                kv_indptr = unified_metadata.hca_indptr
            elif compress_ratio == 4:
                kv_indices = unified_metadata.csa_indices
                kv_indptr = unified_metadata.csa_indptr
                runtime.fill_compress_tail(
                    indices=kv_indices,
                    indptr=kv_indptr,
                    prefix_len=core_attn_metadata.swa_topk_lengths[:T],
                    page_indices=c4_pi[:T],
                    valid_len=core_attn_metadata.c4_sparse_topk_lengths_raw[:T],
                    swa_pages=swa_pages,
                )
            else:
                raise ValueError(f"bad compress_ratio {compress_ratio}")
            return runtime.decode(
                q=q,
                unified_kv=unified,
                kv_indices=kv_indices,
                kv_indptr=kv_indptr,
                attn_sink=attn_sink,
                softmax_scale=self.softmax_scale,
            )

        # prefill / extend
        state_slot = core_attn_metadata.unified.pf_state_slot
        chunk_start = core_attn_metadata.unified.pf_chunk_start
        cu_q = core_attn_metadata.unified.pf_cu_q
        final_pos = core_attn_metadata.unified.pf_final_pos

        # DSA CP (round-robin/interleave): unified_pf_* are built over the GLOBAL
        # token layout, but under CP each rank owns only 1/cp_size of the queries
        # (q/positions are local) while kv was all-gathered to the full sequence.
        # Slice the per-query fields to this rank's tokens so their length matches
        # the local query count T; values stay global so each local query still
        # attends over the full all-gathered KV.
        from sglang.srt.layers.attention.dsa.utils import (
            is_dsa_prefill_cp_round_robin_split,
        )

        # NOTE (AMD/HIP only): this whole DSA-CP prefill handling lives in the
        # HIP backend (DeepseekV4HipRadixBackend, selected only when is_hip()).
        # The NVIDIA path uses DeepseekV4AttnBackend and never reaches here, so
        # these CP changes do not affect B200/H200 execution.
        _cp_size = get_parallel().attn_cp_size
        _cp_active = (
            _cp_size > 1
            and is_dsa_prefill_cp_round_robin_split()
            and kv.shape[0] == _cp_size * T
            and state_slot.shape[0] != T
        )
        state_slot_full = state_slot
        final_pos_full = final_pos
        positions_full = positions
        if _cp_active:
            _sl = slice(get_parallel().attn_cp_rank, None, _cp_size)
            state_slot = state_slot[_sl].contiguous()
            chunk_start = chunk_start[_sl].contiguous()
            cu_q = cu_q[_sl].contiguous()
            final_pos = final_pos[_sl].contiguous()
            # positions for the local queries are this rank's round-robin global
            # positions {r, r+cp, r+2cp, ...}; forward_batch.positions is the full
            # (padded) global layout, so slice it the same way instead of taking
            # the first T entries (which would be the wrong, sequential 0..T-1).
            positions = forward_batch.positions.to(torch.int64)[_sl].contiguous()
            # The SWA ring must hold the FULL window on EVERY rank (decode and
            # later chunks read this rank's ring). kv was all-gathered to the full
            # sequence, so write the full kv with full global positions/state_slot
            # instead of only this rank's 1/cp_size tokens.
            positions_full = forward_batch.positions.to(torch.int64)[
                : state_slot_full.shape[0]
            ].contiguous()

        kpre_i, kpre_p, kext_i, kext_p = runtime.build_prefill_indices(
            compress_ratio=compress_ratio,
            state_slot=state_slot,
            positions=positions,
            chunk_start=chunk_start,
            cu_q=cu_q,
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            c128_page_indices=c128_pi,
            c4_sparse_page_indices=c4_pi,
        )

        if kpre_p.shape[0] < T + 1:
            pad = T + 1 - kpre_p.shape[0]
            kpre_p = torch.cat([kpre_p, kpre_p[-1:].expand(pad)])
            kext_p = torch.cat([kext_p, kext_p[-1:].expand(pad)])
        o = runtime.prefill(
            q=q,
            unified_kv=unified,
            kv_indices_prefix=kpre_i,
            kv_indptr_prefix=kpre_p,
            kv_extend=kv,
            kv_indices_extend=kext_i,
            kv_indptr_extend=kext_p,
            attn_sink=attn_sink,
            softmax_scale=self.softmax_scale,
        )

        # write this chunk's SWA K into the ring for future chunks / decode
        # only the final-window tokens per request
        if save_kv_cache:
            # Under CP, write the FULL all-gathered window so every rank's ring is
            # complete (decode / later chunks read the local ring). Without CP this
            # is just the local kv + local metadata as before.
            _ring_state_slot = state_slot_full if _cp_active else state_slot
            _ring_final_pos = final_pos_full if _cp_active else final_pos
            _ring_positions = positions_full if _cp_active else positions
            n_real = _ring_state_slot.shape[0]
            runtime.store_swa_into_unified(
                kv=kv[:n_real],
                state_slot=_ring_state_slot,
                positions=_ring_positions[:n_real],
                unified_kv=unified,
                win=win,
                ring_stride=ring_stride,
                final_pos=_ring_final_pos,
            )
        return o

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Resolve the SWA KV-store write target for the current forward.

        Fast path: the per-forward value cached by init_forward_metadata_in_graph
        (recorded inside cuda graphs, so replay re-reads live buffers). Fallback:
        translate at store time, matching the pre-cache behavior, for paths that
        never run the in-graph init — eager idle (forward_idle skips attn init),
        runners that only run the out-graph prep (e.g.
        EAGLEDraftExtendCudaGraphRunner) — or whose batch was re-padded after
        init (shape mismatch). Idle always falls back: its metadata is absent or
        left over from a previous forward, and translating the zero-padded
        out_cache_loc writes to the dummy slot.
        """
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

    def get_unified_swa_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """SWA ring write target for unified_kv, shared by all layers.

        Fast path: the per-forward value cached in _attach_unified_kv_decode_streams
        (recorded inside cuda graphs, so replay re-reads live buffers). Fallback:
        recompute at store time, matching the pre-cache per-layer behavior, for
        paths that never ran the decode-stream init (eager prefill/extend, idle,
        or a batch re-padded after init -> shape mismatch).

        Cached swa_loc is computed once from committed positions, so every draft-decode
        step would reuse the same ring slot and break the chain. Recompute from the live
        per-step positions; only the draft path is affected, the rest keeps the fast path.
        """
        positions = forward_batch.positions
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        unified = getattr(core, "unified", None) if core is not None else None
        cached = unified.swa_loc if unified is not None else None
        is_multistep_draft_decode = (
            forward_batch.forward_mode.is_decode_or_idle()
            and self.speculative_num_steps > 1
        )
        if (
            cached is not None
            and not forward_batch.forward_mode.is_idle()
            and cached.shape[0] == positions.shape[0]
            and not is_multistep_draft_decode
        ):
            result = cached
        else:
            ring = self.token_to_kv_pool.unified_swa_ring_size
            req_slot = forward_batch.req_pool_indices.to(torch.int64)
            if req_slot.shape[0] != positions.shape[0]:
                req_slot = req_slot.repeat_interleave(
                    positions.shape[0] // req_slot.shape[0]
                )
            result = (req_slot * ring + positions.to(torch.int64) % ring).to(
                torch.int32
            )
        return result

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        swa_loc = self.get_swa_out_cache_loc(forward_batch)
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                swa_loc=swa_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                swa_loc=swa_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
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

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            return self._forward_unified_kv(
                q=q,
                kv=swa_k,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                attn_sink=attn_sink,
                core_attn_metadata=core_attn_metadata,
                save_kv_cache=save_kv_cache,
            )

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                page_sizes = {
                    4: token_to_kv_pool.page_size // 4,
                    128: token_to_kv_pool.page_size // 128,
                }
                extra_k_cache = extra_k_cache[
                    :, : page_sizes[compress_ratio] * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    page_sizes[compress_ratio],
                    1,
                    k_cache_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if self.mtp_enabled:
                if swa_page_indices.shape[0] != q.shape[0]:
                    swa_page_indices = _pad_tensor_to_size(
                        swa_page_indices, q.shape[0], value=0
                    )

                if swa_topk_lengths.shape[0] != q.shape[0]:
                    swa_topk_lengths = _pad_tensor_to_size(
                        swa_topk_lengths, q.shape[0], value=1
                    )

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            from sglang.srt.layers.attention.hip_flash_mla import (
                flash_mla_with_kvcache_entrypoint,
            )

            backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
            input_dict = dict(
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
            )
            o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
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
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        seq_lens_casual = seq_lens_casual.to(torch.int32)

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata(
                unified_swa_pages=getattr(self.token_to_kv_pool, "unified_swa_pages", 0)
            )
            core_attn_metadata.init_flashmla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_topk_lengths_raw = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c4_sparse_raw_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        # flash_mla attention requires int32 page indices.
        return swa_indices.to(torch.int32)


class DeepseekV4MultiStepBackend(DeepseekV4HipRadixBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4HipRadixBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4HipRadixBackend(
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
