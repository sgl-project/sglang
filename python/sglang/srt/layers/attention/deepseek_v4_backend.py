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

from sglang.jit_kernel.dsv4.online_c128_mtp import OnlineC128MTPController
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4.attn_metadata_kernels import (
    BuildCausalSwaPageIndices,
    BuildPageTablePositions,
    ExpandPrefillCausally,
)
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    _LARGE_INDEXER_QUERY_THRESHOLD,
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
    SparsePrefillWorkspace,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.kernels.dspark_attn_metadata import (
    BuildBlockSeqLensCausal,
    BuildDsparkSwaPageIndices,
    ComputeDsparkWindowGather,
)
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    compute_ragged_extend_lengths,
    compute_target_verify_graph_key,
    compute_uniform_extend_lengths,
    read_ragged_verify_mode,
    resolve_ragged_verify_layout,
)
from sglang.srt.utils import ceil_align, is_xpu
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sgl_kernel.flash_mla import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_is_sm120 = is_sm120_supported()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    # IDLE is a real per-DP-rank mode. Do not let a stale _original_forward_mode
    # from a reused/padded ForwardBatch turn an empty rank into TARGET_VERIFY.
    if forward_batch.forward_mode.is_idle():
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
    if _is_sm120 or _is_xpu:
        return None
    import sgl_kernel.flash_mla as flash_mla

    return flash_mla.get_mla_metadata()[0]


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

    c4_sparse_topk: int
    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
    # once per iteration in make_core_attn_metadata and read by the store path.
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
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
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

    def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> None:
        assert self.c4_sparse_topk == other.c4_sparse_topk
        assert self.page_size == other.page_size
        assert self.cuda_int32_kwargs == other.cuda_int32_kwargs

        tensor_copy_fields = [
            "raw_out_loc",
            "seq_lens_casual",
            "positions_casual",
            "c4_out_loc",
            "c128_out_loc",
            "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1",
            "c4_sparse_topk_lengths",
        ]
        reference_assign_fields = [
            "page_table",
            "swa_page_indices",
            "swa_topk_lengths",
            "c128_page_indices",
            "c128_topk_lengths_clamp1",
            "c1_flashmla_metadata",
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

    def init_compression_metadata(self):
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
            _,
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

    # Lazily populated on the first call to ``_forward_prefill_sparse`` and
    # reused across every layer in the chunk. Reset to ``None`` when graph
    # metadata is refreshed so replay rebuilds it from the live batch.
    sparse_prefill_cache: Optional[SparsePrefillChunkCache] = None

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
        self.sparse_prefill_cache = None

    def refresh_for_breakable_cuda_graph_replay_(self, static_metadata: DSV4Metadata):
        self.core_attn_metadata.refresh_for_breakable_cuda_graph_replay_(
            static_metadata.core_attn_metadata
        )
        maybe_copy_inplace(self.indexer_metadata, src=static_metadata.indexer_metadata)
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
        # Draft-extend and online-c128 verify metadata are host-planned, so
        # spec runs keep the relay publish (the mirror only exists under
        # spec-v2; without spec the flag has no consumer either way).
        # DSPARK is the exception: its draft path carries its own host lens
        # (reserved_seq_lens_cpu) and its verify prep is device-side.
        spec_alg = model_runner.spec_algorithm
        if not spec_alg.is_none() and not spec_alg.is_dspark():
            self.needs_cpu_seq_lens = True
        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)
        self._init_verify_bs_buffers()

        self.is_dspark_draft = model_runner.is_draft_worker and spec_alg.is_dspark()

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
        seq_lens_cpu: List[int],
        extend_seq_lens: torch.Tensor,
        use_prefill_cuda_graph: bool,
        online_c128_state_slot_offset: int,
    ) -> Optional[FusedCompressMetadata]:
        if not self.online_c128_mtp.enabled():
            return None

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
        use_prefill_cuda_graph: bool = False,
    ):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
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
        extend_start_loc: Optional[torch.Tensor] = None,
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
        dspark_block_size: Optional[int] = None,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
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
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(
                core_attn_metadata,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
            if need_compress
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

        c4_compress_metadata = create(compress_ratio=4)
        c128_compress_metadata = create(compress_ratio=128)
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=c4_compress_metadata,
            c128_compress_metadata=c128_compress_metadata,
        )

    def _init_verify_bs_buffers(self) -> None:
        num_reqs = self.req_to_token.shape[0]
        self.extend_seq_lens_buffer = torch.full(
            (num_reqs,),
            self.speculative_num_draft_tokens,
            **self.cuda_int32_kwargs,
        )
        self.extend_start_loc_buffer = torch.zeros(
            num_reqs, **self.cuda_int32_kwargs
        )

    def _ensure_verify_bs_buffers(self) -> None:
        num_reqs = self.req_to_token.shape[0]
        if (
            hasattr(self, "extend_seq_lens_buffer")
            and self.extend_seq_lens_buffer.shape[0] == num_reqs
            and self.extend_start_loc_buffer.shape[0] == num_reqs
        ):
            return
        self._init_verify_bs_buffers()

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
        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            bs = len(seq_lens)
            seq_lens_cpu_list = (
                seq_lens_cpu.tolist() if seq_lens_cpu is not None else None
            )
            if ragged_layout is None:
                self.extend_seq_lens_buffer[:bs].fill_(
                    self.speculative_num_draft_tokens
                )
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
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
                ragged_layout=ragged_layout,
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
        ragged_layout: Optional[RaggedVerifyLayout] = None,
    ) -> DSV4Metadata:
        if ragged_layout is None:
            lengths = compute_uniform_extend_lengths(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_len=self.speculative_num_draft_tokens,
            )
            extend_seq_lens = self._move_to_device(lengths.extend_seq_lens_cpu)
        else:
            lengths = compute_ragged_extend_lengths(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                ragged_layout=ragged_layout,
            )
            extend_seq_lens = ragged_layout.verify_lens
        seq_lens = lengths.seq_lens_extended
        seq_lens_cpu = lengths.seq_lens_cpu_extended
        extend_seq_lens_cpu = lengths.extend_seq_lens_cpu
        num_tokens = lengths.num_tokens
        extend_start_loc = lengths.extend_start_loc
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
            extend_start_loc=extend_start_loc,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            online_c128_state_slot_offset=online_c128_state_slot_offset,
        )

    def init_forward_metadata_dspark_draft_block(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        block_size: int,
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
            num_q_tokens=num_q_tokens,
            online_state_slot_offset=online_c128_state_slot_offset,
        )
        c128_compress_metadata = raw_metadata.c128_compress_metadata
        if c128_compress_metadata is None:
            c128_compress_metadata = create(compress_ratio=128)
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
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
            extend_start_loc=None,
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
                online_c128_state_slot_offset=self.online_c128_mtp.state_slot_offset(),
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

            if self.is_dspark_draft and forward_batch.forward_mode.is_target_verify():
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
            temp_metadata = self.init_forward_metadata_dspark_draft_block(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc_padded,
                block_size=block_size,
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
            draft_extend_seq_lens_cpu = (
                seq_lens_cpu.tolist() if seq_lens_cpu is not None else seq_lens.tolist()
            )
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=draft_extend_seq_lens_cpu,
                num_tokens_per_bs=num_tokens_per_bs,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
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

        self.forward_metadata = self._build_forward_metadata(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

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
            max_seq_len = int(seq_lens.max().item())
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
        elif logical_forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend_v2()
            prefill_seq_lens_cpu = (
                seq_lens_cpu.tolist() if seq_lens_cpu is not None else seq_lens.tolist()
            )
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=prefill_seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                extend_start_loc=forward_batch.extend_start_loc,
                need_compress=not is_draft,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
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
        return self.forward_metadata

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
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

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

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            if forward_batch.forward_mode.is_extend_without_speculative() and (
                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD
                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
            ):
                return self._forward_prefill_sparse(
                    q=q,
                    layer_id=layer_id,
                    compress_ratio=compress_ratio,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    core_attn_metadata=core_attn_metadata,
                    attn_sink=attn_sink,
                )

            if _is_sm120:
                from sglang.srt.layers.attention.flash_mla_sm120 import (
                    flash_mla_with_kvcache_sm120,
                )

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
        compress_ratio: Literal[0, 4, 128],
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
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # q is (b, 1, h_q, d_qk); flash_mla_sparse_fwd takes (s_q, h_q, d_qk).
        q_flat = q.squeeze(1)

        cache = self.forward_metadata.sparse_prefill_cache
        if cache is None:
            seq_lens_cpu = forward_batch.seq_lens_cpu
            assert seq_lens_cpu is not None
            # ``swa_window_size`` on the pool is its storage page size, not
            # the model's SWA window — pass both explicitly.
            cache = SparsePrefillChunkCache.build(
                seq_lens=forward_batch.seq_lens.to(torch.int32),
                extend_seq_lens=forward_batch.extend_seq_lens.to(torch.int32),
                req_pool_indices=forward_batch.req_pool_indices.to(torch.int32),
                req_to_token=self.req_to_token,
                full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                swa_window_size=SWA_WINDOW,
                swa_page_size=token_to_kv_pool.swa_window_size,
                num_qo_tokens=q_flat.shape[0],
                max_seq_len=int(seq_lens_cpu.max().item()),
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
                assert core_attn_metadata.c4_sparse_raw_indices is not None, (
                    "sparse-prefill c4 path requires c4_sparse_raw_indices "
                    "(allocated in init_flashmla_related when is_prefill=True)"
                )
                cache.ensure_c4(core_attn_metadata.page_table, extra_page_size)
                flat_token_ids = cache.c4_flat_token_ids
                combined_indices, combined_lens = cache.combine_c4_layer(
                    c4_sparse_raw_indices=core_attn_metadata.c4_sparse_raw_indices[
                        : cache.num_qo_tokens
                    ],
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
        if dspark_block_size is not None:
            assert (
                self.is_dspark_draft
                and dspark_block_size == self.speculative_num_draft_tokens - 1
            ), (
                f"dspark_block_size={dspark_block_size} must equal gamma = "
                f"speculative_num_draft_tokens-1={self.speculative_num_draft_tokens - 1} "
                f"and is only valid on the DSpark draft backend "
                f"(is_dspark_draft={self.is_dspark_draft})."
            )
            swa_page_indices, swa_topk_lengths = self.get_dspark_swa_page_indices(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool_indices_repeated,
                out_loc=out_loc,
                block_size=dspark_block_size,
            )
        else:
            swa_page_indices = BuildCausalSwaPageIndices.execute(
                req_to_token=self.req_to_token,
                full_to_swa_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
                req_pool_indices_repeated=req_pool_indices_repeated,
                seq_lens_casual=seq_lens_casual,
                swa_window=SWA_WINDOW,
                page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
            )
            swa_topk_lengths = prep.swa_topk_lengths

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
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_flashmla_related(is_prefill=is_prefill)
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c4_sparse_raw_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
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
