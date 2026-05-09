"""
Some comments on the common terms used in DeepSeekV4Backend:

topk_lengths:
    NOTE: TL;DR: topk_lengths == seq_lens
    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
    `topk_lengths` indicates how many tokens each query will attend to.
    This should be named as `seq_lens`, but we simply follow the naming convention.

page_table:
    The page table indicates which pages each request is assigned to.
    Each value in the page table is the page index in the TokenToKVPool.
    This page index is irrelevant to the actual `page_size`.

page_indices:
    The real indices used to index into the KV cache.
    This can be computed from the `page_table` and `page_size`.
    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)
    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.
    In implementation, we don't materialize the full C4 `page_indices`,
    but calculate them from `page_table` on-the-fly in the attention kernel.

positions:
    The position of the last token for each request.
    For compress token, the positions must be times of compress ratio.
    For example, for C4, raw_position=11 will trigger a compression,
    But the RoPE's position, during compression, must be 8 instead of 11.

Some other notes:
    c4_ / c128_: means "compressed by 4" / "compressed by 128".
    c4_page_size: page_size // 4
    c4_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.compressed.compressor import (
    CompressorBackend,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.compressed.indexer import C4IndexerBackend
from sglang.srt.layers.attention.compressed.metadata import (
    PagedIndexerMetadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.debug_flash_mla_adapter import (
    flash_mla_with_kvcache_entrypoint,
)
from sglang.srt.layers.attention.deepseek_v4_backend import _pad_last_dim
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_round_robin_split
from sglang.srt.layers.attention.triton_ops.compressed_metadata import (
    init_compressed_metadata as _init_compressed_metadata_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


def _copy_metadata(
    src,
    dst,
    check_eq_fields: List[str],
    copy_fields: List[str],
    assign_fields: Optional[List[str]] = None,
):
    assign_fields = assign_fields or []

    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f"{field_name=} {src_val=} {dst_val=}"

    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        # Skip if both src and dst are None (e.g., compress fields when need_compress=False)
        if src_val is None and dst_val is None:
            continue
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        if hasattr(dst_val, "copy_"):
            dst_val.copy_(src_val)
        else:
            warnings.warn(
                f"{field_name=} {type(dst_val)=} does not have copy_, use setattr"
            )
            setattr(dst, field_name, src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(
        provided_fields_unique
    ), f"{provided_fields=} has dup"
    all_fields = {f.name for f in dataclasses.fields(src)}
    provided_fields = set(provided_fields)
    assert (
        provided_fields == all_fields
    ), f"{provided_fields - all_fields=}, {all_fields - provided_fields=}"


def _create_flashmla_metadata():
    import flash_mla

    return flash_mla.get_mla_metadata()[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class DSV4AttnMetadataRadix:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    # to calculate compressed metadata
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor  # positions expanded causally

    # sliding window attention (core)
    swa_page_indices: torch.Tensor  # at most (sum_qo_len, 128)
    swa_topk_lengths: torch.Tensor  # clipped to 128

    # NOTE: c4/c128 out_loc will mask the invalid write locations to 0.
    # When no compression happens, out_loc will be 0, which is the "padded slot"
    c4_sparse_topk: int  # must be 512
    c4_out_loc: Optional[torch.Tensor] = None
    c4_positions: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None  # i.e. c4_seq_lens
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)  # clipped to 512
    c4_sparse_page_indices: torch.Tensor = field(init=False)  # (bs, 512)

    # C128 dense attention (core)
    c128_out_loc: Optional[torch.Tensor] = None
    c128_positions: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    # FlashMLA
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

    def copy_(self, other: DSV4AttnMetadataRadix) -> None:
        _copy_metadata(
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
                "c4_positions",
                "c128_positions",
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
            ],
            assign_fields=[
                # For the new API, the metadata has the following lifecycle:
                #
                # Graph capture warmup forward pass:
                # (ignore, we will reset to brand new object after such passes)
                #
                # Graph capture real-capture forward pass:
                # * Layer 0: Set python & tensor objects to metadata
                # * Layer >=1: Read them from metadata
                #
                # Graph replay:
                # * Layer 0: The kernels are in "generate metadata" mode
                # * Layer >=1: The kernels are in "non-generate metadata" mode
                #
                # Thus this field can be ignored.
                # However, to allow running replay w/o in real cuda graph, we do an assignment.
                # (Do we really need that? If no, we can change this field to skip-copy mode)
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compressed_metadata(self):
        """
        Initialize compressed metadata for both C4 and C128 using a single fused Triton kernel.

        NOTE: 0 means "any" in this example
        e.g. seq_lens = [4n - 1, 4n, 4n + 1, 4n + 2]
        raw_out_loc   = [4X + 2, 4X + 3, 4Y, 4Y + 1]
        raw_positions = [4n - 2, 4n - 1, 4n, 4n + 1] (i.e. seq_lens - 1)
        then we have:
        c4_seq_lens   = [n - 1 , n  ,  n   ,   n   ] (i.e. seq_lens // 4)
        c4_positions  = [0  , 4n - 4,    0  ,  0   ] (i.e. positions // 4 * 4)
        c4_out_loc    = [0   ,   X  ,    0  ,  0   ] (i.e. out_loc // 4)
        """
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        # Compute both C4 and C128 metadata in a single kernel launch
        (
            self.c4_out_loc,
            self.c4_positions,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            self.c128_positions,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compressed_metadata_triton(
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
        "c4_positions",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_positions",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_tp_rank()
        cp_size = get_attention_tp_size()
        idx = slice(cp_rank, None, cp_size)
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

    def init_flashmla_related(self):
        assert self.c4_sparse_topk == 512
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
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4MetadataRadix:
    core_attn_metadata: DSV4AttnMetadataRadix
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadataRadix:
        return self.core_attn_metadata

    def copy_(self, other: DSV4MetadataRadix):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4MetadataSimplified:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    # Constant tensor for CUDA graph related
    extend_seq_lens: Optional[torch.Tensor] = None
    real_metadata: Optional[DSV4MetadataRadix] = None

    def copy_(self, other: DSV4MetadataSimplified):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        # constant buffer
        self.extend_seq_lens = other.extend_seq_lens


@dataclass
class _DecodeCudaGraphSharedData:
    pass  # TODO fields


class DeepseekV4BackendRadix(AttentionBackend, C4IndexerBackend, CompressorBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)  # type: ignore
        head_dim = model_runner.model_config.head_dim
        assert head_dim == 512
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.debug_dump_hook: Optional[Callable] = None
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        # Init Pools
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool  # type: ignore
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)

        # Speculative Decoding
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (  # type: ignore
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[DSV4MetadataRadix, DSV4MetadataSimplified] = None

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        # NOTE(dark): need to avoid sync
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    #### Public API ####

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadataRadix):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            # NOTE should use `raw` instead of `clamp1`
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> DSV4MetadataRadix:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

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

        if not envs.SGLANG_OPT_USE_FUSED_PAGED_COMPRESS.get():
            create = _create_dummy_paged_compress_data

        return DSV4MetadataRadix(
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
    ) -> DSV4MetadataRadix:
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
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not (envs.SGLANG_OPT_USE_FUSED_PAGED_COMPRESS.get() and need_compress):
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
        return DSV4MetadataRadix(
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
        use_prefill_cuda_graph: bool = False,
    ) -> Union[DSV4MetadataRadix, DSV4MetadataSimplified]:
        if envs.SGLANG_ADVANCED_CUDA_GRAPH_CAPTURE.get():
            assert out_cache_loc is not None
            # FIXME: Constant tensor
            if not hasattr(self, "extend_seq_lens_buffer"):
                self.extend_seq_lens_buffer = torch.tensor(
                    [self.speculative_num_draft_tokens] * 1025, device=self.device
                )
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]

            return DSV4MetadataSimplified(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
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
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4MetadataRadix:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:  # NOTE: for CUDA graph related
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

    def make_forward_metadata_from_simplified(
        self, simplified_metadata: DSV4MetadataSimplified
    ) -> DSV4MetadataRadix:
        # Extract the real metadata from the simplified metadata
        req_pool_indices = simplified_metadata.req_pool_indices
        seq_lens = simplified_metadata.seq_lens
        out_cache_loc = simplified_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        extend_seq_lens = simplified_metadata.extend_seq_lens

        seq_lens_casual, req_pool_indices_repeated = (
            self.expend_extend_with_same_length(
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
            num_q_tokens=num_draft_tokens,
        )
        return DSV4MetadataRadix(
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
    ) -> DSV4MetadataRadix:
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

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert forward_batch.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
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
            is_draft = forward_batch.forward_mode.is_draft_extend(include_v2=True)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,  # NOTE: draft model is swa only
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        # set metadata
        self.forward_metadata = metadata
        if h := self.debug_dump_hook:
            h("init_forward_metadata_output", metadata)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.decode_cuda_graph_shared_data = _DecodeCudaGraphSharedData()
        self.decode_cuda_graph_metadata_of_bs: Dict[int, DSV4MetadataRadix] = {}
        self.target_verify_cuda_graph_metadata_of_bs: Dict[
            int, Union[DSV4MetadataRadix, DSV4MetadataSimplified]
        ] = {}
        self.draft_extend_cuda_graph_metadata_of_bs: Dict[int, DSV4MetadataRadix] = {}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        if forward_mode.is_decode_or_idle():
            # NOTE: we should use `self.decode_cuda_graph_shared_data` to avoid allocating
            # a pack of tensors per cuda graph, but that is the NEXT step instead of current step.
            # For example, we may write:
            #
            # metadata = compute_decode_metadata()
            # use_shared_tensors(metadata, self.decode_cuda_graph_shared_data)
            #
            # def use_shared_tensors():
            #   for field_name in ...:
            #     getattr(shared_data, field_name).copy_(getattr(metadata, field_name)[..maybe_some_slicing..])
            #     setattr(metadata, field_name, getattr(shared_data, field_name))

            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),  # Dummy value
            )

            self.decode_cuda_graph_metadata_of_bs[bs] = metadata
            self.forward_metadata = metadata
        elif forward_mode.is_target_verify():
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            self.target_verify_cuda_graph_metadata_of_bs[bs] = metadata
            self.forward_metadata = metadata

            # Track the current simplified metadata for resetting after warmup
            self._current_capture_simplified = (
                metadata if isinstance(metadata, DSV4MetadataSimplified) else None
            )
        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
            self.draft_extend_cuda_graph_metadata_of_bs[bs] = metadata
            self.forward_metadata = metadata
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ) -> None:
        # We observe error that len(out_cache_loc)=0 while len(seq_lens)>0.
        # We only support DP attention, thus when IDLE, we will not execute attention backend,
        # thus it is safe to delete it.
        if actual_forward_mode == ForwardMode.IDLE:
            if hasattr(self, "forward_metadata"):
                del self.forward_metadata  # avoid misuse
            return

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        if forward_mode.is_decode_or_idle():
            assert out_cache_loc is not None
            # Future optimization: use real max seq len
            actual_max_seq_len = seq_lens_cpu.max().item()

            chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
            assert actual_max_seq_len <= chosen_max_seq_len

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

            # Future optimization: may not need to `copy` all things,
            # But only copy partially such as `page_table[:, :max_seq_len]`
            chosen_metadata = self.decode_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        elif forward_mode.is_target_verify():
            assert out_cache_loc is not None
            # Future optimization: use real max seq len
            actual_max_seq_len = seq_lens_cpu.max().item()
            chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
            assert actual_max_seq_len <= chosen_max_seq_len
            # NOTE: extend length remains the same during target verify
            num_tokens = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
            )
            chosen_metadata = self.target_verify_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        elif forward_mode.is_draft_extend(include_v2=True):
            actual_max_seq_len = seq_lens_cpu.max().item()
            chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
            assert actual_max_seq_len <= chosen_max_seq_len
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            # NOTE: draft extend doesn't need out_cache_loc since need_compress=False
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
            chosen_metadata = self.draft_extend_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        else:
            raise NotImplementedError

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: DSV4MetadataRadix,
        forward_mode: ForwardMode,
    ) -> None:
        """Copy pre-computed metadata to this backend's cuda graph metadata storage.

        This method is used to avoid redundant computation when multiple backends
        need the same metadata (e.g., in speculative decoding with multiple steps).
        """
        if forward_mode.is_decode_or_idle():
            chosen_metadata = self.decode_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        elif forward_mode.is_target_verify():
            chosen_metadata = self.target_verify_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        elif forward_mode.is_draft_extend(include_v2=True):
            chosen_metadata = self.draft_extend_cuda_graph_metadata_of_bs[bs]
            chosen_metadata.copy_(temp_metadata)
            self.forward_metadata = chosen_metadata
        else:
            raise NotImplementedError

    def get_cuda_graph_seq_len_fill_value(self):
        # FlashMLA, NSA backend, etc, use "1"
        return 1

    # TODO improve naming
    def on_after_cuda_graph_warmup_pass(self):
        metadata: DSV4MetadataRadix = self.forward_metadata
        if isinstance(metadata.core_attn_metadata, DSV4AttnMetadataRadix):
            metadata.core_attn_metadata.c1_flashmla_metadata = (
                _create_flashmla_metadata()
            )
            metadata.core_attn_metadata.c4_flashmla_metadata = (
                _create_flashmla_metadata()
            )
            metadata.core_attn_metadata.c128_flashmla_metadata = (
                _create_flashmla_metadata()
            )

        # For advanced CUDA graph capture, reset forward_metadata back to
        # the current batch size's DSV4MetadataSimplified so that the next
        # pass (including the actual capture pass) re-executes the
        # simplified→real derivation, ensuring those GPU ops are recorded
        # into the CUDA graph.
        current_simplified = getattr(self, "_current_capture_simplified", None)
        if current_simplified is not None:
            self.forward_metadata = current_simplified

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
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
        if isinstance(self.forward_metadata, DSV4MetadataSimplified):
            real_metadata = self.make_forward_metadata_from_simplified(
                simplified_metadata=self.forward_metadata,
            )
            self.forward_metadata = real_metadata

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        # NOTE: here set-kv only applies to swa kv

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        # This sanity check is to avoid, e.g., in CUDA graph capturing, we may accidentally
        # run forward passes multiple times with one init_forward_metadata.
        # If that happens, the real capturing pass will record that layer 0 do not have any meta init operations
        # which is wrong.

        if isinstance(core_attn_metadata, DSV4AttnMetadataRadix):
            # ------- 1. SWA attention k cache -------
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            # ------- 2. Full (C4/C128) attention k cache -------
            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            # ------- Call attention core -------
            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            # view b/c flashmla expect dim=4
            # reference: FlashMLA/tests/test_flash_mla_sparse_prefill.py
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

            # unsqueeze to adapt decode kernel
            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            # compute-sanitizer observe issue if this is not enforced
            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

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

            backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
            o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    #### Helper functions ####

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: expanded follow a `causal` mask pattern
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

        # Padding is generic (out_cache_loc may be ceil_align'd beyond num_tokens).
        # CP always needs it; non-CP can opt in via SGLANG_DSV4_FIX_ATTN_PADDING.
        _need_pad = (
            is_nsa_prefill_cp_round_robin_split()
            or envs.SGLANG_DSV4_FIX_ATTN_PADDING.get()
        )
        if (
            _need_pad
            and padded_num_tokens is not None
            and padded_num_tokens > num_tokens
        ):
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                # TODO: is pad value 1 ok?
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

    def expend_extend_with_same_length(
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
    ) -> DSV4AttnMetadataRadix:
        # NOTE: the full attn page size is 256 and SWA page size is 128,
        # which is OK in current SWA radix tree design
        assert self.swa_page_size == SWA_WINDOW

        # -------------------- START compute SWA metadata --------------------
        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        # -------------------- END compute SWA metadata --------------------
        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadataRadix(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=C4_TOPK,
        )

        if need_compress:
            core_attn_metadata.init_compressed_metadata()
            if is_prefill and is_nsa_prefill_cp_round_robin_split():
                core_attn_metadata.apply_cp_reindex()
            core_attn_metadata.init_flashmla_related()
        else:
            # Draft model doesn't include c4/c128 compressors
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
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
        return swa_indices

    #### Test-only API ####
    def extract_metadata(self, forward_batch: ForwardBatch):
        # NOTE: in the future we may put metadata in the forward_batch itself
        # this function is used for tests. Don't delete it.
        return self.forward_metadata


class DeepseekV4MultiStepBackend(DeepseekV4BackendRadix):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4BackendRadix] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4BackendRadix(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return

        # Compute metadata only once using the first backend
        self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            out_cache_loc=forward_batch.out_cache_loc,
        )
        temp_metadata = self.attn_backends[0].forward_metadata

        # Copy to other backends without recomputing
        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                forward_mode=ForwardMode.DECODE,
            )


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
