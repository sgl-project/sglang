from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, TypeVar

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.debug_flash_mla_adapter import (
    flash_mla_with_kvcache_entrypoint,
)
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

# components
from .compressed import paged_prefill
from .compressed.compressor import CompressorBackend
from .compressed.indexer import C4IndexerBackend
from .compressed.metadata import (
    DeepseekV4Metadata,
    PagedCoreMetadata,
    PagedIndexerMetadata,
    create_flashmla_metadata,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.model_runner import ModelRunner


SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


_HOST_INT32_KWARGS = {"dtype": torch.int32, "pin_memory": True}


@dataclass
class _DecodeCudaGraphSharedData:
    pass  # TODO fields


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None  # type: ignore
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


class DeepseekV4Backend(AttentionBackend, C4IndexerBackend, CompressorBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)  # type: ignore
        head_dim = model_runner.model_config.head_dim
        assert head_dim == 512
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.host_int32_kwargs = _HOST_INT32_KWARGS
        self.debug_dump_hook: Optional[Callable] = None
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_seq_len_for_capture = self.req_to_token.shape[1]
        assert self.page_size == 256, "the system hardcodes page_size=256"

    #### Public API ####

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        batch_size = forward_batch.batch_size
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert forward_batch.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata = self._compute_decode_metadata(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_prefill():
            metadata = self._compute_prefill_metadata(
                max_seq_len=max_seq_len,
                forward_batch=forward_batch,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        # set metadata
        self.forward_metadata = metadata
        if h := self.debug_dump_hook:
            h("init_forward_metadata_output", metadata)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.decode_cuda_graph_shared_data = _DecodeCudaGraphSharedData()
        self.decode_cuda_graph_metadata_of_bs: Dict[int, DeepseekV4Metadata] = {}

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
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

            metadata = self._compute_decode_metadata(
                max_seq_len=self.max_seq_len_for_capture,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                # Dummy value (must be int64 to match real out_cache_loc dtype)
                out_cache_loc=torch.zeros(
                    seq_lens.shape, dtype=torch.int64, device=seq_lens.device
                ),
            )

            self.decode_cuda_graph_metadata_of_bs[bs] = metadata
            self.forward_metadata = metadata
        else:
            raise NotImplementedError(f"unsupported mode {forward_mode=}")

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
    ):
        # We observe error that len(out_cache_loc)=0 while len(seq_lens)>0.
        # We only support DP attention, thus when IDLE, we will not execute attention backend,
        # thus it is safe to delete it.
        if actual_forward_mode == ForwardMode.IDLE:
            if hasattr(self, "forward_metadata"):
                del self.forward_metadata  # avoid misuse
            return

        assert seq_lens_cpu is not None and out_cache_loc is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        if forward_mode.is_decode_or_idle():
            # Future optimization: use real max seq len
            actual_max_seq_len = seq_lens_cpu.max().item()

            chosen_max_seq_len = self.max_seq_len_for_capture
            assert actual_max_seq_len <= chosen_max_seq_len

            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )

            temp_metadata = self._compute_decode_metadata(
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
        else:
            raise NotImplementedError(f"unsupported mode {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        # FlashMLA, NSA backend, etc, use "1"
        return 1

    # TODO improve naming
    def on_after_cuda_graph_warmup_pass(self):
        metadata = self.forward_metadata
        if isinstance(metadata.core_metadata, PagedCoreMetadata):
            metadata.core_metadata.c1_flashmla_metadata = create_flashmla_metadata()
            metadata.core_metadata.c4_flashmla_metadata = create_flashmla_metadata()
            metadata.core_metadata.c128_flashmla_metadata = create_flashmla_metadata()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        compress_ratio: Literal[0, 4, 128],
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:

        # NOTE: here set-kv only applies to swa kv

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_metadata = metadata.core_metadata
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        # This sanity check is to avoid, e.g., in CUDA graph capturing, we may accidentally
        # run forward passes multiple times with one init_forward_metadata.
        # If that happens, the real capturing pass will record that layer 0 do not have any meta init operations
        # which is wrong.

        assert isinstance(core_metadata, PagedCoreMetadata), "TODO: support ragged"
        # ------- 1. SWA attention k cache -------
        if forward_batch.forward_mode.is_prefill():
            # prefill is complex: concat kv and rearrange
            swa_k_cache, swa_k_pack_sliced = (
                paged_prefill.prepare_swa_ring_buffer_cache(
                    swa_k,
                    forward_batch,
                    layer_id,
                    token_to_kv_pool,
                    core_metadata,
                    debug_dump_hook=self.debug_dump_hook,
                )
            )
        else:
            # decode is trivial: no slicing, no rearrangement
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer(layer_id)
            swa_k_pack_sliced = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)

        if save_kv_cache:
            token_to_kv_pool.set_swa_key_buffer(
                layer_id=layer_id,
                loc=core_metadata.swa_out_loc_sliced,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack_sliced,
            )

        # ------- 2. Full (C4/C128) attention k cache -------
        extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
        if compress_ratio == 4:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_metadata.c4_sparse_page_indices
            extra_topk_lengths = core_metadata.c4_sparse_topk_lengths
        elif compress_ratio == 128:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_metadata.c128_page_indices
            extra_topk_lengths = core_metadata.c128_topk_lengths_clamp1

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

        swa_page_indices = core_metadata.swa_page_indices

        # unsqueeze to adapt decode kernel
        if q.ndim == 3:
            q = q.unsqueeze(1)
        if swa_page_indices.ndim == 2:
            swa_page_indices = swa_page_indices.unsqueeze(1)
        if extra_indices is not None and extra_indices.ndim == 2:
            extra_indices = extra_indices.unsqueeze(1)

        assert attn_sink is not None

        flashmla_metadata = core_metadata.get_flashmla_metadata(compress_ratio)

        # compute-sanitizer observe issue if this is not enforced
        assert (
            swa_page_indices.shape[-1] % 64 == 0
        ), f"{swa_page_indices.shape[-1]=} is not aligned to 64"
        if extra_indices is not None:
            assert (
                extra_indices.shape[-1] % 64 == 0
            ), f"{extra_indices.shape[-1]=} is not aligned to 64"

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
            topk_length=core_metadata.swa_topk_lengths,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_lengths,
        )

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "kernel")
        o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]
        o = o.squeeze(1)

        return o

    #### Helper functions ####

    def _compute_prefill_metadata(
        self,
        *,
        max_seq_len: int,
        forward_batch: ForwardBatch,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ) -> DeepseekV4Metadata:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        extend_seq_lens_cpu = extend_seq_lens_cpu or forward_batch.extend_seq_lens_cpu
        assert seq_lens_cpu is not None and extend_seq_lens_cpu is not None
        # NOTE: expanded follow a `causal` mask pattern
        seq_lens_expanded, idx_mapping = paged_prefill.expand_seq_lens(
            seq_lens=seq_lens_cpu.tolist(),
            extend_seq_lens=extend_seq_lens_cpu,
            device=self.device,
        )
        core_metadata = self._make_paged_core_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices[idx_mapping],
            seq_lens=seq_lens_expanded,
            max_seq_len=max_seq_len,
            out_loc=forward_batch.out_cache_loc,
            is_prefill=True,
            forward_batch=forward_batch,
        )
        # NOTE: `raw` does not follow a `causal` mask pattern
        seq_lens_raw_expanded = forward_batch.seq_lens[idx_mapping]
        should_store_swa = (seq_lens_raw_expanded - seq_lens_expanded) < SWA_WINDOW
        swa_slice = torch.nonzero(should_store_swa, as_tuple=False).squeeze(1)
        core_metadata.init_swa_slice(swa_slice)
        indexer_metadata = self._make_indexer_metadata(core_metadata)
        return DeepseekV4Metadata(core_metadata, indexer_metadata, seq_lens_expanded)

    def _compute_decode_metadata(
        self,
        *,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> DeepseekV4Metadata:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"
        core_metadata = self._make_paged_core_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            forward_batch=None,  # not prefill
        )
        indexer_metadata = self._make_indexer_metadata(core_metadata)
        return DeepseekV4Metadata(core_metadata, indexer_metadata, seq_lens)

    def _make_indexer_metadata(self, core_metadata: PagedCoreMetadata):
        # TODO: handle the expanded seqlens for MTP here
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_metadata.page_table,
            # NOTE should use `raw` instead of `clamp1`
            c4_seq_lens=core_metadata.c4_topk_lengths_raw,
        )

    def _make_paged_compress_tensors(
        self,
        *,
        page_table: torch.Tensor,
        page_size: int,
        seq_lens: torch.Tensor,
        out_loc: torch.Tensor,
        compress_ratio: Literal[4, 128],
    ) -> Dict[str, torch.Tensor]:
        # NOTE(dark): c_ prefix means "compressed"
        assert page_table.dim() == 2
        assert out_loc.shape == seq_lens.shape, f"{out_loc.shape=} {seq_lens.shape=}"

        # e.g. seq_lens = [4n - 1, 4n, 4n + 1, 4n + 2]
        # raw_out_loc   = [4X + 2, 4X + 3, 4Y, 4Y + 1]
        # raw_positions = [4n - 2, 4n - 1, 4n, 4n + 1] (i.e. seq_lens - 1)
        # then we have:
        # c4_seq_lens   = [n - 1 , n  ,  n   ,   n   ] (i.e. seq_lens // 4)
        # c4_out_loc    = [0   ,   X  ,    0  ,  0   ] (i.e. out_loc // 4)
        # NOTE: 0 means "any" in this example
        should_compress = seq_lens % compress_ratio == 0
        c_page_size = page_size // compress_ratio
        c_seq_lens_raw = seq_lens // compress_ratio
        c_out_loc = torch.where(should_compress, out_loc // compress_ratio, 0)
        c_seq_lens_clamp1 = torch.clamp(c_seq_lens_raw, min=1)

        # NOTE(dark): c4 does not need page indices
        if compress_ratio == 4:
            return {
                "c_out_loc": c_out_loc,
                "c_seq_lens_raw": c_seq_lens_raw,
                "c_seq_lens_clamp1": c_seq_lens_clamp1,
            }

        max_pages = page_table.size(1)
        c_max_seq_len = c_page_size * max_pages
        # [bs, max_pages] -> [bs, max_pages, c_page_size] -> [bs, c_max_seq_len]
        c_offsets = torch.arange(c_max_seq_len, **self.cuda_int32_kwargs)
        c_page_indices = (
            (page_table.unsqueeze(2) * c_page_size + c_offsets[:c_page_size])
            .to(torch.int32)
            .contiguous()
            .view(-1, c_max_seq_len)
        )
        # TODO(dark): whether this is a must
        # As far as I know, only the padded 0 -> 1 must be filled with -1
        # Should other positions also be masked?
        mask = c_offsets.unsqueeze(0) >= c_seq_lens_raw.unsqueeze(1)
        # NOTE: mask out the extra positions to -1
        c_page_indices.masked_fill_(mask, -1)
        return {
            "c_out_loc": c_out_loc,
            "c_seq_lens_raw": c_seq_lens_raw,
            "c_seq_lens_clamp1": c_seq_lens_clamp1,
            "c_page_indices": c_page_indices,
        }

    def _make_paged_core_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        # extra args for prefill
        is_prefill: bool = False,
        forward_batch: Optional[ForwardBatch] = None,
        skip_compressor: bool = False,
    ) -> PagedCoreMetadata:
        assert self.swa_page_size == SWA_WINDOW  # TODO(dark): relax this

        # -------------------- START compute SWA metadata --------------------
        swa_pages = req_pool_indices.to(torch.int32)
        if is_prefill:
            assert forward_batch is not None
            swa_page_indices = paged_prefill.make_swa_ring_buffer_indices(
                forward_batch=forward_batch,
                device=self.device,
                max_seq_len=max_seq_len,
                swa_window_size=SWA_WINDOW,
            )
        else:
            # NOTE: for decode, we directly index into the ring buffer pool
            # the "page_mapping" for SWA is the req_pool_indices themselves
            offsets = torch.arange(SWA_WINDOW, **self.cuda_int32_kwargs)
            swa_page_indices = swa_pages.unsqueeze(1) * self.swa_page_size + offsets
            # if seq_len < 128, mask out the extra positions to -1
            mask = offsets.unsqueeze(0) >= seq_lens.unsqueeze(1)
            swa_page_indices.masked_fill_(mask, -1)

        positions = seq_lens - 1
        swa_topk_lengths = torch.clamp(seq_lens, max=SWA_WINDOW)
        swa_out_loc = swa_pages * self.swa_page_size + positions % self.swa_page_size

        # -------------------- END compute SWA metadata --------------------

        if not skip_compressor:
            page_table = req_to_token[req_pool_indices, : max_seq_len : self.page_size]
            page_table = page_table.to(torch.int32) // self.page_size
            c4_data = self._make_paged_compress_tensors(
                page_table=page_table,
                page_size=self.page_size,
                seq_lens=seq_lens,
                out_loc=out_loc,
                compress_ratio=4,
            )
            c128_data = self._make_paged_compress_tensors(
                page_table=page_table,
                page_size=self.page_size,
                seq_lens=seq_lens,
                out_loc=out_loc,
                compress_ratio=128,
            )
            c128_page_indices = c128_data["c_page_indices"]
            swa_page_indices = _pad_last_dim(
                swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
            )
            c128_page_indices = _pad_last_dim(
                c128_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
            )
        else:
            # TODO: For draft decode/draft extend
            c4_data = {
                "c_out_loc": None,
                "c_seq_lens_raw": None,
                "c_seq_lens_clamp1": None,
                "c_page_indices": None,
            }
            c128_data = {
                "c_out_loc": None,
                "c_seq_lens_raw": None,
                "c_seq_lens_clamp1": None,
                "c_page_indices": None,
            }
            c128_page_indices = c128_data["c_page_indices"]

        return PagedCoreMetadata(
            positions=positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_out_loc=c4_data["c_out_loc"],
            c4_topk_lengths_raw=c4_data["c_seq_lens_raw"],
            c4_topk_lengths_clamp1=c4_data["c_seq_lens_clamp1"],
            c128_out_loc=c128_data["c_out_loc"],
            c128_page_indices=c128_page_indices,
            c128_topk_lengths_clamp1=c128_data["c_seq_lens_clamp1"],
            c4_sparse_topk=C4_TOPK,
            swa_slice=None,
            swa_out_loc_sliced=swa_out_loc,
        )

    #### Test-only API ####

    def extract_metadata(self, forward_batch: ForwardBatch) -> DeepseekV4Metadata:
        # NOTE: in the future we may put metadata in the forward_batch itself
        # this function is used for tests. Don't delete it.
        return self.forward_metadata
