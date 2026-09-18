from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4 import (
    clear_unaccepted_c128_draft_states,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
)
from sglang.kernels.ops.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import layout
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_dsv4_indexer_bytes_per_token(index_head_dim: int, use_fp4_indexer: bool) -> int:
    """Return payload and quant-scale bytes for one compressed indexer token."""
    if use_fp4_indexer:
        return index_head_dim // 2 + index_head_dim // 32
    return index_head_dim + index_head_dim // 128 * 4


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False, num_draft_tokens: int = 0
) -> int:
    assert compress_ratio in [2, 4, 128], f"Unsupported {compress_ratio = }"
    if compress_ratio == 2:
        # Two positions are one pair, addressed by position % ring_size; a
        # speculative ring must be wider than the draft window: pow2 >= 2 + drafts.
        if not is_speculative:
            return 2
        return 1 << (num_draft_tokens + 1).bit_length()
    # Online c128 keeps one (max, sum, kv) state per index instead of a 128-slot
    # ring of raw tokens, so ring_size collapses to 1.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


def get_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """Largest draft-token count this ring can serve; mirrors `mtp_pad` in
    `c_plan.cuh`, where the bound is derived."""
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return ring_size - window_size + 2 if ring_size > window_size else 0


def get_swa_ring_size(sliding_window: int, is_speculative: bool = False) -> int:
    # A verify batch writes its draft tokens ahead of the committed position.
    spec_extra = (get_spec().speculative_num_draft_tokens - 1) if is_speculative else 0
    return sliding_window + spec_extra


def resolve_compressed_kv_layout(
    kv_layout: KVLayout, compress_ratio: int, option: Optional[str] = None
) -> KVLayout:
    """Layout of one compress ratio's cache next to a ``kv_layout`` main cache.
    The ratio-1/2 latents are already e2m1 with per-16 e4m3 scales, so ``V41_FP4``
    is lossless for them; ratios 4 / 128 are not fp4-rounded and stay fp8."""
    if option is not None:
        option = option.lower()
        assert option in (
            "auto",
            "fp8",
            "fp4",
        ), f"unknown compressed KV layout {option!r}"
        if option == "auto":
            option = None
    if kv_layout is KVLayout.V4:
        assert option in (None, "fp8"), "the V4 main cache only pairs with V4 caches"
        return KVLayout.V4
    assert kv_layout is KVLayout.V41, f"{kv_layout} is not a main-cache layout"
    if option == "fp8":
        return KVLayout.V41
    if option == "fp4":
        return KVLayout.V41_FP4
    return KVLayout.V41_FP4 if compress_ratio in (1, 2) else KVLayout.V41


def flashmla_supports_v41_kv_layouts() -> bool:
    """Whether the installed FlashMLA decode kernel reads the V41 / V41_FP4
    formats; its docstring lists the bytes-per-token it detects."""
    try:
        from sgl_kernel.flash_mla import flash_mla_with_kvcache
    except Exception:
        return False
    return "528" in (flash_mla_with_kvcache.__doc__ or "")


def select_dsv4_kv_layout() -> Tuple[KVLayout, Optional[str]]:
    """The (main-cache layout, compressed-cache option) for a new DeepSeek-V4
    family pool; the V4.1 layouts exist only in SM100 / SM103 FlashMLA."""
    mode = envs.SGLANG_DSV4_KV_LAYOUT.get().lower()
    option = envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT.get().lower()
    if mode == "v4":
        return KVLayout.V4, None if option == "auto" else option
    assert mode in ("v41", "auto"), f"unknown SGLANG_DSV4_KV_LAYOUT={mode!r}"
    is_sm100 = (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] == 10
    )
    supported = flashmla_supports_v41_kv_layouts()
    if mode == "auto":
        if is_sm100 and supported:
            return KVLayout.V41, option
        return KVLayout.V4, None
    assert is_sm100, "the V4.1 KV cache layouts need an SM100 / SM103 GPU"
    if not supported:
        logger.warning(
            "SGLANG_DSV4_KV_LAYOUT=v41 but the installed FlashMLA does not advertise "
            "the V4.1 KV cache formats; the attention kernel will reject the cache."
        )
    return KVLayout.V41, option


class DeepSeekV4SingleKVPool(KVCache):
    # Paged FlashMLA main-KV format of this pool's rows.
    kv_layout: KVLayout = KVLayout.V4

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        kv_layout: Union[str, KVLayout] = KVLayout.V4,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        # Paged FlashMLA layout of this pool's pages; see KVLayout.
        self.kv_layout = KVLayout.parse(kv_layout)
        self.scale_pad = 1
        self.quantize_block_size = self.kv_layout.tile_size
        # V4 keeps its 64 RoPE dims in bf16; the V4.1 layouts quantize them too.
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        if self.kv_layout is not KVLayout.V4:
            assert self.qk_nope_head_dim + self.qk_rope_head_dim == 512
            return self.kv_layout.bytes_per_token
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        self.bytes_per_page_padded = self.kv_layout.page_bytes(self.page_size)

        if self.kv_layout is KVLayout.V4:
            assert bytes_per_token == 448 + 64 * 2 + 8, (
                "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
                "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
            )
            assert (
                self.bytes_per_page_padded
                == ceil_div(self.page_size * bytes_per_token, 576) * 576
            )
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        assert self.kv_layout is KVLayout.V4, (
            "the (fp8 nope, bf16 rope, 7 scales) pack is the V4 layout; "
            f"a {self.kv_layout.value} pool is written through set_key_buffer_fused"
        )
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        """Quantize ``cache_k`` ``[n, 512]`` bf16 into this pool's layout at ``loc``.
        ``freqs_cis`` (V4.1 only) rotates the RoPE tail in-kernel, so the input is
        the un-rotated latent and the fp4 / fp8 rounding happens once."""
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
            layout=self.kv_layout,
            freqs_cis=freqs_cis,
        )

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class DeepSeekV4UniformFP8KVPool(DeepSeekV4SingleKVPool):
    """Uniform 512-dim FP8 (e4m3) variant of the DSv4 single-KV pool.

    Each token is 448 NoPE + 64 RoPE contiguous e4m3 values without in-cache
    scales or per-page padding. The backend supplies the dequant scale.
    """

    def get_bytes_per_token(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        assert bytes_per_token == 512, (
            "DSV4 uniform-FP8 KV layout: qk_nope_head_dim (448) + "
            "qk_rope_head_dim (64), all e4m3 = 512 bytes/token"
        )
        self.kv_cache_total_dim = bytes_per_token
        self.bytes_per_page_padded = self.page_size * bytes_per_token

        return torch.zeros(
            num_pages,
            self.page_size * bytes_per_token,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )

    def get_key_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id]

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        raise NotImplementedError(
            "The packed NopeFp8RopeBf16Pack store does not apply to the "
            "uniform-FP8 pool; use set_key_buffer_fused."
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        """Store normed/roped rows as e4m3 with the backend's fixed unit scale.

        uint8 views work around index_put not supporting FP8 dtypes.
        """

        assert freqs_cis is None, "the uniform-FP8 pool takes finished (rotated) rows"
        assert cache_k.dim() == 2 and cache_k.shape[1] == self.kv_cache_total_dim
        self.kv_buffer[layer_id].view(torch.uint8).view(-1, self.kv_cache_total_dim)[
            loc.long()
        ] = cache_k.to(torch.float8_e4m3fn).view(torch.uint8)


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
        kv_layout: Union[str, KVLayout] = KVLayout.V4,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            kv_layout=kv_layout,
        )
        # The HiSparse transfer kernels hardcode the V4 token layout.
        assert self.kv_layout is KVLayout.V4, (
            f"HiSparse C4 pools support the V4 layout only, got {self.kv_layout}"
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k, freqs_cis)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


# Low-ratio indexer-K pool page, in compressed slots: the DeepGEMM indexer reads
# K in blocks of at most 128 and sglang's JIT metadata builder asserts 64.
def dsv41_index_page_size() -> int:
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        return 128
    return 64


DSV41_INDEX_PAGE_SIZE = dsv41_index_page_size()


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_fp4_indexer: Optional[bool] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        if use_fp4_indexer is None:
            use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        self.use_fp4_indexer = use_fp4_indexer
        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer
        # Low-ratio pools round to nearest even; c4 keeps threshold rounding.
        self.index_k_rne = False

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        return get_dsv4_indexer_bytes_per_token(
            self.index_head_dim, self.use_fp4_indexer
        )

    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        num_pages = (self.size + self.page_size + 1) // self.page_size
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                if self.uses_aiter_fp4_layout:
                    self.index_k_payload_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size, 16),
                            dtype=torch.uint8,
                            device=self.device,
                        ).view(torch.float4_e2m1fn_x2)
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_scale_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_with_scale_buffer = None
                    return

                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        num_pages,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def contiguous_page_row_buffers(self) -> List[torch.Tensor]:
        """Every indexer buffer as 2D page rows, for PD and HiCache transfer.

        FP8 keeps key and scale fused in one buffer per layer; the FP4 layout
        stores payload and scale separately, so it yields two buffers per layer.
        """
        if self.index_k_with_scale_buffer is not None:
            return self.index_k_with_scale_buffer
        return [
            buf.view(torch.uint8).flatten(1)
            for buf in (*self.index_k_payload_buffer, *self.index_k_scale_buffer)
        ]

    def get_index_k_fp4_payload_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_payload_buffer[layer_id]

    def get_index_k_fp4_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
            rne=self.index_k_rne,
        )

    def get_index_k_fp4(
        self, layer_id: int, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed fp4 rows at `slots`: (payload int8 [n, 64], scales int32 [n]),
        from the page layout [page_size * 64 payload | page_size * 4 scale]."""
        assert self.use_fp4_indexer, "packed readback only applies to the fp4 layout"
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        slots = slots.to(torch.int64)
        p = self.page_size
        page, off = (slots // p).unsqueeze(-1), slots % p
        payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
        scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(
            4, device=buf.device
        )
        payload = buf[page, payload_cols].view(torch.int8)  # [n, 64]
        scales = buf[page, scale_cols].contiguous().view(torch.int32).squeeze(-1)
        return payload, scales

    def get_index_k_dequant(
        self, layer_id: int, slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Dequantized bf16 [n, index_head_dim] index K; `slots` None reads the pool."""
        from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

        assert self.use_fp4_indexer, "dequant readback only applies to the fp4 layout"
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if slots is None:
            slots = torch.arange(self.size, device=buf.device)
        slots = slots.to(torch.int64)
        # Page layout: see get_index_k_fp4.
        p = self.page_size
        page, off = (slots // p).unsqueeze(-1), slots % p
        payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
        scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(
            4, device=buf.device
        )
        u = buf[page, payload_cols].view(torch.uint8)  # [n, 64]
        codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)  # [n, 64, 2]
        vals = DSV4_DEQUANT_FP4_TABLE.to(buf.device)[codes.long()].flatten(
            1
        )  # [n, 128]
        exps = buf[page, scale_cols].to(torch.int32) & 0xFF  # [n, 4]
        scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
        return (vals * scales).to(torch.bfloat16)


class _CompressedPoolConfig(NamedTuple):
    kv_size: int
    state_size: int
    state_dtype: torch.dtype
    indexer_size: Optional[int] = None


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 1, 2, 4, 128]
    # Layer index inside compress_kv_pool. Ratios 1/2 share a pool layer across the
    # kv_source layer that writes it and the layers that read it.
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# re-exported: the pool allocates the rows, but the kernels that write them own the
# layout (see unified_kv_kernels/layout.py)
DSV4_FP8_NOPE_ROW_BYTES = layout.DSV4_FP8_NOPE_ROW_BYTES
DSV4_FP8_QUANT_TILE = layout.DSV4_FP8_QUANT_TILE


def dsv4_unified_row_bytes(
    qk_nope_head_dim: int, qk_rope_head_dim: int, fp8: bool
) -> int:
    """Bytes one unified_kv token occupies, summed over both pools."""
    if not fp8:
        return (qk_nope_head_dim + qk_rope_head_dim) * 2
    num_tiles = -(-qk_nope_head_dim // DSV4_FP8_QUANT_TILE)
    scale_bytes = 2 * num_tiles
    # not an assert: sizing runs under -O too, and a silently skipped check here
    # overreports capacity
    if qk_nope_head_dim + scale_bytes > DSV4_FP8_NOPE_ROW_BYTES:
        raise ValueError(
            f"fp8 nope row overflows: {qk_nope_head_dim} latent values at 1 B + "
            f"{scale_bytes} B scales > {DSV4_FP8_NOPE_ROW_BYTES} B stride"
        )
    return DSV4_FP8_NOPE_ROW_BYTES + qk_rope_head_dim * 2


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout (bf16):
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16

    Layout (fp8, ``SGLANG_DSV4_UNIFIED_KV_FP8``) -- two parallel pools with the
    same row count, so a row index means the same thing in both. Named after the
    accessors, which under fp8 each return one half -- ``get_unified_kv`` the
    nope, ``get_unified_kv_rope`` the rope:
    unified_kv[L]      (nope): ``[rows, 512]`` fp8, see DSV4_FP8_NOPE_ROW_BYTES
    unified_kv_rope[L] (rope): ``[rows, qk_rope_head_dim]`` bf16, never quantized

    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
        fp8: bool = False,
    ):
        self.swa_ring_size = swa_ring_size
        self.fp8 = fp8
        self.rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        rope_bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    # Pad by one extra page. The KV pool reserves a null slot
                    # (token indices run 1..size).
                    compress_rows = self.num_blocks * self.k_per_block[ratio]
                    rows_per_page = self.page_size // ratio if ratio else 0
                    padded_compress_rows = compress_rows + rows_per_page
                    rows = self.swa_pages + padded_compress_rows
                    if self.fp8:
                        bufs.append(
                            torch.zeros(
                                rows,
                                DSV4_FP8_NOPE_ROW_BYTES,
                                dtype=torch.float8_e4m3fn,
                                device=device,
                            )
                        )
                        rope_bufs.append(
                            torch.zeros(
                                rows,
                                self.rope_head_dim,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                    else:
                        bufs.append(
                            torch.zeros(
                                rows,
                                self.head_dim,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                        rope_bufs.append(None)
        self.kv_buffer = bufs
        self.kv_buffer_rope = rope_bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_unified_kv_rope(self, local_layer_id: int) -> torch.Tensor:
        assert self.fp8, "rope pool only exists under SGLANG_DSV4_UNIFIED_KV_FP8"
        return self.kv_buffer_rope[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        if self.fp8:
            # same single-pool assumption as the outer get_contiguous_buf_infos:
            # one pointer and one row size per layer describes the nope pool only,
            # so whoever picks this up next would move half a row and not notice.
            # TODO(danli103): report both pools once a consumer needs them.
            raise NotImplementedError(
                "get_buf_infos describes one pool per layer; the fp8 rope pool "
                "would be dropped (SGLANG_DSV4_UNIFIED_KV_FP8=1)."
            )
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):
    # object.__new__ stubs (disagg wire test) skip __init__; False is the env
    # default, so the fp8 PD/HiCache refuses don't AttributeError on them.
    _unified_kv_fp8 = False

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        num_req_slots: Optional[int] = None,
        kv_source_layers: Sequence[int] = (),
        full_size: Optional[int] = None,
        is_draft_worker: bool = False,
        kv_layout: Union[str, KVLayout] = KVLayout.V4,
        compressed_kv_layout: Optional[str] = None,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # Layout of the SWA (main) cache; compressed caches follow
        # resolve_compressed_kv_layout, so valid (main, extra) pairs form only here.
        self.kv_layout = KVLayout.parse(kv_layout)
        assert self.kv_layout in (
            KVLayout.V4,
            KVLayout.V41,
        ), f"{self.kv_layout} is only valid for a compressed (extra) cache"
        self.compressed_kv_layout_option = compressed_kv_layout
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        # PD preallocation can exceed max_num_reqs;
        # the SWA ring must cover every addressable req_pool_idx.
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )

        # Resolve the unified-kv gate before any sizing so the two cannot drift.
        self._unified_kv = is_unified_kv_triton()
        self._unified_kv_fp8 = is_unified_kv_fp8()
        # Uniform 512-dim e4m3 layout for the trtllm attention backend
        self.uniform_fp8 = (
            not self._unified_kv
        ) and get_exec().kernel.dsv4_attn_backend == "trtllm"
        if self.uniform_fp8:
            assert self.kv_layout is KVLayout.V4, (
                "--dsv4-attn-backend trtllm keeps its own uniform 512-byte pages; "
                f"it cannot be combined with SGLANG_DSV4_KV_LAYOUT={self.kv_layout.value}"
            )
        c4_ring_size = self.get_ring_size(4)
        if self._unified_kv:
            # Unified C4 state is request-addressed: one ring per req slot,
            # so the caller-supplied, SWA-scaled size does not apply here.
            c4_state_pool_size = self.num_req_slots * c4_ring_size
        # Non-unified (fp8) keeps the caller-supplied, SWA-addressed size.
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped C128 state must also cover PD preallocation slots.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        # Only the ratios the model has anywhere get a pool config: the backend, PD
        # state transfer and HiCache read the registries as "the ratios this model
        # has", and a PP stage missing one keeps its empty pool so the PD wire aligns.
        model_ratios = set(compression_ratios)
        self.compressed_pool_configs = {
            ratio: config
            for ratio, config in {
                4: _CompressedPoolConfig(
                    kv_size=c4_size,
                    state_size=c4_state_pool_size,
                    state_dtype=c4_state_dtype,
                    indexer_size=c4_logical_size,
                ),
                128: _CompressedPoolConfig(
                    kv_size=c128_size,
                    state_size=c128_state_pool_size,
                    state_dtype=c128_state_dtype,
                ),
            }.items()
            if ratio in model_ratios
        }
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_state_num_req_slots = c128_state_pool_size
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                self.online_c128_state_num_req_slots, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0
        self.sliding_window = sliding_window

        self.swa_size = swa_size
        self.swa_page_size = swa_page_size

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)
        kv_pool_cls: type = DeepSeekV4SingleKVPool

        self.request_window = None
        encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay
        # DSpark's draft shares the target's full-to-SWA mapping, so the target
        # keeps its paged SWA allocator even under encoder replay.
        self.needs_paged_swa_allocator = (
            not encoder_replay
            or is_draft_worker
            or get_spec().speculative_algorithm is not None
        )
        if encoder_replay and not is_draft_worker:
            from sglang.srt.mem_cache.dsv41_request_window import RequestWindow

            def make_window_pool(size, layers):
                return self._make_kv_pool(
                    size=size,
                    page_size=swa_page_size,
                    dtype=dtype,
                    layer_num=layers,
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=swa_page_size,
                    kv_layout=self.kv_layout,
                )

            self.swa_kv_pool = None
            self.unified_kv_pool = None
            from sglang.srt.runtime_context import get_schedule

            chunk = get_schedule().chunked_prefill_size or 0
            self.request_window = RequestWindow(
                make_window_pool,
                num_slots=self.num_req_slots,
                layers=stage_layer_num,
                page_size=swa_page_size,
                capacity=self.sliding_window + (online_mtp_max_draft_tokens or 0),
                workspace_rows=(self.num_req_slots + 1) * self.sliding_window
                + max(
                    chunk,
                    (self.num_req_slots + 1) * (1 + (online_mtp_max_draft_tokens or 0)),
                ),
            )
        elif self._unified_kv:
            assert self.kv_layout is KVLayout.V4, (
                "unified_kv keeps bf16 rows, not a paged FlashMLA layout"
            )
            self.swa_kv_pool = None
            swa_ring_size = get_swa_ring_size(
                self.sliding_window, get_spec().speculative_algorithm is not None
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                page_size=page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=swa_ring_size,
                fp8=self._unified_kv_fp8,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = swa_ring_size
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
            self.swa_req_ring_size = self.unified_swa_ring_size
        else:
            self.unified_kv_pool = None
            if self.uniform_fp8:
                assert dtype == torch.float8_e4m3fn, (
                    "--dsv4-attn-backend trtllm requires "
                    f"kv_cache_dtype=fp8_e4m3, got {dtype}"
                )
                kv_pool_cls = DeepSeekV4UniformFP8KVPool
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
                kv_layout=self.kv_layout,
                cls=kv_pool_cls,
            )

        logger.info(
            "DSV4 SWA storage: worker=%s, storage=%s, paged_allocator=%s",
            "draft" if is_draft_worker else "target",
            "request_window" if self.request_window is not None else "paged",
            self.needs_paged_swa_allocator,
        )
        self.full_size = full_size
        self.kv_source_layers = list(kv_source_layers)
        self.sources_by_ratio = self._collect_sources_by_ratio()
        self._init_compressed_pools(
            stage_ratios=stage_ratios,
            page_size=page_size,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            enable_hisparse=enable_hisparse,
            kv_pool_cls=kv_pool_cls,
        )

        # The distinct compress ratios this stage has, sorted. Registry pools kept
        # for a ratio the model lacks (wire-layout alignment) do not count.
        model_ratios = set(self.compression_ratios)
        self.present_ratios: Tuple[int, ...] = tuple(
            ratio for ratio in sorted(self.kv_pools) if ratio in model_ratios
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def get_unified_kv_rope(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv_rope(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        spec = get_spec()
        return get_compress_state_ring_size(
            compress_ratio,
            spec.speculative_algorithm is not None,
            spec.speculative_num_draft_tokens or 0,
        )

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv_fp8:
            # The page-block transfer below prices one row as buf[0].nbytes and
            # ships a single pointer per layer. Under fp8 that covers the nope
            # pool only -- the parallel bf16 rope pool would be dropped and the
            # remote side would decode rows against stale rope. Refuse instead.
            # TODO(danli103): ship the rope pool as a second per-layer entry.
            raise NotImplementedError(
                "PD disaggregation is not supported with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 (the transfer assumes a single "
                "unified pool; the rope pool would be silently dropped)."
            )

        def append_page_buffer(buf: torch.Tensor) -> None:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        # Registration order defines the PD wire layout: C4 KV, C4 indexer, C128 KV.
        # Keep each indexer immediately after the KV buffers of the same ratio.
        for ratio, kv_pool in self.kv_pools.items():
            if self._unified_kv:
                # Unified buffers store token rows after the SWA ring. Transfer
                # compressed pages from the offset; SWA ships as StateType.SWA_RING.
                swa_pages = self.unified_kv_pool.swa_pages
                for local_layer_id, layer_ratio in enumerate(stage_ratios):
                    if layer_ratio != ratio:
                        continue
                    buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                    assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                    row_bytes = buf[0].nbytes
                    rows_per_page = self.page_size // ratio
                    compress_rows = buf.shape[0] - swa_pages
                    data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                    data_lens.append(compress_rows * row_bytes)
                    item_lens.append(rows_per_page * row_bytes)
            elif kv_pool is not None:
                for buf in kv_pool.kv_buffer:
                    append_page_buffer(buf)

            indexer_pool = self.index_pools.get(ratio)
            if indexer_pool is None:
                continue
            # The transfer addresses every buffer by FULL page id; ratio-1/2 index
            # pools page at DSV41_INDEX_PAGE_SIZE, so one item is the run of index
            # pages holding a FULL page's page_size // ratio slots.
            index_pages_per_full_page = 1
            if ratio in (1, 2):
                slots_per_full_page = self.page_size // ratio
                assert slots_per_full_page % indexer_pool.page_size == 0, (
                    f"ratio-{ratio} index pages of {indexer_pool.page_size} slots do not "
                    f"tile a FULL page of {slots_per_full_page} slots"
                )
                index_pages_per_full_page = (
                    slots_per_full_page // indexer_pool.page_size
                )
            for buf in indexer_pool.contiguous_page_row_buffers():
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes * index_pages_per_full_page)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        # StateType.SWA_RING transfers [0, swa_pages) of each unified_kv layer;
        # its indices address individual ring rows.
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        if self._unified_kv_fp8:
            # Other half of the PD path -- get_contiguous_buf_infos ships the
            # compressed region, this one the ring. Same single-pool assumption,
            # same silently dropped rope, same fix -- land them together.
            raise NotImplementedError(
                "PD disaggregation is not supported with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 (the SWA_RING component assumes a "
                "single unified pool; the rope pool would be silently dropped)."
            )
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def _unified_page_views(
        self, buffers: List[torch.Tensor], ratio: int
    ) -> Tuple[List[torch.Tensor], int]:
        # HiCache expects byte rows containing whole pages;
        # the unified pool stores individual token rows after its SWA region.
        # Bf16 kv layout: [rows, 1024B]
        # Fp8 kv layout:  [rows, 512B] fp8 nope, [rows, 128B] bf16 rope
        swa_pages = self.unified_kv_pool.swa_pages
        rows_per_page = self.page_size // ratio
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]

        views: List[torch.Tensor] = []
        for local_layer_id in local_layer_ids:
            buf = buffers[local_layer_id]
            compress_rows = buf.shape[0] - swa_pages
            assert compress_rows % rows_per_page == 0, (
                f"compressed rows {compress_rows} not a multiple of "
                f"rows_per_page {rows_per_page} for ratio {ratio}"
            )
            num_pages = compress_rows // rows_per_page
            page_view = (
                buf.narrow(0, swa_pages, compress_rows)
                .reshape(num_pages, rows_per_page * buf.shape[1])
                .view(torch.uint8)
            )
            views.append(page_view)

        item_bytes = rows_per_page * buffers[0].shape[1] * buffers[0].element_size()
        return views, item_bytes

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        """
        Main compressed region of one stage: bf16 latents, or fp8 nope.
        """
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"
        return self._unified_page_views(self.unified_kv_pool.kv_buffer, ratio)

    def unified_rope_region_buffers(
        self, ratio: int
    ) -> Optional[Tuple[List[torch.Tensor], int]]:
        """
        The bf16 rope half of an fp8 two-pool row, or None when there isn't one.

        A row index addresses both pools, so this mirrors exactly the rows
        ``unified_region_buffers`` does and only the row width differs. It needs
        its own host pool: offloading the nope half alone leaves whatever rope the
        row held before, which is wrong output rather than a crash.
        """
        if not self._unified_kv_fp8:
            return None
        assert self._unified_kv, "unified_rope_region_buffers requires unified_kv"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"
        return self._unified_page_views(self.unified_kv_pool.kv_buffer_rope, ratio)

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self.swa_kv_pool is not None:
            for buf in self.swa_kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                # Request-scoped state ships as C128_STATE, not with the SWA ring.
                if pool is None or pool.request_scoped:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_request_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Request-scoped state: the c128 raw-token ring (or its single online row)
        and the ratio-2 pending-pair ring. One item is one c128 page / pair ring."""
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or not pool.request_scoped:
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            if pool.ratio == 2:
                item_lens.append(t[0].nbytes * pool.ring_size)
            else:
                item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

    def _init_compressed_pools(
        self,
        *,
        stage_ratios: Sequence[int],
        page_size: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        enable_hisparse: bool,
        kv_pool_cls: type,
    ) -> None:
        """One KV pool (plus packed indexer-K pool) per compress ratio in this stage:
        slot = full-pool loc // ratio, page = page_size // ratio, so pages line up."""
        configs = self.compressed_pool_configs
        layer_counts = {ratio: stage_ratios.count(ratio) for ratio in configs}
        # Keep empty pools and allocation order for PP stages without a given ratio.
        self.kv_pools: dict[int, Optional[DeepSeekV4SingleKVPool]] = {
            ratio: None for ratio in configs
        }
        # The PD wire order stays C4, C128, then the ratio-1/2 kv_source layers.
        low_ratio_sources = {
            ratio: sources
            for ratio, sources in getattr(self, "sources_by_ratio", {}).items()
            if ratio in (1, 2)
        }
        if low_ratio_sources:
            assert self.full_size is not None, (
                "low compress ratios need the full pool size"
            )
            assert not self._unified_kv, "unified_kv has no low compress ratio layout"

        if not self._unified_kv:
            for ratio, config in configs.items():
                pool_cls = kv_pool_cls
                if ratio == 4 and enable_hisparse:
                    assert not self.uniform_fp8, (
                        "enable_hisparse is not supported with --dsv4-attn-backend trtllm."
                    )
                    pool_cls = HiSparseC4DevicePool
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=config.kv_size,
                    page_size=page_size // ratio,
                    dtype=dtype,
                    layer_num=layer_counts[ratio],
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=page_size,
                    cls=pool_cls,
                    kv_layout=self.compressed_kv_layout(ratio),
                )
            for ratio, sources in low_ratio_sources.items():
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=self.full_size // ratio,
                    page_size=page_size // ratio,
                    dtype=dtype,
                    layer_num=len(sources),
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=page_size,
                    cls=kv_pool_cls,
                    kv_layout=self.compressed_kv_layout(ratio),
                )

        self.index_pools: dict[int, DeepSeekV4IndexerPool] = {
            ratio: self._make_indexer_pool(
                config.indexer_size,
                page_size // ratio,
                dtype,
                self.indexer_head_dim,
                layer_counts[ratio],
                device,
                enable_memory_saver,
            )
            for ratio, config in configs.items()
            if config.indexer_size is not None
        }
        for ratio, sources in low_ratio_sources.items():
            # Reserved FULL page 0 pushes real slots past full_size, and one index
            # padding page is too small to cover that gap.
            self.index_pools[ratio] = self._make_indexer_pool(
                (self.full_size + page_size) // ratio,
                DSV41_INDEX_PAGE_SIZE,
                dtype,
                self.indexer_head_dim,
                len(sources),
                device,
                enable_memory_saver,
                force_fp4=True,
            )

        # HiCache and hardware backends still read these per-ratio attributes.
        self.c4_kv_pool = self.kv_pools.get(4)
        self.c128_kv_pool = self.kv_pools.get(128)
        self.c4_indexer_kv_pool = self.index_pools.get(4)

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
        kv_layout: KVLayout = KVLayout.V4,
    ) -> DeepSeekV4SingleKVPool:
        """Build a full / SWA / c4 / c128 single-KV pool. ``global_page_size``
        is the model-wide page_size (== ``page_size`` for the SWA pool, larger
        for the per-ratio c4/c128 pools); the default CUDA pool ignores it.
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the NPU bf16
        PA_ND variant, which needs ``global_page_size`` for its kernel view."""
        del global_page_size  # CUDA pools key only off their own page_size
        return cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kv_layout=kv_layout,
        )

    def compressed_kv_layout(self, compress_ratio: int) -> KVLayout:
        """See :func:`resolve_compressed_kv_layout`."""
        layout = resolve_compressed_kv_layout(
            self.kv_layout, compress_ratio, self.compressed_kv_layout_option
        )
        assert is_valid_kv_layout_pair(self.kv_layout, layout)
        return layout

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        force_fp4: bool = False,
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant. ``force_fp4`` forces the fp4 low-ratio layout."""
        if force_fp4:
            pool = DeepSeekV4IndexerPool(
                size,
                page_size,
                dtype,
                index_head_dim,
                layer_num,
                device,
                enable_memory_saver,
                use_fp4_indexer=True,
            )
            # The dsv41 low-ratio indexer rounds to nearest even (reference rounding).
            pool.index_k_rne = True
            return pool
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _make_compress_state_pool(
        self, ratio: int, *, head_dim: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build attention or indexer state; hardware backends override this factory."""
        config = self.compressed_pool_configs[ratio]
        return CompressStatePool(
            size=config.state_size,
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=head_dim,
            dtype=config.state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            request_scoped=ratio in (2, 128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _make_pair_state_pool(self, enable_memory_saver: bool) -> CompressStatePool:
        """Ratio-2 pending-pair state: one position ring per request slot, holding
        the fp32 (kv, score) of an even token until its odd partner arrives."""
        ring_size = self.get_ring_size(2)
        return CompressStatePool(
            size=self.num_req_slots * ring_size,
            ring_size=ring_size,
            overlap=False,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=torch.float32,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=2,
            request_scoped=True,
            online=False,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio in (0, 1):
                continue

            if ratio == 2:
                # Only a kv_source layer compresses; later ratio-2 layers read it.
                if idx in self.sources_by_ratio.get(2, []):
                    self.compress_state_pools[idx] = self._make_pair_state_pool(
                        enable_memory_saver
                    )
                continue

            self.compress_state_pools[idx] = self._make_compress_state_pool(
                ratio,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )

            if ratio in self.index_pools:
                self.indexer_compress_state_pools[idx] = self._make_compress_state_pool(
                    ratio,
                    head_dim=self.indexer_head_dim,
                    enable_memory_saver=enable_memory_saver,
                )

    def _collect_sources_by_ratio(self) -> dict[int, List[int]]:
        """Layers owning compressed storage: all of ratios 4/128, kv_sources of 1/2."""
        stage = range(self._stage_start, self._stage_end)
        for idx in stage:
            ratio = self.compression_ratios[idx]
            if ratio not in (0, 1, 2, 4, 128):
                raise ValueError(f"Unsupported compression ratio: {ratio}")

        sources_by_ratio: dict[int, List[int]] = {}
        for ratio in (4, 128, 1, 2):
            if ratio in (1, 2):
                layers = [
                    l
                    for l in self.kv_source_layers
                    if l in stage and self.compression_ratios[l] == ratio
                ]
            else:
                layers = [l for l in stage if self.compression_ratios[l] == ratio]
            if layers:
                sources_by_ratio[ratio] = layers
        return sources_by_ratio

    def source_layer_of(self, layer_id: int) -> int:
        """The layer owning this layer's compressed storage: itself for ratios 4/128,
        the nearest preceding kv_source layer for ratios 1/2."""
        ratio = self.compression_ratios[layer_id]
        sources = [l for l in self.sources_by_ratio[ratio] if l <= layer_id]
        assert sources, f"layer {layer_id} (ratio {ratio}) has no kv_source layer"
        return max(sources)

    def _init_compressed_layer_mapping(self):
        layer_counts = {0: 0, **{ratio: 0 for ratio in self.kv_pools}}
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio not in layer_counts:
                raise ValueError(f"Unsupported compression ratio: {ratio}")
            if ratio in (1, 2):
                sources = self.sources_by_ratio[ratio]
                compress_layer_id = sources.index(self.source_layer_of(idx))
            else:
                compress_layer_id = layer_counts[ratio]
                layer_counts[ratio] += 1
            self.layer_mapping[idx] = DeepSeekV4LayerItem(
                compress_ratio=ratio,
                compress_layer_id=compress_layer_id,
                compress_kv_pool=self.kv_pools.get(ratio),
            )

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert compress_state_pool is not None, (
            "Only c4/c128 layers and ratio-2 kv_source layers have attention states."
        )
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_state_num_req_slots(self) -> int:
        return self.online_c128_state_num_req_slots

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def clear_c4_req_states(self, req_pool_indices: Sequence[int]) -> None:
        if not self._unified_kv or not req_pool_indices:
            return

        pools = [
            pool
            for pool in self.compress_state_pools + self.indexer_compress_state_pools
            if pool is not None and pool.ratio == 4
        ]
        if not pools:
            return

        ring_size = self.get_ring_size(4)
        device = pools[0].kv_score_buffer.kv_score.device
        req_indices = torch.as_tensor(req_pool_indices, dtype=torch.long, device=device)
        state_locs = (
            req_indices[:, None] * ring_size
            + torch.arange(ring_size, dtype=torch.long, device=device)
        ).flatten()

        for pool in pools:
            state = pool.kv_score_buffer.kv_score
            half = state.shape[-1] // 2
            state[state_locs, :half] = 0
            state[state_locs, half:] = float("-inf")

    def request_state_transfer_indices(self, req_pool_idx: int, seq_len: int):
        """PD transfer indices of the request-state component for one request."""
        pools = [
            p for p in self.compress_state_pools if p is not None and p.request_scoped
        ]
        assert pools, "no request-scoped state pool"
        # One index list addresses every request-state buffer (one per layer), so
        # the request-scoped pools must share a ring layout.
        layout = (pools[0].ratio, pools[0].online, pools[0].ring_size)
        assert all((p.ratio, p.online, p.ring_size) == layout for p in pools), (
            "request-scoped state pools must share one ring layout"
        )
        return pools[0].transfer_indices(req_pool_idx, seq_len)

    def clear_request_scoped_state(self, req_pool_idx: int) -> None:
        """Reset one req slot's C128 ring and ratio-2 pending-pair state."""
        for pool in self.compress_state_pools:
            if pool is None or not pool.request_scoped:
                continue

            if pool.ratio == 128 and ONLINE_C128:
                row = pool.kv_score_buffer.kv_score[req_pool_idx]
                head_dim = row.shape[-1] // 3
                row[:head_dim].fill_(float("-inf"))
                row[head_dim:].zero_()
                continue

            start = req_pool_idx * pool.ring_size
            pool.kv_score_buffer[start : start + pool.ring_size].clear()

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        # C128 compression can read rejected draft slots at a boundary;
        # C4 overwrites its draft slots before reading them.
        if ONLINE_C128 or num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        bs = req_pool_indices.numel()
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            clear_unaccepted_c128_draft_states(
                pool.kv_score_buffer.kv_score,
                req_pool_indices,
                seq_lens,
                accept_lens,
                ring_size=pool.ring_size,
                num_draft_tokens=num_draft_tokens,
            )

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert indexer_compress_state_pool is not None, (
            "Only c4 layers have indexer states."
        )
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        if self.request_window is not None:
            return self.request_window.buffer(self._swa_local_layer_id(layer_id))
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        if self.request_window is not None:
            return self.get_swa_raw_buffer(layer_id).view(
                self.request_window.state.dtype
            )
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        assert self.kv_layout is KVLayout.V4, (
            "the (fp8 nope, bf16 rope, 7 scales) pack is the V4 layout; "
            f"a {self.kv_layout.value} pool is written through the fused setters"
        )
        if self.request_window is not None:
            dsv4_index_buf_accessor.SetKAndS.execute(
                pool=self.request_window.state,
                buf=self.get_swa_raw_buffer(layer_id),
                loc=loc,
                nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            )
        else:
            self.swa_kv_pool.set_key_buffer(
                self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack
            )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_layout(self, layer_id: int) -> KVLayout:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.kv_layout

    def get_extra_key_bytes_per_token(self, layer_id: int) -> int:
        """Last dim of the ``(pages, page_size, 1, bytes)`` view the attention
        kernel detects the extra cache's format from."""
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.kv_cache_total_dim

    def get_swa_key_layout(self) -> KVLayout:
        # swa_kv_pool is None under the request window and unified_kv.
        return self.kv_layout

    def get_swa_key_bytes_per_token(self) -> int:
        """Last dim of the ``(pages, page_size, 1, bytes)`` view the attention
        kernel detects the SWA cache's format from."""
        if self.uniform_fp8:
            # The trtllm uniform-FP8 pool has no paged FlashMLA layout: 512 B/token.
            return self.swa_kv_pool.kv_cache_total_dim
        return self.kv_layout.bytes_per_token

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def _indexer_pool(self, compress_ratio: int) -> DeepSeekV4IndexerPool:
        pool = self.index_pools.get(compress_ratio)
        assert pool is not None, (
            f"No indexer pool for compression ratio {compress_ratio}"
        )
        return pool

    def get_low_ratio_index_k_dequant(
        self, layer_id: int, slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Index-K rows at `slots` from the layer's latent source."""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_dequant(
            compress_layer_id, slots
        )

    def get_low_ratio_index_k_fp4(
        self, layer_id: int, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed fp4 index-K rows at `slots`: (payload int8 [n, 64], ue8m0 scales
        packed int32 [n]), the input layout of quantize_fp4_indexer_tensor."""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4(
            compress_layer_id, slots
        )

    def get_index_k_page_size(self, compress_ratio: int = 4) -> int:
        return self._indexer_pool(compress_ratio).page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_with_scale_buffer(
            compress_layer_id
        )

    def get_index_k_fp4_payload_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4_payload_buffer(
            compress_layer_id
        )

    def get_index_k_fp4_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4_scale_buffer(
            compress_layer_id
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_scale_buffer(
            compress_layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        self._indexer_pool(compress_ratio).set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.set_swa_key_buffer(layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack)

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        if self.request_window is not None:
            return self.get_swa_raw_buffer(layer_id).view(
                self.request_window.state.dtype
            )
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.get_swa_raw_buffer(layer_id),
            indices=swa_loc,
            page_size=self.swa_page_size,
            type="flashmla",
            layout=self.kv_layout,
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self.uniform_fp8:
            # Uniform-FP8 (trtllm-gen) layout: norm + RoPE with the existing
            # Triton kernel (in-place on kv; safe -- kv is not read again),
            # then a plain e4m3 cast + scatter in the pool setter (per-tensor
            # scale 1.0). Fusing the store is deferred to the perf phase.
            from sglang.kernels.ops.attention.deepseek_v4_rope import (
                fused_norm_rope_inplace_triton,
            )

            fused_norm_rope_inplace_triton(
                kv,
                kv_weight,
                eps,
                freqs_cis,
                positions=positions,
            )
            self.swa_kv_pool.set_key_buffer_fused(
                self._swa_local_layer_id(layer_id), swa_loc, kv
            )
            return
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.get_swa_raw_buffer(layer_id),
            page_size=self.swa_page_size,
            layout=self.kv_layout,
        )

    def set_unified_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """unified_kv counterpart of set_swa_key_buffer_radix_fused_norm_rope.

        Under unified_kv the (fp8, paged) swa_kv_pool is None -- SWA K lives in
        the shared bf16 unified_kv ring instead. Norm+RoPE the draft KV in place
        (the same freqs_cis path the main model uses via _compute_kv_bf16) and
        scatter it into ``unified_kv[swa_loc]``. Rows with swa_loc < 0
        (uncommitted verify tokens) are skipped by the scatter.
        """
        from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        fused_norm_rope_inplace(kv, kv_weight, eps, freqs_cis, positions)
        runtime.scatter_bf16_into_unified(
            kv=kv,
            loc=swa_loc,
            unified_kv=self.get_unified_kv(layer_id),
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        """Write ``cache_k`` ``[n, 512]`` bf16 into the layer's compressed cache.
        For an fp4 (``V41_FP4``) cache pass the *un-quantized* latent, plus
        ``freqs_cis`` if it is not rotated yet: the kernel rounds to e2m1 once."""
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        if freqs_cis is not None:
            assert compress_kv_pool.kv_layout is KVLayout.V41_FP4, (
                "in-kernel RoPE is for the fp4 cache; fp8 caches take the finished value"
            )
        return compress_kv_pool.set_key_buffer_fused(
            compress_layer_id, loc, cache_k, freqs_cis
        )

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).set_index_fused(
            compress_layer_id, loc, cache_k
        )

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).set_index_fp4(
            compress_layer_id, loc, cache_k
        )
