# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV pool for ERNIE 5.5 Flash: the compressed-KV pool plus a latent sub-pool
for the MLA layers.

The HCA layers (``compress_ratio == 128``) keep the compressed-KV layout
unchanged. An MLA layer is presented to the compressed-KV pool as ratio 0 (no
compressed state) and excluded from its SWA buffers by ``swa_layer_ids``; it
stores its latent KV plus the DSA lightning-indexer key in
:attr:`Ernie5_5TokenToKVPool.mla_kv_pool`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.ops.quantization.dsa_quant_k_cache import fp8_ds_mla_row_width
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    ExtraHostRegion,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.runtime_context import get_exec, get_spec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

# FlashMLA sparse and the DSA indexer kernels are built for 64-token pages.
DSA_PAGE_SIZE = 64


def mla_layer_bytes_per_token(
    hf_config, kv_cache_dtype_str: str, use_fp4_indexer: bool = False
) -> int:
    """Bytes one token adds to one MLA layer: latent + quantized index key.

    Only ``fp8_e4m3`` and ``bfloat16`` get here; the model override rejects the rest.
    """
    kv_lora_rank = hf_config.hybrid_mla_kv_lora_rank
    rope_dim = hf_config.hybrid_mla_qk_rope_head_dim
    # A bf16 latent row, or the FlashMLA FP8 one (656 B for 512 + 64).
    latent = (
        fp8_ds_mla_row_width(kv_lora_rank, rope_dim)
        if kv_cache_dtype_str.startswith("fp8")
        else (kv_lora_rank + rope_dim) * torch.bfloat16.itemsize
    )
    index_head_dim = hf_config.index_head_dim
    index_key = DSATokenToKVPool.index_bytes_per_token_for(
        index_head_dim, use_fp4_indexer
    )
    return latent + index_key


class _LayerSubsetDSATokenToKVPool(DSATokenToKVPool):
    """DSA pool indexed by model-wide layer id, with storage only for
    ``owned_layer_ids``; every other layer gets a 0-row placeholder."""

    def __init__(self, *, owned_layer_ids: Sequence[int], layer_num: int, **kwargs):
        self._owned_layer_ids = frozenset(owned_layer_ids)
        super().__init__(
            layer_num=layer_num,
            skip_topk_layers=[i not in self._owned_layer_ids for i in range(layer_num)],
            **kwargs,
        )

    def layer_buffer_rows(self, layer_index: int) -> int:
        if layer_index not in self._owned_layer_ids:
            return 0
        return super().layer_buffer_rows(layer_index)


class _StageLocalLayerCounter:
    """A layer-done counter read with global layer ids, shifted to the stage's."""

    def __init__(self, counter, stage_start: int):
        self._counter = counter
        self._stage_start = stage_start

    def wait_until(self, layer_id: int) -> None:
        self._counter.wait_until(layer_id - self._stage_start)


class Ernie5_5TokenToKVPool(DeepSeekV4TokenToKVPool):
    @staticmethod
    def stage_kwargs(hf_config, start_layer: int, end_layer: int) -> dict:
        return dict(
            mla_layer_ids=[
                i for i in hf_config.mla_layer_ids if start_layer <= i < end_layer
            ],
            mla_kv_lora_rank=hf_config.hybrid_mla_kv_lora_rank,
            mla_qk_rope_head_dim=hf_config.hybrid_mla_qk_rope_head_dim,
            index_head_dim=hf_config.index_head_dim,
            use_fp4_indexer=get_exec().kernel.enable_fp4_indexer,
        )

    def __init__(
        self,
        *,
        mla_layer_ids: Sequence[int],
        mla_kv_lora_rank: int,
        mla_qk_rope_head_dim: int,
        index_head_dim: int,
        use_fp4_indexer: bool = False,
        **dsv4_kwargs,
    ):
        # Read by swa_layer_ids, which the base __init__ calls.
        self._mla_layer_ids = frozenset(mla_layer_ids)
        super().__init__(**dsv4_kwargs)
        self._swa_buffer_index = {
            layer: index for index, layer in enumerate(self.swa_layer_ids())
        }
        assert self.full_size is not None, "the compressed-KV pool sets full_size"
        if self.page_size % DSA_PAGE_SIZE:
            raise ValueError(
                f"ERNIE 5.5 Flash needs a page size that is a multiple of "
                f"{DSA_PAGE_SIZE}, got {self.page_size}"
            )
        # The model-wide allocator hands out slots up to full_size + page_size - 1,
        # while this pool pads by one of its own 64-token pages.
        size = self.full_size + self.page_size - DSA_PAGE_SIZE
        fp8 = self.dtype == torch.float8_e4m3fn
        self.mla_kv_pool = _LayerSubsetDSATokenToKVPool(
            owned_layer_ids=mla_layer_ids,
            layer_num=self._stage_end,
            size=size,
            page_size=DSA_PAGE_SIZE,
            kv_lora_rank=mla_kv_lora_rank,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            qk_rope_head_dim=mla_qk_rope_head_dim,
            device=self.device,
            index_head_dim=index_head_dim,
            use_fp4_indexer=use_fp4_indexer,
            enable_memory_saver=dsv4_kwargs["enable_memory_saver"],
            kv_cache_dim=(
                fp8_ds_mla_row_width(mla_kv_lora_rank, mla_qk_rope_head_dim)
                if fp8
                else mla_kv_lora_rank + mla_qk_rope_head_dim
            ),
        )

    def swa_layer_ids(self) -> list[int]:
        # The MLA layers attend over the full context and keep no SWA buffer.
        return [
            layer
            for layer in super().swa_layer_ids()
            if layer not in self._mla_layer_ids
        ]

    def _swa_local_layer_id(self, layer_id: int) -> int:
        return self._swa_buffer_index[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter) -> None:
        super().register_layer_transfer_counter(layer_transfer_counter)
        # The sub-pool getters wait on a HiCache layer load only if it holds the
        # counter too; it waits on global layer ids, the counter on stage-local ones.
        self.mla_kv_pool.register_layer_transfer_counter(
            None
            if layer_transfer_counter is None
            else _StageLocalLayerCounter(layer_transfer_counter, self._stage_start)
        )

    def _mla_page_regions(self) -> list[tuple[PoolName, int, torch.Tensor, int]]:
        """``(pool name, layer_id, buffer, bytes per main page)``: every latent
        buffer, then every index-K one, each in layer order.

        Both kinds are addressed by the *main* pool's page id, so one item is
        one full main page.
        """
        regions = []
        for pool_name, buffers, rows_per_main_page in (
            (PoolName.ERNIE5_5_MLA, self.mla_kv_pool.kv_buffer, self.page_size),
            (
                PoolName.ERNIE5_5_MLA_INDEXER,
                self.mla_kv_pool.index_k_with_scale_buffer,
                self.page_size // DSA_PAGE_SIZE,
            ),
        ):
            for layer_id, buf in enumerate(buffers):
                if buf.shape[0] == 0:
                    continue
                # Latent is [tokens, 1, kv_cache_dim], index-K [pages, row];
                # either way buf[0] is one unit of rows_per_main_page.
                assert buf.ndim in (2, 3), (
                    f"expected a 2D or 3D buffer, got {buf.ndim}D"
                )
                item_len = buf[0].nbytes * rows_per_main_page
                assert buf.nbytes >= item_len, (
                    f"a {buf.nbytes} B buffer cannot hold one {item_len} B page"
                )
                regions.append((pool_name, layer_id, buf, item_len))
        return regions

    def get_contiguous_buf_infos(self):
        """The compressed-KV regions, then :meth:`_mla_page_regions`. Prefill and
        decode must register the same regions in the same order."""
        data_ptrs, data_lens, item_lens = super().get_contiguous_buf_infos()
        for _, _, buf, item_len in self._mla_page_regions():
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(item_len)
        return data_ptrs, data_lens, item_lens

    def extra_host_regions(self) -> tuple[ExtraHostRegion, ...]:
        views: dict[PoolName, list[tuple[int, torch.Tensor]]] = {}
        for pool_name, layer_id, buf, item_len in self._mla_page_regions():
            # The allocator never hands out a trailing page that does not fit whole.
            num_pages = buf.nbytes // item_len
            flat = buf.view(torch.uint8).reshape(-1)[: num_pages * item_len]
            views.setdefault(pool_name, []).append(
                (layer_id, flat.view(num_pages, item_len))
            )
        return tuple(
            ExtraHostRegion(
                pool_name=pool_name,
                device_pool=self.mla_kv_pool,
                device_buffers=[view for _, view in layer_views],
                item_bytes=layer_views[0][1].shape[1],
                # Transfer ids are stage-local, the sub-pool's layer ids global.
                layer_mapping={
                    layer_id - self._stage_start: index
                    for index, (layer_id, _) in enumerate(layer_views)
                },
            )
            for pool_name, layer_views in views.items()
        )


class Ernie5_5PoolConfigurator(DSV4PoolConfigurator):
    """ERNIE 5.5 Flash: the MLA layers keep a latent + index key per full token
    and no SWA buffer (Ernie5_5TokenToKVPool.swa_layer_ids)."""

    def __init__(self, kvc: KVCacheConfigurator):
        self._hf_config = kvc.model_config.hf_config
        self.use_fp4_indexer = get_exec().kernel.enable_fp4_indexer
        stage = range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
        self._num_mla_layers = sum(i in stage for i in kvc.model_config.mla_layer_ids)
        # The MTP draft is one MLA layer, priced here instead of by inflation.
        self._draft_kv_priced_exactly = (
            get_spec().speculative_algorithm is not None
            and not kvc.spec_algorithm.is_dspark()
        )
        super().__init__(kvc)

    @property
    def num_swa_layers(self) -> int:
        return self.num_layers_total - self._num_mla_layers

    def _get_bytes_per_full_token(self) -> float:
        mla_layers = self._num_mla_layers + int(self._draft_kv_priced_exactly)
        return super()._get_bytes_per_full_token() + mla_layers * (
            mla_layer_bytes_per_token(
                self._hf_config, self.kv_cache_dtype_str, self.use_fp4_indexer
            )
        )
