from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from sglang.jit_kernel.deepseek_v4 import tilelang_make_swa_prefill_indices
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor_v4
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.metadata import PagedCoreMetadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_HOST_INT32_KWARGS: Dict = dict(dtype=torch.int32, device="cpu", pin_memory=True)

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def prepare_swa_ring_buffer_cache(
    swa_k: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    core_metadata: PagedCoreMetadata,
    debug_dump_hook: Any,
) -> Tuple[torch.Tensor, index_buf_accessor_v4.NopeFp8RopeBf16Pack]:

    pool_swa_k_cache = token_to_kv_pool.get_swa_key_buffer(layer_id)
    num_pool_pages = forward_batch.batch_size
    num_newly_gen_tokens, _ = swa_k.shape

    swa_kv_pool = token_to_kv_pool.swa_kv_pool
    swa_page_size = swa_kv_pool.page_size
    assert swa_page_size == 128
    effective_swa_k_cache = swa_kv_pool.create_buffer(
        num_pages=num_pool_pages + ceil_align(num_newly_gen_tokens, swa_page_size),
    )

    loc_swa = forward_batch.req_pool_indices
    assert loc_swa.shape[0] == forward_batch.batch_size == num_pool_pages
    effective_swa_k_cache[:num_pool_pages, :] = pool_swa_k_cache[loc_swa, :].view(
        effective_swa_k_cache.dtype
    )

    swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
    offset = num_pool_pages * swa_page_size
    loc_newly_gen = torch.arange(
        offset,
        offset + num_newly_gen_tokens,
        device=loc_swa.device,
    )
    index_buf_accessor_v4.SetKAndS.execute(
        pool=swa_kv_pool,
        buf=effective_swa_k_cache,
        loc=loc_newly_gen,
        nope_fp8_rope_bf16_pack=swa_k_pack,
    )

    if h := debug_dump_hook:
        h(
            "forward__swa_info",
            dict(
                loc_swa=loc_swa,
                loc_newly_gen=loc_newly_gen,
            ),
        )

    return effective_swa_k_cache, swa_k_pack.slice_pack(core_metadata.swa_slice)
