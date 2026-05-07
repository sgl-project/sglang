from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from sglang.jit_kernel.deepseek_v4 import (
    tilelang_make_swa_prefill_indices,
    triton_make_swa_prefill_indices,
)
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


def expand_seq_lens(
    *,
    seq_lens: List[int],
    extend_seq_lens: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = sum(extend_seq_lens)
    seq_lens_expanded = torch.empty(num_tokens, **_HOST_INT32_KWARGS)
    expanded_idx_to_unexpanded_idx = torch.empty(num_tokens, **_HOST_INT32_KWARGS)
    offset = 0
    for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
        out = seq_lens_expanded[offset : offset + qo_len]
        offset += qo_len
        torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
        expanded_idx_to_unexpanded_idx[offset - qo_len : offset].fill_(i)
    return (
        seq_lens_expanded.to(device, non_blocking=True),
        expanded_idx_to_unexpanded_idx.to(device, non_blocking=True),
    )


# NOTE: about the ring buffer layout:
# TODO(dark): add doc


def make_swa_ring_buffer_indices(
    forward_batch: ForwardBatch,
    device: torch.device,
    *,
    max_seq_len: int,
    swa_window_size: int,
) -> torch.Tensor:
    SWA_WINDOW = swa_window_size
    extend_num_tokens = forward_batch.extend_num_tokens
    assert extend_num_tokens is not None
    if envs.SGLANG_OPT_USE_TILELANG_SWA_PREPARE.get():
        seq_lens = forward_batch.seq_lens
        extend_lens = forward_batch.extend_seq_lens
        assert extend_lens is not None
        seq_lens_k = seq_lens.to(torch.int32)
        seq_lens_q = extend_lens.to(torch.int32)
        swa_indices = torch.empty(
            (extend_num_tokens, SWA_WINDOW), device=device, dtype=torch.int32
        )
        return tilelang_make_swa_prefill_indices(
            seq_lens_k=seq_lens_k,
            seq_lens_q=seq_lens_q,
            swa_indices=swa_indices,
        )
    elif envs.SGLANG_OPT_USE_TRITON_SWA_PREPARE.get():
        seq_lens = forward_batch.seq_lens
        extend_lens = forward_batch.extend_seq_lens
        assert extend_lens is not None
        seq_lens_k = seq_lens.to(torch.int32)
        seq_lens_q = extend_lens.to(torch.int32)
        swa_indices = torch.empty(
            (extend_num_tokens, SWA_WINDOW), device=device, dtype=torch.int32
        )
        return triton_make_swa_prefill_indices(
            seq_lens_k=seq_lens_k,
            seq_lens_q=seq_lens_q,
            swa_indices=swa_indices,
        )

    seq_lens = forward_batch.seq_lens_cpu
    extend_lens = forward_batch.extend_seq_lens_cpu
    assert seq_lens is not None and extend_lens is not None
    batch_size = len(seq_lens)
    num_tokens = extend_num_tokens
    swa_indices = torch.full((num_tokens, swa_window_size), -1, **_HOST_INT32_KWARGS)
    cum_qo_len = 0
    abs_pos_buf = torch.arange(max_seq_len, dtype=torch.int32)
    for seq_idx, (kv_len, qo_len) in enumerate(zip(seq_lens.tolist(), extend_lens)):
        # already existing KV
        old_kv_start = seq_idx * SWA_WINDOW
        # newly computed KV
        new_kv_start = batch_size * SWA_WINDOW + cum_qo_len
        prefix_len = kv_len - qo_len
        for curr_seq_qo_idx in range(qo_len):
            # layout | prefix_len (cached) | qo_len                  |
            #        | 0 ... prefix_len-1  | prefix_len ... kv_len-1 |
            #
            # Step 1: compute chosen_abs_positions - absolute positions to look at for this specific query token
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = max(end_abs_pos - SWA_WINDOW, 0)
            chosen_abs_positions = abs_pos_buf[start_abs_pos:end_abs_pos]
            # Step 2: compute swa_indices
            # For one abs_pos in chosen_abs_positions, the swa_indices will be:
            # 1. abs_pos < prefix_len  -> old_kv_start + abs_pos % SWA_WINDOW
            # 2. abs_pos >= prefix_len -> new_kv_start + (abs_pos - prefix_len)
            torch.where(
                chosen_abs_positions < prefix_len,
                old_kv_start + chosen_abs_positions % SWA_WINDOW,
                new_kv_start + (chosen_abs_positions - prefix_len),
                out=swa_indices[
                    cum_qo_len + curr_seq_qo_idx, : end_abs_pos - start_abs_pos
                ],
            )
        cum_qo_len += qo_len
    return swa_indices.to(device, non_blocking=True)


def prepare_swa_ring_buffer_cache(
    swa_k: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    core_metadata: PagedCoreMetadata,
    debug_dump_hook: Any,
) -> Tuple[torch.Tensor, index_buf_accessor_v4.NopeFp8RopeBf16Pack]:
    # Quick example: A prefill batch, with:
    # * request 0/1: generates 3 token, cache_len = 5 (i.e. seq_len = 5 + 3 = 8)
    # * req 2: generate 2 token, cache_len=1000
    #
    # Then, the temporary KV Cache has:
    # * indices [0, 128) = request 0 window.
    #       abs_pos=0,1,2,3,4 <-> pool_idx=0,1,2,3,4
    #       pool_idx=5...127 contains no data
    # * indices [128, 256) = request 1 window.
    #       abs_pos=0,1,2,3,4 <-> pool_idx=128,...,132
    #       pool_idx=128...255 contains no data
    # * indices [256, 384) = request 2 window.
    #       it is a ring buffer, thus abs_pos % 128 = pool_idx_inside_the_block
    #       abs_pos=0...,781 <-> no valid corresponding pool idx
    #       abs_pos=872...,895 <-> pool_idx=360,...,383
    #       abs_pos=896,...,999 <-> pool_idx=256,...,359
    # * indices [384, 387) = request 0 newly gen 3 kv token
    #       abs_pos=5,6,7 <-> pool_idx=384,...,386
    # * indices [387, 390) = request 1 newly gen 3 kv token
    #       abs_pos=5,6,7 <-> pool_idx=387,...,389
    # * indices [390, 392) = request 2 newly gen 2 kv token
    #       abs_pos=1000,1001 <-> pool_idx=390,391

    pool_swa_k_cache = token_to_kv_pool.get_swa_key_buffer(layer_id)
    num_pool_pages = forward_batch.batch_size
    num_newly_gen_tokens, _ = swa_k.shape

    swa_kv_pool = token_to_kv_pool.swa_kv_pool
    swa_page_size = swa_kv_pool.page_size
    assert swa_page_size == 128
    effective_swa_k_cache = swa_kv_pool.create_buffer(
        num_pages=num_pool_pages + ceil_align(num_newly_gen_tokens, swa_page_size),
    )

    # a. SWA data in real kv cache
    loc_swa = forward_batch.req_pool_indices
    assert loc_swa.shape[0] == forward_batch.batch_size == num_pool_pages
    effective_swa_k_cache[:num_pool_pages, :] = pool_swa_k_cache[loc_swa, :].view(
        effective_swa_k_cache.dtype
    )

    # b. Newly generated data
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
