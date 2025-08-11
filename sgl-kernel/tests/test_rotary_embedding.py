from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace
from sgl_kernel.testing.rotary_embedding import (
    FlashInferRotaryEmbedding,
    MHATokenToKVPool,
    RotaryEmbedding,
    create_inputs,
)


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads, save_kv_cache",
    [
        # GPT-OSS cases
        *[
            (
                64,
                64,
                4096,
                8000,
                False,
                torch.bfloat16,
                "cuda",
                batch_size,
                seq_len,
                64,
                8,
                save_kv_cache,
            )
            for save_kv_cache in (False, True)
            for batch_size, seq_len in (
                (1, 1),
                (32, 1),
                (2, 512),
            )
        ],
        # Other cases
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1, False),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2, False),
        (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4, False),
        (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
    ],
)
def test_correctness(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    save_kv_cache: bool,
):
    config = dict(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )

    rope_ref = RotaryEmbedding(**config).to(device)
    rope_flashinfer = FlashInferRotaryEmbedding(**config).to(device)

    inputs = create_inputs(
        head_size=head_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )

    if save_kv_cache:
        pool_ref = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)
        pool_flashinfer = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)

    query_ref, key_ref = inputs["query"].clone(), inputs["key"].clone()
    query_flashinfer, key_flashinfer = inputs["query"].clone(), inputs["key"].clone()

    query_ref_out, key_ref_out = rope_ref.forward_native(
        inputs["pos_ids"], query_ref, key_ref
    )
    if save_kv_cache:
        pool_ref.set_kv_buffer(
            loc=inputs["out_cache_loc"],
            cache_k=key_ref_out.view(-1, num_kv_heads, head_size),
            cache_v=inputs["value"].view(-1, num_kv_heads, head_size),
        )

    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(
        inputs["pos_ids"],
        query_flashinfer,
        key_flashinfer,
        fused_set_kv_buffer_arg=(
            FusedSetKVBufferArg(
                value=inputs["value"],
                k_buffer=pool_flashinfer.k_buffer[0].view(-1, num_kv_heads * head_size),
                v_buffer=pool_flashinfer.v_buffer[0].view(-1, num_kv_heads * head_size),
                k_scale=None,
                v_scale=None,
                cache_loc=inputs["out_cache_loc"],
            )
            if save_kv_cache
            else None
        ),
    )

    torch.testing.assert_close(
        query_ref_out, query_flashinfer_out, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(key_ref_out, key_flashinfer_out, atol=1e-2, rtol=1e-2)
    if save_kv_cache:
        assert torch.all(pool_ref.k_buffer[0] == pool_flashinfer.k_buffer[0])
        assert torch.all(pool_ref.v_buffer[0] == pool_flashinfer.v_buffer[0])


if __name__ == "__main__":
    pytest.main([__file__])
