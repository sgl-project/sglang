from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TurboQuantDenseKVCodec,
    TurboQuantDenseKVConfig,
    pack_indices,
    unpack_indices,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_indices_round_trip(bits):
    dim = 19 if bits == 3 else 17
    indices = torch.randint(0, 1 << bits, (5, dim), dtype=torch.uint8)
    packed = pack_indices(indices, bits)
    assert torch.equal(unpack_indices(packed, dim, bits), indices)


def test_pack_indices_2p5_round_trip():
    indices = torch.empty(5, 256, dtype=torch.uint8)
    grouped = indices.reshape(5, 2, 128)
    grouped[..., :32] = torch.randint(0, 8, (5, 2, 32), dtype=torch.uint8)
    grouped[..., 32:] = torch.randint(0, 4, (5, 2, 96), dtype=torch.uint8)
    packed = pack_indices(indices, 2.5)
    assert torch.equal(unpack_indices(packed, 256, 2.5), indices)


def test_wht_rotation_round_trip():
    codec = TurboQuantDenseKVCodec(
        TurboQuantDenseKVConfig(latent_dim=512, rope_dim=64),
        torch.device("cpu"),
    )
    x = torch.randn(7, 512)
    torch.testing.assert_close(codec.inverse_rotate(codec.rotate(x)), x)


@pytest.mark.parametrize(
    ("preset", "slot_bytes"),
    [
        ("latent_k8", 640),
        ("latent_4bit_nc", 386),
        ("latent_k3_nc", 322),
        ("latent_2p5bit_nc", 274),
    ],
)
def test_slot_bytes(preset, slot_bytes):
    config = TurboQuantDenseKVConfig(latent_dim=512, rope_dim=64, preset=preset)
    assert config.slot_bytes == slot_bytes


def test_codec_round_trip_preserves_rope():
    codec = TurboQuantDenseKVCodec(
        TurboQuantDenseKVConfig(latent_dim=512, rope_dim=64, preset="latent_4bit_nc"),
        torch.device("cpu"),
    )
    latent = torch.randn(8, 1, 512, dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    restored = codec.decompress(codec.compress(latent, rope), torch.bfloat16)
    restored_latent = restored[..., :512].float()
    latent_cos = torch.nn.functional.cosine_similarity(
        restored_latent.reshape(8, 512),
        latent.float().reshape(8, 512),
        dim=-1,
    )
    assert torch.all(latent_cos > 0.95)
    assert torch.equal(restored[..., 512:], rope)


def test_codec_2p5_round_trip_preserves_rope():
    codec = TurboQuantDenseKVCodec(
        TurboQuantDenseKVConfig(
            latent_dim=512,
            rope_dim=64,
            preset="latent_2p5bit_nc",
        ),
        torch.device("cpu"),
    )
    latent = torch.randn(8, 1, 512, dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    compressed = codec.compress(latent, rope)
    restored = codec.decompress(compressed, torch.bfloat16)
    assert compressed.shape[-1] == 274
    assert torch.equal(restored[..., 512:], rope)
    assert torch.all(
        torch.nn.functional.cosine_similarity(
            restored[..., :512].float().reshape(8, 512),
            latent.float().reshape(8, 512),
            dim=-1,
        )
        > 0.85
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dequantize_selected_2p5_matches_codec():
    from sglang.jit_kernel.turboquant_dense_kv import dequantize_selected_2p5

    codec = TurboQuantDenseKVCodec(
        TurboQuantDenseKVConfig(
            latent_dim=512,
            rope_dim=64,
            preset="latent_2p5bit_nc",
        ),
        torch.device("cuda"),
    )
    latent = torch.randn(8, 1, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 64, device="cuda", dtype=torch.bfloat16)
    compressed = codec.compress(latent, rope)
    locs = torch.tensor([6, 1, 4, 1], device="cuda", dtype=torch.int64)
    out = torch.empty((locs.numel(), 1, 576), device="cuda", dtype=torch.bfloat16)

    dequantize_selected_2p5(
        compressed,
        locs,
        out,
        codec.centroids_high,
        codec.centroids_low,
        codec.signs1,
        codec.signs2,
    )
    torch.testing.assert_close(
        out,
        codec.decompress(compressed[locs], torch.bfloat16),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_turboquant_nsa_pool_preserves_indexer_buffer():
    from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool

    pool = TurboQuantNSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        turboquant_dense_kv_preset="latent_4bit_nc",
        turboquant_execution_mode="materialize",
    )
    before = pool.get_index_k_with_scale_buffer(0).clone()
    loc = torch.arange(4, device="cuda", dtype=torch.int64)
    latent = torch.randn(4, 1, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, latent, rope)
    restored_latent, restored_rope = pool.get_mla_kv_buffer(
        SimpleNamespace(layer_id=0),
        loc,
    )

    assert torch.equal(before, pool.get_index_k_with_scale_buffer(0))
    assert torch.equal(restored_rope, rope)
    assert torch.all(
        torch.nn.functional.cosine_similarity(
            restored_latent.float().reshape(4, 512),
            latent.float().reshape(4, 512),
            dim=-1,
        )
        > 0.95
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_turboquant_fused_decode_keeps_indexer_native():
    from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool

    pool = TurboQuantNSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
    )
    before = pool.get_index_k_with_scale_buffer(0).clone()
    loc = torch.arange(4, device="cuda", dtype=torch.int64)
    latent = torch.randn(4, 1, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, latent, rope)
    kv_cache, page_table = pool.get_turboquant_selected_kv_buffer(
        0,
        loc.reshape(1, -1).to(torch.int32),
    )

    assert pool._deq_buffer is None
    assert torch.equal(before, pool.get_index_k_with_scale_buffer(0))
    assert torch.equal(page_table, loc.reshape(1, -1).to(torch.int32))
    assert torch.equal(kv_cache[..., 512:], rope)
    assert torch.all(
        torch.nn.functional.cosine_similarity(
            kv_cache[..., :512].float().reshape(4, 512),
            latent.float().reshape(4, 512),
            dim=-1,
        )
        > 0.85
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_turboquant_fused_mla_decode_matches_dequantized_reference():
    from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool

    pool = TurboQuantNSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
    )
    loc = torch.arange(64, device="cuda", dtype=torch.int64)
    latent = torch.randn(64, 1, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(64, 1, 64, device="cuda", dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, latent, rope)

    page_table = torch.arange(64, device="cuda", dtype=torch.int32).reshape(2, 32)
    q_nope = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    sm_scale = 0.04

    out = pool.forward_turboquant_dense_mla_decode(
        0,
        q_nope,
        q_rope,
        page_table,
        sm_scale,
    )
    dequantized = pool.turboquant_codec.decompress(
        pool.kv_buffer[0][loc],
        torch.bfloat16,
    )
    ref_latent = dequantized[..., :512].squeeze(1).float()
    ref_rope = dequantized[..., 512:].squeeze(1).float()
    rows = page_table.long()
    scores = (
        torch.einsum("rhd,rkd->rhk", q_nope.float(), ref_latent[rows])
        + torch.einsum("rhe,rke->rhk", q_rope.float(), ref_rope[rows])
    ) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("rhk,rkd->rhd", weights, ref_latent[rows])

    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
