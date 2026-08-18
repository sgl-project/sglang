import importlib.util
import math
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

# CPU CI exercises the bit-exact codec/reference contract. The CUDA
# registration targets Blackwell because the writer, 200K chunk chain, and
# native FP4 dtype are part of the SM100 production path.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)


def _load_codec_module():
    codec_path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/attention/dsa/nvfp4_k_cache.py"
    )
    spec = importlib.util.spec_from_file_location("dsa_nvfp4_k_cache_test", codec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CODEC = _load_codec_module()
NVFP4_BYTES_PER_TOKEN = _CODEC.NVFP4_BYTES_PER_TOKEN
NVFP4_LATENT_DIM = _CODEC.NVFP4_LATENT_DIM
NVFP4_PACKED_LATENT_BYTES = _CODEC.NVFP4_PACKED_LATENT_BYTES
NVFP4_ROPE_BYTES = _CODEC.NVFP4_ROPE_BYTES
NVFP4_ROPE_DIM = _CODEC.NVFP4_ROPE_DIM
NVFP4_SCALE_BYTES = _CODEC.NVFP4_SCALE_BYTES
dequantize_nvfp4_k_cache_paged_reference = (
    _CODEC.dequantize_nvfp4_k_cache_paged_reference
)
dequantize_nvfp4_k_cache_paged = _CODEC.dequantize_nvfp4_k_cache_paged
quantize_nvfp4_k_cache_into = _CODEC.quantize_nvfp4_k_cache_into
quantize_nvfp4_k_cache_into_reference = _CODEC.quantize_nvfp4_k_cache_into_reference


def _make_inputs(num_tokens: int, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(42)
    latent = torch.randn(
        num_tokens,
        1,
        NVFP4_LATENT_DIM,
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    rope = torch.randn(
        num_tokens,
        1,
        NVFP4_ROPE_DIM,
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    return latent, rope


def _unpack_codes(row: torch.Tensor):
    packed = row[:NVFP4_PACKED_LATENT_BYTES]
    codes = torch.empty(NVFP4_LATENT_DIM, dtype=torch.uint8)
    codes[0::2] = packed & 0x0F
    codes[1::2] = packed >> 4
    return codes


def test_layout_and_dsa_byte_accounting():
    assert NVFP4_BYTES_PER_TOKEN == 256 + 32 + 128 == 416
    assert NVFP4_PACKED_LATENT_BYTES == 256
    assert NVFP4_SCALE_BYTES == 32
    assert NVFP4_ROPE_BYTES == 128

    # Existing DSA indexer storage is 128 E4M3 values + one FP32 scale.
    indexer_bytes = 128 + 4
    assert NVFP4_BYTES_PER_TOKEN + indexer_bytes == 548
    # TRTLLM-GEN stores the full 576-dimensional key as raw E4M3.
    assert 576 + indexer_bytes == 708
    assert math.isclose(708 / 548, 1.291970802919708, rel_tol=0, abs_tol=1e-15)
    # Open FlashMLA stores 512 E4M3 latent bytes, four FP32 scales, and
    # 64 BF16 RoPE values, for a 656-byte main row.
    assert 656 + indexer_bytes == 788
    assert math.isclose(788 / 548, 1.437956204379562, rel_tol=0, abs_tol=1e-15)


@pytest.mark.parametrize("global_scale", [0.03125, 0.5, 1.0, 1.375])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reference_roundtrip_uses_stored_bytes(global_scale, input_dtype):
    latent, rope = _make_inputs(5, input_dtype)
    rows = torch.zeros(8, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
    loc = torch.tensor([6, 1, 4, 0, 3], dtype=torch.int64)
    quantize_nvfp4_k_cache_into_reference(latent, rope, rows, loc, global_scale)

    decoded = dequantize_nvfp4_k_cache_paged_reference(
        rows, loc.to(torch.int32), global_scale, torch.float32
    )
    stored_rope = rows[loc, 0, -NVFP4_ROPE_BYTES:].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(
        decoded[..., NVFP4_LATENT_DIM:].squeeze(1),
        stored_rope.float(),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(decoded).all()


def test_midpoint_rne_signed_zero_and_scale_before_code():
    latent = torch.zeros(1, 1, NVFP4_LATENT_DIM, dtype=torch.float32)
    # amax=6 makes the rounded block scale exactly 1 when global_scale=1.
    latent[0, 0, :9] = torch.tensor([-0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    rope = torch.zeros(1, 1, NVFP4_ROPE_DIM)
    rows = torch.zeros(1, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
    quantize_nvfp4_k_cache_into_reference(
        latent, rope, rows, torch.tensor([0], dtype=torch.int32), 1.0
    )
    codes = _unpack_codes(rows[0, 0])
    assert codes[:9].tolist() == [8, 0, 2, 2, 4, 4, 6, 6, 7]
    scale = rows[0, 0, 256:288].contiguous().view(torch.float8_e4m3fn)
    assert scale[0].float().item() == 1.0


def test_zero_block_nonfinite_policy_and_invalid_locations():
    latent = torch.zeros(4, 1, NVFP4_LATENT_DIM, dtype=torch.float32)
    rope = torch.zeros(4, 1, NVFP4_ROPE_DIM, dtype=torch.float32)
    latent[1, 0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
    rope[1, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    rows = torch.full((4, 1, NVFP4_BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8)
    before = rows.clone()
    loc = torch.tensor([0, 2, -1, 4], dtype=torch.int64)
    quantize_nvfp4_k_cache_into_reference(latent, rope, rows, loc, 0.5)

    # Invalid destinations are screened before access and do not mutate rows.
    assert torch.equal(rows[1], before[1])
    assert torch.equal(rows[3], before[3])
    assert _unpack_codes(rows[0, 0]).eq(0).all()
    decoded = dequantize_nvfp4_k_cache_paged_reference(
        rows, torch.tensor([2, -1, 4]), 0.5, torch.float32
    )
    assert torch.isfinite(decoded).all()
    assert decoded[1:].eq(0).all()
    assert decoded[0, 0, NVFP4_LATENT_DIM : NVFP4_LATENT_DIM + 3].tolist() == [
        0.0,
        torch.finfo(torch.bfloat16).max,
        torch.finfo(torch.bfloat16).min,
    ]


@pytest.mark.parametrize("num_tokens", [0, 1, 31, 32, 33, 63, 64, 65, 1024])
def test_cpu_reference_token_boundaries(num_tokens):
    latent, rope = _make_inputs(num_tokens)
    rows = torch.zeros(max(1, num_tokens), 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
    loc = torch.arange(num_tokens, dtype=torch.int32)
    quantize_nvfp4_k_cache_into_reference(latent, rope, rows, loc, 1.375)
    decoded = dequantize_nvfp4_k_cache_paged_reference(rows, loc, 1.375)
    assert decoded.shape == (num_tokens, 1, 576)


def test_duplicate_location_with_identical_value_is_deterministic():
    latent, rope = _make_inputs(1)
    latent = latent.expand(2, -1, -1).contiguous()
    rope = rope.expand(2, -1, -1).contiguous()
    rows = torch.zeros(2, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
    quantize_nvfp4_k_cache_into_reference(latent, rope, rows, torch.tensor([1, 1]), 1.0)
    expected = torch.zeros_like(rows)
    quantize_nvfp4_k_cache_into_reference(
        latent[:1], rope[:1], expected, torch.tensor([1]), 1.0
    )
    assert torch.equal(rows, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 31, 32, 33, 64, 65, 1024])
def test_cuda_writer_is_bit_exact_to_cpu_reference(num_tokens):
    latent, rope = _make_inputs(num_tokens, torch.bfloat16, "cuda")
    loc = torch.randperm(num_tokens, device="cuda", dtype=torch.int64)
    cuda_rows = torch.zeros(
        num_tokens, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda"
    )
    quantize_nvfp4_k_cache_into(latent, rope, cuda_rows, loc, 1.375)
    torch.cuda.synchronize()

    reference = torch.zeros_like(cuda_rows, device="cpu")
    quantize_nvfp4_k_cache_into_reference(
        latent.cpu(), rope.cpu(), reference, loc.cpu(), 1.375
    )
    assert torch.equal(cuda_rows.cpu(), reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cuda_paged_dequant_matches_stored_byte_reference(output_dtype):
    latent, rope = _make_inputs(65, torch.bfloat16, "cuda")
    rows = torch.zeros(65, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda")
    locations = torch.arange(65, dtype=torch.int64, device="cuda")
    quantize_nvfp4_k_cache_into(latent, rope, rows, locations, 1.375)
    indices = torch.tensor([64, 1, 63, 1, -1, 65, 0], dtype=torch.int32, device="cuda")
    actual = dequantize_nvfp4_k_cache_paged(rows, indices, 1.375, dtype=output_dtype)
    expected = dequantize_nvfp4_k_cache_paged_reference(
        rows, indices, 1.375, dtype=output_dtype
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "chunk_size,expected_full_chunks",
    [(8192, 24), (16384, 12), (32768, 6)],
)
def test_cuda_chunked_200k_chain_matches_one_shot(chunk_size, expected_full_chunks):
    """Chunk boundaries must not change the final 200K physical cache bytes."""

    num_tokens = 200_000
    tail = num_tokens % chunk_size
    assert tail == 3392
    assert num_tokens // chunk_size == expected_full_chunks

    generator = torch.Generator(device="cuda").manual_seed(20260810 + chunk_size)
    latent = torch.randn(
        num_tokens,
        1,
        NVFP4_LATENT_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope = torch.randn(
        num_tokens,
        1,
        NVFP4_ROPE_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    locations = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    one_shot = torch.zeros(
        num_tokens,
        1,
        NVFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device="cuda",
    )
    chunked = torch.zeros_like(one_shot)

    quantize_nvfp4_k_cache_into(latent, rope, one_shot, locations, global_scale=1.375)
    starts = list(range(0, num_tokens, chunk_size))
    assert len(starts) == expected_full_chunks + 1
    for start in starts:
        end = min(start + chunk_size, num_tokens)
        quantize_nvfp4_k_cache_into(
            latent[start:end],
            rope[start:end],
            chunked,
            locations[start:end],
            global_scale=1.375,
        )
    torch.cuda.synchronize()
    assert torch.equal(chunked, one_shot)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_global_scale_rejected(scale):
    latent, rope = _make_inputs(1)
    rows = torch.zeros(1, 1, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
    with pytest.raises(ValueError):
        quantize_nvfp4_k_cache_into_reference(
            latent, rope, rows, torch.tensor([0]), scale
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
