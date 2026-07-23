from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    NVFP4_BLOCK_SIZE,
    NVFP4_BYTES_PER_TOKEN,
    NVFP4_LATENT_DIM,
    NVFP4_PACKED_LATENT_BYTES,
    NVFP4_ROPE_BYTES,
    NVFP4_ROPE_DIM,
    NVFP4_SCALE_BYTES,
    _decode_e2m1_torch,
    _dequantize_nvfp4_k_cache_paged_reference,
    _quantize_nvfp4_k_cache_into_reference,
    dequantize_nvfp4_k_cache_paged,
    quantize_nvfp4_k_cache_into,
)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes


def _exact_quant_inputs(
    num_tokens: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = (
        torch.arange(num_tokens * NVFP4_LATENT_DIM, device=device)
        .reshape(num_tokens, NVFP4_LATENT_DIM)
        .remainder(16)
        .to(torch.uint8)
    )
    k_nope = _decode_e2m1_torch(codes).to(dtype)
    if num_tokens:
        # With global_scale=.5 this makes every block scale exactly 2 and its
        # E2M1 denominator exactly 1, avoiding threshold ambiguity while still
        # exercising every nonzero code and sign.
        k_nope.view(num_tokens, -1, NVFP4_BLOCK_SIZE)[:, :, 0] = 6.0
    k_rope = (
        torch.arange(num_tokens * NVFP4_ROPE_DIM, device=device)
        .reshape(num_tokens, NVFP4_ROPE_DIM)
        .remainder(97)
        .sub(48)
        .div(16)
        .to(dtype)
    )
    return k_nope, k_rope


def _cuda_nvfp4_available() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


def test_dsa_nvfp4_layout_constants() -> None:
    assert NVFP4_BLOCK_SIZE == 16
    assert NVFP4_LATENT_DIM == 512
    assert NVFP4_ROPE_DIM == 64
    assert NVFP4_PACKED_LATENT_BYTES == 256
    assert NVFP4_SCALE_BYTES == 32
    assert NVFP4_ROPE_BYTES == 128
    assert NVFP4_BYTES_PER_TOKEN == 416
    assert (
        NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES + NVFP4_ROPE_BYTES
        == NVFP4_BYTES_PER_TOKEN
    )


def test_dsa_nvfp4_cpu_scatter_rne_and_minus_one_gather() -> None:
    # Every block has max(abs(x)) == 6 and global_scale == 1, so its rounded
    # E4M3 scale is exactly 1.  The first block probes every E2M1 midpoint.
    midpoints = torch.tensor(
        [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            6.0,
            -6.0,
        ],
        dtype=torch.bfloat16,
    )
    expected_codes = torch.tensor(
        [0, 2, 2, 4, 4, 6, 6, 8, 10, 10, 12, 12, 14, 14, 7, 15],
        dtype=torch.uint8,
    )
    k_nope = torch.zeros((1, NVFP4_LATENT_DIM), dtype=torch.bfloat16)
    k_nope.view(-1, NVFP4_BLOCK_SIZE)[:, -1] = 6.0
    k_nope[0, :NVFP4_BLOCK_SIZE] = midpoints
    k_rope = torch.arange(NVFP4_ROPE_DIM, dtype=torch.bfloat16).unsqueeze(0)
    cache = torch.full((4, 1, NVFP4_BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8)

    quantize_nvfp4_k_cache_into(
        k_nope, k_rope, cache, torch.tensor([2]), global_scale=1.0
    )

    # Direct scatter only changes row 2.
    assert torch.all(cache[0] == 0xA5)
    assert torch.all(cache[1] == 0xA5)
    assert torch.all(cache[3] == 0xA5)
    row = cache.view(-1, NVFP4_BYTES_PER_TOKEN)[2]
    actual_codes = _unpack(row[:NVFP4_PACKED_LATENT_BYTES])
    torch.testing.assert_close(actual_codes[:16], expected_codes, rtol=0, atol=0)
    scales = row[
        NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + NVFP4_SCALE_BYTES
    ].view(torch.float8_e4m3fn)
    torch.testing.assert_close(
        scales.float(), torch.ones(NVFP4_SCALE_BYTES), rtol=0, atol=0
    )
    torch.testing.assert_close(
        row[-NVFP4_ROPE_BYTES:].view(torch.bfloat16), k_rope[0], rtol=0, atol=0
    )

    gathered = dequantize_nvfp4_k_cache_paged(
        cache, torch.tensor([2, -1, 2, 4], dtype=torch.int32), global_scale=1.0
    )
    assert gathered.shape == (4, 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM)
    assert torch.count_nonzero(gathered[1]) == 0
    assert torch.count_nonzero(gathered[3]) == 0
    torch.testing.assert_close(
        gathered[0, 0, NVFP4_LATENT_DIM:], k_rope[0], rtol=0, atol=0
    )
    torch.testing.assert_close(gathered[0], gathered[2], rtol=0, atol=0)


def test_dsa_nvfp4_rounds_e4m3_scale_before_e2m1() -> None:
    k_nope = torch.zeros((1, NVFP4_LATENT_DIM), dtype=torch.bfloat16)
    # 6.375 / 6 = 1.0625, exactly halfway between E4M3 values 1 and
    # 1.125. RNE selects 1.0. Therefore 5.1875 normalizes above the E2M1
    # midpoint 5 and must become code 7 (6.0). Using the unrounded scale
    # would incorrectly produce code 6 (4.0).
    k_nope[0, 0] = 6.375
    k_nope[0, 1] = 5.1875
    k_rope = torch.zeros((1, NVFP4_ROPE_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8)

    quantize_nvfp4_k_cache_into(
        k_nope, k_rope, cache, torch.tensor([0]), global_scale=1.0
    )

    row = cache[0]
    scale = (
        row[NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + 1]
        .view(torch.float8_e4m3fn)
        .float()[0]
    )
    torch.testing.assert_close(scale, torch.tensor(1.0), rtol=0, atol=0)
    codes = _unpack(row[:NVFP4_PACKED_LATENT_BYTES])
    torch.testing.assert_close(
        codes[:2], torch.tensor([7, 7], dtype=torch.uint8), rtol=0, atol=0
    )


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_cuda_scaled_midpoints_use_exact_rne() -> None:
    """A reciprocal multiply must not move exact E2M1 midpoint ties."""

    device = torch.device("cuda")
    k_nope = torch.zeros(
        (1, NVFP4_LATENT_DIM), dtype=torch.bfloat16, device=device
    )
    # max/6/global_scale = .9375 (exact E4M3), hence denominator=.46875.
    # The following values normalize to exact ties 2.5 and 1.25.  The even
    # codes are 4 and 2 respectively; their negative counterparts retain the
    # sign bit, and negative zero must remain code 8.
    k_nope[0, :16] = torch.tensor(
        [
            2.8125,
            0.1171875,
            0.3515625,
            0.5859375,
            0.8203125,
            1.171875,
            1.640625,
            2.34375,
            -0.1171875,
            -0.3515625,
            -0.5859375,
            -0.8203125,
            -1.171875,
            -1.640625,
            -2.34375,
            -0.0,
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    # A zero-scale block canonicalizes every value, including -0, to +0.
    k_nope[0, NVFP4_BLOCK_SIZE] = -0.0
    k_rope = torch.zeros(
        (1, NVFP4_ROPE_DIM), dtype=torch.bfloat16, device=device
    )
    cache = torch.zeros(
        (1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )

    quantize_nvfp4_k_cache_into(
        k_nope, k_rope, cache, torch.tensor([0], device=device), global_scale=0.5
    )

    row = cache[0]
    scale = (
        row[NVFP4_PACKED_LATENT_BYTES : NVFP4_PACKED_LATENT_BYTES + 1]
        .view(torch.float8_e4m3fn)
        .float()[0]
    )
    torch.testing.assert_close(
        scale, torch.tensor(0.9375, device=device), rtol=0, atol=0
    )
    codes = _unpack(row[:NVFP4_PACKED_LATENT_BYTES])
    torch.testing.assert_close(
        codes[:16],
        torch.tensor(
            [7, 0, 2, 2, 4, 4, 6, 6, 8, 10, 10, 12, 12, 14, 14, 8],
            dtype=torch.uint8,
            device=device,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        codes[NVFP4_BLOCK_SIZE : 2 * NVFP4_BLOCK_SIZE],
        torch.zeros(NVFP4_BLOCK_SIZE, dtype=torch.uint8, device=device),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "global_scale",
    [0.0, -1.0, float("nan"), float("inf"), torch.tensor([0.0])],
)
def test_dsa_nvfp4_rejects_invalid_cpu_global_scale(global_scale) -> None:
    k_nope = torch.zeros((1, NVFP4_LATENT_DIM), dtype=torch.bfloat16)
    k_rope = torch.zeros((1, NVFP4_ROPE_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8)
    with pytest.raises(ValueError, match="global_scale"):
        quantize_nvfp4_k_cache_into(
            k_nope, k_rope, cache, torch.tensor([0]), global_scale
        )


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("global_scale_value", [0.03125, 0.5, 1.375])
def test_dsa_nvfp4_triton_matches_pytorch_reference(
    index_dtype: torch.dtype, global_scale_value: float
) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_inputs = 6
    capacity = 12
    k_nope = (torch.randn((num_inputs, 1, NVFP4_LATENT_DIM), device=device) * 1.75).to(
        torch.bfloat16
    )
    # Exercise an exact E4M3 midpoint on the Triton path as well: RNE must
    # round the block scale 1.0625 to 1.0 before deriving the E2M1 codes.
    k_nope[0, 0, :NVFP4_BLOCK_SIZE] = 0
    k_nope[0, 0, 0] = 6.375
    k_nope[0, 0, 1] = 5.1875
    k_rope = torch.randn(
        (num_inputs, NVFP4_ROPE_DIM), dtype=torch.bfloat16, device=device
    )
    # A padded -1 entry must not write.  The other indices are deliberately
    # non-monotonic to exercise direct scatter addressing.
    loc = torch.tensor([5, 1, -1, 9, 3, capacity], dtype=index_dtype, device=device)
    global_scale = torch.tensor(
        [global_scale_value], dtype=torch.float32, device=device
    )
    actual = torch.full(
        (capacity, 1, NVFP4_BYTES_PER_TOKEN),
        0x5A,
        dtype=torch.uint8,
        device=device,
    )
    expected = actual.clone()

    quantize_nvfp4_k_cache_into(k_nope, k_rope, actual, loc, global_scale)
    _quantize_nvfp4_k_cache_into_reference(k_nope, k_rope, expected, loc, global_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    page_indices = torch.tensor(
        [9, -1, 1, 7, 5, 3, -1, capacity],
        dtype=index_dtype,
        device=device,
    )
    actual_dequant = dequantize_nvfp4_k_cache_paged(actual, page_indices, global_scale)
    expected_dequant = _dequantize_nvfp4_k_cache_paged_reference(
        expected, page_indices, global_scale
    )
    torch.testing.assert_close(actual_dequant, expected_dequant, rtol=0, atol=0)
    assert torch.count_nonzero(actual_dequant[1]) == 0
    assert torch.count_nonzero(actual_dequant[-2]) == 0
    assert torch.count_nonzero(actual_dequant[-1]) == 0


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_dequant_output_dtype() -> None:
    device = torch.device("cuda")
    k_nope = torch.randn((2, NVFP4_LATENT_DIM), dtype=torch.bfloat16, device=device)
    k_rope = torch.randn((2, NVFP4_ROPE_DIM), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((2, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=device)
    loc = torch.arange(2, dtype=torch.int32, device=device)
    quantize_nvfp4_k_cache_into(k_nope, k_rope, cache, loc, 0.125)

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        output = dequantize_nvfp4_k_cache_paged(cache, loc, 0.125, dtype=dtype)
        assert output.dtype == dtype
        assert output.shape == (2, 1, NVFP4_LATENT_DIM + NVFP4_ROPE_DIM)


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_codec_cuda_graph_replay_with_invalid_indices() -> None:
    device = torch.device("cuda")
    capacity = 8
    num_inputs = 4
    static_nope = torch.zeros(
        (num_inputs, NVFP4_LATENT_DIM), dtype=torch.bfloat16, device=device
    )
    static_rope = torch.zeros(
        (num_inputs, NVFP4_ROPE_DIM), dtype=torch.bfloat16, device=device
    )
    replay_nope = torch.randn_like(static_nope) * 0.25
    replay_rope = torch.randn_like(static_rope) * 0.25
    cache = torch.zeros(
        (capacity, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    loc = torch.tensor([0, 2, -1, capacity], dtype=torch.int32, device=device)
    page_indices = torch.tensor([2, -1, capacity, 0], dtype=torch.int32, device=device)
    global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        quantize_nvfp4_k_cache_into(static_nope, static_rope, cache, loc, global_scale)
        dequantize_nvfp4_k_cache_paged(cache, page_indices, global_scale)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_nvfp4_k_cache_into(static_nope, static_rope, cache, loc, global_scale)
        graph_output = dequantize_nvfp4_k_cache_paged(cache, page_indices, global_scale)

    captured_output = graph_output.clone()
    static_nope.copy_(replay_nope)
    static_rope.copy_(replay_rope)
    graph.replay()
    torch.cuda.synchronize()

    expected_cache = torch.zeros_like(cache)
    quantize_nvfp4_k_cache_into(
        replay_nope, replay_rope, expected_cache, loc, global_scale
    )
    expected_output = dequantize_nvfp4_k_cache_paged(
        expected_cache, page_indices, global_scale
    )
    assert not torch.equal(graph_output, captured_output)
    torch.testing.assert_close(graph_output, expected_output, rtol=0, atol=0)
    assert torch.count_nonzero(graph_output[1]) == 0
    assert torch.count_nonzero(graph_output[2]) == 0


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_dequant_large_shuffled_four_dimensional_pages() -> None:
    """Cover the single-CTA geometry around warp and page boundaries."""

    torch.manual_seed(29)
    page_size = 64
    num_pages = 4
    capacity = num_pages * page_size
    global_scale = 0.5
    k_nope = torch.randn(
        (capacity, NVFP4_LATENT_DIM), dtype=torch.bfloat16
    )
    k_rope = torch.randn((capacity, NVFP4_ROPE_DIM), dtype=torch.bfloat16)
    cache_cpu = torch.zeros(
        (num_pages, page_size, 1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8
    )
    loc = torch.arange(capacity, dtype=torch.int32)
    _quantize_nvfp4_k_cache_into_reference(
        k_nope, k_rope, cache_cpu, loc, global_scale
    )
    cache = cache_cpu.cuda()

    shuffled = torch.randperm(capacity, dtype=torch.int32)
    pattern = torch.cat(
        (
            shuffled,
            torch.tensor([-1, capacity, 7, 7], dtype=torch.int32),
        )
    )
    for num_outputs in (0, 1, 31, 32, 33, 63, 64, 65, 1024):
        repeats = (num_outputs + pattern.numel() - 1) // pattern.numel()
        indices = pattern.repeat(repeats)[:num_outputs].cuda()
        actual = dequantize_nvfp4_k_cache_paged(cache, indices, global_scale)
        expected = _dequantize_nvfp4_k_cache_paged_reference(
            cache, indices, global_scale
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    all_invalid = torch.tensor(
        [-1, capacity, capacity + 17] * 23, dtype=torch.int64, device="cuda"
    )
    invalid_output = dequantize_nvfp4_k_cache_paged(
        cache, all_invalid, global_scale
    )
    assert torch.count_nonzero(invalid_output) == 0


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_dequant_graph_replay_updates_indices_and_scale() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    capacity = 96
    num_outputs = 65
    k_nope = torch.randn(
        (capacity, NVFP4_LATENT_DIM), dtype=torch.bfloat16, device=device
    )
    k_rope = torch.randn(
        (capacity, NVFP4_ROPE_DIM), dtype=torch.bfloat16, device=device
    )
    cache = torch.zeros(
        (capacity, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    loc = torch.arange(capacity, dtype=torch.int32, device=device)
    quant_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    quantize_nvfp4_k_cache_into(k_nope, k_rope, cache, loc, quant_scale)

    static_indices = torch.arange(num_outputs, dtype=torch.int32, device=device)
    static_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        dequantize_nvfp4_k_cache_paged(cache, static_indices, static_scale)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = dequantize_nvfp4_k_cache_paged(
            cache, static_indices, static_scale
        )

    output_ptr = graph_output.data_ptr()
    captured_output = graph_output.clone()
    replay_indices = torch.arange(
        capacity - 1, capacity - 1 - num_outputs, -1, dtype=torch.int32, device=device
    )
    replay_indices[3] = -1
    replay_indices[17] = capacity
    static_indices.copy_(replay_indices)
    static_scale.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()

    expected = _dequantize_nvfp4_k_cache_paged_reference(
        cache, replay_indices, static_scale
    )
    assert graph_output.data_ptr() == output_ptr
    assert not torch.equal(graph_output, captured_output)
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
    assert torch.count_nonzero(graph_output[3]) == 0
    assert torch.count_nonzero(graph_output[17]) == 0


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_quant_grouped_launch_boundaries_and_four_dimensional_pages() -> None:
    device = torch.device("cuda")
    global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    for num_inputs in (0, 1, 31, 32, 33, 63, 64, 65, 1024):
        num_pages = max(1, (num_inputs + 79) // 64)
        capacity = num_pages * 64
        k_nope, k_rope = _exact_quant_inputs(
            num_inputs, torch.bfloat16, device
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(41 + num_inputs)
        loc = torch.randperm(capacity, generator=generator, dtype=torch.int32)[
            :num_inputs
        ].to(device)
        if num_inputs >= 3:
            loc[-2] = -1
            loc[-1] = capacity
        actual = torch.full(
            (num_pages, 64, 1, NVFP4_BYTES_PER_TOKEN),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        expected = actual.clone()

        quantize_nvfp4_k_cache_into(
            k_nope, k_rope, actual, loc, global_scale
        )
        _quantize_nvfp4_k_cache_into_reference(
            k_nope, k_rope, expected, loc, global_scale
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dsa_nvfp4_quant_grouped_supports_input_dtype(
    input_dtype: torch.dtype,
) -> None:
    device = torch.device("cuda")
    num_inputs = 65
    capacity = 96
    k_nope, k_rope = _exact_quant_inputs(num_inputs, input_dtype, device)
    loc = torch.arange(num_inputs, dtype=torch.int64, device=device)
    global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    actual = torch.full(
        (capacity, NVFP4_BYTES_PER_TOKEN), 0x5A, dtype=torch.uint8, device=device
    )
    expected = actual.clone()

    quantize_nvfp4_k_cache_into(k_nope, k_rope, actual, loc, global_scale)
    _quantize_nvfp4_k_cache_into_reference(
        k_nope, k_rope, expected, loc, global_scale
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _cuda_nvfp4_available(), reason="requires an SM90+ CUDA device")
def test_dsa_nvfp4_quant_graph_replay_updates_loc_and_global_scale() -> None:
    device = torch.device("cuda")
    num_inputs = 65
    capacity = 128
    static_nope, static_rope = _exact_quant_inputs(
        num_inputs, torch.bfloat16, device
    )
    replay_nope = static_nope.roll(7, dims=1)
    replay_rope = static_rope.roll(11, dims=1)
    static_loc = torch.arange(num_inputs, dtype=torch.int32, device=device)
    replay_loc = torch.arange(
        capacity - 1, capacity - 1 - num_inputs, -1, dtype=torch.int32, device=device
    )
    replay_loc[5] = -1
    replay_loc[29] = capacity
    static_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    cache = torch.full(
        (capacity, NVFP4_BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8, device=device
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        quantize_nvfp4_k_cache_into(
            static_nope, static_rope, cache, static_loc, static_scale
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_nvfp4_k_cache_into(
            static_nope, static_rope, cache, static_loc, static_scale
        )

    cache_ptr = cache.data_ptr()
    captured_cache = cache.clone()
    cache.fill_(0xA5)
    static_nope.copy_(replay_nope)
    static_rope.copy_(replay_rope)
    static_loc.copy_(replay_loc)
    static_scale.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.full_like(cache, 0xA5)
    _quantize_nvfp4_k_cache_into_reference(
        replay_nope, replay_rope, expected, replay_loc, static_scale
    )
    assert cache.data_ptr() == cache_ptr
    assert not torch.equal(cache, captured_cache)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
