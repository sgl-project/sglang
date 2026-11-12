"""Routed-expert, prefill MoE, NVFP4, and expert-parallel kernels."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.inkling2.nvfp4 import (
    expert_matmul,
    nvfp4_quantize,
    swizzle_weight_scales,
)
from sglang.kernels.ops.moe.inkling_moe import (
    compute_expert_block_metadata,
    compute_grouped_gemm_metadata,
)
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.inkling2_test_utils import precision_group

register_cuda_ci(est_time=200, stage="base-c", runner_config="4-gpu-gb300")

sm100_or_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires an SM100 or SM103 Blackwell GPU",
)


EXPERTS, LOCAL, TOP_K = 384, 96, 8


def _route(tokens, skewed=False, seed=0):
    from sglang.kernels.ops.inkling2.route import route_metadata

    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(EXPERTS, device="cuda", generator=g)[:TOP_K]
            for _ in range(tokens)
        ]
    ).int()
    if skewed:
        ids = (ids % 40).int()
    weights = torch.rand(tokens, TOP_K, device="cuda", generator=g)
    return route_metadata(ids.flatten(), weights.flatten(), LOCAL, LOCAL, EXPERTS)


def _scale_rows(x_scales, offsets, rows, width):
    """Unswizzle the grouped-GEMM activation scales of rows [offsets[0], offsets[-1])."""
    scale_k = (width + 63) // 64 * 4
    offsets = offsets.long()
    rows = rows.long()
    expert = torch.searchsorted(offsets, rows, right=True) - 1
    local = rows - offsets[expert]
    base = (offsets[expert] - offsets[0] + expert * 127) // 128 * 128 * scale_k
    cols = torch.arange(width // 16, device=rows.device)
    index = (
        base[:, None]
        + ((local[:, None] // 128) * (scale_k // 4) + cols // 4) * 512
        + (local[:, None] % 32) * 16
        + ((local[:, None] // 32) % 4) * 4
        + cols % 4
    )
    return x_scales[index]


def _x_scales(rows, width):
    scale_k = (width + 63) // 64 * 4
    size = (rows + 127 * LOCAL + 127) // 128 * 128 * scale_k
    return torch.full((size,), 0x7F, device="cuda", dtype=torch.uint8)


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


@sm100_or_sm103
@pytest.mark.parametrize("tokens", [1, 8, 64, 512])
def test_routed_quantize_matches_token_quantize(tokens):
    from sglang.kernels.ops.inkling2.nvfp4 import (
        nvfp4_quantize,
        routed_quant_module,
    )

    width = 4608
    _, dst2src, _, counts, offsets, _, _, _ = _route(tokens, seed=tokens)
    x = torch.randn(tokens, width, device="cuda").bfloat16() * 3
    rows = dst2src.numel()
    values = torch.empty(rows, width // 2, device="cuda", dtype=torch.uint8)
    x_scales = _x_scales(rows, width)
    amax = torch.empty(rows, device="cuda")
    routed_quant_module(width).quantize(
        x, dst2src, values, x_scales, amax, offsets, TOP_K, _sm_count()
    )
    ref_values, ref_scales, ref_amax = nvfp4_quantize(x)
    local = torch.arange(offsets[0].item(), offsets[-1].item(), device="cuda")
    source = dst2src[local].long() // TOP_K
    assert torch.equal(values[local], ref_values[source])
    assert torch.equal(amax[local], ref_amax[source])
    assert torch.equal(
        _scale_rows(x_scales, offsets, local, width),
        ref_scales[source].view(torch.uint8),
    )
    _assert_zero_tails(x_scales, offsets, counts, width)


def _assert_zero_tails(x_scales, offsets, counts, width):
    for e in torch.nonzero(counts % 128).flatten().tolist():
        start, count = offsets[e].item(), counts[e].item()
        padded = (count + 127) // 128 * 128
        tail = torch.arange(start + count, start + padded, device="cuda")
        fake = offsets.clone()
        fake[e + 1 :] += padded - count
        assert not _scale_rows(x_scales, fake, tail, width).any()


@sm100_or_sm103
@pytest.mark.parametrize(
    "tokens,skewed",
    [(1, False), (64, False), (129, True), (256, False), (512, False), (512, True)],
)
def test_route_metadata_cuda_matches_triton(monkeypatch, tokens, skewed):
    from sglang.kernels.ops.inkling2 import route

    monkeypatch.setattr(route, "ROUTE_CUDA_MIN_IDS", 1 << 30)
    expected = _route(tokens, skewed, seed=tokens)
    monkeypatch.setattr(route, "ROUTE_CUDA_MIN_IDS", 0)
    actual = _route(tokens, skewed, seed=tokens)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


@sm100_or_sm103
@pytest.mark.parametrize("tokens", [8, 64, 512])
def test_routed_partial_matches_post_reorder(tokens):
    from sglang.kernels.ops.inkling2.moe import (
        routed_partial,
        routed_rows,
        unweighted_post_reorder,
    )

    width = 4608
    src2dst, _, _, _, offsets, _, _, _ = _route(tokens, seed=tokens)
    rows = src2dst.numel()
    down = torch.randn(rows, width, device="cuda") * 7
    scales = (
        torch.rand(rows, device="cuda") + 0.5,
        torch.rand(LOCAL, device="cuda") + 0.5,
        torch.tensor([3.0], device="cuda"),
    )
    expected = unweighted_post_reorder(
        routed_rows(down, offsets, out_dtype=torch.bfloat16, nvfp4_scales=scales),
        src2dst,
        TOP_K,
    )
    actual = routed_partial(down, offsets, src2dst, TOP_K, nvfp4_scales=scales)
    assert torch.equal(actual.bfloat16(), expected)


@sm100_or_sm103
@pytest.mark.parametrize(
    "tokens,skewed,width",
    [(1, False, 4608), (16, False, 4608), (256, False, 4608), (512, True, 4608)]
    + [(64, False, 2304), (512, True, 2304)],
)
def test_cute_routed_gemm_matches_cutlass(tokens, skewed, width):
    """The CuTe DSL routed NVFP4 GEMM equals the CUTLASS swap-AB grouped GEMM over
    reordered rows; skewed routing fills multi-block experts and partial subtiles."""
    from sglang.kernels.ops.inkling2 import nvfp4
    from sglang.kernels.ops.inkling2.route import route_metadata
    from sglang.kernels.ops.moe.inkling_moe import pre_reorder

    g = torch.Generator(device="cuda").manual_seed(tokens + width)
    ids = torch.stack(
        [
            torch.randperm(EXPERTS, device="cuda", generator=g)[:TOP_K]
            for _ in range(tokens)
        ]
    ).int()
    if skewed:
        ids = (LOCAL + ids % 16).int()
    weights = torch.rand(tokens, TOP_K, device="cuda", generator=g)
    src2dst, dst2src, _, counts, offsets, _, schedule, _ = route_metadata(
        ids.flatten(), weights.flatten(), LOCAL, LOCAL, EXPERTS
    )
    assert dst2src.numel() <= nvfp4.SWAP_AB_MAX_ROWS
    n = 2 * 2304 if width == 4608 else 4608
    x = torch.randn(tokens, width, device="cuda", generator=g).bfloat16()
    weight = torch.randint(
        0, 256, (LOCAL, n, width // 2), device="cuda", generator=g
    ).to(torch.uint8)
    scales = nvfp4.swizzle_weight_scales(
        (torch.rand(LOCAL, n, width // 16, device="cuda", generator=g) * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    )
    acc, x_amax = nvfp4.routed_expert_matmul(
        x, dst2src, TOP_K, weight, scales, counts, offsets, schedule
    )
    expected_acc, expected_amax = nvfp4.expert_matmul(
        pre_reorder(x, src2dst, TOP_K),
        weight,
        scales,
        torch.ones(LOCAL, device="cuda"),
        torch.ones(1, device="cuda"),
        metadata=(counts, offsets, schedule),
        raw=True,
    )
    lo, hi = int(offsets[0]), int(offsets[-1])
    assert hi > lo
    assert torch.equal(
        acc[lo:hi].view(torch.int32), expected_acc[lo:hi].view(torch.int32)
    )
    assert torch.equal(x_amax[lo:hi], expected_amax[lo:hi])


blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires a Blackwell GPU",
)


try:
    # The server imports this, which points Triton at the system ptxas; kernel
    # numerics are only comparable under the same ptxas.
    import flashinfer.triton  # noqa: F401
except ImportError:
    pass


E, TOPK, H, I = 96, 8, 4608, 2304


def _routing(tokens, rank):
    torch.manual_seed(tokens + rank)
    ids = torch.stack(
        [torch.randperm(4 * E, device="cuda")[:TOPK] for _ in range(tokens)]
    ).to(torch.int16)
    sorted_ids, dst2src = torch.sort(ids.flatten(), stable=True)
    dst2src = dst2src.int()
    counts, offsets, _, _ = compute_grouped_gemm_metadata(
        sorted_ids, 4 * E, block_size_m=128
    )
    counts = counts[rank * E : (rank + 1) * E]
    offsets = offsets[rank * E : (rank + 1) * E + 1]
    _, schedule = compute_expert_block_metadata(counts, ids.numel(), block_size_m=128)
    gammas = torch.rand(tokens * TOPK, device="cuda")[dst2src]
    return dst2src, counts, offsets, schedule, gammas


@blackwell
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 3),
    reason="NVFP4 experts require SM103",
)
@pytest.mark.parametrize("tokens,rank", [(4096, 1), (16384, 2)])
def test_nvfp4_prefill_gemm(tokens, rank):
    """The routed prefill GEMM (gathered quantization, 128x192 tiles) equals the
    grouped path over reordered rows (128x128 tiles) beyond the swap-AB size."""
    from sglang.kernels.ops.inkling2 import nvfp4
    from sglang.kernels.ops.moe.inkling_moe import get_src2dst, pre_reorder

    dst2src, counts, offsets, schedule, _ = _routing(tokens, rank)
    assert dst2src.numel() > nvfp4.SWAP_AB_MAX_ROWS
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    weight = torch.randint(0, 256, (E, 2 * I, H // 2), device="cuda").to(torch.uint8)
    scales = nvfp4.swizzle_weight_scales(
        (torch.rand(E, 2 * I, H // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    )
    amax = torch.rand(E, device="cuda") + 1
    denominator = torch.tensor([2.0], device="cuda")
    acc, x_amax = nvfp4.routed_expert_matmul(
        x, dst2src, TOPK, weight, scales, counts, offsets, schedule
    )
    expected_acc, expected_amax = nvfp4.expert_matmul(
        pre_reorder(x, get_src2dst(dst2src), TOPK),
        weight,
        scales,
        amax,
        denominator,
        metadata=(counts, offsets, schedule),
        raw=True,
    )
    lo, hi = int(offsets[0]), int(offsets[-1])
    assert torch.equal(acc[lo:hi], expected_acc[lo:hi])
    assert torch.equal(x_amax[lo:hi], expected_amax[lo:hi])


@blackwell
@pytest.mark.parametrize("tokens,rank", [(1280, 2), (16384, 1)])
def test_mxfp8_prefill_chain(tokens, rank):
    from sglang.kernels.ops.inkling2 import mxfp8

    dst2src, counts, offsets, schedule, gammas = _routing(tokens, rank)
    metadata = (counts, offsets, schedule)
    x = torch.randn(tokens, H, device="cuda").bfloat16()

    def weights(n, k):
        values = (torch.randn(E, n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        scales = torch.randint(118, 130, (E, n, k // 32), device="cuda").to(torch.uint8)
        return values, scales

    w13, s13 = weights(2 * I, H)
    w2, s2 = weights(H, I)
    gate_up = mxfp8.routed_expert_matmul(x, dst2src, TOPK, w13, s13, metadata)
    down = mxfp8.routed_swiglu_expert_matmul(
        gate_up, gammas, 4.0, 25.0, w2, s2, metadata, tokens
    )
    prefill_gate_up = mxfp8.prefill_expert_matmul(
        x, dst2src, TOPK, w13, mxfp8.cutlass_weight_scales(s13), counts, offsets
    )
    prefill_down = mxfp8.prefill_swiglu_expert_matmul(
        prefill_gate_up, gammas, 4.0, 25.0, w2, mxfp8.cutlass_weight_scales(s2),
        counts, offsets,
    )  # fmt: skip
    lo, hi = int(offsets[0]), int(offsets[-1])
    assert torch.equal(prefill_gate_up[lo:hi], gate_up[lo:hi])
    assert torch.equal(prefill_down[lo:hi], down[lo:hi])


def _quantize_reference(x):
    x = x.float()
    amax = x.abs().amax(-1)
    encoder = (x.new_tensor(1536.0) / amax).clamp(max=torch.finfo(torch.float32).max)
    decoder = encoder.reciprocal()
    outer = amax * (1.0 / 1536.0)
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    maximum = blocks.abs().amax(-1)
    grid = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    boundaries = (grid[:-1] + grid[1:]) / 2
    candidates = []
    for inverse_max in (1.0 / 6.0, 0.25):
        scale = (maximum * (encoder[..., None] * inverse_max)).clamp(max=448)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = (
            (scale * decoder[..., None])
            .reciprocal()
            .clamp(max=torch.finfo(torch.float32).max)
        )
        values = blocks * inverse[..., None]
        magnitude = values.abs()
        code = torch.bucketize(magnitude, boundaries)
        ties = magnitude == boundaries[code.clamp(max=6)]
        code += (ties & (code % 2 == 1)).long()
        rounded = grid[code] * torch.where(values.signbit(), -1.0, 1.0)
        error = ((rounded * scale[..., None]) * outer[..., None, None] - blocks).abs()
        for _ in range(4):
            error = error[..., ::2] + error[..., 1::2]
        code = code.to(torch.uint8) | (values.signbit().to(torch.uint8) << 3)
        packed = code[..., ::2] | (code[..., 1::2] << 4)
        candidates.append((packed, scale, error.squeeze(-1)))
    packed6, scale6, error6 = candidates[0]
    packed4, scale4, error4 = candidates[1]
    choose4 = error4 < error6
    packed = torch.where(choose4[..., None], packed4, packed6)
    scale = torch.where(choose4, scale4, scale6).to(torch.float8_e4m3fn)
    return packed.reshape(*x.shape[:-1], x.shape[-1] // 2), scale, amax


@sm100_or_sm103
@pytest.mark.parametrize("width", [16, 384, 4096, 12288])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_activation_quantization(width, dtype):
    torch.manual_seed(8)
    x = torch.randn(9, width, device="cuda", dtype=dtype)
    x[0].zero_()
    x[1].fill_(-0.0)
    x[2] *= 1e-8
    x[3] *= 1e8
    x[4, :16].zero_()
    actual = nvfp4_quantize(x)
    expected = _quantize_reference(x)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
        )


@sm100_or_sm103
@pytest.mark.parametrize("layout", ["batched", "grouped", "offset"])
def test_packed_expert_weights(layout):
    x = torch.ones(2, 17, 128, dtype=torch.bfloat16, device="cuda")
    x[1].neg_()
    weights = torch.full((2, 128, 64), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.ones((2, 128, 8), dtype=torch.float8_e4m3fn, device="cuda")
    amax = torch.full((2,), 1536.0, dtype=torch.float32, device="cuda")
    denominator = torch.full((1,), 1536.0, dtype=torch.float32, device="cuda")
    metadata = None
    if layout != "batched":
        x = x.reshape(34, 128)
        offset = 5 if layout == "offset" else 0
        if offset:
            x = torch.cat((x.new_zeros(5, 128), x, x.new_zeros(7, 128)))
        metadata = (
            torch.tensor([17, 17], dtype=torch.int32, device="cuda"),
            torch.tensor([offset, offset + 17], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, -1], dtype=torch.int32, device="cuda"),
        )
    output = expert_matmul(
        x, weights, swizzle_weight_scales(scales), amax, denominator, metadata=metadata
    )
    if layout == "offset":
        output, x = output[5:39], x[5:39]
    torch.testing.assert_close(output, x * 128, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Quantized expert computation requires Blackwell",
)
@pytest.mark.parametrize("precision", ["bfloat16", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("empty_rank", [False, True])
def test_expert_parallel_partials_preserve_full_expert_computation(
    monkeypatch, precision, empty_rank
):
    if precision == "nvfp4" and torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("NVFP4 expert computation requires SM103")
    parallel = SimpleNamespace(
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(all_reduce=lambda x: x),
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    prefix = "model.llm.layers.0.mlp.experts."
    quant = Inkling2QuantConfig(
        {
            "model": {
                "parameter_groups": [
                    precision_group(
                        precision, [prefix + name + "_weight" for name in ("w13", "w2")]
                    )
                ]
            }
        }
    )
    config = SimpleNamespace(
        hidden_size=128,
        intermediate_size=256,
        mlp_gate_softcap=4,
        mlp_up_softcap=25,
        num_experts_per_tok=2,
        route_scale=8.0,
    )
    torch.manual_seed(9)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        full = moe.Inkling2Experts(config, 4, prefix, quant)
        for name, value in full.named_parameters():
            if name.endswith("outer_scale_denominator"):
                value.fill_(2688)
            elif value.dtype == torch.uint8:
                value.copy_(
                    torch.randint(0, 256, value.shape, device="cuda", dtype=torch.uint8)
                    if name.endswith("weight")
                    else torch.full_like(value, 124)
                )
            else:
                value.copy_(torch.randn(value.shape, device="cuda").abs() + 0.125)
        x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 2 if empty_rank else 4, (17, 2), device="cuda")
        gammas = torch.rand(17, 2, device="cuda", dtype=torch.float32)
        parallel.moe_ep_size = 2
        shards = []
        for rank in range(2):
            parallel.moe_ep_rank = rank
            local = moe.Inkling2Experts(config, 4, prefix, quant)
            for name, value in local.named_parameters():
                value.weight_loader(value, full.get_parameter(name))
            local.swizzle_nvfp4_scales()
            shards.append(local)
        full.swizzle_nvfp4_scales()
        for rank, local in enumerate(shards):
            expected = full(x, gammas * (ids // 2 == rank), ids)
            actual = local(x, gammas, ids)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="NVFP4 expert computation requires SM103",
)
def test_nvfp4_reload_reaches_captured_cuda_graph(monkeypatch):
    """Reloaded NVFP4 weights must be used by a CUDA graph captured before the reload."""
    parallel = SimpleNamespace(
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(all_reduce=lambda x: x),
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    prefix = "model.llm.layers.0.mlp.experts."
    quant = Inkling2QuantConfig(
        {
            "model": {
                "parameter_groups": [
                    precision_group(
                        "nvfp4", [prefix + name + "_weight" for name in ("w13", "w2")]
                    )
                ]
            }
        }
    )
    config = SimpleNamespace(
        hidden_size=128,
        intermediate_size=256,
        mlp_gate_softcap=4,
        mlp_up_softcap=25,
        num_experts_per_tok=2,
        route_scale=8.0,
    )
    torch.manual_seed(9)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        experts = moe.Inkling2Experts(config, 4, prefix, quant)
        checkpoints = []
        for _ in range(2):
            checkpoint = {}
            for name, value in experts.named_parameters():
                if name.endswith("outer_scale_denominator"):
                    checkpoint[name] = torch.full_like(value, 2688)
                elif value.dtype == torch.uint8:
                    checkpoint[name] = torch.randint(
                        0, 256, value.shape, device="cuda", dtype=torch.uint8
                    )
                else:
                    checkpoint[name] = (torch.rand(value.shape) + 0.125).to(value.dtype)
            checkpoints.append(checkpoint)

        def load(checkpoint):
            for name, value in experts.named_parameters():
                value.weight_loader(value, checkpoint[name])
            experts.swizzle_nvfp4_scales()

        x = torch.randn(17, 128)
        ids = torch.randint(0, 4, (17, 2))
        gammas = torch.rand(17, 2, dtype=torch.float32)
        load(checkpoints[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            experts(x, gammas, ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = experts(x, gammas, ids)
        graph.replay()
        before_reload = captured.clone()
        # Keep the captured storages alive so the allocator cannot hand them back.
        previous = [param.data for param in experts.parameters()]
        load(checkpoints[1])
        graph.replay()
        expected = experts(x, gammas, ids)
    assert not torch.equal(before_reload, expected)
    assert torch.equal(captured.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_domain_paged_empty():
    from cuda.bindings import driver as cu

    out = moe.domain_paged_empty((8, 4608, 1152), torch.uint8)
    if out is None:
        pytest.skip("The GPU has one locality domain")
    weight = torch.randint(0, 256, out.shape, dtype=torch.uint8, device="cuda")
    assert out.is_contiguous() and out.device == weight.device
    out.copy_(weight)
    assert torch.equal(out, weight)
    n, page, half = out.numel(), 2 << 20, 4608 // 2 * 1152
    domains = [
        cu.cuPointerGetAttribute(
            cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
            out.data_ptr() + p,
        )[1]
        for p in range(0, n, page)
    ]
    local = sum(
        max(0, min(p + page, s + half) - max(p, s))
        for p, d in zip(range(0, n, page), domains)
        for s in range(d * half, n, 2 * half)
    )
    assert local / n > 0.75


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
