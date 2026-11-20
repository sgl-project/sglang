"""Routed-expert, prefill MoE, NVFP4, and expert-parallel kernels."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.inkling_moe import (
    compute_expert_block_metadata,
    compute_grouped_gemm_metadata,
)
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.models.inkling_common.inkling2_attn_res import (
    Inkling2AttnRes,
    Inkling2AttnResState,
)
from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm
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
    from sglang.kernels.ops.moe.inkling2_route import route_metadata

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


_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_nvfp4(packed, scales):
    """Packed E2M1 pairs (low nibble first) times their 16-wide FP8 block scales."""
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    grid = torch.tensor(_FP4_GRID, device=packed.device)
    values = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return values * scales.float().repeat_interleave(16, -1)


def _dequantize_mxfp8(values, scales):
    """FP8 E4M3 values times their 32-wide E8M0 block scales."""
    exponents = scales.view(torch.uint8).float() - 127
    return values.float() * torch.exp2(exponents).repeat_interleave(32, -1)


def _grouped_reference(x_rows, dequantize_weight, offsets):
    """FP32 x_rows[r] @ W[e].T for the rows r of each local expert e; other rows 0."""
    out = None
    for e in range(offsets.numel() - 1):
        start, end = int(offsets[e]), int(offsets[e + 1])
        if start == end:
            continue
        product = x_rows[start:end] @ dequantize_weight(e).T
        if out is None:
            out = x_rows.new_zeros(x_rows.shape[0], product.shape[1])
        out[start:end] = product
    return out


def _assert_relative_error(actual, expected, tolerance):
    error = (actual.float() - expected).norm() / expected.norm()
    assert error < tolerance, f"relative error {error.item():.3g} >= {tolerance}"


@sm100_or_sm103
@pytest.mark.parametrize(
    "tokens,skewed",
    [(1, False), (64, False), (129, True), (256, False), (512, False), (512, True)],
)
def test_route_metadata_cuda_matches_triton(monkeypatch, tokens, skewed):
    from sglang.kernels.ops.moe import inkling2_route as route

    monkeypatch.setattr(route, "ROUTE_CUDA_MIN_IDS", 1 << 30)
    expected = _route(tokens, skewed, seed=tokens)
    monkeypatch.setattr(route, "ROUTE_CUDA_MIN_IDS", 0)
    actual = _route(tokens, skewed, seed=tokens)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


@sm100_or_sm103
@pytest.mark.parametrize(
    "tokens,skewed,width",
    [(1, False, 4608), (16, False, 4608), (256, False, 4608), (512, True, 4608)]
    + [(64, False, 2304), (512, True, 2304)],
)
def test_cute_routed_gemm_matches_reference(tokens, skewed, width):
    """The routed NVFP4 GEMM's raw accumulator is the FP32 product of the 4over6
    quantized token rows and the dequantized expert weights; skewed routing fills
    multi-block experts and partial subtiles."""
    from sglang.kernels.ops.gemm import inkling2_nvfp4 as nvfp4
    from sglang.kernels.ops.moe.inkling2_route import route_metadata

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
    _, dst2src, _, counts, offsets, _, schedule, _ = route_metadata(
        ids.flatten(), weights.flatten(), LOCAL, LOCAL, EXPERTS
    )
    n = 2 * 2304 if width == 4608 else 4608
    x = torch.randn(tokens, width, device="cuda", generator=g).bfloat16()
    weight = torch.randint(
        0, 256, (LOCAL, n, width // 2), device="cuda", generator=g
    ).to(torch.uint8)
    weight_scales = (
        torch.rand(LOCAL, n, width // 16, device="cuda", generator=g) * 2 + 0.25
    ).to(torch.float8_e4m3fn)
    acc, x_amax = nvfp4.routed_expert_matmul(
        x,
        dst2src,
        TOP_K,
        weight,
        swizzle_blockscale(weight_scales).flatten(1),
        counts,
        offsets,
        schedule,
    )
    _assert_nvfp4_routed_rows(
        acc, x_amax, x, dst2src, TOP_K, weight, weight_scales, offsets
    )


def _assert_nvfp4_routed_rows(acc, x_amax, x, dst2src, top_k, weight, scales, offsets):
    values, x_scales, amax = _quantize_reference(x)
    source = dst2src.long() // top_k
    expected = _grouped_reference(
        _dequantize_nvfp4(values, x_scales)[source],
        lambda e: _dequantize_nvfp4(weight[e], scales[e]),
        offsets,
    )
    lo, hi = int(offsets[0]), int(offsets[-1])
    assert hi > lo
    _assert_relative_error(acc[lo:hi], expected[lo:hi], 1e-5)
    assert torch.equal(x_amax[lo:hi], amax[source[lo:hi]])


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
@pytest.mark.parametrize("tokens,rank", [(1280, 2), (16384, 1)])
def test_mxfp8_prefill_chain(tokens, rank):
    """The routed MXFP8 GEMMs, which also serve expert-parallel prefill, match an
    FP32 reference: token-quantized rows, then the capped swiglu into w2."""
    from sglang.kernels.ops.gemm import inkling2_mxfp8 as mxfp8
    from sglang.kernels.ops.moe.inkling2_moe import mxfp8_quantize

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
    lo, hi = int(offsets[0]), int(offsets[-1])
    x_rows = _dequantize_mxfp8(*mxfp8_quantize(x))[dst2src.long() // TOPK]
    expected_gate_up = _grouped_reference(
        x_rows, lambda e: _dequantize_mxfp8(w13[e], s13[e]), offsets
    )
    _assert_relative_error(gate_up[lo:hi], expected_gate_up[lo:hi], 1e-2)
    # The kernel quantizes the activation to MXFP8; the reference does not.
    activated = moe._capped_swiglu(gate_up.float(), gammas, 4.0, 25.0, offsets)
    expected_down = _grouped_reference(
        activated, lambda e: _dequantize_mxfp8(w2[e], s2[e]), offsets
    )
    _assert_relative_error(down[lo:hi], expected_down[lo:hi], 5e-2)


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


@blackwell
@pytest.mark.parametrize("tokens,rank", [(1028, 0), (4099, 3), (16384, 2)])
@pytest.mark.parametrize("layer", [6, 7])
def test_mega_moe_local_input_matches_full_batch_rows(tokens, rank, layer):
    """The MoE input chain (AttnRes, norm, router, top-k) on the rows a rank
    dispatches to MegaMoE equals those rows of the chain on every token: from
    MEGA_MOE_LOCAL_INPUT_MIN_ROWS up, no kernel in it is picked by row count.
    Layer 6 opens an AttnRes block; layer 7 adds the attention output instead."""
    start, count = moe.Inkling2Experts.mega_token_slice(
        SimpleNamespace(ep_size=4, ep_rank=rank), tokens
    )
    assert count >= moe.MEGA_MOE_LOCAL_INPUT_MIN_ROWS
    rows = slice(start, start + count)
    torch.manual_seed(tokens + rank)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        attn_res = Inkling2AttnRes(12, H, 6, [], normalize_values=True)
        norm = RMSNorm(H, 1e-6, output_dtype=torch.bfloat16)
        gate = moe.Inkling2Gate(
            SimpleNamespace(
                num_experts_per_tok=TOPK,
                route_scale=8.0,
                n_routed_experts=4 * E,
                hidden_size=H,
                use_gate_bias=True,
            )
        )
        for parameter in (attn_res.mlp_queries[layer], gate.weight, gate.bias):
            parameter.data.copy_(torch.randn(parameter.shape, dtype=torch.float32))
        norm.weight.data.copy_(torch.rand(H, dtype=torch.float32) + 0.5)
        blocks = [torch.randn(tokens, H) for _ in range(2)]
        partial, attention = torch.randn(tokens, H), torch.randn(tokens, H)
    state = Inkling2AttnResState(attn_res, partial, blocks)

    full = norm(state.mlp_input(layer, attention))
    local = norm(state.mlp_input(layer, attention, (start, count)))
    assert torch.equal(local, full[rows])
    full_weights, full_ids = gate(full)
    weights, ids = gate(local)
    assert torch.equal(ids, full_ids[rows])
    assert torch.equal(weights, full_weights[rows])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Quantized expert computation requires Blackwell",
)
@pytest.mark.parametrize("empty_rank", [False, True])
def test_expert_parallel_partials_preserve_full_expert_computation(
    monkeypatch, empty_rank
):
    """Quantized experts need expert parallelism, so the EP1 reference is BF16."""
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
                        "bfloat16",
                        [prefix + name + "_weight" for name in ("w13", "w2")],
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
        moe_ep_size=2,
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
                    continue
                # Checkpoint tensors hold every rank's experts.
                shape = (2 * value.shape[0], *value.shape[1:])
                if value.dtype == torch.uint8:
                    checkpoint[name] = torch.randint(
                        0, 256, shape, device="cuda", dtype=torch.uint8
                    )
                else:
                    checkpoint[name] = (torch.rand(shape) + 0.125).to(value.dtype)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
