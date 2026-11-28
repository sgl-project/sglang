"""Muse Spark localized decode MoE: per-domain N split against the stock launch."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-small",
    disabled="Needs an SM107 GPU with two locality domains; no CI runner has one. "
    "Run manually on SM107.",
)

NUM_EXPERTS, LOCAL_EXPERTS, LOCAL_OFFSET, TOP_K = 64, 16, 16, 8
HIDDEN, INTER = 4096, 4096
MAX_TOKENS = 512


def _streams():
    if not torch.cuda.is_available():
        return None
    from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
        _localized_moe_streams,
    )

    return _localized_moe_streams(torch.cuda.current_device())


pytestmark = pytest.mark.skipif(
    _streams() is None,
    reason="needs SM107 with two locality domains and FlashInfer localized MoE",
)


def _sf(*shape):
    return (torch.rand(shape, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn)


def _fp4(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")


def _layer():
    e = LOCAL_EXPERTS
    params = dict(
        gate_up_proj=_fp4(e, 2 * INTER, HIDDEN // 2),
        gate_up_proj_weight_scale=_sf(e, 2 * INTER, HIDDEN // 16),
        down_proj=_fp4(e, HIDDEN, INTER // 2),
        down_proj_weight_scale=_sf(e, HIDDEN, INTER // 16),
    )
    layer = SimpleNamespace(
        **{k: torch.nn.Parameter(v, requires_grad=False) for k, v in params.items()}
    )
    scalar = torch.full((e,), 1 / 256, device="cuda")
    layer.g1_scale_c, layer.g1_alphas, layer.g2_alphas = scalar, scalar, scalar
    return layer, {k: v.clone() for k, v in params.items()}


def _inputs(num_tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device="cuda", generator=gen)[:TOP_K]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    scores = torch.rand(num_tokens, TOP_K, device="cuda", generator=gen)
    hs = torch.randint(
        0,
        256,
        (num_tokens, HIDDEN // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    hs_sf = (
        torch.rand(num_tokens, HIDDEN // 16, device="cuda", generator=gen) * 0.5 + 0.5
    ).to(torch.float8_e4m3fn)
    return ids, scores, hs, hs_sf


def _moe(layer, weights, inputs, **localized):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    ids, scores, hs, hs_sf = inputs
    out, _, permuted = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, scores),
        routing_bias=None,
        hidden_states=hs,
        hidden_states_scale=hs_sf,
        gemm1_weights=weights["gate_up_proj"],
        gemm1_weights_scale=weights["gate_up_proj_weight_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=weights["down_proj"],
        gemm2_weights_scale=weights["down_proj_weight_scale"],
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c,
        output1_scale_gate_scalar=layer.g1_alphas,
        output2_scale_scalar=layer.g2_alphas,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTER,
        local_expert_offset=LOCAL_OFFSET,
        local_num_experts=LOCAL_EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=1,
        do_finalize=False,
        activation_type=3,
        **localized,
    )[:3]
    # Per-(token, slot) expert outputs; slots routed elsewhere are zero.
    idx = permuted.reshape(-1, 1).long()
    return torch.where(idx >= 0, out[idx.squeeze(1).clamp(min=0)].float(), 0.0)


def _run(layer, weights, num_tokens, **localized):
    return _moe(layer, weights, _inputs(num_tokens, num_tokens), **localized)


@pytest.fixture(scope="module")
def localized_layer():
    from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
        _localize_fp4_moe_weights,
    )

    torch.manual_seed(0)
    layer, original = _layer()
    _localize_fp4_moe_weights(layer, MAX_TOKENS)
    return layer, original


def test_mapped_twice_keeps_full_tensor(localized_layer):
    layer, original = localized_layer
    for name, tensor in original.items():
        assert torch.equal(
            getattr(layer, name).view(torch.uint8), tensor.view(torch.uint8)
        )


@pytest.mark.parametrize("num_tokens", [16, 32, 64, 128, 256, MAX_TOKENS])
def test_localized_matches_stock_bitwise(localized_layer, num_tokens):
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    layer, original = localized_layer
    localized = MuseSpark1xFp4Experts._localized_moe_kwargs(layer, num_tokens)
    assert localized
    ref = _run(layer, original, num_tokens)
    weights = {name: getattr(layer, name) for name in original}
    stock = _run(layer, weights, num_tokens)
    loc = _run(layer, weights, num_tokens, **localized)
    assert ref.isfinite().all() and ref.abs().sum() > 0
    assert torch.equal(stock, ref)
    assert torch.equal(loc, ref)


@pytest.mark.parametrize("num_tokens", [16, MAX_TOKENS])
def test_localized_graph_replay_with_new_inputs(localized_layer, num_tokens):
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    layer, original = localized_layer
    localized = MuseSpark1xFp4Experts._localized_moe_kwargs(layer, num_tokens)
    weights = {name: getattr(layer, name) for name in original}
    static = _inputs(num_tokens, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # Warm up outside the capture.
        _moe(layer, weights, static, **localized)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _moe(layer, weights, static, **localized)
    # Each replay sees new routing and activations written into the captured buffers.
    for seed in (1, 2, 3):
        fresh = _inputs(num_tokens, seed)
        for buf, value in zip(static, fresh):
            buf.copy_(value)
        graph.replay()
        assert torch.equal(captured, _moe(layer, original, fresh))


def test_reload_keeps_addresses_and_graphs():
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
    from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
        _localize_fp4_moe_weights,
    )

    torch.manual_seed(1)
    layer, _ = _layer()
    _localize_fp4_moe_weights(layer, MAX_TOKENS)
    names = (
        "gate_up_proj",
        "gate_up_proj_weight_scale",
        "down_proj",
        "down_proj_weight_scale",
    )
    weights = {name: getattr(layer, name) for name in names}

    def addresses():
        shards = [t for shard in layer.localized_moe_weights for t in shard.values()]
        return [t.data_ptr() for t in (*weights.values(), *shards)]

    before = addresses()
    num_tokens = 128
    localized = MuseSpark1xFp4Experts._localized_moe_kwargs(layer, num_tokens)
    static = _inputs(num_tokens, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _moe(layer, weights, static, **localized)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _moe(layer, weights, static, **localized)
    # A reload writes every expert parameter in place, then reruns the preparation.
    _, fresh = _layer()
    for name, tensor in fresh.items():
        getattr(layer, name).data.copy_(tensor)
    _localize_fp4_moe_weights(layer, MAX_TOKENS)
    assert addresses() == before
    graph.replay()
    assert torch.equal(captured, _moe(layer, fresh, static))


@pytest.mark.parametrize("num_tokens", [1, 8, MAX_TOKENS + 1, 4096])
def test_outside_token_range_runs_stock(localized_layer, num_tokens):
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    layer, _ = localized_layer
    assert MuseSpark1xFp4Experts._localized_moe_kwargs(layer, num_tokens) == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
