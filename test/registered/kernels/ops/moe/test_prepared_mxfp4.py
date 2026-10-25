"""Prepared MXFP4 routing (Triton and fused metadata) must match FlashInfer's routed MoE."""

import pytest
import torch

from sglang.kernels.ops.moe.trtllm_prepared_fp4 import prepared_routing_sm_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, INTER, K = 896, 3584, 384, 16


def _moe_kwargs(m, do_finalize):
    from flashinfer.fused_moe import core as c

    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )

    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
    xq, xs = per_token_group_quant(x, group_size=32, scale_ue8m0=True)

    def weights(shape):
        return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)

    def scales(shape):
        return torch.full(shape, 120, device="cuda", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )

    return x, dict(
        topk_ids=torch.empty((m, K), device="cuda", dtype=torch.int32),
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs.view(torch.float8_e4m3fn),
        gemm1_weights=weights((E, 2 * INTER, H // 2)),
        gemm1_weights_scale=scales((E, 2 * INTER, H // 32)),
        gemm1_bias=None,
        gemm1_alpha=torch.full((E,), 4.0, device="cuda"),
        gemm1_beta=torch.full((E,), 25.0, device="cuda"),
        gemm1_clamp_limit=None,
        gemm2_weights=weights((E, H, INTER // 2)),
        gemm2_weights_scale=scales((E, H, INTER // 32)),
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=E,
        top_k=K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=c.RoutingMethodType.TopK.value,
        activation_type=c.ActivationType.Situ.value,
        tune_max_num_tokens=m,
        output=torch.empty((m, H), device="cuda", dtype=torch.bfloat16),
        do_finalize=do_finalize,
        enable_pdl=True,
    )


def _moe_output(kwargs, result):
    if kwargs["do_finalize"]:
        return kwargs["output"].float()
    y, expert_weights, mapping = result
    m = kwargs["hidden_states"].shape[0]
    rows = (
        y[mapping.flatten().long()].float() * expert_weights.flatten().float()[:, None]
    )
    return rows.view(m, K, -1).sum(1)


def _run_reference(kwargs):
    from flashinfer.fused_moe import core as c

    out = _moe_output(kwargs, c.trtllm_fp4_block_scale_routed_moe(**kwargs)).clone()
    kwargs["output"].fill_(float("nan"))
    return out


def _scores(m, shared):
    if not shared:
        return torch.randn((m, E), device="cuda")
    return torch.randn((1, E), device="cuda") + 0.05 * torch.randn(
        (m, E), device="cuda"
    )


def _check_metadata(actual, packed):
    from sglang.kernels.ops.moe.moe_route_quant_fused import allocate_metadata
    from sglang.kernels.ops.moe.trtllm_prepared_fp4 import (
        prepare_expert_tile_metadata,
    )

    m = packed.shape[0]
    expected = allocate_metadata(packed.device, m)
    prepare_expert_tile_metadata[(1,)](
        packed,
        packed,
        *[expected[i] for i in (0, 1, 2, 3, 6, 7, 8)],
        TILE=8,
        ROUTES=m * K,
        TOP_K=K,
        PACKED=True,
        num_warps=4,
    )
    count = int(expected[8])
    for index in (0, 1, 8):
        torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
    assert torch.equal(actual[3].view(torch.int16), expected[3].view(torch.int16))
    for index in (6, 7):
        torch.testing.assert_close(
            actual[index][:count], expected[index][:count], rtol=0, atol=0
        )
    live = expected[1].long()
    torch.testing.assert_close(actual[2][live], expected[2][live], rtol=0, atol=0)


@pytest.mark.skipif(
    not prepared_routing_sm_supported(), reason="Requires native SiTU expert kernels"
)
@pytest.mark.parametrize(
    "m,do_finalize", [(1, False), (8, False), (8, True)], ids=["m1", "m8", "m8-final"]
)
def test_prepared_mxfp4_matches_routed_moe(m, do_finalize):
    from types import SimpleNamespace

    from sglang.kernels.ops.moe.moe_route_quant_fused import (
        _arrival_counter,
        allocate_metadata,
        route_quant_fused,
    )
    from sglang.kernels.ops.moe.trtllm_prepared_fp4 import try_prepared_k3_mxfp4

    torch.manual_seed(204 + m)
    x, kwargs = _moe_kwargs(m, do_finalize)
    owner = SimpleNamespace()
    # A small bias lets unshared scores pick mostly distinct experts per token.
    bias = 0.01 * torch.randn(E, device="cuda")

    # Triton metadata from packed ids, including tokens that share experts.
    for shared in (False, True, False, True):
        _, _, packed, _, _ = route_quant_fused(
            _scores(m, shared), bias, x, K, True, 1.0, False
        )
        kwargs["topk_ids"].copy_(packed)
        expected = _run_reference(kwargs)
        result = try_prepared_k3_mxfp4(owner, **kwargs)
        assert result is not None
        torch.testing.assert_close(
            _moe_output(kwargs, result), expected, atol=1e-5, rtol=1e-5
        )

    # Fused metadata with uninitialized padding, eagerly and under graph replay.
    metadata = allocate_metadata(x.device, m)
    metadata[2].fill_(-12345)
    scores = torch.randn((m, E), device="cuda")

    def fused_route():
        _, _, packed, quantized, quant_scales = route_quant_fused(
            scores, bias, x, K, True, 1.0, False, metadata=metadata
        )
        return dict(
            kwargs,
            topk_ids=packed,
            hidden_states=quantized,
            hidden_states_scale=quant_scales.view(torch.float8_e4m3fn),
        )

    def fused_body(fused_kwargs):
        result = try_prepared_k3_mxfp4(
            owner, prepared_metadata=metadata, **fused_kwargs
        )
        assert result is not None
        return result

    def check(result, fused_kwargs):
        actual = _moe_output(fused_kwargs, result).clone()
        torch.testing.assert_close(
            actual, _run_reference(fused_kwargs), atol=1e-5, rtol=1e-5
        )
        assert int(_arrival_counter(x.device)) == 0

    # The expert body may rewrite the map, so compare metadata before it runs.
    for shared in (False, True):
        scores.copy_(_scores(m, shared))
        fused_kwargs = fused_route()
        torch.cuda.synchronize()
        _check_metadata(metadata, fused_kwargs["topk_ids"])
        check(fused_body(fused_kwargs), fused_kwargs)

    def fused_run():
        fused_kwargs = fused_route()
        return fused_body(fused_kwargs), fused_kwargs

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, fused_kwargs = fused_run()
    for shared in (False, True, False, True):
        scores.copy_(_scores(m, shared))
        x.copy_(torch.randn_like(x))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        check(result, fused_kwargs)

    # Every token picks the same experts: each eight-token tile is full.
    scores.copy_(torch.randn((1, E), device="cuda").expand(m, -1))
    graph.replay()
    torch.cuda.synchronize()
    check(result, fused_kwargs)


if __name__ == "__main__":
    for case in [(1, False), (8, False), (8, True)]:
        test_prepared_mxfp4_matches_routed_moe(*case)
