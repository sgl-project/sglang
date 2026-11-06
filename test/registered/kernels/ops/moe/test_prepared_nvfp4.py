"""Prepared NVFP4 expert-tile routing versus FlashInfer's routed FP4 MoE.

``try_prepared_k3_nvfp4`` runs FlashInfer's native FP4 MoE body with
device-prepared expert-tile metadata (unpacked precomputed routing: separate
int32 ids and fp32 weights). Its output must match
``trtllm_fp4_block_scale_routed_moe`` on the same inputs, deferred and
finalized, with normalized and unnormalized routing weights.
"""

import pytest
import torch

from sglang.kernels.ops.moe.trtllm_prepared_fp4 import prepared_routing_sm_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, INTER, K = 896, 3584, 384, 16


def _moe_kwargs(m, do_finalize):
    from flashinfer.fused_moe import core as c

    def u8(shape):
        return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)

    def fp8_scales(shape):
        return torch.full(shape, 120, device="cuda", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )

    return dict(
        topk_ids=None,  # (ids int32 [m, K], weights fp32 [m, K]) set per case
        routing_bias=None,
        hidden_states=u8((m, H // 2)),
        hidden_states_scale=fp8_scales((m, H // 16)),
        gemm1_weights=u8((E, 2 * INTER, H // 2)),
        gemm1_weights_scale=fp8_scales((E, 2 * INTER, H // 16)),
        gemm1_bias=None,
        gemm1_alpha=torch.full((E,), 4.0, device="cuda"),
        gemm1_beta=torch.full((E,), 25.0, device="cuda"),
        gemm1_clamp_limit=None,
        gemm2_weights=u8((E, H, INTER // 2)),
        gemm2_weights_scale=fp8_scales((E, H, INTER // 16)),
        gemm2_bias=None,
        output1_scale_scalar=torch.full((E,), 0.5, device="cuda"),
        output1_scale_gate_scalar=torch.full((E,), 0.25, device="cuda"),
        output2_scale_scalar=torch.full((E,), 0.125, device="cuda"),
        per_token_scale=None,
        num_experts=E,
        top_k=K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=1,
        activation_type=c.ActivationType.Situ.value,
        tune_max_num_tokens=m,
        output=torch.empty((m, H), device="cuda", dtype=torch.bfloat16),
        do_finalize=do_finalize,
        enable_pdl=True,
    )


def _routing(m, shared, normalized):
    scores = torch.randn((1 if shared else m, E), device="cuda").expand(m, -1)
    scores = scores + 0.05 * torch.randn((m, E), device="cuda")
    weights, ids = torch.topk(torch.sigmoid(scores), K, dim=-1)
    if normalized:
        weights = weights / weights.sum(-1, keepdim=True)
    else:
        weights = weights * 2.5  # routed_scaling_factor already applied upstream
    return ids.to(torch.int32).contiguous(), weights.float().contiguous()


def _moe_output(kwargs, result):
    if kwargs["do_finalize"]:
        return kwargs["output"].float()
    y, expert_weights, mapping = result[:3]
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


@pytest.mark.skipif(
    not prepared_routing_sm_supported(), reason="Requires native SiTU expert kernels"
)
@pytest.mark.parametrize(
    "m,do_finalize", [(1, False), (8, False), (8, True)], ids=["m1", "m8", "m8-final"]
)
@pytest.mark.parametrize("normalized", [True, False], ids=["norm", "scaled"])
def test_prepared_nvfp4_matches_routed_moe(m, do_finalize, normalized):
    from types import SimpleNamespace

    from sglang.kernels.ops.moe.trtllm_prepared_fp4 import try_prepared_k3_nvfp4

    torch.manual_seed(304 + m)
    kwargs = _moe_kwargs(m, do_finalize)
    owner = SimpleNamespace()
    for shared in (False, True, False, True):
        kwargs["topk_ids"] = _routing(m, shared, normalized)
        expected = _run_reference(kwargs)
        result = try_prepared_k3_nvfp4(owner, **kwargs)
        assert result is not None, "prepared NVFP4 path not taken"
        actual = _moe_output(kwargs, result).clone()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # Same routing under CUDA-graph replay: metadata is rebuilt every replay.
    kwargs["topk_ids"] = _routing(m, False, normalized)
    ids, weights = kwargs["topk_ids"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = try_prepared_k3_nvfp4(owner, **kwargs)
    assert result is not None
    for shared in (True, False):
        new_ids, new_weights = _routing(m, shared, normalized)
        ids.copy_(new_ids)
        weights.copy_(new_weights)
        kwargs["hidden_states"].copy_(
            torch.randint_like(kwargs["hidden_states"], 0, 256)
        )
        graph.replay()
        torch.cuda.synchronize()
        actual = _moe_output(kwargs, result).clone()
        torch.testing.assert_close(actual, _run_reference(kwargs), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not prepared_routing_sm_supported(), reason="Requires native SiTU expert kernels"
)
@pytest.mark.parametrize(
    "m,do_finalize", [(1, False), (8, False), (8, True)], ids=["m1", "m8", "m8-final"]
)
def test_prepared_nvfp4_fresh_buffers_each_call(m, do_finalize):
    """The cached body workspace must not pin the first call's activation,
    scale, routing or output buffers: every call here uses new tensors."""
    from types import SimpleNamespace

    from sglang.kernels.ops.moe.trtllm_prepared_fp4 import try_prepared_k3_nvfp4

    torch.manual_seed(404 + m)
    owner = SimpleNamespace()
    base = _moe_kwargs(m, do_finalize)
    for shared in (False, True, False, True, False):
        kwargs = dict(
            base,
            hidden_states=torch.randint_like(base["hidden_states"], 0, 256),
            hidden_states_scale=torch.full_like(
                base["hidden_states_scale"].view(torch.uint8),
                120 + (2 if shared else 0),
            ).view(torch.float8_e4m3fn),
            output=torch.empty_like(base["output"]),
            topk_ids=_routing(m, shared, True),
        )
        expected = _run_reference(dict(kwargs))
        result = try_prepared_k3_nvfp4(owner, **kwargs)
        assert result is not None
        actual = _moe_output(kwargs, result).clone()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    for normalized in (True, False):
        for case in [(1, False), (8, False), (8, True)]:
            test_prepared_nvfp4_matches_routed_moe(*case, normalized)
            print("ok", case, "normalized" if normalized else "scaled")
    for case in [(1, False), (8, False), (8, True)]:
        test_prepared_nvfp4_fresh_buffers_each_call(*case)
        print("ok fresh-buffers", case)
