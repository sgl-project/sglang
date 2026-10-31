"""Muse Spark routed MoE split: metadata + body must match the ordinary FlashInfer call bitwise."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("SM10x required", allow_module_level=True)

from flashinfer.fused_moe import (  # noqa: E402
    QuantConfig,
    QuantFormat,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_routed_moe,
)

from sglang.srt.layers.muse_spark.routed_moe_split import (  # noqa: E402
    get_routed_moe_split,
    split_enabled,
    split_supported,
)

# Muse Spark experts: 256 routed experts, top-8, hidden 4096.
H, I, E, TOP_K = 4096, 4096, 256, 8
# (local experts, M): 64 / 32 local experts are TP4 / TP8 with EP; tiles 8 to 128.
SPLIT_CASES = [(64, m) for m in (1, 16, 128, 512, 1024)] + [
    (32, m) for m in (1, 16, 128, 512, 768)
]


@pytest.fixture(scope="module")
def weights():
    quant = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    views = {}
    for local in (64, 32):
        torch.manual_seed(local)
        w1 = (torch.randn(local, 2 * I, H, device="cuda") * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(local, H, I, device="cuda") * 0.02).to(torch.bfloat16)
        views[local] = TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=local,
            hidden_size=H,
            intermediate_size=I,
            device=torch.device("cuda"),
        )
        del w1, w2
    yield quant, views
    torch.cuda.empty_cache()


def _inputs(quant, m: int, seed: int, per_token: bool):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g) * 2
    hs, sf = TrtllmFp4Config.prepare_activations(x.to(torch.bfloat16), quant=quant)
    ids = torch.stack(
        [torch.randperm(E, device="cuda", generator=g)[:TOP_K] for _ in range(m)]
    ).to(torch.int32)
    w = torch.rand(m, TOP_K, device="cuda", generator=g).to(torch.bfloat16)
    packed = (ids << 16) | (w.view(torch.int16).to(torch.int32) & 0xFFFF)
    scale = torch.rand(m, device="cuda", generator=g) + 0.5 if per_token else None
    return hs, sf, packed, scale


def _kwargs(view, local, hs, sf, packed, scale):
    return dict(
        topk_ids=packed,
        routing_bias=None,
        hidden_states=hs,
        hidden_states_scale=sf,
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=view.get("gemm1_alpha"),
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=view.get("output1_scale_scalar"),
        output1_scale_gate_scalar=view.get("output1_scale_gate_scalar"),
        output2_scale_scalar=view.get("output2_scale_scalar"),
        per_token_scale=scale,
        num_experts=E,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=I,
        local_expert_offset=local,
        local_num_experts=local,
        routed_scaling_factor=1.0,
        routing_method_type=1,
        do_finalize=False,
        activation_type=3,
    )


def _finalize(m, gemm2_out, expert_weights, expanded_idx_to_permuted_idx):
    """Per-token weighted sum in top-k order; padded FC2 rows are not compared."""
    idx = expanded_idx_to_permuted_idx.long().view(m, TOP_K)
    # Padded slots (-1) may hold uninitialized FC2 rows and expert weights (NaN * 0 is NaN);
    # zero their products explicitly.
    rows = gemm2_out.float()[idx.clamp(min=0)] * expert_weights.float().view(
        m, TOP_K, 1
    )
    return torch.where((idx >= 0).view(m, TOP_K, 1), rows, 0.0).sum(dim=1)


def _ordinary(kwargs, m):
    return _finalize(m, *trtllm_fp4_block_scale_routed_moe(**kwargs)[:3])


@pytest.mark.parametrize("per_token", [False, True], ids=["static", "per_token"])
@pytest.mark.parametrize("local, m", SPLIT_CASES)
def test_split_matches_ordinary_call(weights, local, m, per_token):
    """Populate on a side stream and join before the body, as serving does; then replay a graph
    of the same sequence with new routing and activations copied into its inputs."""
    assert split_supported(num_tokens=m, top_k=TOP_K, num_local_experts=local)
    quant, views = weights
    hs, sf, packed, scale = _inputs(quant, m, seed=m, per_token=per_token)
    kwargs = _kwargs(views[local], local, hs, sf, packed, scale)
    split = get_routed_moe_split(
        packed, num_experts=E, local_expert_offset=local, num_local_experts=local
    )
    side = torch.cuda.Stream()

    def run():
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            split.populate(packed)
        main.wait_stream(side)
        return split.run_body(**kwargs)

    assert torch.equal(_finalize(m, *run()), _ordinary(kwargs, m)), "eager"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for seed in (m + 1, m + 2):
        new_hs, new_sf, new_packed, new_scale = _inputs(quant, m, seed, per_token)
        hs.copy_(new_hs)
        sf.copy_(new_sf)
        packed.copy_(new_packed)
        if per_token:
            scale.copy_(new_scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_finalize(m, *captured), _ordinary(kwargs, m)), (
            f"replay seed={seed}"
        )


def test_split_gate():
    """The gate admits the tiles measured bitwise (<= 128) and keeps the rest on the ordinary call."""
    for local, m in SPLIT_CASES:
        assert split_supported(num_tokens=m, top_k=TOP_K, num_local_experts=local)
    # Tile 192 at 32 local experts, and above the measured token range.
    assert not split_supported(num_tokens=960, top_k=TOP_K, num_local_experts=32)
    assert not split_supported(num_tokens=1025, top_k=TOP_K, num_local_experts=64)


def test_split_enabled_per_gpu_and_tp():
    """Only measured (GPU, TP) pairs take the split; GB300 keeps the fused routing at every size."""
    for tp in (4, 8):
        assert split_enabled(num_tokens=8, tp_size=tp, capability=(10, 7))
        assert split_enabled(num_tokens=1024, tp_size=tp, capability=(10, 7))
        assert not split_enabled(num_tokens=1024, tp_size=tp, capability=(10, 3))
    assert not split_enabled(num_tokens=4, tp_size=4, capability=(10, 7))
    assert not split_enabled(num_tokens=64, tp_size=2, capability=(10, 7))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
