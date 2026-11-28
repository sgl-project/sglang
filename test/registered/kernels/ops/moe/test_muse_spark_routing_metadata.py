"""Muse Spark routing metadata built on a side stream: the routed MoE must match the fused routing bitwise.

The metadata has the tile the routed call runs: the autotuned one when FlashInfer has it, else the
default."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("SM10x required", allow_module_level=True)

from flashinfer.autotuner import AutoTuner, autotune  # noqa: E402
from flashinfer.fused_moe import (  # noqa: E402
    QuantConfig,
    QuantFormat,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_moe_routing_tile_n,
    trtllm_fp4_block_scale_routed_moe,
)

from sglang.srt.layers.muse_spark import muse_spark_fp4_moe  # noqa: E402
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import (  # noqa: E402
    MuseSpark1xFp4Experts,
)

# Muse Spark experts: 256 routed experts, top-8, hidden 4096.
H, I, E, TOP_K = 4096, 4096, 256, 8
# (local experts, M): 64 / 32 local experts are TP4 / TP8 with EP.
SPLIT_CASES = [(64, m) for m in (8, 16, 128, 512, 1024)] + [
    (32, m) for m in (8, 16, 128, 512, 1024)
]


@pytest.fixture
def this_gpu_enabled(monkeypatch):
    """List this GPU at TP4 / TP8 as the measured SM107 entries are."""
    cap = torch.cuda.get_device_capability()
    monkeypatch.setitem(muse_spark_fp4_moe._ROUTING_METADATA_MIN_TOKENS, (cap, 4), 8)
    monkeypatch.setitem(muse_spark_fp4_moe._ROUTING_METADATA_MIN_TOKENS, (cap, 8), 8)


def _populate(
    packed: torch.Tensor,
    local: int,
    view=None,
    per_token: bool = False,
    num_experts: int = E,
    offset=None,
):
    experts = SimpleNamespace(
        use_marlin=False,
        global_num_experts=num_experts,
        local_expert_offset=local if offset is None else offset,
        num_local_experts=local,
        hidden_size=H,
        intermediate_size_per_partition=I,
        quant_config=SimpleNamespace(use_per_token_activation=per_token),
        gate_up_proj=None if view is None else view["gemm1_weights"],
        gate_up_proj_weight_scale=None if view is None else view["gemm1_weights_scale"],
    )
    experts._prepared_input_meta = lambda m: MuseSpark1xFp4Experts._prepared_input_meta(
        experts, m
    )
    return MuseSpark1xFp4Experts.populate_routing_metadata(
        experts, packed, tp_size=num_experts // local
    )


def _record_fp4_tactics(monkeypatch):
    """Tactics FlashInfer's FP4 MoE op chooses (the routed call and the tile query)."""
    chosen = []
    choose_one = AutoTuner.choose_one

    def record(self, custom_op, *args, **kwargs):
        runner, tactic = choose_one(self, custom_op, *args, **kwargs)
        if custom_op == "flashinfer::trtllm_fp4_block_scale_moe":
            chosen.append(tactic)
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", record)
    return chosen


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


@pytest.mark.parametrize("tuned", [False, True], ids=["untuned", "tuned"])
@pytest.mark.parametrize("per_token", [False, True], ids=["static", "per_token"])
@pytest.mark.parametrize("local, m", SPLIT_CASES)
def test_routing_metadata_matches_fused_routing(
    weights, this_gpu_enabled, monkeypatch, local, m, per_token, tuned
):
    """Populate on a side stream and join before the experts, as serving does; then replay a graph
    of the same sequence with new routing and activations copied into its inputs."""
    quant, views = weights
    hs, sf, packed, scale = _inputs(quant, m, seed=m, per_token=per_token)
    kwargs = _kwargs(views[local], local, hs, sf, packed, scale)
    side = torch.cuda.Stream()
    AutoTuner.get().clear_cache()
    if tuned:
        with autotune(True):
            assert _populate(packed, local, views[local], per_token) is None
            trtllm_fp4_block_scale_routed_moe(**kwargs)
    chosen = _record_fp4_tactics(monkeypatch)
    want = _ordinary(kwargs, m)
    # The tactic the ordinary call runs: tuned ([tile_N, config]) or not (-1).
    (tactic,) = chosen
    assert (tactic != -1) == tuned
    # The tile for the real activations; the experts pass shape-only ones.
    tile_n = trtllm_fp4_block_scale_moe_routing_tile_n(
        packed,
        hs,
        sf,
        kwargs["gemm1_weights"],
        kwargs["gemm1_weights_scale"],
        E,
        TOP_K,
        I,
        local,
        do_finalize=False,
        activation_type=3,
        per_token_scale=scale,
    )
    if tuned:
        assert tile_n == int(tactic[0])

    def run():
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            metadata = _populate(packed, local, views[local], per_token)
        main.wait_stream(side)
        assert metadata is not None and metadata.tile_n == tile_n
        return trtllm_fp4_block_scale_routed_moe(**kwargs, routing_metadata=metadata)

    assert torch.equal(_finalize(m, *run()[:3]), want), "eager"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()[:3]
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


def test_routing_metadata_cache_keys_expert_geometry(weights, this_gpu_enabled):
    """Expert splits at the same offset (256/64 and 128/32 at 0) get their own metadata."""
    _, views = weights
    packed = torch.zeros(128, TOP_K, dtype=torch.int32, device="cuda")
    a = _populate(packed, 64, views[64], num_experts=256, offset=0)
    b = _populate(packed, 32, views[32], num_experts=128, offset=0)
    by_slot = {
        id(md.slots[0]): md for md in muse_spark_fp4_moe._routing_metadata.values()
    }
    md_a, md_b = by_slot[id(a)], by_slot[id(b)]
    assert md_a is not md_b
    assert (md_a.num_experts, md_a.num_local_experts) == (256, 64)
    assert (md_b.num_experts, md_b.num_local_experts) == (128, 32)


def test_routing_metadata_declined_tile_keeps_fused_routing(
    weights, this_gpu_enabled, monkeypatch
):
    """No tile can run: keep the fused routing, which raises FlashInfer's error."""
    import flashinfer.fused_moe

    _, views = weights
    monkeypatch.setattr(
        flashinfer.fused_moe,
        "trtllm_fp4_block_scale_moe_routing_tile_n",
        lambda *args, **kwargs: None,
    )
    packed = torch.zeros(128, TOP_K, dtype=torch.int32, device="cuda")
    assert _populate(packed, 64, views[64]) is None


def test_routing_metadata_gate(this_gpu_enabled, monkeypatch):
    """Only measured (GPU, TP) pairs and token counts leave the fused routing."""

    def ids(m):
        return torch.zeros(m, TOP_K, dtype=torch.int32, device="cuda")

    assert _populate(ids(4), 64) is None  # below the measured minimum
    assert _populate(ids(1025), 64) is None  # above the measured range
    monkeypatch.setattr(muse_spark_fp4_moe, "_ROUTING_METADATA_MIN_TOKENS", {})
    assert _populate(ids(128), 64) is None  # unlisted GPU and TP


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
