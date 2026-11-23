"""Muse Spark BF16 experts on SM90: interleaved gate-up Triton and grouped vs stock."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.muse_spark.fused import glue
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, H, I, K, G, D = 16, 4096, 4096, 8, "cuda", torch.float64  # local experts 8 .. 15
sm90 = pytest.mark.skipif(not glue.hopper_kernels_enabled(), reason="SM90")


@pytest.fixture(scope="module", autouse=True)
def _runtime():
    """fused_experts reads server args and allocates under the TP group (TP = 1)."""
    import os

    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.test.test_utils import publish_build_topology

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    for k, v in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT="29643").items():
        os.environ.setdefault(k, v)
    if not torch.distributed.is_initialized():
        ps.init_distributed_environment(1, 0, local_rank=0, backend="gloo")
    if not ps.model_parallel_is_initialized():
        publish_build_topology(tp_size=1, ep_size=1, pp_size=1)
        ps.initialize_model_parallel(backend="gloo")


def _setup(M):
    torch.manual_seed(M)
    mod = MuseSpark1xBf16Experts(
        num_experts=E,
        num_local_experts=E // 2,
        local_expert_offset=E // 2,
        expert_hidden_size=H,
        intermediate_size=I,
        top_k=K,
    ).to(G, torch.bfloat16)
    mod.gate_up_proj.data.normal_(0, H**-0.5)
    mod.down_proj.data.normal_(0, I**-0.5)
    norm = SimpleNamespace(weight=(torch.rand(H, device=G) + 0.5).bfloat16(), eps=1e-8)
    return mod, norm, *_inputs(M)


def _inputs(M):
    x = torch.randn(M, H, device=G, dtype=torch.bfloat16)
    ids = torch.rand(M, E, device=G).argsort(1)[:, :K].int()  # half remote
    return x, ids, torch.rand(M, K, device=G)


def _ref(x, ids, tw, w13, w2, nw):  # fp64; RMS, then the norm weight
    ref = torch.zeros(x.shape, dtype=D, device=G)
    for e in range(E // 2):
        t, j = (ids == E // 2 + e).nonzero(as_tuple=True)
        g, u = (x[t].to(D) @ w13[e].to(D).T).chunk(2, -1)
        y = (g * g.sigmoid() * u) @ w2[e].to(D).T
        y *= (y.square().mean(-1, True) + 1e-8).rsqrt() * nw.to(D) * tw[t, j, None]
        ref.index_add_(0, t, y)
    return ref


@sm90
@pytest.mark.parametrize("M", [1, 100, 300, 4096])  # 4096: grouped path
def test_not_less_accurate_than_stock(M):
    mod, norm, x, ids, tw = _setup(M)
    w13 = mod.gate_up_proj.data.clone()
    mod._triton_runner_config = mod._build_triton_runner_config()
    mod._triton_interleaved = False  # stock: [gate; up] rows, Triton fused_moe
    stock = mod._forward_triton(x, tw, ids, norm)
    mod.prepare_trtllm_weights(norm)  # Hopper: interleave gate / up rows
    gu = mod.gate_up_proj.data
    assert mod._triton_interleaved and torch.equal(gu[:, 0::2], w13[:, :I])
    assert torch.equal(gu[:, 1::2], w13[:, I:])
    fused = mod._forward_triton_interleaved(x, tw, ids, norm, grouped=False)
    ref = _ref(x, ids, tw, w13, mod.down_proj.data, norm.weight)
    err = lambda o: (o.to(D) - ref).norm().item()
    # the SiLU in the up-GEMM epilogue skips stock's BF16 rounding of gate / up
    assert err(fused) <= err(stock) <= 2**-7 * ref.norm().item()
    # grouped rounds where stock does; 1 %: sampling noise of an RMS over 2^24 outputs
    if M >= 4096:
        assert err(mod._forward_triton(x, tw, ids, norm)) <= 1.01 * err(stock)


@sm90
def test_cuda_graph_replay():
    """Decode batches captured once, replayed on new inputs: as eager."""
    mod, norm, x, ids, tw = _setup(64)
    mod.prepare_trtllm_weights(norm)
    run = lambda: mod._forward_triton(x, tw, ids, norm)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(3):
        for t, v in zip((x, ids, tw), _inputs(64)):
            t.copy_(v)
        graph.replay()
        assert torch.equal(out, run())
