"""Single-kernel MoE router front end vs the stock chain it replaces.

Given the same bf16 logits, the softmax / top-10 / renormalize output must equal the
Triton router bit for bit, the NVFP4 input quantization must equal fp4_quantize
(cute-dsl, linear scales), and the routing tables must equal moe_sort on every entry the
grouped GEMMs read. The fused GEMV reproduces the cuBLAS router GEMM bit for bit; the
tail kernel takes the stock router GEMM's logits.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, EXPERTS, TOP_K = 4096, 512, 10

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)


def _inputs(m: int, seed: int, clustered: bool):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    weight = torch.randn(EXPERTS, HIDDEN, device="cuda", generator=g) * 0.02
    if clustered:
        # Experts 0..7 share a lane of the top-k layout: forces the exact fallback.
        weight[:8] *= 6.0
    h = torch.randn(m, HIDDEN, device="cuda", generator=g)
    h[:, torch.randint(0, HIDDEN, (24,), device="cuda", generator=g)] *= 8.0
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    gs = torch.tensor([300.0], device="cuda")
    return x.bfloat16().contiguous(), weight.bfloat16().contiguous(), gs


@requires_blackwell
@pytest.mark.parametrize("m", [1, 2, 7, 8, 9, 14, 16])
@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("clustered", [False, True])
def test_router_front_matches_stock_chain(m: int, tile: int, clustered: bool) -> None:
    from sglang.kernels.ops.moe.qwen_router_front import (
        RouterFrontBuffers,
        router_front,
    )

    x, weight, gs = _inputs(m, seed=1000 + m, clustered=clustered)
    out = RouterFrontBuffers(m, tile, "cuda")
    for _ in range(3):  # repeated launches reuse the monotonic arrival counter
        router_front(x, weight, gs, out)
    torch.cuda.synchronize()

    # The GEMV follows the cuBLAS router GEMM's accumulation order: logits must be bitwise.
    assert torch.equal(out.logits, torch.nn.functional.linear(x, weight))
    _assert_matches_stock_chain(out, out.logits, x, gs, m, tile)


@requires_blackwell
@pytest.mark.parametrize("m", [1, 7, 9, 16])
def test_router_gemm_quant_matches_stock(m: int) -> None:
    """Logits and NVFP4 input quantization for runners that route internally: bitwise to stock."""
    from sglang.kernels.ops.moe.qwen_router_front import router_gemm_quant
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    x, weight, gs = _inputs(m, seed=4000 + m, clustered=False)
    logits, x_fp4, x_sf = router_gemm_quant(x, weight, gs)
    torch.cuda.synchronize()
    assert torch.equal(logits, torch.nn.functional.linear(x, weight))
    xq, xsf = fp4_quantize(x, gs, 16, False, False)
    assert torch.equal(x_fp4, xq.view(torch.uint8).reshape(m, -1))
    assert torch.equal(x_sf, xsf.view(torch.uint8).reshape(m, -1))


@requires_blackwell
@pytest.mark.parametrize("m", [1, 7, 9, 16])
@pytest.mark.parametrize("tile", [128, 256])
def test_router_tail_matches_stock_chain(m: int, tile: int) -> None:
    """Behind the stock router GEMM the whole front end must be bitwise to the stock chain."""
    from sglang.kernels.ops.moe.qwen_router_front import RouterFrontBuffers, router_tail

    x, weight, gs = _inputs(m, seed=3000 + m, clustered=False)
    logits = torch.nn.functional.linear(x, weight)
    out = RouterFrontBuffers(m, tile, "cuda")
    router_tail(logits, x, gs, out)
    torch.cuda.synchronize()
    _assert_matches_stock_chain(out, logits, x, gs, m, tile)


def _assert_matches_stock_chain(out, logits, x, gs, m: int, tile: int) -> None:
    from flashinfer import fp4_quantize
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        moe_sort,
    )

    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    w_ref, ids_ref = moe_fused_gate(
        logits, None, TOP_K, scoring_func="softmax", renormalize=True
    )
    assert torch.equal(out.topk_ids, ids_ref)
    assert torch.equal(out.topk_weights.view(torch.int32), w_ref.view(torch.int32))

    xq, xsf = fp4_quantize(
        x,
        gs,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_sf_8x4_layout=False,
        enable_pdl=None,
        backend="cute-dsl",
    )
    assert torch.equal(out.x_fp4, xq.view(torch.uint8).reshape(m, -1))
    assert torch.equal(out.x_sf, xsf.view(torch.uint8).reshape(m, -1))

    bufs = allocate_moe_sort_buffers(m, EXPERTS, TOP_K, EXPERTS, tile, "cuda")
    tabs = moe_sort(
        token_selected_experts=out.topk_ids,
        token_final_scales=out.topk_weights,
        num_experts=EXPERTS,
        top_k=TOP_K,
        local_expert_offset=0,
        num_local_experts=EXPERTS,
        tile_tokens_dim=tile,
        **bufs,
    )
    n = int(tabs[5][0])
    assert n == int(out.num_non_exiting_tiles[0])
    assert int(tabs[4][0]) == int(out.total_num_padded_tokens[0])
    assert torch.equal(tabs[0][:n], out.tile_idx_to_expert_idx[:n])
    assert torch.equal(tabs[1][:n], out.tile_idx_to_mn_limit[:n])
    assert torch.equal(tabs[2], out.expanded_idx_to_permuted_idx)
    perm = tabs[2].flatten().long()
    assert torch.equal(tabs[3][perm], out.permuted_idx_to_expanded_idx[perm])


@requires_blackwell
@pytest.mark.parametrize("m", [1, 7, 8])
@pytest.mark.parametrize("n_shared", [256, 512])
def test_router_gemm_quant_shared(m: int, n_shared: int) -> None:
    """The shared gate/up stage: no worse than the served cuBLASLt path against an fp64 reference,
    and it must leave the router logits and the NVFP4 input bitwise unchanged."""
    from flashinfer import bmm_fp8

    from sglang.kernels.ops.moe.qwen_router_front import (
        router_gemm_quant,
        router_gemm_quant_shared,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    x, weight, gs = _inputs(m, seed=5000 + m, clustered=False)
    g = torch.Generator(device="cuda")
    g.manual_seed(6000 + n_shared)
    w_sh = (torch.randn(n_shared, HIDDEN, device="cuda", generator=g) * 64).to(
        torch.float8_e4m3fn
    )
    in_scale = torch.tensor([0.05], device="cuda")
    w_scale = torch.tensor([3.7e-4], device="cuda")
    logits, x_fp4, x_sf, gate_up = router_gemm_quant_shared(
        x, weight, gs, w_sh, in_scale, w_scale
    )
    logits_ref, x_fp4_ref, x_sf_ref = router_gemm_quant(x, weight, gs)
    torch.cuda.synchronize()
    assert torch.equal(logits, logits_ref)
    assert torch.equal(x_fp4, x_fp4_ref) and torch.equal(x_sf, x_sf_ref)

    qx, qs = static_quant_fp8(x, in_scale)
    stock = bmm_fp8(
        qx.unsqueeze(0),
        w_sh.t().unsqueeze(0),
        qs.reshape(1),
        w_scale,
        torch.bfloat16,
        backend="cublas",
    ).view(m, n_shared)
    ref = (qx.double() @ w_sh.double().t()) * (float(in_scale) * float(w_scale))
    err = lambda y: float((y.double() - ref).norm() / ref.norm())
    assert err(gate_up) <= 1.05 * err(stock)


@requires_blackwell
@pytest.mark.parametrize("m", [1, 7, 16])
def test_router_topk_only_matches_full(m: int) -> None:
    """The top-k-only launch must produce the same ids / weights as the full launch."""
    from sglang.kernels.ops.moe.qwen_router_front import (
        RouterFrontBuffers,
        router_front,
        router_topk,
    )

    x, weight, gs = _inputs(m, seed=2000 + m, clustered=False)
    full = RouterFrontBuffers(m, 128, "cuda")
    router_front(x, weight, gs, full)
    arrival = torch.zeros(1, dtype=torch.int32, device="cuda")
    for _ in range(2):
        w, ids = router_topk(x, weight, arrival)
    torch.cuda.synchronize()
    assert torch.equal(ids, full.topk_ids)
    assert torch.equal(w.view(torch.int32), full.topk_weights.view(torch.int32))


def _expected_tables(ids: torch.Tensor, tile: int):
    """trtllm-gen routing tables in the single-block routing kernels' layout: experts in id order, each
    expert's rows tile-padded, tokens inside an expert in ascending order, -1 in the padding rows."""
    m = ids.shape[0]
    flat = ids.flatten().tolist()
    counts = [0] * EXPERTS
    rank = []
    for e in flat:
        rank.append(counts[e])
        counts[e] += 1
    ctas, limits, offset, cta = [], [], [0] * EXPERTS, 0
    for e in range(EXPERTS):
        offset[e] = cta * tile
        for c in range((counts[e] + tile - 1) // tile):
            ctas.append(e)
            limits.append(min((cta + c + 1) * tile, cta * tile + counts[e]))
        cta += (counts[e] + tile - 1) // tile
    e2p = [offset[e] + r for e, r in zip(flat, rank)]
    p2t = [-1] * (cta * tile)
    for i, p in enumerate(e2p):
        p2t[p] = i // TOP_K
    return ctas, limits, e2p, p2t, counts


@requires_blackwell
@pytest.mark.parametrize("m", [1, 2, 4, 5, 7, 14, 16])
@pytest.mark.parametrize("tile", [8, 16, 64])
@pytest.mark.parametrize("early_trigger", [False, True])
def test_router_topk_tables_small(m: int, tile: int, early_trigger: bool) -> None:
    """Same ids / weights as the multi-CTA tables kernel and the single-block routing layout entry for entry."""
    from sglang.kernels.ops.moe.qwen_router_tables import (
        RouterTables,
        router_topk_tables,
        router_topk_tables_small,
    )

    g = torch.Generator(device="cuda")
    g.manual_seed(3000 + m)
    logits = (torch.randn(m, EXPERTS, device="cuda", generator=g) * 1.5).bfloat16()
    logits[0, 100:140] = logits[0].max()  # ties resolve to the smaller expert id
    small, ref = RouterTables(m, tile, "cuda"), RouterTables(m, tile, "cuda")
    for _ in range(2):
        router_topk_tables_small(logits, small, early_trigger=early_trigger)
    router_topk_tables(logits, ref)
    torch.cuda.synchronize()
    assert torch.equal(small.topk_ids, ref.topk_ids)
    assert torch.equal(
        small.expert_weights.view(torch.int16), ref.expert_weights.view(torch.int16)
    )

    ctas, limits, e2p, p2t, counts = _expected_tables(small.topk_ids.cpu(), tile)
    n = len(ctas)
    assert int(small.num_non_exiting_ctas.item()) == n
    assert int(small.total_num_padded_tokens.item()) == n * tile
    assert small.cta_idx_xy_to_batch_idx[:n].tolist() == ctas
    assert small.cta_idx_xy_to_mn_limit[:n].tolist() == limits
    assert small.expanded_idx_to_permuted_idx[: m * TOP_K].tolist() == e2p
    assert small.permuted_idx_to_token_idx[: n * tile].tolist() == p2t
    assert small.num_tokens_per_expert.tolist() == counts


def test_routed_rt_dispatch_order(monkeypatch) -> None:
    """The fused MKM engine takes the deferred-finalize calls it covers before the trtllm runner;
    only ROUTE_PARQ admits calls above the router front, and those skip the fused front."""
    import contextlib
    from types import SimpleNamespace

    from sglang.kernels.ops.moe import qwen_router_front as kernels
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import qwen_router_front as front
    from sglang.srt.model_executor import runner

    calls = []

    def record(name):
        def fn(*args, **kwargs):
            calls.append(name)
            return "logits", "x_fp4", "x_sf"

        return fn

    monkeypatch.setattr(kernels, "router_gemm_quant", record("front"))
    monkeypatch.setattr(front, "_logits_and_quant_parallel", record("parq"))
    monkeypatch.setattr(front, "_forward_mkm_fused", record("mkm"))
    monkeypatch.setattr(front, "_run_rt_experts", record("runner"))
    monkeypatch.setattr(front, "_use_shared", lambda *args: False)
    block = SimpleNamespace(
        experts=SimpleNamespace(w13_input_scale_quant=torch.ones(())),
        gate=SimpleNamespace(weight=None),
    )
    for m, defer, mkm, expected in (
        (7, True, True, ["front", "mkm"]),
        (7, False, True, ["front", "runner"]),
        (16, True, False, ["front", "runner"]),
        (56, True, True, ["parq", "runner"]),
    ):
        calls.clear()
        monkeypatch.setattr(front, "_mkm_supported", lambda *args, on=mkm: on)
        hidden = torch.zeros(m, 8, dtype=torch.bfloat16)
        front.forward_routed_rt(block, hidden, defer_finalize=defer)
        assert calls == expected, (m, defer, mkm)
    # At the ROUTE_TABLES / ROUTE2 defaults the runner keeps FlashInfer's own routing.
    for m in (1, 16, 56):
        assert isinstance(front._tables_context(m), contextlib.nullcontext), m

    # Without ROUTE_PARQ, 17..21-token calls stay off this path (MKM_FUSED_M21 takes them).
    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: True)
    block = SimpleNamespace(_route_rt_static_ok=True, alt_stream=object())
    hidden = torch.zeros(17, 8, dtype=torch.bfloat16)
    with envs.SGLANG_Q35MF_ROUTE_PARQ.override(False):
        assert not front.supported_rt(block, hidden)
    with envs.SGLANG_Q35MF_ROUTE_PARQ.override(True):
        assert front.supported_rt(block, hidden)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
