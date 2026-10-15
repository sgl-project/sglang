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
    from sglang.kernels.ops.moe.qwen_router_front import RouterFrontBuffers, router_front

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
    from flashinfer.fused_moe.cute_dsl.moe_utils import allocate_moe_sort_buffers, moe_sort

    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    w_ref, ids_ref = moe_fused_gate(logits, None, TOP_K, scoring_func="softmax", renormalize=True)
    assert torch.equal(out.topk_ids, ids_ref)
    assert torch.equal(out.topk_weights.view(torch.int32), w_ref.view(torch.int32))

    xq, xsf = fp4_quantize(x, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=False,
                           is_sf_8x4_layout=False, enable_pdl=None, backend="cute-dsl")
    assert torch.equal(out.x_fp4, xq.view(torch.uint8).reshape(m, -1))
    assert torch.equal(out.x_sf, xsf.view(torch.uint8).reshape(m, -1))

    bufs = allocate_moe_sort_buffers(m, EXPERTS, TOP_K, EXPERTS, tile, "cuda")
    tabs = moe_sort(token_selected_experts=out.topk_ids, token_final_scales=out.topk_weights,
                    num_experts=EXPERTS, top_k=TOP_K, local_expert_offset=0,
                    num_local_experts=EXPERTS, tile_tokens_dim=tile, **bufs)
    n = int(tabs[5][0])
    assert n == int(out.num_non_exiting_tiles[0])
    assert int(tabs[4][0]) == int(out.total_num_padded_tokens[0])
    assert torch.equal(tabs[0][:n], out.tile_idx_to_expert_idx[:n])
    assert torch.equal(tabs[1][:n], out.tile_idx_to_mn_limit[:n])
    assert torch.equal(tabs[2], out.expanded_idx_to_permuted_idx)
    perm = tabs[2].flatten().long()
    assert torch.equal(tabs[3][perm], out.permuted_idx_to_expanded_idx[perm])


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
