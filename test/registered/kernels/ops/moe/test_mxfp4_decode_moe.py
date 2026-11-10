"""MXFP4 decode MoE kernels (``mxfp4_decode_moe``): FC2 rows against an fp64
reference on random MXFP4 weights for the Kimi-K3.1 and DeepSeek-V4 shapes,
EP offsets with non-local and -1 ids, the packed top-k handoff, flashinfer's
SwiGLU-limit MoE, determinism and CUDA graph replays."""

import functools
import sys

import msgspec
import pytest
import torch

from sglang.kernels.arch import current_arch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Mean error relative to the reference RMS. The kernel and flashinfer both land
# near 1.1e-3 on these inputs (MXFP8 activations and intermediate).
MEAN_REL_TOL = 3e-3
# Rows of the kernel vs flashinfer's rows (same MXFP8 intermediate).
VS_FLASHINFER_TOL = 2e-3
SWIGLU_LIMIT = 10.0


class Shape(msgspec.Struct, frozen=True):
    name: str
    num_experts: int  # local
    global_experts: int
    top_k: int
    hidden: int
    inter: int  # per rank, padded to 128
    swiglu: bool  # SwiGLU limit 10 (DeepSeek-V4), else SiTU (Kimi-K3.1)
    inter_real: int = 0  # unpadded intermediate (0: = inter)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(0, self.global_experts, self.num_experts))


SHAPES = [
    Shape("k31_tp8", 896, 896, 16, 3584, 384, False),
    Shape("k31_tp4ep2", 448, 896, 16, 3584, 768, False),
    Shape("v41flash_tp4", 384, 384, 6, 5120, 640, True, inter_real=576),
    Shape("v4flash_tp4", 256, 256, 6, 4096, 512, True),
    Shape("v4flash_tp8", 256, 256, 6, 4096, 256, True),
    Shape("v4pro_tp4", 384, 384, 6, 7168, 768, True),
    Shape("v4pro_tp8", 384, 384, 6, 7168, 384, True),
]
SHAPE_IDS = [s.name for s in SHAPES]
BLOCK_NS = [16, 32]
TOKENS = [1, 7, 48, 96, 256]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = current_arch()
    return arch.is_sm(10, 0) or arch.is_sm(10, 3)


pytestmark = pytest.mark.skipif(not _supported(), reason="SM100 / SM103 kernels")


def _op():
    from sglang.kernels.ops.moe import mxfp4_decode_moe

    return mxfp4_decode_moe


def _activation(shape: Shape):
    return ("swiglu", (SWIGLU_LIMIT,)) if shape.swiglu else ("situ", (4.0, 25.0))


@pytest.fixture(scope="module", autouse=True)
def _prebuild():
    if not _supported():
        return
    from sglang.kernels.jit.utils.compile.parallel import parallel_compile

    op = _op()
    parallel_compile(
        [
            functools.partial(
                op._jit_module,
                s.num_experts,
                s.top_k,
                s.hidden,
                s.inter,
                bn,
                *_activation(s),
            )
            for s in SHAPES
            for bn in BLOCK_NS
        ]
    )


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float64)


def _dequant_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] (element 2j in the low nibble), E8M0 [..., K/32] -> f64 [..., K]."""
    lut = _E2M1.to(packed.device)
    v = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1)
    s = torch.pow(2.0, scales.double() - 127.0)
    return (v.flatten(-2).unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2)


def _dequant_mxfp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    sc = torch.pow(2.0, s.double() - 127.0)
    return (q.double().unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2)


def _quant_dequant_mxfp8(a: torch.Tensor) -> torch.Tensor:
    """Blocks of 32: UE8M0 scale 2^ceil(log2(amax / 448)), E4M3 round to nearest."""
    blk = a.unflatten(-1, (-1, 32))
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blk / torch.pow(2.0, e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.double() * torch.pow(2.0, e)).flatten(-2)


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * (25.0 * torch.tanh(up / 25.0))
    )


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = gate.clamp(max=SWIGLU_LIMIT)
    return g * torch.sigmoid(g) * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)


@functools.cache
def _weights(shape: Shape):
    """Random checkpoint-layout MXFP4 weights (w13 rows: [gate; up]) and the
    flashinfer_mxfp4 layout the engine prepares from them: the K3.1 path
    (Mxfp4MoEMethod: explicit (up_i, gate_i) pairs + the w2 permute) or the
    DeepSeek-V4 path (Mxfp4FlashinferTrtllmMoEMethod: [w3; w1] halves + the
    w3_w1 permute), so the test also checks that both preps agree."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    E, H, I = shape.num_experts, shape.hidden, shape.inter
    g = torch.Generator(device="cuda").manual_seed(shape.hidden + shape.inter + E)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    # SiTU: FC1 outputs O(1). SwiGLU: FC1 outputs O(10), so the limit clamps a share of them.
    lo13 = (118 if shape.swiglu else 117) - (H // 3584 > 1)
    s13 = torch.randint(
        lo13 + (2 if shape.swiglu else 0),
        lo13 + (7 if shape.swiglu else 5),
        (E, 2 * I, H // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    s2 = torch.randint(
        113 if shape.swiglu else 118,
        118 if shape.swiglu else 123,
        (E, H, I // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    if shape.inter_real and shape.inter_real < I:
        # as _pad_intermediate_size: gate and up halves each get a zero tail (scale 1.0)
        r = shape.inter_real
        for half in (0, I):
            w13[:, half + r : half + I] = 0
            s13[:, half + r : half + I] = 127
        w2[:, :, r // 2 :] = 0
        s2[:, :, r // 32 :] = 127

    cache = {}
    if shape.swiglu:
        pw13 = torch.cat([w13[:, I:], w13[:, :I]], dim=1)  # reorder_w1w3_to_w3w1
        ps13 = torch.cat([s13[:, I:], s13[:, :I]], dim=1)
        p13 = _maybe_get_cached_w3_w1_permute_indices(cache, pw13[0], 128).cuda()
        q13 = _maybe_get_cached_w3_w1_permute_indices(
            cache, ps13[0], 128, num_elts_per_sf=16
        ).cuda()
    else:
        pair = torch.empty(2 * I, dtype=torch.long)
        pair[0::2] = torch.arange(I) + I
        pair[1::2] = torch.arange(I)
        pw13, ps13 = w13[:, pair.cuda()], s13[:, pair.cuda()]
        p13 = get_w2_permute_indices_with_cache(cache, pw13[0], 128).cuda()
        q13 = get_w2_permute_indices_with_cache(
            cache, ps13[0], 128, num_elts_per_sf=16
        ).cuda()
    p2 = get_w2_permute_indices_with_cache(cache, w2[0], 128).cuda()
    q2 = get_w2_permute_indices_with_cache(cache, s2[0], 128, num_elts_per_sf=16).cuda()
    fw13, fw2 = pw13[:, p13].contiguous(), w2[:, p2].contiguous()
    fs13, fs2 = torch.empty_like(ps13), torch.empty_like(s2)
    for e in range(E):
        fs13[e] = nvfp4_block_scale_interleave(ps13[e][q13].contiguous()).reshape(
            2 * I, H // 32
        )
        fs2[e] = nvfp4_block_scale_interleave(s2[e][q2].contiguous()).reshape(
            H, I // 32
        )
    return (w13, s13, w2, s2), (fw13, fs13, fw2, fs2)


def _inputs(shape: Shape, num_tokens: int, seed: int, invalid: float = 0.0):
    """MXFP8 activations and a top-k routing over the global experts; a share
    `invalid` of the slots is replaced by -1."""
    from flashinfer import mxfp8_quantize

    H = shape.hidden
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, H, generator=g).to(torch.bfloat16).cuda()
    xq, xs = mxfp8_quantize(x, False)
    xs = xs.view(torch.uint8).reshape(num_tokens, H // 32)
    scores = torch.rand(num_tokens, shape.global_experts, generator=g)
    ids = scores.topk(shape.top_k, dim=1).indices.to(torch.int32)
    if invalid > 0:
        ids[torch.rand(ids.shape, generator=g) < invalid] = -1
    w = torch.softmax(torch.randn(num_tokens, shape.top_k, generator=g), dim=1)
    return xq, xs, ids.cuda(), w.cuda()


def _reference_rows(shape: Shape, xq, xs, ids, offset: int) -> torch.Tensor:
    """FC2 output of every (token, k) slot, unweighted, [T * top-k, H] f64; NaN
    rows for slots whose id is not a local expert."""
    (w13, s13, w2, s2), _ = _weights(shape)
    I, K = shape.inter, shape.top_k
    act_fn = _swiglu if shape.swiglu else _situ
    x = _dequant_mxfp8(xq, xs)
    flat = ids.flatten().long() - offset
    valid = (flat >= 0) & (flat < shape.num_experts)
    act = torch.zeros(flat.numel(), I, dtype=torch.float64, device="cuda")
    y = torch.full(
        (flat.numel(), shape.hidden), float("nan"), dtype=torch.float64, device="cuda"
    )
    experts = torch.unique(flat[valid]).tolist()
    for e in experts:
        rows = (flat == e).nonzero().flatten()
        h = x[rows // K] @ _dequant_fp4(w13[e], s13[e]).T
        act[rows] = act_fn(h[:, :I], h[:, I:])
    act = _quant_dequant_mxfp8(act)
    for e in experts:
        rows = (flat == e).nonzero().flatten()
        y[rows] = act[rows] @ _dequant_fp4(w2[e], s2[e]).T
    return y


def _pack(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wbits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return ((ids << 16) | wbits).contiguous()


def _run(shape: Shape, xq, xs, ids, w, *, packed: bool, offset: int, block_n: int):
    _, (fw13, fs13, fw2, fs2) = _weights(shape)
    act, constants = _activation(shape)
    return _op().mxfp4_decode_moe(
        xq,
        xs,
        _pack(ids, w) if packed else ids,
        fw13,
        fs13,
        fw2,
        fs2,
        activation=act,
        activation_constants=constants,
        expert_offset=offset,
        packed=packed,
        block_n=block_n,
    )


def _mean_rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).abs().mean() / ref.pow(2).mean().sqrt()).item()


def _check(shape: Shape, gemm2_out, expanded_idx, w, ref_rows) -> None:
    valid = ~ref_rows[:, 0].isnan()
    num_valid = int(valid.sum())
    idx = expanded_idx.long()
    assert torch.all(idx[~valid] == -1), "non-local slot without row -1"
    assert torch.equal(
        torch.sort(idx[valid])[0], torch.arange(num_valid, device="cuda")
    ), "expanded_idx of the local slots is not a permutation"
    if num_valid == 0:
        return
    rows = gemm2_out[idx[valid]]
    assert _mean_rel(rows, ref_rows[valid]) < MEAN_REL_TOL
    # the engine's finalize: sum_k w[t, k] * row (t, k), weight 0 for row -1
    wv = w.double().flatten() * valid
    full = torch.zeros_like(ref_rows)
    full[valid] = rows.double()
    ref = torch.where(valid.unsqueeze(1), ref_rows, 0.0)
    K, H = shape.top_k, shape.hidden
    out = (full * wv.unsqueeze(1)).view(-1, K, H).sum(1)
    ref_out = (ref * wv.unsqueeze(1)).view(-1, K, H).sum(1)
    assert _mean_rel(out, ref_out) < MEAN_REL_TOL


def _cases(shape: Shape):
    """(offset, share of -1 ids): EP shapes run every rank's offset with -1 ids."""
    if shape.num_experts < shape.global_experts:
        return [(o, 0.05) for o in shape.offsets]
    return [(0, 0.0), (0, 0.05)]


@pytest.mark.parametrize("block_n", BLOCK_NS)
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mxfp4_decode_moe(shape: Shape, num_tokens: int, block_n: int) -> None:
    for i, (offset, invalid) in enumerate(_cases(shape)):
        packed = (num_tokens + i) % 2 == 1  # both forms over the T sweep
        xq, xs, ids, w = _inputs(shape, num_tokens, num_tokens * 10 + i, invalid)
        gemm2_out, expanded_idx, weights = _run(
            shape, xq, xs, ids, w, packed=packed, offset=offset, block_n=block_n
        )
        torch.cuda.synchronize()
        if packed:
            assert torch.equal(weights, w.to(torch.bfloat16))
        _check(
            shape,
            gemm2_out,
            expanded_idx,
            w,
            _reference_rows(shape, xq, xs, ids, offset),
        )


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mxfp4_decode_moe_deterministic(shape: Shape) -> None:
    offset, invalid = _cases(shape)[-1]
    xq, xs, ids, w = _inputs(shape, 96, seed=7, invalid=invalid)
    a = _run(shape, xq, xs, ids, w, packed=False, offset=offset, block_n=16)
    b = _run(shape, xq, xs, ids, w, packed=False, offset=offset, block_n=16)
    torch.cuda.synchronize()
    assert torch.equal(a[1], b[1])
    n = int((a[1] >= 0).sum())
    assert torch.equal(a[0][:n], b[0][:n])


@pytest.mark.parametrize("num_tokens", [7, 96, 256])
@pytest.mark.parametrize("shape", [s for s in SHAPES if s.swiglu], ids=lambda s: s.name)
def test_mxfp4_decode_moe_vs_flashinfer(shape: Shape, num_tokens: int) -> None:
    """DeepSeek-V4: rows against flashinfer's trtllm-gen MXFP4 MoE with the
    engine's SwiGLU parameters (gemm1_clamp_limit = swiglu_limit, alpha / beta
    None, unit output scales) and deferred finalize."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import RoutingMethodType

    _, (fw13, fs13, fw2, fs2) = _weights(shape)
    xq, xs, ids, w = _inputs(shape, num_tokens, seed=1000 + num_tokens)
    E = shape.num_experts
    ones = torch.ones(E, dtype=torch.float32, device="cuda")
    fi_out = trtllm_fp4_block_scale_routed_moe(
        topk_ids=_pack(ids, w),
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs.view(torch.float8_e4m3fn),
        gemm1_weights=fw13,
        gemm1_weights_scale=fs13.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=torch.full_like(ones, SWIGLU_LIMIT),
        gemm2_weights=fw2,
        gemm2_weights_scale=fs2.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        num_experts=shape.global_experts,
        top_k=shape.top_k,
        n_group=1,
        topk_group=1,
        intermediate_size=shape.inter,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.TopK.value,
        do_finalize=False,
    )
    fi_rows, _, fi_idx = fi_out
    gemm2_out, expanded_idx, _ = _run(
        shape, xq, xs, ids, w, packed=False, offset=0, block_n=16
    )
    torch.cuda.synchronize()
    ref = fi_rows[fi_idx.long()].double()
    rel = _mean_rel(gemm2_out[expanded_idx.long()], ref)
    print(f"{shape.name} T={num_tokens}: vs flashinfer mean rel {rel:.2e}")
    assert rel < VS_FLASHINFER_TOL


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mxfp4_decode_moe_graph_replays(shape: Shape) -> None:
    """Many replays of a graph that chains several calls of different sizes: a
    lost handoff between the kernels' roles shows up as a hang or a wrong row."""
    offset, invalid = _cases(shape)[-1]
    cases = [_inputs(shape, t, seed=100 + t, invalid=invalid) for t in (6, 37, 240)]
    for bn in BLOCK_NS:
        run = functools.partial(_run, shape, packed=False, offset=offset, block_n=bn)
        for case in cases:  # warm the allocator outside the capture
            run(*case)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [run(*case) for case in cases]
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        for (xq, xs, ids, w), (gemm2_out, expanded_idx, _) in zip(cases, outs):
            _check(
                shape,
                gemm2_out,
                expanded_idx,
                w,
                _reference_rows(shape, xq, xs, ids, offset),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
