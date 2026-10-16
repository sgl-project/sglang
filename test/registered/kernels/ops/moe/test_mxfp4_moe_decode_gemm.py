"""MXFP4 MoE decode GEMM (mxfp4_moe_decode_gemm) and its use in the Marlin LoRA
MoE runner (SGLANG_OPT_MXFP4_MOE_DECODE_GEMM).

The kernel reads Marlin-packed MXFP4 weights in place and differs from Marlin
only in the fp32 summation order over K. Checked against Marlin and against an
fp64 reference built from the exact dequantized weights, at the per-rank decode
shapes of a 896-expert, top-16 MoE at TP16 (hidden 3584, intermediate 192
Marlin-padded to 256).
"""

import pytest
import torch

from sglang.kernels.ops.moe.moe_wna16_marlin import (
    moe_wna16_marlin_gemm,
    mxfp4_moe_decode_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import get_scalar_type
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    moe_align_block_size,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.lora import lora_moe_runner_marlin
from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore
from sglang.srt.lora.lora_moe_runners import LoRAHooks
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, TOP_K, K, N, N_REAL = 896, 16, 3584, 256, 192
GATE_UP_CFG, DOWN_CFG = (1, 8, 4), (4, 1, 4)  # (warps_n, warps_k, stages)

_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
pytestmark = pytest.mark.skipif(not _SUPPORTED, reason="requires sm90+")


def _make_layer(seed: int, extreme_scales: bool = False) -> dict:
    """Random Marlin-packed MXFP4 expert weights with the Marlin padding of the
    intermediate dimension zeroed, as a checkpoint load leaves it."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def bits(*shape):
        return torch.randint(
            -(2**31), 2**31 - 1, shape, generator=g, device="cuda", dtype=torch.int32
        )

    def e8m0(*shape):
        return torch.randint(
            118, 127, shape, generator=g, device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)

    def bias(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 0.05).bfloat16()

    layer = dict(
        w13=bits(E, K // 16, 2 * N * 16 // 8),
        s13=e8m0(E, K // 32, 2 * N),
        b13=bias(E, 2 * N),
        w2=bits(E, N // 16, K * 16 // 8),
        s2=e8m0(E, N // 32, K),
        b2=bias(E, K),
    )
    # Marlin keeps 64-column tiles together: the padded intermediate columns
    # 192..255 of each half of w13 are tiles 3 and 7, and k16 rows 12..15 of w2.
    for tile in (3, 7):
        layer["w13"][:, :, 128 * tile : 128 * (tile + 1)] = 0
        layer["b13"][:, 64 * tile : 64 * (tile + 1)] = 0
    layer["w2"][:, N_REAL // 16 :, :] = 0
    if extreme_scales:
        # E8M0 bytes outside the range the kernel folds the FP4 bias into
        # (0 and 129..135) on 0.5 % of the scales: the exact fallback path
        for key in ("s13", "s2"):
            raw = layer[key].view(torch.uint8)
            pick = torch.rand(raw.shape, generator=g, device="cuda") < 0.005
            ext = torch.randint(
                128, 136, raw.shape, generator=g, device="cuda", dtype=torch.uint8
            )
            raw[pick] = torch.where(ext == 128, torch.zeros_like(ext), ext + 1)[pick]
    layer["scalar_type"] = get_scalar_type(4, False, layer["s13"], None)
    return layer


def _make_inputs(m: int, seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    topk_weights = torch.rand(m, TOP_K, generator=g, device="cuda") + 0.05
    inputs = dict(
        m=m,
        x=torch.randn(m, K, generator=g, device="cuda").bfloat16(),
        inter=torch.randn(m * TOP_K, N, generator=g, device="cuda").bfloat16(),
        topk_ids=torch.rand(m, E, generator=g, device="cuda")
        .topk(TOP_K, dim=1)
        .indices.to(torch.int32),
        topk_weights=topk_weights / topk_weights.sum(1, keepdim=True),
    )
    return _align(inputs)


def _align(inputs: dict) -> dict:
    inputs["sorted"], inputs["expert_ids"], inputs["num_post_padded"] = (
        moe_align_block_size(inputs["topk_ids"], 8, E)
    )
    return inputs


def _gemm_args(layer: dict, inputs: dict, which: str):
    if which == "gate_up":
        return (inputs["x"], layer["w13"], layer["b13"], layer["s13"]), dict(
            top_k=TOP_K,
            mul_topk_weights=False,
            size_m=inputs["m"],
            size_n=2 * N,
            size_k=K,
        )
    return (inputs["inter"], layer["w2"], layer["b2"], layer["s2"]), dict(
        top_k=1, mul_topk_weights=True, size_m=inputs["m"] * TOP_K, size_n=K, size_k=N
    )


def _marlin(layer: dict, inputs: dict, which: str) -> torch.Tensor:
    (a, w, b, s), kw = _gemm_args(layer, inputs, which)
    out = torch.empty(inputs["m"] * TOP_K, kw["size_n"], device="cuda").bfloat16()
    workspace = marlin_make_workspace(torch.device("cuda"), max_blocks_per_sm=4)
    return moe_wna16_marlin_gemm(
        a,
        out,
        w,
        b,
        s,
        None,
        None,
        None,
        None,
        workspace,
        inputs["sorted"],
        inputs["expert_ids"],
        inputs["num_post_padded"],
        inputs["topk_weights"],
        moe_block_size=8,
        is_ep=False,
        b_q_type=layer["scalar_type"],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
        **kw,
    )


def _decode(layer: dict, inputs: dict, which: str, skip_padding=False):
    (a, w, b, s), kw = _gemm_args(layer, inputs, which)
    out = torch.empty(inputs["m"] * TOP_K, kw["size_n"], device="cuda").bfloat16()
    warps_n, warps_k, stages = GATE_UP_CFG if which == "gate_up" else DOWN_CFG
    if skip_padding and which == "gate_up":
        kw["n_half_valid"] = N_REAL
    elif skip_padding:
        kw["k_valid"] = N_REAL
    return mxfp4_moe_decode_gemm(
        a,
        out,
        w,
        b,
        s,
        inputs["sorted"],
        inputs["expert_ids"],
        inputs["num_post_padded"],
        inputs["topk_weights"],
        is_ep=False,
        warps_n=warps_n,
        warps_k=warps_k,
        stages=stages,
        **kw,
    )


def _ulp_stats(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """Error in units of |ref| * 2^-7 (one bf16 ulp up to a factor 2)."""
    r, o = ref.float(), got.float()
    ulps = (r - o).abs() / torch.where(
        r == 0, torch.full_like(r, 2.0**-133), r.abs() * 2.0**-7
    )
    return dict(
        bit_equal=(ref == got).float().mean().item(),
        within1=(ulps <= 1).float().mean().item(),
        within2=(ulps <= 2).float().mean().item(),
        mean_abs=(r - o).abs().mean().item(),
        finite=bool(torch.isfinite(o).all()),
    )


def _marlin_rows(a, w, s, b, expert: int, size_k: int, size_n: int):
    """Marlin output of the rows of ``a`` through one expert."""
    m = a.shape[0]
    ids = torch.full((m, 1), expert, device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(ids, 64, E)
    out = torch.empty(m, size_n, device="cuda").bfloat16()
    return moe_wna16_marlin_gemm(
        a,
        out,
        w,
        b,
        s,
        None,
        None,
        None,
        None,
        marlin_make_workspace(torch.device("cuda"), max_blocks_per_sm=4),
        sorted_ids,
        expert_ids,
        num_post_padded,
        torch.ones(m, 1, device="cuda"),
        moe_block_size=64,
        top_k=1,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=get_scalar_type(4, False, s, None),
        size_m=m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def _exact(layer: dict, inputs: dict, which: str) -> torch.Tensor:
    """fp64 GEMM of the exact dequantized weights, through Marlin's bf16
    epilogue (one rounding of the sum, + bias, * bf16 top-k weight)."""
    (a, w, b, s), kw = _gemm_args(layer, inputs, which)
    size_k, size_n = kw["size_k"], kw["size_n"]
    pairs = inputs["m"] * TOP_K
    experts = inputs["topk_ids"].long().flatten()  # pair p = token * TOP_K + k
    if which == "gate_up":
        a = a[torch.arange(pairs, device="cuda") // TOP_K]
    y = torch.empty(pairs, size_n, device="cuda", dtype=torch.float64)
    bias = torch.empty(pairs, size_n, device="cuda").bfloat16()
    eye = torch.eye(size_k, device="cuda").bfloat16()
    zero = torch.zeros(1, size_k, device="cuda").bfloat16()
    for e in experts.unique().tolist():
        rows = (experts == e).nonzero().flatten()
        # one-hot rows give each weight times 1.0 exactly
        weights = _marlin_rows(eye, w, s, None, e, size_k, size_n).double()
        y[rows] = a[rows].double() @ weights
        bias[rows] = _marlin_rows(zero, w, s, b, e, size_k, size_n)[0]
    out = (y.float().bfloat16().double() + bias.double()).bfloat16()
    if which == "down":
        scale = inputs["topk_weights"].flatten()[:, None].bfloat16().float()
        out = (out.float() * scale).bfloat16()
    return out


@pytest.mark.parametrize("which", ["gate_up", "down"])
@pytest.mark.parametrize("m", [1, 4, 8, 16])
def test_decode_gemm_matches_marlin(which: str, m: int):
    layer, inputs = _make_layer(0), _make_inputs(m, seed=m)
    ref = _marlin(layer, inputs, which)
    got = _decode(layer, inputs, which)
    stats = _ulp_stats(ref, got)
    print(f"{which} m={m} vs Marlin: {stats}")
    assert stats["finite"] and stats["within2"] > 0.999
    # no output is shared between threadblocks: run-to-run identical, and the
    # padding skip reads the same nonzero weights
    assert torch.equal(got, _decode(layer, inputs, which))
    assert torch.equal(got, _decode(layer, inputs, which, skip_padding=True))


@pytest.mark.parametrize("which", ["gate_up", "down"])
@pytest.mark.parametrize("m", [8, 16])
def test_decode_gemm_vs_exact_reference(which: str, m: int):
    layer, inputs = _make_layer(1), _make_inputs(m, seed=100 + m)
    exact = _exact(layer, inputs, which)
    marlin = _ulp_stats(exact, _marlin(layer, inputs, which))
    decode = _ulp_stats(exact, _decode(layer, inputs, which))
    print(f"{which} m={m} vs exact: Marlin {marlin}\n    decode {decode}")
    assert decode["within2"] > 0.9999
    # as close to the exact result as Marlin, up to single elements
    assert decode["within1"] >= marlin["within1"] - 1e-5


@pytest.mark.parametrize("which", ["gate_up", "down"])
def test_decode_gemm_batch_invariant(which: str):
    layer, inputs = _make_layer(2), _make_inputs(16, seed=3)
    alone = _align(
        dict(
            m=1,
            x=inputs["x"][:1].clone(),
            inter=inputs["inter"][:TOP_K].clone(),
            topk_ids=inputs["topk_ids"][:1].clone(),
            topk_weights=inputs["topk_weights"][:1].clone(),
        )
    )
    in_batch = _decode(layer, inputs, which)[:TOP_K]
    assert torch.equal(in_batch, _decode(layer, alone, which))


@pytest.mark.parametrize("which", ["gate_up", "down"])
def test_decode_gemm_extreme_scales(which: str):
    layer, inputs = _make_layer(3, extreme_scales=True), _make_inputs(8, seed=4)
    stats = _ulp_stats(_marlin(layer, inputs, which), _decode(layer, inputs, which))
    print(f"{which} extreme scales vs Marlin: {stats}")
    assert stats["finite"] and stats["within2"] > 0.999


_RUNNER_CFG = dict(
    num_experts=E,
    num_local_experts=E,
    hidden_size=K,
    top_k=TOP_K,
    activation="situ",
    is_gated=True,
    gemm1_alpha=4.0,
    gemm1_clamp_limit=25.0,
)


def _run_runner(monkeypatch, layer, inputs, *, decode_gemm: bool, inter_real=N_REAL):
    monkeypatch.setattr(lora_moe_runner_marlin, "_MXFP4_DECODE_GEMM", decode_gemm)
    config = MoeRunnerConfig(intermediate_size_per_partition=inter_real, **_RUNNER_CFG)
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=layer["w13"],
        w2_qweight=layer["w2"],
        w13_scales=layer["s13"],
        w2_scales=layer["s2"],
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        w13_bias=layer["b13"],
        w2_bias=layer["b2"],
    )
    dispatch = StandardDispatchOutput(
        hidden_states=inputs["x"],
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            inputs["topk_weights"], inputs["topk_ids"], None
        ),
    )
    return (
        MarlinLoraRunnerCore(config)
        .run_from_dispatch(dispatch, quant_info, config, hooks=LoRAHooks())
        .hidden_states
    )


@pytest.mark.parametrize("m", [1, 4, 8, 16])
def test_runner_decode_path(monkeypatch, m: int):
    monkeypatch.setattr(lora_moe_runner_marlin, "_zero_padding", {})
    layer, inputs = _make_layer(4), _make_inputs(m, seed=200 + m)
    ref = _run_runner(monkeypatch, layer, inputs, decode_gemm=False).clone()
    got = _run_runner(monkeypatch, layer, inputs, decode_gemm=True).clone()
    stats = _ulp_stats(ref, got)
    print(f"runner m={m} decode GEMM vs Marlin: {stats}")
    assert stats["finite"] and stats["within2"] > 0.999
    assert list(lora_moe_runner_marlin._zero_padding.values()) == [True]
    # skipping the zero padding is bit-identical to reading it
    padded = _run_runner(monkeypatch, layer, inputs, decode_gemm=True, inter_real=N)
    assert torch.equal(got, padded)
    # the padding check ran eagerly above; a captured graph replays the same bits
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run_runner(monkeypatch, layer, inputs, decode_gemm=True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _run_runner(monkeypatch, layer, inputs, decode_gemm=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, got)


def test_runner_nonzero_padding_is_read(monkeypatch):
    """A layer whose Marlin padding is not zero keeps the padded size."""
    monkeypatch.setattr(lora_moe_runner_marlin, "_zero_padding", {})
    layer = _make_layer(5)
    layer["w13"][:, :, 128 * 3] = 0x11111111
    layer["w2"][:, N_REAL // 16, 0] = 0x22222222
    inputs = _make_inputs(8, seed=99)
    got = _run_runner(monkeypatch, layer, inputs, decode_gemm=True)
    assert list(lora_moe_runner_marlin._zero_padding.values()) == [False]
    padded = _run_runner(monkeypatch, layer, inputs, decode_gemm=True, inter_real=N)
    assert torch.equal(got, padded)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
