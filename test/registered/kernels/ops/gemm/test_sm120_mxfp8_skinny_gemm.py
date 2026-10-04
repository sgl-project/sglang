"""SM120 small-M MXFP8 GEMM: FP64 reference, bit-identity with FlashInfer's
activation quantization, determinism, batch invariance, CUDA-graph replay, and
dispatch through Fp8LinearMethod.apply."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip("SM120 MXFP8 skinny GEMM requires SM 12.0.", allow_module_level=True)

from sglang.kernels.ops.gemm.sm120_mxfp8_skinny_gemm import (  # noqa: E402
    MAX_M,
    mxfp8_skinny_gemm,
    tuned_config,
)

# The tuned configs: split-K (BN 32), BN 64, BK 256. Engram wkv (25600 x 6144)
# shares the BN 64 config and is left out for memory.
SHAPES = [
    (1792, 5120),
    (8192, 1280),
    (5120, 2048),
    (1280, 5120),
    (512, 5120),
    (5120, 15360),
]


def _weight(n, k):
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).mul_(0.02)
    w = w.to(torch.float8_e4m3fn)
    e = torch.randint(-12, -4, ((n + 31) // 32, k // 32), device="cuda").float()
    return w, torch.pow(2.0, e)


def _counters(n, k):
    bn, split, _ = tuned_config(n, k)
    if split == 1:
        return None
    return torch.zeros((n + bn - 1) // bn, dtype=torch.int32, device="cuda")


def _dequant_x(x):
    """MXFP8 round trip: scale = amax / 448 rounded up to a power of two."""
    m, k = x.shape
    xb = x.float().view(m, k // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(min=-127)
    e = torch.where(amax > 0, e, torch.full_like(e, -127))
    q = (xb * torch.pow(2.0, -e)).to(torch.float8_e4m3fn)
    return (q.double() * torch.pow(2.0, e).double()).view(m, k)


def _reference(x, w, s, chunk=256):
    """FP64 product of the dequantized MXFP8 operands, N in chunks of 32-row blocks."""
    xd = _dequant_x(x)
    out = []
    for n0 in range(0, w.shape[0], chunk):
        wc = w[n0 : n0 + chunk].double()
        sc = s[n0 // 32 : (n0 + chunk) // 32].double().repeat_interleave(32, 0)
        sc = sc.repeat_interleave(32, 1)[: wc.shape[0]]
        out.append(xd @ (wc * sc).t())
    return torch.cat(out, dim=1)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 6, 16, 48])
def test_matches_reference(n, k, m):
    torch.manual_seed(m * 7 + n)
    w, s = _weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    counters = _counters(n, k)
    out = mxfp8_skinny_gemm(x, w, s, counters)
    ref = _reference(x, w, s)
    err = (out.double() - ref).abs().max() / ref.abs().max()
    assert err < 8e-3, err
    if counters is not None:
        assert not counters.any(), "split-K counters must return to zero"


@pytest.mark.parametrize("backend", ["cute-dsl", "cuda"])
@pytest.mark.parametrize("m", [1, 6, 16, 48, 128])
def test_fused_quant_matches_flashinfer(m, backend):
    flashinfer = pytest.importorskip("flashinfer")
    n, k = 8192, 1280
    # FlashInfer's CUDA backend flushes BF16 subnormal inputs to zero; the
    # CuTe-DSL backend (the one SGLang calls) and this kernel keep them.
    lo = -120 if backend == "cute-dsl" else -100
    x = torch.randn(m, k, device="cuda")
    x *= torch.logspace(lo, 30, m, base=2.0, device="cuda")[:, None]
    x = x.bfloat16()
    x[0, 0:32] = 0
    x[0, 32:64] = 3.5  # amax / 448 is exactly 2^-7
    x[0, 64:96] = 448.0 * 2.0**-127  # amax / 448 at the smallest UE8M0 scale
    try:
        q, sf = flashinfer.mxfp8_quantize(x, True, 32, backend=backend)
        q_lin, sf_lin = flashinfer.mxfp8_quantize(x, False, 32, backend=backend)
    except RuntimeError as e:
        pytest.skip(f"FlashInfer {backend} quantize unavailable: {e}")
    # Selector weight (row j picks activation j) with block scale 2^64:
    # out[:, :K] is the activation as the GEMM saw it, exact in FP32 for every
    # BF16 input, so this compares operands.
    w = torch.zeros(n, k, device="cuda")
    w[:k] = torch.eye(k, device="cuda")
    w = w.to(torch.float8_e4m3fn)
    s = torch.full((n // 32, k // 32), 2.0**64, device="cuda")
    fused = mxfp8_skinny_gemm(x, w, s, out_dtype=torch.float32)[:, :k]
    prequant = mxfp8_skinny_gemm(q, w, s, a_sf=sf, out_dtype=torch.float32)[:, :k]
    sf_lin = sf_lin.view(m, k // 32)[:, :, None].float()
    expect = q_lin.float().view(m, k // 32, 32) * torch.exp2(sf_lin - 127 + 64)
    assert torch.equal(prequant, expect.view(m, k))
    assert torch.equal(fused, prequant)


@pytest.mark.parametrize("n,k", SHAPES)
def test_deterministic_and_batch_invariant(n, k):
    w, s = _weight(n, k)
    x = torch.randn(MAX_M, k, device="cuda", dtype=torch.bfloat16)
    counters = _counters(n, k)
    first = mxfp8_skinny_gemm(x, w, s, counters)
    for _ in range(50):
        assert torch.equal(mxfp8_skinny_gemm(x, w, s, counters), first)
    for m in range(1, MAX_M):
        assert torch.equal(mxfp8_skinny_gemm(x[:m], w, s, counters), first[:m]), m


def test_cuda_graph_replay():
    n, k, m = 1792, 5120, 6
    assert tuned_config(n, k)[1] > 1, "exercise the split-K reduction"
    w, s = _weight(n, k)
    counters = _counters(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mxfp8_skinny_gemm(x, w, s, counters)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mxfp8_skinny_gemm(x, w, s, counters)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(out, mxfp8_skinny_gemm(x, w, s, counters))
    assert not counters.any()


@pytest.fixture
def build_linear(monkeypatch):
    """A real block-FP8 (32x32, UE8M0) ColumnParallelLinear served as MXFP8."""
    from sglang.srt import runtime_context
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.test.layer_ut_utils import (
        init_single_process_dist,
        load_linear_weights,
        make_tp1_column_parallel_linear,
    )

    init_single_process_dist()
    monkeypatch.setattr(
        fp8_utils,
        "FP8_GEMM_RUNNER_BACKEND",
        fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
    )

    def build(n, k, deterministic=False, block_n=32):
        cfg = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )
        monkeypatch.setattr(runtime_context, "get_exec", lambda: cfg)
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[block_n, 32],
            scale_fmt="ue8m0",
        )
        layer = make_tp1_column_parallel_linear(
            quant_config, n, k, skip_block_quant_check=True
        )
        w, s = _weight(n, k)
        s = s[:: block_n // 32].contiguous()
        load_linear_weights(
            layer, weight=w, weight_scale_inv=s.to(torch.float8_e8m0fnu)
        )
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.block_fp8_mxfp8_ready
        return layer, s

    return build


@pytest.fixture
def skinny_rows(monkeypatch):
    """Row counts of the calls that reach the skinny kernel."""
    from sglang.kernels.ops.gemm import sm120_mxfp8_skinny_gemm as module

    rows = []

    def counted(a, *args, **kwargs):
        rows.append(a.shape[0])
        return mxfp8_skinny_gemm(a, *args, **kwargs)

    monkeypatch.setattr(module, "mxfp8_skinny_gemm", counted)
    return rows


@pytest.mark.parametrize("n,k", [(1792, 5120), (8192, 1280)])
def test_linear_apply(build_linear, skinny_rows, n, k):
    flashinfer = pytest.importorskip("flashinfer")
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    layer, s = build_linear(n, k)
    split = tuned_config(n, k)[1]
    assert (layer.mxfp8_skinny_counters is not None) == (split > 1)
    x = torch.randn(MAX_M + 1, k, device="cuda", dtype=torch.bfloat16)
    cutlass = layer(x)[0]  # above MAX_M: the existing CUTLASS path
    assert skinny_rows == []
    for m in (1, 6, MAX_M):
        out = layer(x[:m])[0]
        assert torch.equal(
            out, mxfp8_skinny_gemm(x[:m], layer.weight, s.float(), _counters(n, k))
        )
        err = (
            out.float() - cutlass[:m].float()
        ).abs().max() / cutlass.float().abs().max()
        assert err < 1e-2, (m, err)
    q, sf = flashinfer.mxfp8_quantize(x[:6], True, 32)
    out_q = layer.quant_method.apply(layer, Mxfp8SwizzledInput(q, sf))
    assert torch.equal(out_q, layer(x[:6])[0])
    out_3d = layer(x[:6].view(2, 3, k))[0]
    assert torch.equal(out_3d.view(6, n), layer(x[:6])[0])
    assert skinny_rows == [1, 6, MAX_M, 6, 6, 6, 6]


def test_linear_gates(build_linear, skinny_rows):
    from sglang.srt.environ import envs

    x = torch.randn(6, 2048, device="cuda", dtype=torch.bfloat16)
    # (N, block N, deterministic inference, env switch): untuned shape, 128x32
    # blocks, the two off switches, then the tuned shape with all gates open.
    for n, block_n, deterministic, enabled in (
        (1024, 32, False, True),
        (5120, 128, False, True),
        (5120, 32, True, True),
        (5120, 32, False, False),
        (5120, 32, False, True),
    ):
        with envs.SGLANG_ENABLE_SM120_MXFP8_SKINNY_GEMM.override(enabled):
            layer, _ = build_linear(n, 2048, deterministic, block_n)
        layer(x)
    assert skinny_rows == [6]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
