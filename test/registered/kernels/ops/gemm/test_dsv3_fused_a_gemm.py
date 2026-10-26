"""Tests for BF16 and block-FP8 JIT fused-A GEMMs."""

import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
)
from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm,
    dsv3_fused_a_gemm_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-gb300")

# hd_in must be a multiple of 256; 2048/6144/7168 cover the real fused-A shapes
# (2048 is q_b_proj TP4/TP8, 6144/7168 are qkv_a).
HD_INS = [2048, 6144, 7168]
# hd_out must be a multiple of 16; 2048/2112/2624/4096 cover real fused-A variants
# (2048/4096 are q_b_proj TP8/TP4, 2112/2624 are qkv_a).
HD_OUTS = [2048, 2112, 2624, 4096]
NUM_TOKENS = get_ci_test_range(list(range(1, 17)), [1, 8, 16])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hd_out", HD_OUTS)
@pytest.mark.parametrize("hd_in", HD_INS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_dsv3_fused_a_gemm(num_tokens, hd_in, hd_out):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    mat_a = torch.randn(num_tokens, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(hd_out, hd_in, dtype=torch.bfloat16, device="cuda").transpose(
        0, 1
    )

    ref = F.linear(mat_a, mat_b.T)
    out = dsv3_fused_a_gemm(mat_a, mat_b)

    assert out.shape == (num_tokens, hd_out)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available() or is_hip_runtime():
        pytest.skip("CUDA required")


@pytest.fixture
def cuda_sm103(cuda_device):
    arch = get_jit_cuda_arch()
    if (arch.major, arch.minor) != (10, 3):
        pytest.skip("SM103 (GB300) required")


def _fp8_case(m, n, k, scale_format):
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = (16 * torch.randn((n, k), device="cuda")).to(torch.float8_e4m3fn)
    shape = ((n + 127) // 128, k // 128)
    if scale_format == "fp32":
        scales = torch.rand(shape, device="cuda") / 64 + 1 / 128
        stored = scales
    else:
        exponents = torch.randint(119, 125, shape, device="cuda", dtype=torch.int32)
        scales = torch.exp2(exponents.float() - 127)
        # Build the packed layout independently of the production scale transform.
        # Poison unused K bytes to expose reading beyond the last block.
        padded = torch.full(
            (n, (k // 128 + 3) // 4 * 4),
            255,
            device="cuda",
            dtype=torch.uint8,
        )
        padded[:, : k // 128] = exponents.repeat_interleave(128, dim=0)[:n]
        packed = padded.view(torch.int32)
        stored = packed.T.contiguous().T
    effective_w = w.double() * scales.double().repeat_interleave(128, dim=0)[
        :n
    ].repeat_interleave(128, dim=1)
    return x, w, stored, effective_w


@pytest.mark.usefixtures("cuda_sm103")
@pytest.mark.parametrize("scale_format", ["fp32", "ue8m0"])
@pytest.mark.parametrize("m", [1, 3, 8, 9, 16])
@pytest.mark.parametrize(
    "n,k",
    [
        (16, 128),
        (144, 640),
        (2624, 6144),
        (4096, 2048),
        (2112, 7168),
        (6144, 4096),
        (1024, 6144),
        (6144, 512),
    ],
)
def test_fp8_fused_a_gemm(m, n, k, scale_format):
    x, w, scales, effective_w = _fp8_case(m, n, k, scale_format)
    ref = (x.double() @ effective_w.T).bfloat16()
    out = torch.empty_like(ref)
    result = dsv3_fused_a_gemm_fp8(x, w, scales, output=out)
    assert result is out
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.usefixtures("cuda_sm103")
@pytest.mark.parametrize("m", [6, 12])
def test_fp8_fused_a_gemm_large_activations(m):
    # The tuned shapes run FP16 operands: activations in the thousands are exact, and the sums (far beyond
    # the FP16 range) stay correct only if the accumulation is FP32.
    x, w, scales, effective_w = _fp8_case(m, 2624, 6144, "fp32")
    x = x * 2000
    ref = (x.double() @ effective_w.T).bfloat16()
    out = dsv3_fused_a_gemm_fp8(x, w, scales)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.usefixtures("cuda_sm103")
@pytest.mark.parametrize("scale_format", ["fp32", "ue8m0"])
def test_fp8_fused_a_gemm_strided_scales(scale_format):
    x, w, scales, effective_w = _fp8_case(9, 144, 640, scale_format)
    storage = torch.empty(
        (scales.shape[0] * 2, scales.shape[1] * 3),
        device="cuda",
        dtype=scales.dtype,
    )
    strided = storage[::2, ::3]
    strided.copy_(scales)
    out = dsv3_fused_a_gemm_fp8(x, w, strided)
    ref = (x.double() @ effective_w.T).bfloat16()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.usefixtures("cuda_sm103")
@pytest.mark.parametrize("scale_format", ["fp32", "ue8m0"])
def test_fp8_fused_a_gemm_graph_and_compile(scale_format):
    x, w, scales, effective_w = _fp8_case(9, 144, 640, scale_format)
    # Compile JIT specializations before capture, as server graph warmup does.
    dsv3_fused_a_gemm_fp8(x, w, scales)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = dsv3_fused_a_gemm_fp8(x, w, scales)
    torch.cuda.current_stream().wait_stream(stream)
    x.normal_()
    graph.replay()
    ref = (x.double() @ effective_w.T).bfloat16()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)

    compiled = torch.compile(dsv3_fused_a_gemm_fp8, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, w, scales), ref, rtol=1e-2, atol=1e-3)


@triton.jit
def _late_writer(src, dst, n_elements, one, zero, spin, BLOCK: tl.constexpr):
    # Releases its dependent grid first, then stores long after, like a producer whose tail is still running.
    tl.extra.cuda.gdc_launch_dependents()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    for _ in range(spin):
        x = x * one + zero
    tl.store(dst + offs, x.to(tl.bfloat16), mask=mask)


@pytest.mark.usefixtures("cuda_sm103")
def test_fp8_fused_a_gemm_waits_for_its_pdl_producer():
    """In a graph, behind a producer that releases its dependent early and stores late, the GEMM reads the new output.

    Regression: activation loads were scheduled above the PDL wait, so each replay used the previous replay's input.
    """
    m, n, k = 6, 2624, 6144
    _, w, scales, effective_w = _fp8_case(m, n, k, "ue8m0")
    src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    act = torch.zeros_like(src)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def step():
        _late_writer[(triton.cdiv(m * k, 1024),)](
            src, act, m * k, 1.0, 0.0, 16384, BLOCK=1024, launch_pdl=True
        )
        dsv3_fused_a_gemm_fp8(act, w, scales, output=out)

    step()  # compile before capture, as server graph warmup does
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(4):
        src.normal_()
        graph.replay()
        ref = (act.double() @ effective_w.T).bfloat16()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.usefixtures("cuda_sm103")
@pytest.mark.parametrize("m", [0, 17])
def test_fp8_fused_a_gemm_rejects_m(m):
    x, w, scales, _ = _fp8_case(m, 16, 128, "fp32")
    with pytest.raises(RuntimeError, match="num_tokens"):
        dsv3_fused_a_gemm_fp8(x, w, scales)


@pytest.fixture
def fp8_projection(cuda_device, monkeypatch):
    from types import SimpleNamespace

    from sglang.kernels.ops.gemm import fused_a_gemm as dispatch
    from sglang.srt import runtime_context
    from sglang.srt.layers.quantization import fp8

    # Backend selection is unrelated to the stored-layout dispatch contract.
    monkeypatch.setattr(fp8, "dispatch_w8a8_block_fp8_linear", lambda **kw: None)
    monkeypatch.setattr(fp8, "can_serve_block_fp8_as_mxfp8", lambda *a: False)
    # No server runtime context is published in a unit test.
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )
    monkeypatch.setattr(dispatch, "_DEVICE_SM", 103)
    monkeypatch.setattr(dispatch, "_FP8_SMALL_M_SHAPES", frozenset({(144, 640)}))
    method = fp8.Fp8LinearMethod(
        fp8.Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    )

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            _, weight, scales, _ = _fp8_case(1, 144, 640, "fp32")
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
            self.bias = None
            self.quant_method = method

    return Projection()


def _fp8_projection_reference(layer, x):
    scales = layer.weight_scale_inv.double()
    weight = layer.weight.double() * scales.repeat_interleave(128, dim=0)[
        : layer.weight.shape[0]
    ].repeat_interleave(128, dim=1)
    return (x.double() @ weight.T).bfloat16()


@pytest.mark.parametrize(
    "case",
    [
        "eligible",
        "unlisted_shape",
        "sm90",
        "sm100",
        "bias",
        "deterministic",
        "mxfp8_view",
        "block32",
        "bad_scales",
    ],
)
def test_fp8_small_m_eligible(fp8_projection, monkeypatch, case):
    from types import SimpleNamespace

    from sglang.kernels.ops.gemm import fused_a_gemm as dispatch
    from sglang.srt import runtime_context

    block_size, mxfp8_view = [128, 128], False
    if case == "unlisted_shape":
        monkeypatch.setattr(dispatch, "_FP8_SMALL_M_SHAPES", frozenset())
    elif case in ("sm90", "sm100"):
        monkeypatch.setattr(dispatch, "_DEVICE_SM", int(case[2:]))
    elif case == "bias":
        fp8_projection.bias = torch.ones(144, device="cuda")
    elif case == "deterministic":
        monkeypatch.setattr(
            runtime_context,
            "get_exec",
            lambda: SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=True)
            ),
        )
    elif case == "mxfp8_view":
        mxfp8_view = True
    elif case == "block32":
        block_size = [1, 32]
    elif case == "bad_scales":
        fp8_projection.weight_scale_inv = torch.nn.Parameter(
            torch.ones((2, 4), device="cuda"), requires_grad=False
        )
    assert dispatch.fp8_small_m_eligible(
        layer=fp8_projection,
        weight_block_size=block_size,
        block_fp8_as_mxfp8=mxfp8_view,
    ) == (case == "eligible")


@pytest.mark.parametrize("case", ["too_many_rows", "strided", "fp16"])
def test_fp8_small_m_linear_declines(fp8_projection, case):
    from sglang.kernels.ops.gemm.fused_a_gemm import fp8_small_m_linear

    x = torch.randn(
        (17 if case == "too_many_rows" else 3, 640),
        device="cuda",
        dtype=torch.float16 if case == "fp16" else torch.bfloat16,
    )
    if case == "strided":
        x = x.T.contiguous().T
    assert fp8_small_m_linear(fp8_projection, x) is None


@pytest.mark.usefixtures("cuda_sm103")
def test_fp8_small_m_apply_after_weight_loading(fp8_projection):
    method = fp8_projection.quant_method
    assert not method.small_m_w8a16
    method.process_weights_after_loading_block_quant(fp8_projection)
    assert method.small_m_w8a16
    x = torch.randn((3, 640), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(fp8_projection, x),
        _fp8_projection_reference(fp8_projection, x),
        rtol=1e-2,
        atol=1e-3,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
