"""Dispatch, fallback, and SM120 parity tests for small-M rowwise FP8 linear."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _is_live_sm120() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and fp8_utils.get_cuda_version() >= (12, 8)
    )


def _make_rowwise_inputs(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    col_major_weight: bool = True,
    with_bias: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device="cuda").mul_(0.2)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device="cuda").mul_(0.2)
    weight_scale = (
        w_bf16.abs().amax(dim=1, keepdim=True).float().div_(448.0).clamp_min_(1e-6)
    )
    qw_nk = (w_bf16.float() / weight_scale).to(torch.float8_e4m3fn)
    weight = qw_nk.t() if col_major_weight else qw_nk.t().contiguous()
    bias = (
        torch.randn((n,), dtype=dtype, device="cuda").mul_(0.1) if with_bias else None
    )
    return x, weight, weight_scale, bias


@pytest.mark.parametrize(
    "major,minor,sm_count,cuda_ver,expected",
    [
        (12, 0, 156, (12, 8), 6656),  # RTX 6000D
        (12, 0, 170, (13, 0), 7168),  # RTX 5090
        (12, 0, 188, (13, 0), 7936),  # RTX PRO 6000 Blackwell
        (12, 1, 48, (13, 0), 0),  # SM121 / GB10 excluded
        (10, 0, 148, (13, 0), 0),  # SM100 excluded
        (9, 0, 132, (12, 8), 0),  # SM90 excluded
        (8, 9, 142, (12, 8), 0),  # SM89 excluded
        (12, 0, 156, (12, 6), 0),  # CUDA < 12.8 excluded
    ],
)
def test_sm120_torch_fp8_max_n_by_arch_and_cuda(
    major: int,
    minor: int,
    sm_count: int,
    cuda_ver: tuple[int, int],
    expected: int,
):
    props = SimpleNamespace(major=major, minor=minor, multi_processor_count=sm_count)
    fp8_utils._sm120_torch_fp8_max_n.cache_clear()
    try:
        with (
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
            mock.patch.object(fp8_utils, "get_cuda_version", return_value=cuda_ver),
        ):
            assert fp8_utils._sm120_torch_fp8_max_n(torch.device("cuda:0")) == expected
    finally:
        fp8_utils._sm120_torch_fp8_max_n.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "m,k,n,dtype,col_major,with_bias,max_n,compiling,tuned_cfg,expected_backend",
    [
        # Compatible shapes route to torch._scaled_mm.
        (1, 4096, 4096, torch.bfloat16, True, False, 6656, False, None, "torch"),
        (16, 4096, 1536, torch.bfloat16, True, False, 6656, False, None, "torch"),
        (128, 5120, 6656, torch.float16, True, False, 6656, False, None, "torch"),
        # Shape boundaries fall back to CUTLASS.
        (129, 4096, 4096, torch.bfloat16, True, False, 6656, False, None, "cutlass"),
        (16, 3968, 4096, torch.bfloat16, True, False, 6656, False, None, "cutlass"),
        (16, 4096, 1408, torch.bfloat16, True, False, 6656, False, None, "cutlass"),
        (16, 4096, 6784, torch.bfloat16, True, False, 6656, False, None, "cutlass"),
        # Non-SM120, bias, non-col-major weight, and torch.compile fall back.
        (16, 4096, 4096, torch.bfloat16, True, False, 0, False, None, "cutlass"),
        (16, 4096, 4096, torch.bfloat16, True, True, 6656, False, None, "cutlass"),
        (16, 4096, 4096, torch.bfloat16, False, False, 6656, False, None, "cutlass"),
        (16, 4096, 4096, torch.bfloat16, True, False, 6656, True, None, "cutlass"),
        # Tuned Triton configs keep precedence over the PyTorch route.
        (
            16,
            4096,
            4096,
            torch.bfloat16,
            True,
            False,
            6656,
            False,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "num_warps": 4,
                "num_stages": 3,
            },
            "triton",
        ),
    ],
)
def test_apply_fp8_linear_sm120_dispatch_and_fallbacks(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    col_major: bool,
    with_bias: bool,
    max_n: int,
    compiling: bool,
    tuned_cfg: dict | None,
    expected_backend: str,
):
    x, weight, weight_scale, bias = _make_rowwise_inputs(
        m, k, n, dtype=dtype, col_major_weight=col_major, with_bias=with_bias
    )
    called: list[str] = []

    def fake_quant(inp: torch.Tensor):
        q = torch.empty(inp.shape, dtype=torch.float8_e4m3fn, device=inp.device)
        s = torch.ones((inp.shape[0], 1), dtype=torch.float32, device=inp.device)
        return q, s

    def fake_torch_scaled_mm(q, w, *, scale_a, scale_b, out_dtype):
        called.append("torch")
        assert scale_a.shape == (m, 1)
        assert scale_b.shape == (1, n)
        assert out_dtype == dtype
        return torch.zeros((m, n), dtype=out_dtype, device=q.device)

    def fake_cutlass(q, w, sa, sb, *, out_dtype, bias=None):
        called.append("cutlass")
        return torch.zeros((m, n), dtype=out_dtype, device=q.device)

    def fake_triton(q, w, sa, sb, out_dtype, bias, **kwargs):
        called.append("triton")
        return torch.zeros((m, n), dtype=out_dtype, device=q.device)

    with (
        mock.patch.object(fp8_utils, "_is_cuda", True),
        mock.patch.object(fp8_utils, "use_triton_w8a8_fp8_kernel", False),
        mock.patch.object(fp8_utils, "_sm120_torch_fp8_max_n", return_value=max_n),
        mock.patch.object(
            fp8_utils, "get_w8a8_channelwise_fp8_config", return_value=tuned_cfg
        ),
        mock.patch.object(
            fp8_utils, "sglang_per_token_quant_fp8", side_effect=fake_quant
        ),
        mock.patch.object(torch.compiler, "is_compiling", return_value=compiling),
        mock.patch.object(torch, "_scaled_mm", side_effect=fake_torch_scaled_mm),
        mock.patch.object(
            fp8_utils, "fp8_scaled_mm", side_effect=fake_cutlass, create=True
        ),
        mock.patch.object(fp8_utils, "triton_scaled_mm", side_effect=fake_triton),
    ):
        out = fp8_utils.apply_fp8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            cutlass_fp8_supported=True,
        )

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert called == [expected_backend]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_invariant_mode_falls_back_to_cutlass():
    from sglang.srt import batch_invariant_ops

    m, k, n = 16, 4096, 4096
    x, weight, weight_scale, _ = _make_rowwise_inputs(m, k, n, dtype=torch.bfloat16)
    qx = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    x_scale = torch.full((m, 1), 0.05, dtype=torch.float32, device="cuda")
    called: list[str] = []

    def fake_torch_scaled_mm(*args, **kwargs):
        called.append("torch")
        return torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    def fake_cutlass(*args, **kwargs):
        called.append("cutlass")
        return torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    with (
        mock.patch.object(fp8_utils, "_is_cuda", True),
        mock.patch.object(fp8_utils, "use_triton_w8a8_fp8_kernel", False),
        mock.patch.object(fp8_utils, "_sm120_torch_fp8_max_n", return_value=6656),
        mock.patch.object(
            fp8_utils, "get_w8a8_channelwise_fp8_config", return_value=None
        ),
        mock.patch.object(
            fp8_utils, "sglang_per_token_quant_fp8", return_value=(qx, x_scale)
        ),
        mock.patch.object(
            batch_invariant_ops, "is_batch_invariant_mode_enabled", return_value=True
        ),
        mock.patch.object(torch, "_scaled_mm", side_effect=fake_torch_scaled_mm),
        mock.patch.object(
            fp8_utils, "fp8_scaled_mm", side_effect=fake_cutlass, create=True
        ),
    ):
        fp8_utils.apply_fp8_linear(
            x,
            weight,
            weight_scale,
            cutlass_fp8_supported=True,
        )

    assert called == ["cutlass"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_scalar_input_scale_falls_back_to_cutlass():
    m, k, n = 16, 4096, 4096
    _, weight, weight_scale, _ = _make_rowwise_inputs(m, k, n, dtype=torch.bfloat16)
    qx = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    scalar_scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    called: list[str] = []

    def fake_torch_scaled_mm(*args, **kwargs):
        called.append("torch")
        return torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    def fake_cutlass(*args, **kwargs):
        called.append("cutlass")
        return torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    fake_platform = SimpleNamespace(is_sm90=False, is_sm100=False, is_sm120=True)
    with (
        mock.patch.object(fp8_utils, "_is_cuda", True),
        mock.patch.object(fp8_utils, "use_triton_w8a8_fp8_kernel", False),
        mock.patch.object(fp8_utils, "get_platform", return_value=fake_platform),
        mock.patch.object(fp8_utils, "_sm120_torch_fp8_max_n", return_value=6656),
        mock.patch.object(
            fp8_utils, "get_w8a8_channelwise_fp8_config", return_value=None
        ),
        mock.patch.object(torch, "_scaled_mm", side_effect=fake_torch_scaled_mm),
        mock.patch.object(
            fp8_utils, "fp8_scaled_mm", side_effect=fake_cutlass, create=True
        ),
    ):
        fp8_utils.apply_fp8_linear(
            qx,
            weight,
            weight_scale,
            input_scale=scalar_scale,
            cutlass_fp8_supported=True,
            pre_quant_output_dtype=torch.bfloat16,
        )

    assert called == ["cutlass"]


@pytest.mark.skipif(
    not _is_live_sm120(), reason="requires CUDA SM120 with CUDA >= 12.8"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,k,n", [(1, 4096, 4096), (16, 4096, 4096), (128, 5120, 5120)]
)
def test_sm120_live_numerics_match_cutlass(m: int, k: int, n: int, dtype: torch.dtype):
    max_n = fp8_utils._sm120_torch_fp8_max_n(torch.device("cuda:0"))
    if n > max_n:
        pytest.skip(f"N={n} exceeds SM120 max_n={max_n} on this GPU")

    x, weight, weight_scale, _ = _make_rowwise_inputs(m, k, n, dtype=dtype, seed=7)
    with mock.patch.object(
        fp8_utils, "get_w8a8_channelwise_fp8_config", return_value=None
    ):
        actual = fp8_utils.apply_fp8_linear(
            x, weight, weight_scale, cutlass_fp8_supported=True
        )
        with mock.patch.object(fp8_utils, "_sm120_torch_fp8_max_n", return_value=0):
            expected = fp8_utils.apply_fp8_linear(
                x, weight, weight_scale, cutlass_fp8_supported=True
            )

    rel_l2 = (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-8)
    ).item()
    assert rel_l2 < 1e-3, f"M={m}, K={k}, N={n}, dtype={dtype}: rel_l2={rel_l2:.3e}"
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not _is_live_sm120(), reason="requires CUDA SM120 with CUDA >= 12.8"
)
def test_sm120_live_cuda_graph_replay():
    m, k, n = 16, 4096, 4096
    max_n = fp8_utils._sm120_torch_fp8_max_n(torch.device("cuda:0"))
    if n > max_n:
        pytest.skip(f"N={n} exceeds SM120 max_n={max_n} on this GPU")

    x, weight, weight_scale, _ = _make_rowwise_inputs(
        m, k, n, dtype=torch.bfloat16, seed=13
    )

    with mock.patch.object(
        fp8_utils, "get_w8a8_channelwise_fp8_config", return_value=None
    ):
        # Warm up before graph capture.
        _ = fp8_utils.apply_fp8_linear(
            x, weight, weight_scale, cutlass_fp8_supported=True
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fp8_utils.apply_fp8_linear(
                x, weight, weight_scale, cutlass_fp8_supported=True
            )

        graph.replay()
        first = out.clone()

        torch.manual_seed(29)
        x.copy_(torch.randn_like(x).mul_(0.3))
        graph.replay()
        expected = fp8_utils.apply_fp8_linear(
            x, weight, weight_scale, cutlass_fp8_supported=True
        )

    assert not torch.equal(out, first)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
