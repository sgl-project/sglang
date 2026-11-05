"""CuTe DSL mHC seam (mhc_mega_cute) against DeepGEMM's mega_mhc: every output
bitwise equal, in both scale layouts, through CUDA-graph replay; and the
_post_fusion gate that selects it."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

H, HC = 5120, 4
_NAMES = ("new_residual", "y_bf16", "y_fp8", "y_sf", "pre", "post", "comb")


def _sm100_family_with_deep_gemm() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False
    return True


gpu = pytest.mark.skipif(
    not _sm100_family_with_deep_gemm(),
    reason="requires an SM100-family GPU and DeepGEMM",
)


def _sinkhorn(logits, iters=20, eps=1e-6):
    v = torch.softmax(logits, -1) + eps
    v = v / (v.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        v = v / (v.sum(-1, keepdim=True) + eps)
        v = v / (v.sum(-2, keepdim=True) + eps)
    return v


def _case(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    residual = torch.randn(rows, HC, H, device="cuda", generator=g).bfloat16()
    # A zero group, a negative zero and a large-magnitude group.
    x[0, :33] = 0.0
    residual[0, :, :33] = 0.0
    x[1, 7] = -0.0
    residual[1, 2, 100:140] *= 1e4
    pre = torch.sigmoid(torch.randn(rows, HC, device="cuda", generator=g)) + 1e-6
    post = 2 * torch.sigmoid(torch.randn(rows, HC, device="cuda", generator=g))
    comb = _sinkhorn(torch.randn(rows, HC, HC, device="cuda", generator=g))
    fn = torch.randn(24, HC * H, device="cuda", generator=g) * 0.01
    scale = torch.tensor([0.13, 0.017, 0.083], device="cuda")
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    weight = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
    return (x, residual, pre, post, comb.contiguous(), fn, scale, base, weight)


def _run(boundary, case, flashinfer):
    from sglang.srt.models.deepseek_v4_mhc import empty_deepgemm_mxfp8_pair

    x = case[0]
    y_fp8, y_sf = empty_deepgemm_mxfp8_pair(*x.shape, x.device, flashinfer=flashinfer)
    # Neither kernel writes the 128x4 layout's padding rows.
    y_sf.zero_()
    updated, normalized, stats = boundary(
        *case, 1e-6, 1e-6, 1e-20, 20, y_fp8=y_fp8, y_gemm_sf=y_sf
    )
    return (updated, normalized, y_fp8.view(torch.uint8), y_sf) + tuple(stats)


def _assert_bits_equal(out, ref, tag):
    for name, a, b in zip(_NAMES, out, ref, strict=True):
        assert a.shape == b.shape, f"{name} {tag}"
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), f"{name} {tag}"


@gpu
@pytest.mark.parametrize("flashinfer", [False, True])
@pytest.mark.parametrize("rows", [1024, 1025, 2048, 2049, 8191, 8192])
def test_bitwise_vs_mega_mhc(rows, flashinfer):
    from sglang.kernels.ops.layernorm import mhc_mega_cute
    from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary

    sms = mhc_mega_cute.num_sms(torch.device("cuda"))
    if not mhc_mega_cute.deepgemm_uses_16_splits(rows, sms):
        pytest.skip(f"DeepGEMM splits K other than 16 ways at {rows} rows")
    case = _case(rows, seed=rows)
    _assert_bits_equal(
        _run(mhc_mega_cute.mhc_mega_boundary_cute, case, flashinfer),
        _run(mhc_mega_boundary, case, flashinfer),
        f"{rows=} {flashinfer=}",
    )


@gpu
def test_graph_replay():
    from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary
    from sglang.kernels.ops.layernorm.mhc_mega_cute import mhc_mega_boundary_cute

    case = _case(3000, seed=1)
    _run(mhc_mega_boundary_cute, case, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(mhc_mega_boundary_cute, case, True)
    for seed in (2, 3, 4):
        for dst, src in zip(case, _case(3000, seed=seed), strict=True):
            dst.copy_(src)
        graph.replay()
        _assert_bits_equal(out, _run(mhc_mega_boundary, case, True), f"replay {seed=}")


def test_gate():
    from sglang.kernels.ops.layernorm import mhc_mega_cute
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as mhc

    hc = SimpleNamespace(fn=torch.empty(24, HC * H))
    residual = torch.empty(8192, HC, H, dtype=torch.bfloat16)
    y = torch.empty(8192, H, dtype=torch.bfloat16)
    sm107 = SimpleNamespace(cuda=SimpleNamespace(capability=(10, 7)))
    sm103 = SimpleNamespace(cuda=SimpleNamespace(capability=(10, 3)))
    with (
        mock.patch.object(mhc_mega_cute, "num_sms", return_value=212),
        mock.patch("sglang.kernels.arch.current_arch", return_value=sm107),
    ):
        # Unset: on for SM107, from SGLANG_OPT_DSV41_MHC_CUTE_MIN_ROWS rows.
        assert mhc._use_mhc_cute(y, residual, hc)
        assert not mhc._use_mhc_cute(y[:8191], residual[:8191], hc)
        with envs.SGLANG_OPT_DSV41_MHC_CUTE_MIN_ROWS.override(1):
            assert mhc._use_mhc_cute(y[:641], residual[:641], hc)
            # DeepGEMM splits K 20 ways at 640 rows on 212 SMs.
            assert not mhc._use_mhc_cute(y[:640], residual[:640], hc)
        assert not mhc._use_mhc_cute(y[:, :4096], residual, hc)
        assert not mhc._use_mhc_cute(y, residual, SimpleNamespace(fn=hc.fn.bfloat16()))
        with envs.SGLANG_OPT_DSV41_MHC_CUTE.override(False):
            assert not mhc._use_mhc_cute(y, residual, hc)
    with (
        mock.patch.object(mhc_mega_cute, "num_sms", return_value=148),
        mock.patch("sglang.kernels.arch.current_arch", return_value=sm103),
    ):
        assert not mhc._use_mhc_cute(y, residual, hc)
        with envs.SGLANG_OPT_DSV41_MHC_CUTE.override(True):
            assert mhc._use_mhc_cute(y, residual, hc)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
