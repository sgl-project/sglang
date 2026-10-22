"""The WO-A epilogue must match the existing quantizer and WO-B consumer."""

import inspect
import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_kernel import w8a8_mxfp8_matmul_deepgemm
from sglang.kernels.ops.gemm.fp8_wo_a import fp8_wo_a_mxfp8
from sglang.kernels.ops.quantization.fp8_group_major_quant_ue8m0 import (
    sglang_per_token_group_quant_fp8_dsv4_wo_a,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FP8 einsum output requires SM10x",
)


@pytest.fixture(autouse=True)
def require_configurable_fp8_epilogue():
    import deep_gemm

    epilogue_type = getattr(
        getattr(deep_gemm, "epilogue", None), "FP8CastWithClamp", None
    )
    if (
        epilogue_type is None
        or "amax_eps" not in inspect.signature(epilogue_type).parameters
    ):
        pytest.skip("Requires the configurable FP8 floor in sgl-project/DeepGEMM#111")


def _quantize(x):
    return sglang_per_token_group_quant_fp8(
        x, 32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
    )


def _quantize_woa(x, block_size):
    if block_size == 128:
        return sglang_per_token_group_quant_fp8_dsv4_wo_a(x)
    rows, groups, _ = x.shape
    q, scales = sglang_per_token_group_quant_fp8(
        x.transpose(0, 1).contiguous(),
        32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        masked_m=torch.full((groups,), rows, device=x.device, dtype=torch.int32),
    )
    return q.transpose(0, 1).contiguous(), scales.transpose(0, 1)


@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("rows,groups", [(1, 1), (6, 2), (24, 4), (96, 8), (1025, 2)])
def test_codes_scales_and_consumer(rows, groups, block_size):
    import deep_gemm

    torch.manual_seed(42)
    rank, hidden = 1024, 4096
    x = torch.randn(rows, groups, hidden, device="cuda", dtype=torch.bfloat16)
    # Include exact zero and very small activation rows, not just unit variance.
    x[0].zero_()
    if rows > 1:
        x[1].mul_(1e-8)
    x_q, x_s = _quantize_woa(x, block_size)
    weight = torch.randn(rank, groups, hidden, device="cuda", dtype=torch.bfloat16)
    w_q, w_s = _quantize_woa(weight, block_size)
    w_q, w_s = w_q.transpose(0, 1).contiguous(), w_s.transpose(0, 1)
    reference = torch.empty(rows, groups, rank, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (x_q, x_s), (w_q, w_s), reference, recipe=(1, 1, block_size)
    )
    ref_q, ref_s = _quantize(reference.flatten(1))
    actual_q, actual_s = fp8_wo_a_mxfp8(x_q, x_s, w_q, w_s, (1, 1, block_size))
    torch.testing.assert_close(
        actual_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_s, ref_s, rtol=0, atol=0)
    assert actual_q.is_contiguous()
    assert actual_s.stride() == (1, (rows + 3) // 4 * 4)
    wb, sb = _quantize(
        torch.randn(5120, groups * rank, device="cuda", dtype=torch.bfloat16)
    )
    expected = w8a8_mxfp8_matmul_deepgemm(ref_q, wb, ref_s, sb, torch.bfloat16)
    actual = w8a8_mxfp8_matmul_deepgemm(actual_q, wb, actual_s, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cuda_graph_replay_updates_outputs():
    torch.manual_seed(123)
    x = torch.randn(7, 2, 4096, device="cuda", dtype=torch.bfloat16)
    xq, xs = sglang_per_token_group_quant_fp8_dsv4_wo_a(x)
    w = torch.randn(1024, 2, 4096, device="cuda", dtype=torch.bfloat16)
    wq, ws = sglang_per_token_group_quant_fp8_dsv4_wo_a(w)
    wq, ws = wq.transpose(0, 1).contiguous(), ws.transpose(0, 1)

    def run():
        return fp8_wo_a_mxfp8(xq, xs, wq, ws, (1, 1, 128))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()
    for scale in (1.0, -0.25, 0.0):
        q, s = sglang_per_token_group_quant_fp8_dsv4_wo_a(x * scale)
        xq.copy_(q)
        xs.copy_(s)
        graph.replay()
        expected = run()
        torch.testing.assert_close(
            actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 6, 24, 96, 384])
def test_block32_wo_b_consumer(rows, monkeypatch):
    """V4.1 block32 WO-B consumes the same packed activation as its BF16 path."""
    import functools
    from types import SimpleNamespace

    import deep_gemm

    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.mxfp8_input import (
        accepts_mxfp8_deepgemm_input,
    )

    torch.manual_seed(20261022)
    x = torch.randn(rows, 2, 4096, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    if rows > 1:
        x[1].mul_(1e-8)
    w = torch.randn(1024, 2, 4096, device="cuda", dtype=torch.bfloat16)
    xq, xs = _quantize_woa(x, 32)
    wq, ws = _quantize_woa(w, 32)
    wq, ws = wq.transpose(0, 1).contiguous(), ws.transpose(0, 1)
    reference = torch.empty(rows, 2, 1024, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (xq, xs), (wq, ws), reference, recipe=(1, 1, 32)
    )
    actual_q, actual_s = fp8_wo_a_mxfp8(xq, xs, wq, ws, (1, 1, 32))

    # Block32 weights retain one scale per 32x32 block, unlike native MXFP8.
    wb = torch.randn(5120, 2048, device="cuda", dtype=torch.bfloat16)
    blocks = wb.float().view(160, 32, 64, 32)
    sf = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-10) / 448))
    )
    qb = (blocks / sf[:, None, :, None]).reshape_as(wb).to(torch.float8_e4m3fn)
    sf.power_of_two = True
    consumer = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
    layer = SimpleNamespace(
        weight=qb,
        weight_scale_inv=sf,
        quant_method=SimpleNamespace(
            use_mxfp8=False,
            use_marlin=False,
            weight_block_size=[32, 32],
            w8a8_block_fp8_linear=functools.partial(consumer, act_scale_ue8m0=True),
        ),
    )
    assert accepts_mxfp8_deepgemm_input(layer)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Packed UE8M0 block32 input must reach DeepGEMM")

    monkeypatch.setattr(fp8_utils, "triton_w8a8_block_fp8_linear", unexpected_fallback)
    expected = consumer(reference.flatten(1), qb, [32, 32], sf, act_scale_ue8m0=True)
    actual = consumer(
        actual_q, qb, [32, 32], sf, input_scale=actual_s, act_scale_ue8m0=True
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not fp8_utils._ue8m0_deepgemm_skips_block32([32, 32], sf, actual_s)
    assert fp8_utils._ue8m0_deepgemm_skips_block32([32, 32], sf, sf)
    sf.power_of_two = False
    assert not accepts_mxfp8_deepgemm_input(layer)
    assert fp8_utils._ue8m0_deepgemm_skips_block32([32, 32], sf, actual_s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
