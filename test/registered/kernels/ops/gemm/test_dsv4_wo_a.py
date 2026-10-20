"""MXFP8 epilogues stay bitwise identical to FlashInfer's standalone quantizer."""

import sys

import pytest
import torch
from flashinfer import mxfp8_quantize

from sglang.kernels.ops.gemm.dsv4_wo_a import (
    _quantize_partial,
    _wo_a_reduce,
    wo_a_bf16_small_batch,
    wo_a_bf16_small_batch_mxfp8,
)
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

DEVICE = "cuda"

HIDDEN = 5120

STREAMS = 4


def test_wo_a_partial_quant_matches_quantize():
    torch.manual_seed(0)
    for rows in range(2, 9):
        for magnitude in (0.0, 1e-37, 1e-7, 1.0, 448.0, 1e10):
            partial = torch.randn(8, rows, 2, 1024, device=DEVICE) * magnitude
            bf16 = torch.empty(rows, 2048, dtype=torch.bfloat16, device=DEVICE)
            _wo_a_reduce[(rows * 8,)](partial, bf16, rows * 2048, num_warps=4)
            expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
            actual_q, actual_s = _quantize_partial(partial)
            torch.testing.assert_close(
                actual_q.view(torch.uint8),
                expected_q.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

        x = torch.randn(rows, 64, 512, device=DEVICE, dtype=torch.bfloat16)[
            :, :16
        ].view(rows, 2, 4096)
        wo_a = torch.randn(2, 1024, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
        bf16 = wo_a_bf16_small_batch(x, wo_a).flatten(1)
        q, s = wo_a_bf16_small_batch_mxfp8(x, wo_a)
        expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)

    partial = torch.randn(8, 6, 2, 1024, device=DEVICE)
    _quantize_partial(partial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = _quantize_partial(partial)
    for _ in range(3):
        partial.normal_()
        # The replay must regenerate scale padding as well as live rows.
        s.fill_(255)
        graph.replay()
        bf16 = torch.empty(6, 2048, dtype=torch.bfloat16, device=DEVICE)
        _wo_a_reduce[(48,)](partial, bf16, 6 * 2048, num_warps=4)
        eq, es = mxfp8_quantize(bf16, True, alignment=32)
        torch.testing.assert_close(
            q.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(s, es, rtol=0, atol=0)


@pytest.mark.parametrize("groups", [2, 8])
def test_wo_a_strided_bmm_matches_einsum(groups):
    """The strided bmm returns the einsum's values in the token-major layout wo_b reads."""
    from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

    torch.manual_seed(0)
    wo_a = torch.randn(groups, 1024, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
    cases = [("is_target_verify", rows) for rows in (9, 64, 100, 128, 384)]
    cases += [("is_prefill", rows) for rows in (4096, 8192, 16384)]
    if groups == 8:
        # 2 groups run 1-8 rows through the gemv and small-batch kernels instead
        cases += [("is_decode", 1)]
        cases += [("is_target_verify", rows) for rows in range(1, 9)]
    for mode, rows in cases:
        # attention may hand over a head slice of a padded buffer
        for padded_heads in (groups * 8, 128):
            x = torch.randn(
                rows, padded_heads, 512, device=DEVICE, dtype=torch.bfloat16
            )[:, : groups * 8].view(rows, groups, 4096)
            flags = dict(is_decode=False, is_target_verify=False, is_prefill=False)
            flags[mode] = True
            out = _apply_wo_a_bf16_matmul(x, wo_a, fast_path=True, **flags)
            case = (groups, mode, rows, padded_heads)
            assert out.is_contiguous(), case
            assert torch.equal(out, torch.einsum("tgd,grd->tgr", x, wo_a)), case


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
