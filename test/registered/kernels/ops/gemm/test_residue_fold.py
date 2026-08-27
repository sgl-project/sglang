"""GPU tests for the residue NVFP4 mext_r1 fold GEMM (sm100/sm103).

The fold computes fold(A[2M, K] @ W[N, K]^T) where A holds row-pair
interleaved (base, residue) activation rows and the kernel sums each output
pair: out[m] = (x_q[m] + r_q[m]) @ W_q^T * alpha. The reference is the same
contraction done in BF16/FP32 on the dequantized operands, so the test
isolates the GEMM (quantization error cancels out by construction).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("sm100/sm103 (datacenter Blackwell) required", allow_module_level=True)
pytest.importorskip("cutlass", reason="nvidia-cutlass-dsl required")

from sglang.kernels.ops.gemm.residue_fold import run_fold, warmup
from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
    scaled_fp4_quant_mext_r1,
)
from sglang.test.kernels.residue_nvfp4 import (
    base_row,
    decode_fp4,
    dequant_nvfp4_weight,
    quantize_nvfp4_weight,
    residue_row,
    sf_unswizzle,
)

MAJOR, MINOR = torch.cuda.get_device_capability()


@pytest.mark.parametrize("m", [1, 4, 16, 64, 128])
@pytest.mark.parametrize("n,k", [(512, 1024), (2048, 4096)])
def test_fold_matches_bf16_decomposition(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)

    x_global = ((448.0 * 6.0) / x.float().abs().max()).reshape(1).cuda()
    act_fp4, act_sf = scaled_fp4_quant_mext_r1(x, x_global, layout_mode="row_pair")

    w_packed, w_sf_swz, w_global = quantize_nvfp4_weight(w)
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()

    out = run_fold(
        MAJOR,
        MINOR,
        w_packed,
        act_fp4,
        w_sf_swz.view(torch.float8_e4m3fn),
        act_sf.view(torch.float8_e4m3fn),
        alpha,
        torch.bfloat16,
    )
    assert out.shape == (m, n)

    # Reference: dequantize both operands and contract in fp32. The fold sums
    # the (base, residue) row pair before the weight contraction.
    r = torch.arange(m, device="cuda")
    act_sf_grid = sf_unswizzle(act_sf, 2 * m, k)
    a_base = decode_fp4(act_fp4[base_row(r, m, 1)]) * act_sf_grid[
        base_row(r, m, 1)
    ].repeat_interleave(16, dim=1)
    a_res = decode_fp4(act_fp4[residue_row(r, m, 1)]) * act_sf_grid[
        residue_row(r, m, 1)
    ].repeat_interleave(16, dim=1)
    a_deq = (a_base + a_res) / x_global
    w_deq = dequant_nvfp4_weight(w_packed, w_sf_swz, w_global)

    ref = a_deq @ w_deq.T

    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 5e-3, f"fold vs decomposition rel err {rel:.2e} (m={m} n={n} k={k})"


def test_warmup_compiles_static_table():
    count = warmup(MAJOR, MINOR)
    assert count > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
