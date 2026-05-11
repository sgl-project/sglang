import sys

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import (
    check_marlin_supported,
    marlin_make_workspace,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import awq_marlin_quantize, marlin_quantize

register_cuda_ci(est_time=13, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (13, 17, 67),
    (257, 13, 11),
]


@pytest.mark.parametrize("k_chunk", [128])
@pytest.mark.parametrize("n_chunk", [64, 256])
@pytest.mark.parametrize("quant_type", [scalar_types.uint4, scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_marlin_gemm(
    k_chunk,
    n_chunk,
    quant_type,
    group_size,
    mnk_factors,
    act_order,
):
    m_factor, n_factor, k_factor = mnk_factors
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return
        if has_zp:
            return

    if size_k % group_size != 0:
        return

    a_input = torch.randn((size_m, size_k), dtype=torch.float16, device="cuda")
    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    if has_zp:
        w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
            b_weight, quant_type, group_size
        )
        g_idx = None
        sort_indices = None
        marlin_s2 = None
    else:
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            b_weight, quant_type, group_size, act_order
        )
        marlin_zp = None
        marlin_s2 = None

    workspace = marlin_make_workspace(w_ref.device)

    output = gptq_marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        marlin_s,
        marlin_s2,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )

    output_ref = torch.matmul(a_input, w_ref)
    torch.cuda.synchronize()

    # JIT kernel should produce approximately correct results vs torch.matmul
    max_diff = torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )
    assert max_diff < 0.04


def _is_sm80_sm90_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (80, 86, 90)


@pytest.mark.skipif(
    not _is_sm80_sm90_cuda(),
    reason="NVFP4 Marlin fallback tests require CUDA SM80, SM86, or SM90",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_support_and_scale_transforms_sm80_sm90(dtype):
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    assert check_marlin_supported(
        scalar_types.float4_e2m1f,
        group_size=16,
        has_zp=False,
        device_capability=capability,
    )

    scales = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
        dtype=dtype,
        device="cuda",
    )
    expected_scales = scales.view(-1, 4)[:, [0, 2, 1, 3]].view(scales.size(0), -1)
    expected_scales = (expected_scales.to(torch.half) * (2**7)).view(torch.int16) << 1
    expected_scales = expected_scales.view(torch.float8_e4m3fn)[:, 1::2].contiguous()

    actual_scales = nvfp4_marlin_process_scales(scales)
    assert actual_scales.is_cuda
    assert actual_scales.dtype == torch.float8_e4m3fn
    assert torch.equal(actual_scales.view(torch.uint8), expected_scales.view(torch.uint8))

    global_scale = torch.tensor(1.0, dtype=dtype, device="cuda")
    actual_global_scale = nvfp4_marlin_process_global_scale(global_scale)
    assert actual_global_scale.is_cuda
    assert actual_global_scale.ndim == 1
    assert actual_global_scale.numel() == 1
    if dtype == torch.float16:
        assert actual_global_scale.item() == 128.0
    else:
        assert actual_global_scale.item() == 2.0**119


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
