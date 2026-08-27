import pytest
import torch
from sgl_kernel import gptq_marlin_gemm
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.test.test_marlin_utils import awq_marlin_quantize, marlin_quantize

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]


# uint4 for awq
# uint4b8 for gptq
@pytest.mark.parametrize("k_chunk", [128])
@pytest.mark.parametrize("n_chunk", [64, 256])
@pytest.mark.parametrize("quant_type", [scalar_types.uint4, scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("is_k_full", [False, True])
@pytest.mark.parametrize("use_atomic_add", [False, True])
@pytest.mark.parametrize("use_fp32_reduce", [False, True])
def test_gptq_marlin_gemm(
    k_chunk,
    n_chunk,
    quant_type,
    group_size,
    mnk_factors,
    act_order,
    is_k_full,
    use_atomic_add,
    use_fp32_reduce,
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
        # AWQ style, unsigned + runtime zero-point
        if group_size == 16:
            return
        w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
            b_weight, quant_type, group_size
        )
        g_idx = None
        sort_indices = None
        marlin_s2 = None
    else:
        # GPTQ style, unsigned + symmetric bias
        if group_size == 16:
            return
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            b_weight, quant_type, group_size, act_order
        )
        marlin_zp = None
        marlin_s2 = None

    workspace = marlin_make_workspace(w_ref.device)

    # marlin gemm
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
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    # ref gemm
    output_ref = torch.matmul(a_input, w_ref)

    torch.cuda.synchronize()

    max_diff = torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )

    assert max_diff < 0.04


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
