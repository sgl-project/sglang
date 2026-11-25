import pytest
import torch

from sglang.kernels.ops.moe.dsv4 import silu_and_mul_masked_post_quant
from sglang.kernels.ops.moe.ep_moe_kernels import moe_ep_deepgemm_preprocess
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.deep_gemm import _varlen_deep_gemm_silu_mul_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not deep_gemm_wrapper.DEEPGEMM_MASKED_FP32_MN_MAJOR,
    reason="Direct masked FP32 scale stores require SM90 DeepGEMM",
)


def _check_layout(scales, direct):
    _, m, g = scales.shape
    assert scales.dtype == torch.float32
    aligned = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(scales)
    assert aligned.stride() == (m * g, 1, m)
    if direct:
        assert aligned is scales
    else:
        assert scales.stride() == (m * g, g, 1)
        assert aligned.data_ptr() != scales.data_ptr()


@pytest.mark.parametrize("direct", [False, True], ids=["default", "direct"])
def test_gateup_scatter_preserves_fp32_group_scales(direct):
    hidden, group = 512, 128
    x = torch.randn(5, hidden, dtype=torch.bfloat16, device="cuda")
    experts = torch.tensor(
        [[2, -1], [0, 2], [2, 0], [-1, -1], [0, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    _, _, dst, values, scales = moe_ep_deepgemm_preprocess(
        experts,
        3,
        x,
        experts.size(1),
        [group, group],
        **({"use_mn_major_fp32_scales": True} if direct else {}),
    )
    _check_layout(scales, direct)
    want_values, want_scales = sglang_per_token_group_quant_fp8(x, group)
    token, slot = torch.where(experts >= 0)
    expert, row = (
        experts[token, slot].long(),
        dst.view_as(experts)[token, slot].long() % scales.size(1),
    )
    assert torch.equal(
        values.view(torch.uint8)[expert, row], want_values.view(torch.uint8)[token]
    )
    assert torch.equal(scales[expert, row], want_scales[token])


@pytest.mark.parametrize("direct", [False, True], ids=["default", "direct"])
def test_masked_clamped_down_preserves_fp32_group_scales(direct):
    e, m, hidden, group = 4, 16, 512, 128
    gate_up = torch.randn(e, m, 2 * hidden, dtype=torch.bfloat16, device="cuda") * 8
    counts = torch.tensor([0, 1, m - 1, m], dtype=torch.int32, device="cuda")
    old_values = torch.empty((e, m, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    old_scales = torch.empty((e, m, hidden // group), device="cuda")
    silu_and_mul_masked_post_quant(
        gate_up, old_values, old_scales, group, counts, topk=4, swiglu_limit=10.0
    )
    new_values, new_scales = _varlen_deep_gemm_silu_mul_quant(
        gate_up,
        counts,
        group,
        topk=4,
        swiglu_limit=10.0,
        **({"use_mn_major_fp32_scales": True} if direct else {}),
    )
    _check_layout(new_scales, direct)
    live = torch.arange(m, device="cuda")[None, :] < counts[:, None]
    assert torch.equal(
        old_values.view(torch.uint8)[live], new_values.view(torch.uint8)[live]
    )
    assert torch.equal(old_scales[live], new_scales[live])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
