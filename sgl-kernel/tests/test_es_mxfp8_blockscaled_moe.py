import random

import pytest
import torch
from sgl_kernel import (
    es_sm100_mxfp8_blockscaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_quant,
)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def is_sm100_supported(device=None) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="test_es_sm100_mxfp8_blockscaled_grouped_mm at sgl-kernel is only supported on sm100",
)
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_es_sm100_mxfp8_blockscaled_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128
    n_g = random.randint(1, 64) * alignment
    k_g = random.randint(1, 64) * alignment

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    a_blockscale_offset = 0
    a_blockscale_offsets = []
    b_blockscale_offset = 0
    b_blockscale_offsets = []
    problem_sizes = []
    aux_problem_sizes = []
    a_list = []
    b_list = []
    ref_d_list = []

    for g in range(num_experts):
        m_g = random.randint(1, 512)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += align(m_g, 128)
        b_blockscale_offsets.append(b_blockscale_offset)
        b_blockscale_offset += n_g  # n_g already align to 128
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        a = torch.normal(
            0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype
        )  # (M, K):(K, 1)
        b = torch.normal(
            0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype
        )  # (N, K):(K, 1)

        a_list.append(a)
        b_list.append(b)
        ref_d = a @ b.T
        ref_d_list.append(ref_d)
    a = torch.concat(a_list, dim=0)
    b = torch.concat(b_list, dim=0)

    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(aux_problem_sizes).to(
        device=device, dtype=torch.int32
    )
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(
        device=device, dtype=torch.int32
    )
    _a_blockscale_offsets = torch.tensor(a_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _b_blockscale_offsets = torch.tensor(b_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )

    a_quant = torch.zeros_like(a, dtype=torch.float8_e4m3fn, device=device)
    a_scale_factor = torch.zeros(
        (a_blockscale_offset, k_g // 32), dtype=torch.uint8, device=device
    )

    b_quant = torch.zeros_like(b, dtype=torch.float8_e4m3fn, device=device)
    b_scale_factor = torch.zeros(
        (num_experts, n_g, k_g // 32), dtype=torch.uint8, device=device
    )

    es_sm100_mxfp8_blockscaled_grouped_quant(
        a,
        _problem_sizes,
        _expert_offsets,
        _a_blockscale_offsets,
        a_quant,
        a_scale_factor,
    )

    es_sm100_mxfp8_blockscaled_grouped_quant(
        b,
        _aux_problem_sizes,
        _aux_expert_offsets,
        _b_blockscale_offsets,
        b_quant,
        b_scale_factor,
    )
    b_quant = b_quant.view(num_experts, n_g, k_g).transpose(1, 2)
    b_scale_factor = b_scale_factor.view(num_experts, n_g, k_g // 32).transpose(1, 2)

    d = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)
    es_sm100_mxfp8_blockscaled_grouped_mm(
        d,
        a_quant,
        b_quant,
        a_scale_factor,
        b_scale_factor,
        _problem_sizes,
        _expert_offsets,
        _a_blockscale_offsets,
    )

    for g in range(num_experts):
        baseline = ref_d_list[g]
        actual = d[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
        diff = calc_diff(actual, baseline)
        assert diff < 0.001
        print(
            f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, out_dtype={out_dtype}, diff={diff:.5f}: OK"
        )


if __name__ == "__main__":
    pytest.main([__file__])
