import random

import pytest
import torch
from sgl_kernel import fp8_blockwise_scaled_grouped_mm


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
) -> torch.Tensor:

    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    return torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)


def is_sm100_supported(device=None) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="fp8_blockwise_scaled_grouped_mm at sgl-kernel is only supported on sm100",
)
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_fp8_blockwise_scaled_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 16
    n_g = alignment * random.randint(1, 5) * 128
    k_g = alignment * random.randint(1, 5) * 128

    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)

    expert_offsets = torch.zeros((num_experts + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)
    layout_sfa = torch.zeros((num_experts, 5), device=device, dtype=torch.int32)
    layout_sfb = torch.zeros((num_experts, 5), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    baseline_tensors = []

    for g in range(num_experts):
        m_g = alignment * random.randint(1, 64)
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        b_g = to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        scale_a_shape = scale_shape(a_g.shape, scale_a_group_shape)
        scale_b_shape = scale_shape(b_g.shape, scale_b_group_shape)

        a_scales_tensors.append(torch.randn(scale_a_shape, device=device) * 0.001)
        b_scales_tensors.append(torch.randn(scale_b_shape, device=device) * 0.001)

        baseline = baseline_scaled_mm(
            a_g, b_g, a_scales_tensors[-1], b_scales_tensors[-1], out_dtype
        )
        baseline_tensors.append(baseline)

    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn
    )
    b_stack = torch.empty(
        (num_experts, n_g, k_g), device=device, dtype=torch.float8_e4m3fn
    )

    for g in range(num_experts):
        a_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)

    a_scale_stack = torch.empty(
        (expert_offsets[-1], k_g // 128), device=device, dtype=torch.float32
    )
    b_scale_stack = torch.empty(
        (num_experts, n_g // 128, k_g // 128), device=device, dtype=torch.float32
    )

    for g in range(num_experts):
        a_scale_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_scales_tensors[g]
        b_scale_stack[g] = b_scales_tensors[g].t()
    b_scale_stack = b_scale_stack.transpose(1, 2)

    c_out = torch.empty((expert_offsets[-1], n_g), device=device, dtype=out_dtype)
    a_strides = torch.full(
        (num_experts,), a_stack.stride(0), device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_experts,), c_out.stride(0), device=device, dtype=torch.int64
    )
    workspace = torch.empty((1024 * 1024 * 1024), device=device, dtype=torch.uint8)
    a_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    b_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    out_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    a_scales_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    b_scales_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)

    fp8_blockwise_scaled_grouped_mm(
        c_out,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a_stack,
        b_stack,
        a_scale_stack,
        b_scale_stack,
        a_strides,
        a_strides,
        c_strides,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets[:-1],
        workspace,
    )

    for g in range(num_experts):
        baseline = baseline_tensors[g]
        actual = c_out[expert_offsets[g] : expert_offsets[g + 1]]
        torch.testing.assert_close(actual, baseline, rtol=1e-2, atol=5e-4)
        print(f"num_experts={num_experts}, out_dtype={out_dtype}: OK")


if __name__ == "__main__":
    pytest.main([__file__])
