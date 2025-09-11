import random
from typing import Tuple

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


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
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


def is_sm90_supported(device=None) -> bool:
    return (torch.cuda.get_device_capability(device)[0] == 9) and (
        torch.version.cuda >= "12.3"
    )


@pytest.mark.skipif(
    not (is_sm100_supported() or is_sm90_supported()),
    reason="fp8_blockwise_scaled_grouped_mm at sgl-kernel is only supported on sm100 or sm90",
)
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_fp8_blockwise_scaled_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128
    n_g = random.randint(1, 64) * 128
    k_g = random.randint(1, 64) * 128

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
        m_g = random.randint(1, 256)
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a = torch.randn((m_g, k_g), device=device, dtype=out_dtype)  # (M, K):(K, 1)
        b = torch.randn((n_g, k_g), device=device, dtype=out_dtype).t()  # (K, N):(1, K)

        a_g, a_scale = per_token_cast_to_fp8(
            a
        )  # ag -- (M, K):(K, 1), a_scale() -- (M, k):(k, 1)
        b_g, b_scale = per_block_cast_to_fp8(
            b
        )  # bg -- (K, N):(N, 1), b_scale() -- (k, n):(n, 1)
        a_tensors.append(a_g)
        b_tensors.append(b_g)
        a_scales_tensors.append(a_scale)
        b_scales_tensors.append(b_scale)

        baseline = torch.mm(a, b)
        baseline_tensors.append(baseline)
    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn
    )
    b_stack = torch.empty(
        (num_experts, n_g, k_g), device=device, dtype=torch.float8_e4m3fn
    )
    a_scale_stack = torch.empty(
        (expert_offsets[-1], (k_g // 128)), device=device, dtype=torch.float32
    )
    b_scale_stack = torch.empty(
        (num_experts, n_g // 128, k_g // 128), device=device, dtype=torch.float32
    )

    for g in range(num_experts):
        # Matrix A is Row-Major.
        a_stack[expert_offsets[g] : expert_offsets[g + 1], :] = a_tensors[
            g
        ]  # a_stack[expert_offsets[g] : expert_offsets[g + 1], :] -- (M, K):(K, 1)
        b_stack[g] = b_tensors[g].t()  # b_stack[g] -- (N, K):(K, 1)

        # We need K-Major scale factor
        a_scale_stack[expert_offsets[g] : expert_offsets[g + 1], :] = a_scales_tensors[
            g
        ]
        b_scale_stack[g] = b_scales_tensors[
            g
        ].t()  # b_scale_stack[g] -- (k, n):(n, 1), we need transpose & contiguous later
    b_stack = b_stack.transpose(1, 2)  # Transpose Matrix B to Column-Major.
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
        diff = calc_diff(actual, baseline)
        assert diff < 0.001
        print(
            f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, out_dtype={out_dtype}, diff={diff:.5f}: OK"
        )


if __name__ == "__main__":
    pytest.main([__file__])
