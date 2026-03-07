import pytest
import torch

from sglang.jit_kernel.nvfp4 import (
    cutlass_fp4_group_mm,
    scaled_fp4_experts_quant,
    scaled_fp4_quant,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _build_expert_offsets(
    m_per_expert: list[int], device: torch.device
) -> torch.Tensor:
    offsets = [0]
    for m in m_per_expert:
        offsets.append(offsets[-1] + m)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _build_blockscale_offsets(
    m_per_expert: list[int], device: torch.device
) -> torch.Tensor:
    offsets = [0]
    for m in m_per_expert:
        offsets.append(offsets[-1] + _round_up(m, 128))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


@pytest.mark.skipif(
    not _nvfp4_supported(), reason="NVFP4 requires compute capability >= 10.0"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_blockwise_moe_grouped_mm(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")

    num_experts = 4
    m_per_expert = [33, 17, 48, 29]
    n = 256
    k = 128

    expert_offsets_full = _build_expert_offsets(m_per_expert, device)
    blockscale_offsets_full = _build_blockscale_offsets(m_per_expert, device)

    total_m = int(expert_offsets_full[-1].item())
    a = torch.randn((total_m, k), device=device, dtype=dtype) * 0.1
    b = torch.randn((num_experts, n, k), device=device, dtype=dtype) * 0.1

    a_global_scale = torch.empty((num_experts,), device=device, dtype=torch.float32)
    for i in range(num_experts):
        start = int(expert_offsets_full[i].item())
        end = int(expert_offsets_full[i + 1].item())
        amax = a[start:end].abs().max().to(torch.float32)
        a_global_scale[i] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax

    b_global_scale = torch.empty((num_experts,), device=device, dtype=torch.float32)
    for i in range(num_experts):
        bmax = b[i].abs().max().to(torch.float32)
        b_global_scale[i] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / bmax

    a_fp4, a_blockscale = scaled_fp4_experts_quant(
        a,
        a_global_scale,
        expert_offsets_full,
        blockscale_offsets_full,
        topk=1,
    )

    b_fp4 = torch.empty((num_experts, n, k // 2), device=device, dtype=torch.uint8)
    b_blockscale = torch.empty(
        (num_experts, _round_up(n, 128), _round_up(k // 16, 4)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    for i in range(num_experts):
        b_fp4_i, b_scale_i = scaled_fp4_quant(b[i], b_global_scale[i])
        b_fp4[i].copy_(b_fp4_i)
        b_blockscale[i].copy_(b_scale_i)

    alphas = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

    params = {
        "ab_strides": torch.full((num_experts,), k, dtype=torch.int64, device=device),
        "c_strides": torch.full((num_experts,), n, dtype=torch.int64, device=device),
        "problem_sizes": torch.tensor(
            [[m, n, k] for m in m_per_expert], dtype=torch.int32, device=device
        ),
        "expert_offsets": expert_offsets_full[:-1].contiguous(),
        "blockscale_offsets": blockscale_offsets_full[:-1].contiguous(),
        "a_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "b_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "out_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "a_scales_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "b_scales_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "alpha_ptrs": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "layout_sfa": torch.empty((num_experts, 5), dtype=torch.int64, device=device),
        "layout_sfb": torch.empty((num_experts, 5), dtype=torch.int64, device=device),
    }

    out = cutlass_fp4_group_mm(
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        dtype,
        params,
    )

    ref = torch.empty((total_m, n), device=device, dtype=dtype)
    for i in range(num_experts):
        start = int(expert_offsets_full[i].item())
        end = int(expert_offsets_full[i + 1].item())
        ref[start:end] = torch.matmul(a[start:end], b[i].t())

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
