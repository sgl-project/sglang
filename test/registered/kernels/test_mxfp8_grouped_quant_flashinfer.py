import random
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="nightly-4-gpu-b200", nightly=True)


def _cuda_version_at_least(major: int, minor: int) -> bool:
    if torch.version.cuda is None:
        return False
    version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return version >= (major, minor)


# This test validates quantized output with the SM100 CUTLASS grouped GEMM ops.
# SM120 MXFP8 MoE uses a different GEMM path, and is intentionally excluded.
if not (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and _cuda_version_at_least(12, 8)
):
    pytest.skip(
        "FlashInfer MXFP8 grouped quantization requires SM100 with CUDA 12.8+.",
        allow_module_level=True,
    )

try:
    from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_mm
except Exception as exc:
    pytest.skip(f"sgl_kernel is unavailable: {exc}", allow_module_level=True)

from sglang.srt.layers.quantization import mxfp8_grouped_quant

if not mxfp8_grouped_quant.is_flashinfer_mxfp8_grouped_quant_available():
    pytest.skip(
        "FlashInfer MXFP8 grouped quantization backend is unavailable.",
        allow_module_level=True,
    )


def _align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _set_flashinfer_backend():
    mxfp8_grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
        mxfp8_grouped_quant.Mxfp8GroupedQuantBackend.FLASHINFER
    )
    mxfp8_grouped_quant._load_flashinfer_mxfp8_grouped_quant.cache_clear()


def _reset_backend():
    mxfp8_grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = None
    mxfp8_grouped_quant._load_flashinfer_mxfp8_grouped_quant.cache_clear()


@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_flashinfer_mxfp8_grouped_quant_backend(num_experts, out_dtype):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(num_experts)

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
    m_per_expert = []

    for _ in range(num_experts):
        m_g = random.randint(1, 512)
        m_per_expert.append(m_g)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += _align(m_g, 128)
        b_blockscale_offsets.append(b_blockscale_offset)
        b_blockscale_offset += n_g
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        a = torch.normal(0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype)
        b = torch.normal(0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype)

        a_list.append(a)
        b_list.append(b)
        ref_d_list.append(a @ b.T)

    a = torch.concat(a_list, dim=0)
    b = torch.concat(b_list, dim=0)

    _problem_sizes = torch.tensor(problem_sizes, device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(
        aux_problem_sizes, device=device, dtype=torch.int32
    )
    _expert_offsets = torch.tensor(expert_offsets, device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(
        aux_expert_offsets, device=device, dtype=torch.int32
    )
    _a_blockscale_offsets = torch.tensor(
        a_blockscale_offsets, device=device, dtype=torch.int32
    )
    _b_blockscale_offsets = torch.tensor(
        b_blockscale_offsets, device=device, dtype=torch.int32
    )

    a_quant = torch.zeros_like(a, dtype=torch.float8_e4m3fn, device=device)
    a_scale_factor = torch.zeros(
        (a_blockscale_offset, k_g // 32), dtype=torch.uint8, device=device
    )

    b_quant = torch.zeros_like(b, dtype=torch.float8_e4m3fn, device=device)
    b_scale_factor = torch.zeros(
        (num_experts, n_g, k_g // 32), dtype=torch.uint8, device=device
    )

    _set_flashinfer_backend()
    try:
        mxfp8_grouped_quant.mxfp8_grouped_quant(
            a,
            _problem_sizes,
            _expert_offsets,
            _a_blockscale_offsets,
            a_quant,
            a_scale_factor,
        )
        mxfp8_grouped_quant.mxfp8_grouped_quant(
            b,
            _aux_problem_sizes,
            _aux_expert_offsets,
            _b_blockscale_offsets,
            b_quant,
            b_scale_factor,
        )
    finally:
        _reset_backend()

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

    for g, m_g in enumerate(m_per_expert):
        baseline = ref_d_list[g]
        actual = d[expert_offsets[g] : expert_offsets[g] + m_g]
        diff = _calc_diff(actual, baseline)
        assert diff < 0.001


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
