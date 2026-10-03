"""Opt-in ROCm device parity for the real FP8 helper and extracted wrappers.

Not run by the CPU suite. On a disposable ROCm test machine:
  SGL_RUN_DSV4_FP8_GPU=1 python -m pytest -q <this-file>

Expected bytes come from the independent C++ CPU golden decoder/search, not a
Python reimplementation. This tests device code generation, NOT deployment of
sgl_kernel.common_ops. test_dsv4_norm_rope.py has an installed-AOT byte check.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from test_dsv4_fp8_cpu import JIT_INCLUDE, compile_cpu, wrapper_source


@pytest.mark.skipif(
    os.environ.get("SGL_RUN_DSV4_FP8_GPU") != "1",
    reason="Opt-in only: set SGL_RUN_DSV4_FP8_GPU=1 on a ROCm test machine",
)
def test_fp8_device_bytes_match_cpu_golden():
    import torch
    from torch.utils.cpp_extension import load_inline

    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires ROCm and a GPU")
    with tempfile.TemporaryDirectory(prefix="dsv4-fp8-gpu-") as directory:
        binary = Path(directory) / "cpu_golden"
        compile_cpu(binary)
        emitted = subprocess.run(
            [str(binary), "--emit"], check=True, capture_output=True, text=True
        ).stdout
    rows = [list(map(int, line.split())) for line in emitted.splitlines()]
    source = (
        wrapper_source()
        + r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void convert(const int32_t* input, int32_t* output, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t bits = static_cast<uint32_t>(input[i]);
  const float x = __uint_as_float(bits);
  const float y = __uint_as_float(static_cast<uint32_t>(input[(i + 997) % n]));
  output[i * 8 + 0] = sglang::deepseek_v4::fp8::f32_to_fp8_e4m3_bits<false>(bits);
  output[i * 8 + 1] = sglang::deepseek_v4::fp8::f32_to_fp8_e4m3_bits<true>(bits);
  output[i * 8 + 2] = aot::cvt_float_to_fp8_e4m3(x);
  output[i * 8 + 3] = jit_fn::cvt_float_to_fp8_e4m3(x);
  output[i * 8 + 4] = jit_fnuz::cvt_float_to_fp8_e4m3(x);
  output[i * 8 + 5] = aot::pack_fp8(x, y);
  output[i * 8 + 6] = jit_fn::pack_fp8(x, y);
  output[i * 8 + 7] = jit_fnuz::pack_fp8(x, y);
}
torch::Tensor convert_fp8(torch::Tensor input) {
  auto output = torch::empty({input.numel(), 8}, input.options());
  convert<<<(input.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int32_t>(), output.data_ptr<int32_t>(), input.numel());
  return output;
}
"""
    )
    extension = load_inline(
        name="dsv4_fp8_software_parity",
        cpp_sources="torch::Tensor convert_fp8(torch::Tensor input);",
        cuda_sources=source,
        functions=["convert_fp8"],
        extra_include_paths=[str(JIT_INCLUDE)],
        extra_cuda_cflags=["-O3", "-DUSE_ROCM"],
    )
    inputs = torch.tensor(
        [r[0] if r[0] < 2**31 else r[0] - 2**32 for r in rows],
        dtype=torch.int32,
        device="cuda",
    )
    expected = []
    for i, (_, raw_fn, raw_fnuz, clip_fn, clip_fnuz) in enumerate(rows):
        other = rows[(i + 997) % len(rows)]
        pack_fn = clip_fn | (other[3] << 8)
        pack_fnuz = clip_fnuz | (other[4] << 8)
        expected.append(
            [raw_fn, raw_fnuz, clip_fn, clip_fn, clip_fnuz, pack_fn, pack_fn, pack_fnuz]
        )
    torch.testing.assert_close(
        extension.convert_fp8(inputs).cpu(),
        torch.tensor(expected, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
