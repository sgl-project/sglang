"""CPU-only FP8 regression tests; no torch, HIP/CUDA compiler, or GPU required.

Run directly with Python, or via pytest. CXX and FP8_TEST_CXXFLAGS can select a
host compiler/sanitizer. The C++ test includes the production integer helper.
This runner also extracts the actual clipping/packing wrapper bodies from the
AOT .cu and JIT .cuh, changing only device annotations/intrinsics for the host.
It deliberately does not maintain a second Python copy of the encoder.
"""

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[2]
JIT_INCLUDE = KERNELS / "jit/include"
CPP_TEST = Path(__file__).with_suffix(".cpp")


def extract_function(source, name, *, last=False):
    """Copy a complete real function, including nested braces/preprocessor code."""
    matches = list(re.finditer(rf"^.*\b{re.escape(name)}\([^\n]*\) \{{", source, re.M))
    if not matches:
        raise AssertionError(f"Cannot locate production function {name}")
    start = matches[-1 if last else 0].start()
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def wrapper_source(aot=None, jit=None):
    """Host or device test wrappers; never copy/reimplement encoding logic."""
    if aot is None:
        aot = (KERNELS / "aot/csrc/elementwise/dsv4_norm_rope.cu").read_text()
    if jit is None:
        jit = (JIT_INCLUDE / "sgl_kernel/deepseek_v4/fp8_utils.cuh").read_text()
    types = (JIT_INCLUDE / "sgl_kernel/type.cuh").read_text()
    max_start = types.rindex("#ifndef USE_ROCM")
    max_end = types.index("\n}  // namespace sglang", max_start)
    jit_max = types[max_start:max_end]
    aot_max = re.search(r"static constexpr float kFP8Max = [^;]+;", aot).group()
    code = f"""
#include <sgl_kernel/deepseek_v4/fp8_e4m3.h>
#include <cstdint>
#include <cmath>
#ifndef __HIPCC__
#include <cstring>
#define __device__
#define __forceinline__ inline
inline uint32_t __float_as_uint(float x) {{
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}}
#endif
#define SGL_DEVICE __device__ __forceinline__
#ifndef USE_ROCM
#define USE_ROCM 1
#endif
using fp8x2_e4m3_t = uint16_t;
namespace aot {{
{aot_max}
{extract_function(aot, "cvt_float_to_fp8_e4m3")}
{extract_function(aot, "pack_fp8", last=True)}
}}
"""
    for namespace, fnuz in [("jit_fn", 0), ("jit_fnuz", 1)]:
        code += f"""
#undef HIP_FP8_TYPE_FNUZ
#define HIP_FP8_TYPE_FNUZ {fnuz}
namespace {namespace} {{
using sglang::deepseek_v4::fp8::f32_to_fp8_e4m3_bits;
{jit_max}
{extract_function(jit, "fp8_e4m3_clip")}
{extract_function(jit, "cvt_float_to_fp8_e4m3")}
{extract_function(jit, "pack_fp8", last=True)}
}}
"""
    return code


def compile_cpu(output, *, wrappers=True, aot=None, jit=None):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        raise RuntimeError(f"Host C++ compiler unavailable: {compiler[0]}")
    flags = shlex.split(os.environ.get("FP8_TEST_CXXFLAGS", "-O2"))
    command = compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror"] + flags
    if wrappers:
        header = output.with_suffix(".h")
        header.write_text(wrapper_source(aot, jit))
        command.append(f'-DSGL_FP8_WRAPPER_HEADER="{header}"')
    command += [f"-I{JIT_INCLUDE}", str(CPP_TEST), "-o", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"Host compile failed:\n{completed.stdout}\n{completed.stderr}"
        )


class TestDSV4FP8CPU(unittest.TestCase):
    def test_real_helper_and_callers(self):
        with tempfile.TemporaryDirectory(prefix="dsv4-fp8-cpu-") as directory:
            binary = Path(directory) / "test_fp8"
            compile_cpu(binary)
            completed = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertIn("PASS:", completed.stdout)
            print(completed.stdout, end="")


if __name__ == "__main__":
    unittest.main()
