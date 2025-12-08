# export MAX_JOBS=4
# python setup.py build_ext --inplace

import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

def gather_include_dirs():
    includes = []
    # Local SGLang kernel include (if present)
    sgl_include = os.path.normpath(os.path.join(this_dir, "..", "..", "include"))
    if os.path.isdir(sgl_include):
        includes.append(sgl_include)
    # Prefer CUTLASS under diffusion/sglang/ (repo root)
    sglang_root = os.path.normpath(os.path.join(this_dir, "..", "..", ".."))
    for cutlass_root in (
        os.path.join(sglang_root, "cutlass"),
        os.path.join(sglang_root, "third_party", "cutlass"),
    ):
        c_inc = os.path.join(cutlass_root, "include")
        c_util = os.path.join(cutlass_root, "tools", "util", "include")
        for p in (c_inc, c_util):
            if os.path.isdir(p):
                includes.append(p)
    # CUTLASS from local third_party/cmake deps (if present)
    cutlass_root = os.path.normpath(os.path.join(this_dir, "..", "..", "build", "_deps", "repo-cutlass-src"))
    cutlass_include = os.path.join(cutlass_root, "include")
    cutlass_tools_util_include = os.path.join(cutlass_root, "tools", "util", "include")
    for p in (cutlass_include, cutlass_tools_util_include):
        if os.path.isdir(p):
            includes.append(p)
    # CUTLASS from env CUTLASS_PATH (if provided)
    env_cutlass = os.environ.get("CUTLASS_PATH")
    if env_cutlass:
        c_inc = os.path.join(env_cutlass, "include")
        c_util = os.path.join(env_cutlass, "tools", "util", "include")
        for p in (c_inc, c_util):
            if os.path.isdir(p):
                includes.append(p)
    # Fallback commonly used paths (if exist)
    for p in ("/workspace/cutlass/include", "/workspace/cutlass/tools/util/include"):
        if os.path.isdir(p):
            includes.append(p)
    return includes

setup(
    name='fused_layernorm_scale_shift',
    ext_modules=[
        CUDAExtension(
            name='fused_layernorm_scale_shift',
            sources=['fused_layernorm_scale_shift.cu', 'bindings.cpp'],
            include_dirs=gather_include_dirs(),
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--use_fast_math'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)