from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sgl-kernel",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "sgl_kernel.ops.warp_reduce_cuda",
            [
                "src/sgl-kernel/csrc/warp_reduce.cc",
                "src/sgl-kernel/csrc/warp_reduce_kernel.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-Xcompiler",
                    "-fPIC",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
                "cxx": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
