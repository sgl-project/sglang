import pathlib
from setuptools import setup, find_packages
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent
print(root)


def glob(pattern):
  return [str(p) for p in root.glob(pattern)]

def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass


remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        "sglang._kernels",
        ["sglang/srt/csrc/lora_ops.cc"] +
        glob("sglang/srt/csrc/bgmv/*.cu"),
        extra_compile_args=['-std=c++17'],
    ))

setup(
    name="sglang",
    version="0.1.11",
    description="A structured generation language for LLMs.",
    long_description="",
    long_description_content_type="text/markdown",
    license="Apache Software License",
    url="https://github.com/sgl-project/sglang",
    project_urls={
        "Bug Tracker": "https://github.com/sgl-project/sglang/issues",
        "Homepage": "https://github.com/sgl-project/sglang",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    packages=find_packages(
        exclude=("assets*", "benchmark*", "build", "csrc", "dist*", "docs*",
                 "include", "playground*", "scripts*", "tests*", "sglang.egg-info")
    ),
    include_package_data=True,
    install_requires=[
        "requests",
    ],
    extras_require={
        "srt": [
            "aiohttp", "fastapi", "psutil", "rpyc", "torch", "uvloop", "uvicorn",
            "zmq", "vllm>=0.2.5", "interegular", "lark", "numba",
            "pydantic", "referencing", "diskcache", "cloudpickle", "pillow",
        ],
        "openai": [
            "openai>=1.0", "numpy",
        ],
        "anthropic": [
            "anthropic", "numpy",
        ],
        "all": [
            "sglang[srt]", "sglang[openai]", "sglang[anthropic]",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
