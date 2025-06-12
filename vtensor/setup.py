import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension

setup(
    name="vTensor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch >= 2.0",
    ],
    ext_modules=[
        CppExtension(
            name="vTensor",
            sources=[
                "vtensor.cpp",
            ],
            include_dirs=[os.path.join(CUDA_HOME, "include")],
            library_dirs=[
                os.path.join(CUDA_HOME, "lib64"),
                os.path.join(CUDA_HOME, "lib64", "stubs"),
            ],
            libraries=["cuda", "cudart"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    author="antgroup",
    author_email="@antgroup.com",
    description="VMM-based Tensor library for FlowMLA",
)
