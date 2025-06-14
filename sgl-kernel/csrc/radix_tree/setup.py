from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="radix_tree_cpp",
    ext_modules=[
        CppExtension(
            name="radix_tree_cpp",
            sources=[
                "tree_v2.cpp",
                "tree_v2_debug.cpp",
                "tree_v2_binding.cpp",
            ],  # your C++ file
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
