from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
INCLUDE_DIR = BASE_DIR / "include"
SYSTEM_INCLUDE_DIR = "/usr/include"  # cppzmq header path, e.g. /usr/include/zmq.hpp
zmq_lib_dir = "/usr/lib"  # ZMQ library path
zmq_libs = ["zmq"]  # Link against libzmq.so


ext_modules = [
    Extension(
        "spectre_zmq",  # Name of the generated Python module
        sources=[
            str(SRC_DIR / "spectre_zmq.cpp"),
            str(SRC_DIR / "spectre_zmq_logging.cpp"),
            str(SRC_DIR / "spectre_zmq_serialization.cpp"),
            str(SRC_DIR / "spectre_zmq_endpoints.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),  # pybind11 headers
            str(INCLUDE_DIR),
            SYSTEM_INCLUDE_DIR,
        ],
        library_dirs=[zmq_lib_dir],
        libraries=zmq_libs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall", "-fPIC"],
        extra_link_args=[],
    )
]


setup(
    name="spectre_zmq",
    version="0.1.0",
    author="JD",
    description="Full duplex ZMQ C++ module for Python",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
