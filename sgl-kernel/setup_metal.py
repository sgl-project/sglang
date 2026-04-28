# Copyright 2026 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

root = Path(__file__).parent.resolve()


_BUILD_REQUIRES = [
    ("setuptools", "setuptools"),
    ("mlx", "mlx==0.31.1"),
    ("nanobind", "nanobind"),
]


def _ensure_toolchain():
    if sys.platform != "darwin":
        raise SystemExit("setup_metal.py only supports macOS (Apple Silicon).")
    if shutil.which("c++") is None or shutil.which("xcrun") is None:
        raise SystemExit(
            "Apple toolchain not found. Install the Xcode Command Line Tools "
            "with `xcode-select --install` (or a full Xcode install) and retry."
        )
    try:
        subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Apple Metal shader compiler not found. Install a full Xcode "
            "(not just Command Line Tools) so that `xcrun -sdk macosx metal` "
            "is available, then retry."
        ) from exc


def _ensure_build_requires():
    missing = []
    for import_name, pip_name in _BUILD_REQUIRES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        return
    print(
        f"[sgl-kernel:metal] installing build requirements: {missing}",
        flush=True,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *missing]
    )


# Section 1: Prerequisites
_ensure_toolchain()
_ensure_build_requires()
os.chdir(root)


# Section 2: Build and install
from setuptools import Extension, find_packages, setup  # noqa: E402
from setuptools.command.build_ext import build_ext  # noqa: E402


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
metallib_name = "sgl_metal_kernels.metallib"

# Metal shader sources (compiled with `xcrun metal`) and C++ host sources
# (compiled with `c++`). Add new kernels by appending to these lists.
metal_shader_sources = [
    "csrc/metal/placeholder.metal",
]
cxx_sources = [
    "csrc/metal/placeholder.cpp",
]

# Header search paths shared by both the Metal shader compiler and the C++
# host compiler.
include_dirs = [
    root / "csrc",
    root / "csrc" / "metal",
]

cxx_flags = ["-std=c++17", "-O3", "-fvisibility=hidden"]
metal_flags = ["-O3"]
frameworks = ["Metal", "Foundation", "QuartzCore"]
libraries = ["mlx"]


class BuildMetalExtension(build_ext):
    def build_extension(self, ext):
        if sys.platform != "darwin":
            raise RuntimeError("setup_metal.py only supports macOS")

        ext_path = Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        python_exe = Path(sys.executable)
        python_include = Path(sysconfig.get_paths()["include"])
        python_lib = Path(sysconfig.get_config_var("LIBDIR"))
        # Match the deployment target that Python itself was built against
        # unless the user overrides it. MLX's prebuilt wheels may require a
        # higher minimum; in that case set MACOSX_DEPLOYMENT_TARGET explicitly.
        deployment_target = os.environ.get(
            "MACOSX_DEPLOYMENT_TARGET",
            str(sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "11.0"),
        )

        def _python_eval(expr: str) -> str:
            return subprocess.check_output(
                [str(python_exe), "-c", expr], text=True
            ).strip()

        nanobind_dir = Path(
            _python_eval("import nanobind; print(nanobind.__path__[0])")
        )
        mlx_dir = Path(_python_eval("import mlx.core as mx; print(mx.__file__)"))
        mlx_site = mlx_dir.parent
        mlx_include = mlx_site / "include"
        mlx_lib = mlx_site / "lib"

        generated_dir = root / "build" / "metal"
        generated_dir.mkdir(parents=True, exist_ok=True)

        metallib_path = generated_dir / metallib_name
        metal_std = os.environ.get("SGL_METAL_STD", "metal3.1")

        ext_include_dirs = [Path(p) for p in (ext.include_dirs or [])]
        host_includes = [
            python_include,
            nanobind_dir / "include",
            nanobind_dir / "ext" / "robin_map" / "include",
            mlx_include,
            mlx_include / "metal_cpp",
        ]
        all_includes = ext_include_dirs + host_includes
        include_args = [f"-I{p}" for p in all_includes]
        # `xcrun metal` accepts `-I` for header search; reuse the project
        # include dirs so shaders can include shared MSL headers.
        metal_include_args = [f"-I{p}" for p in ext_include_dirs]

        if not metal_shader_sources:
            raise RuntimeError("metal_shader_sources is empty; nothing to compile")

        air_paths = []
        for rel in metal_shader_sources:
            metal_src = root / rel
            if not metal_src.is_file():
                raise RuntimeError(f"metal shader source not found: {metal_src}")
            air_path = generated_dir / (metal_src.stem + ".air")
            self.spawn(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    f"-std={metal_std}",
                    *metal_flags,
                    *metal_include_args,
                    "-c",
                    str(metal_src),
                    "-o",
                    str(air_path),
                ]
            )
            air_paths.append(str(air_path))

        self.spawn(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metallib",
                *air_paths,
                "-o",
                str(metallib_path),
            ]
        )

        cflags = [
            *cxx_flags,
            f"-mmacosx-version-min={deployment_target}",
            *include_args,
        ]

        ldflags = [
            "-shared",
            "-undefined",
            "dynamic_lookup",
            f"-mmacosx-version-min={deployment_target}",
            f"-L{python_lib}",
            f"-L{mlx_lib}",
            f"-Wl,-rpath,{mlx_lib}",
            *[f"-l{lib}" for lib in libraries],
            *[arg for fw in frameworks for arg in ("-framework", fw)],
        ]

        objects = []
        for src in ext.sources:
            src_path = Path(src)
            obj_path = generated_dir / (src_path.stem + ".o")
            compile_cmd = [
                "c++",
                *cflags,
                "-c",
                str(src_path),
                "-o",
                str(obj_path),
            ]
            self.spawn(compile_cmd)
            objects.append(str(obj_path))

        nanobind_src = nanobind_dir / "src" / "nb_combined.cpp"
        nanobind_obj = generated_dir / "nb_combined.o"
        nanobind_cmd = [
            "c++",
            *cflags,
            "-DNB_COMPACT_ASSERTIONS",
            "-DNB_BUILD",
            "-DNB_SHARED",
            "-c",
            str(nanobind_src),
            "-o",
            str(nanobind_obj),
        ]
        self.spawn(nanobind_cmd)
        objects.append(str(nanobind_obj))

        link_cmd = [
            "c++",
            *objects,
            *ldflags,
            "-o",
            str(ext_path),
        ]
        self.spawn(link_cmd)

        # Stage the metallib next to the freshly-linked extension so that
        # `install_lib` picks it up via `package_data={"sgl_kernel": ["*.metallib"]}`.
        staged_metallib = ext_path.parent / metallib_path.name
        if metallib_path.resolve() != staged_metallib.resolve():
            shutil.copy2(metallib_path, staged_metallib)


ext_modules = [
    Extension(
        name=f"{operator_namespace}._metal",
        sources=cxx_sources,
        include_dirs=[str(p) for p in include_dirs],
        language="c++",
    )
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"sgl_kernel": ["*.metallib"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildMetalExtension},
)
