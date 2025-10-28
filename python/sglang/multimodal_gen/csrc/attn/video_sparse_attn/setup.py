import os
import subprocess
import sys
from pathlib import Path

# Ensure local imports and relative paths resolve when run under PEP 517
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from config_vsa import kernels, sources, target
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

target = target.lower()

# Package metadata
PACKAGE_NAME = "vsa"
VERSION = "0.0.3"
AUTHOR = "Hao AI Lab"
DESCRIPTION = "Video Sparse Attention Kernel Used in sgl-diffusion"
URL = (
    "https://github.com/hao-ai-lab/sgl-diffusion/tree/main/csrc/attn/video_sparse_attn"
)

_tk_env = os.getenv("THUNDERKITTENS_ROOT", "").strip()
candidate_tk_roots = [
    (ROOT_DIR / "tk"),
    (ROOT_DIR / "../sliding_tile_attn/tk").resolve(),
    (Path(_tk_env) if _tk_env else None),
]
candidate_tk_roots = [p for p in candidate_tk_roots if p]

def _find_tk_root() -> str | None:
    for cand in candidate_tk_roots:
        if (cand / "include" / "kittens.cuh").exists():
            return str(cand.resolve())
    return None

tk_root = _find_tk_root()
if tk_root is None:
    # Try to initialize submodules if available
    repo = ROOT_DIR
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    if (repo / ".git").exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=str(repo), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception:
            pass
        tk_root = _find_tk_root()

if tk_root is None:
    raise RuntimeError(
        "ThunderKittens headers not found. Set THUNDERKITTENS_ROOT or ensure tk submodule exists under video_sparse_attn/tk or sliding_tile_attn/tk"
    )

# Python and Torch include paths
python_include = (
    subprocess.check_output(
        ["python", "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
    )
    .decode()
    .strip()
)
torch_include = (
    subprocess.check_output(
        [
            "python",
            "-c",
            "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))",
        ]
    )
    .decode()
    .strip()
)
print("vsa tk root:", tk_root)
print("Python include:", python_include)
print("Torch include directories:", torch_include)

# CUDA flags
cuda_flags = [
    "-DNDEBUG",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-forward-unknown-to-host-compiler",
    "--use_fast_math",
    "-std=c++20",
    "-O3",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    f"-I{tk_root}/include",
    f"-I{tk_root}/prototype",
    f"-I{python_include}",
    "-DTORCH_COMPILE",
] + torch_include.split()
cpp_flags = ["-std=c++20", "-O3"]

if target == "h100":
    cuda_flags.append("-DKITTENS_HOPPER")
    cuda_flags.append("-arch=sm_90a")
else:
    raise ValueError(f"Target {target} not supported")

source_files = ["vsa.cpp"]
for k in kernels:
    if target not in sources[k]["source_files"]:
        raise KeyError(f"Target {target} not found in source files for kernel {k}")
    if isinstance(sources[k]["source_files"][target], list):
        source_files.extend(sources[k]["source_files"][target])
    else:
        source_files.append(sources[k]["source_files"][target])
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')


ext_modules = [
    CUDAExtension(
        "vsa_cuda",
        sources=source_files,
        extra_compile_args={"cxx": cpp_flags, "nvcc": cuda_flags},
        include_dirs=[python_include, f"{tk_root}/include", f"{tk_root}/prototype"],
        libraries=["cuda"],
    )
]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    url=URL,
    packages=["vsa"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.10",
    install_requires=["torch>=2.5.0"],
)
