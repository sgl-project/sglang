# Standalone DeepEP Wheel Builder Design

## Goal

Add a host-side build script that turns a clean CUDA-enabled SGLang container
into a DeepEP build environment and produces a wheel without invoking Docker or
depending on CI setup scripts.

## Interface

The entry point is `scripts/build_sgl_deepep.sh [OUTPUT_DIR]`. It runs in the
current container, uses the current `python3` and CUDA toolkit, and writes the
wheel to `OUTPUT_DIR` (default: `$PWD/dist`). It must be run as root or by a user
with passwordless `sudo`, because it installs apt and dpkg packages.

The following environment variables are supported:

- `CUDA_HOME`, default `/usr/local/cuda`.
- `PYTHON_BIN`, default `python3`.
- `MAX_JOBS`, default the number of online CPUs.

## Platform Selection

The script obtains the host architecture from `uname -m` and the toolkit major
version from `${CUDA_HOME}/bin/nvcc --version`. Only this matrix is supported:

| Architecture | CUDA major | DeepEP branch |
| --- | --- | --- |
| `x86_64` | 12 or 13 | `sgl-deepep-x86` |
| `aarch64` | 13 | `sgl-deepep-arm` |
| `aarch64` | 12 | `sgl-deepep-cu12-arm` |

Any other architecture or CUDA major fails before dependency installation.

## Build Flow

1. Validate `git`, the selected Python interpreter, pip, CUDA, and PyTorch.
2. Remove an installed `deep_ep` distribution and old `/opt/gdrcopy` and
   temporary DeepEP source directories. Do not remove PyTorch or SGLang.
3. Install the system, RDMA, compiler, and packaging dependencies derived from
   `scripts/ci/cuda/ci_install_deepep.sh`.
4. Build and install GDRCopy v2.5.1 Debian packages, then ensure the unversioned
   `libmlx5.so` link and `libfabric-dev` are available.
5. Clone `https://github.com/sgl-project/DeepEP.git` at the selected branch into
   a temporary directory.
6. For CUDA 13, append `${CUDA_HOME}/include/cccl` to the traditional DeepEP
   extension's `include_dirs` in `setup.py`; fail if the expected insertion
   point or CCCL directory is absent.
7. Run the selected Python interpreter's `setup.py bdist_wheel` with
   `TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3'` and `MAX_JOBS`.
8. Verify exactly one `deep_ep-*.whl` was produced and print its absolute path.

The cloned source is removed on exit. Installed system dependencies and GDRCopy
remain available because they are runtime prerequisites.

## Error Handling and Repeatability

The script uses Bash strict mode and stops on failed package installation,
clone, source patch, or build. Apt failures use the CI installer's fallback:
continue only when every requested package is already installed. Re-running the
script uninstalls an existing DeepEP package, rebuilds GDRCopy from the pinned
tag, and replaces only same-named wheel artifacts in the requested output
directory.

## Verification

Local tests source and execute the real script functions to verify all four
platform cells, unsupported platform rejection, CUDA 13 CCCL injection, output
handling, and command failures. The wheel-build test uses a real minimal
setuptools project whose build rejects the wrong architecture list. Static
checks include `bash -n` and ShellCheck when available.

End-to-end validation uses two 4-GPU devboxes, reusing each devbox across two
images after reprovisioning:

| GPU / architecture | `latest-cu129` | `latest-cu130` |
| --- | --- | --- |
| 4x B200 / x86_64 | build, install wheel, test | build, install wheel, test |
| 4x GB300 / aarch64 | build, install wheel, test | build, install wheel, test |

Before every matrix cell, remove installed DeepEP and prior DeepEP/GDRCopy build
artifacts. After building, install the generated wheel and run
`TestDSV4FlashFP4B200Balanced` from
`test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`. A matrix cell
passes only when wheel build, installation/import, and that test all succeed.
