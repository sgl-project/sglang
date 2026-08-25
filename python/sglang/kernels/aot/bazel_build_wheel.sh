#!/usr/bin/env bash
set -euo pipefail

backend="$1"
output_dir="$2"
source_root="$3"
shift 3

exec_root="${PWD}"
resolve_executable() {
  local executable="$1"
  if [[ "${executable}" == */* ]]; then
    if [[ "${executable}" != /* ]]; then
      executable="${exec_root}/${executable}"
    fi
    [[ -x "${executable}" ]] || return 1
    printf '%s\n' "${executable}"
  else
    command -v "${executable}"
  fi
}

if ! python_bin="$(resolve_executable "${PYTHON_BIN_PATH:-python3}")"; then
  echo "Python toolchain executable not found: ${PYTHON_BIN_PATH:-python3}" >&2
  exit 2
fi

cuda_version=""
cuda_architectures=""
torch_cxx11_abi=""
dependency_specs=()
for argument in "$@"; do
  case "${argument}" in
    --cuda-version=*)
      [[ -z "${cuda_version}" ]] || {
        echo "duplicate --cuda-version" >&2
        exit 2
      }
      cuda_version="${argument#*=}"
      ;;
    --cuda-architectures=*)
      [[ -z "${cuda_architectures}" ]] || {
        echo "duplicate --cuda-architectures" >&2
        exit 2
      }
      cuda_architectures="${argument#*=}"
      ;;
    --torch-cxx11-abi=*)
      [[ -z "${torch_cxx11_abi}" ]] || {
        echo "duplicate --torch-cxx11-abi" >&2
        exit 2
      }
      torch_cxx11_abi="${argument#*=}"
      ;;
    *=*) dependency_specs+=("${argument}") ;;
    *)
      echo "invalid kernel wheel argument: ${argument}" >&2
      exit 2
      ;;
  esac
done

case "${output_dir}" in
  /*) ;;
  *) output_dir="${PWD}/${output_dir}" ;;
esac

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export UV_OFFLINE=1

stage="$(mktemp -d "${TMPDIR:-/tmp}/sgl-kernel-bazel.XXXXXX")"
trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/src"
cp -aL "${source_root}/." "${stage}/src/"

if [[ "${backend}" == "cuda" ]]; then
  required_dependencies=(
    REPO-CUTLASS
    REPO-FLASH-ATTENTION
    REPO-FLASHINFER
    REPO-FLASHMLA
    REPO-FLASHMLA-CUTLASS
    REPO-FMT
    REPO-TRITON
  )
  configuration_values=(
    "${cuda_version}"
    "${cuda_architectures}"
    "${torch_cxx11_abi}"
  )
  configured_values=0
  for value in "${configuration_values[@]}"; do
    [[ -z "${value}" ]] || ((configured_values += 1))
  done
  if [[ "${configured_values}" -ne 0 && "${configured_values}" -ne 3 ]]; then
    echo "CUDA version, architectures, and PyTorch C++ ABI must be configured together" >&2
    exit 2
  fi

  cmake_configuration_args=()
  if [[ "${configured_values}" -eq 3 ]]; then
    if [[ ! "${cuda_version}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "invalid declared CUDA toolkit version: ${cuda_version}" >&2
      exit 2
    fi
    if [[ ! "${cuda_architectures}" =~ ^[0-9]+[af]?(,[0-9]+[af]?)*$ ]]; then
      echo "invalid declared CUDA architectures: ${cuda_architectures}" >&2
      exit 2
    fi
    if [[ "${torch_cxx11_abi}" != "0" && "${torch_cxx11_abi}" != "1" ]]; then
      echo "invalid declared PyTorch C++ ABI: ${torch_cxx11_abi}" >&2
      exit 2
    fi

    cuda_compiler="${CUDACXX:-nvcc}"
    if ! cuda_compiler="$(command -v "${cuda_compiler}")"; then
      echo "declared CUDA ${cuda_version}, but ${CUDACXX:-nvcc} was not found" >&2
      exit 1
    fi
    nvcc_version="$("${cuda_compiler}" --version)"
    if [[ ! "${nvcc_version}" =~ release[[:space:]]+([0-9]+\.[0-9]+) ]]; then
      echo "could not determine CUDA toolkit version from ${cuda_compiler}" >&2
      exit 1
    fi
    if [[ "${BASH_REMATCH[1]}" != "${cuda_version}" ]]; then
      echo "declared CUDA ${cuda_version}, but ${cuda_compiler} reports ${BASH_REMATCH[1]}" >&2
      exit 1
    fi
    export CUDACXX="${cuda_compiler}"

    python_executable="${PYTHON:-${python_bin}}"
    if ! python_executable="$(resolve_executable "${python_executable}")"; then
      echo "cannot validate declared PyTorch C++ ABI: ${PYTHON:-${python_bin}} was not found" >&2
      exit 1
    fi
    if ! detected_torch_cxx11_abi="$(
      "${python_executable}" -c \
        'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))'
    )"; then
      echo "cannot validate declared PyTorch C++ ABI with ${python_executable}" >&2
      exit 1
    fi
    if [[ "${detected_torch_cxx11_abi}" != "${torch_cxx11_abi}" ]]; then
      echo "declared PyTorch C++ ABI ${torch_cxx11_abi}, but ${python_executable} reports ${detected_torch_cxx11_abi}" >&2
      exit 1
    fi

    cmake_configuration_args=(
      "-DSGL_KERNEL_EXPECTED_CUDA_VERSION=${cuda_version}"
      "-DSGL_KERNEL_EXPECTED_CUDA_ARCHITECTURES=${cuda_architectures//,/;}"
      "-DSGL_KERNEL_TORCH_CXX11_ABI=${torch_cxx11_abi}"
    )
  fi

  declare -A dependency_roots=()
  for dependency_spec in "${dependency_specs[@]}"; do
    dependency="${dependency_spec%%=*}"
    source_dir="${dependency_spec#*=}"
    dependency_dir="${stage}/deps/${dependency}"
    mkdir -p "${dependency_dir}"
    cp -aL "${source_dir}/." "${dependency_dir}/"
    dependency_roots["${dependency}"]="${dependency_dir}"
  done
  for dependency in "${required_dependencies[@]}"; do
    if [[ ! -v "dependency_roots[${dependency}]" ]]; then
      echo "missing CMake source dependency: ${dependency}" >&2
      exit 2
    fi
  done

  # FlashMLA expects its separately pinned CUTLASS submodule in-tree and patches
  # both source trees for CUDA 13, so keep all CMake inputs in the writable stage.
  flashmla_cutlass="${dependency_roots[REPO-FLASHMLA]}/csrc/cutlass"
  rm -rf "${flashmla_cutlass}"
  mkdir -p "$(dirname "${flashmla_cutlass}")"
  mv "${dependency_roots[REPO-FLASHMLA-CUTLASS]}" "${flashmla_cutlass}"
  dependency_roots[REPO-FLASHMLA-CUTLASS]="${flashmla_cutlass}"

  cmake_source_args=("-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
  cmake_source_args+=("${cmake_configuration_args[@]}")
  for dependency in "${required_dependencies[@]}"; do
    cmake_source_args+=(
      "-DFETCHCONTENT_SOURCE_DIR_${dependency}=${dependency_roots[${dependency}]}"
    )
  done
  export CMAKE_ARGS="${CMAKE_ARGS:-} ${cmake_source_args[*]}"
elif [[ -n "${cuda_version}${cuda_architectures}${torch_cxx11_abi}" ]]; then
  echo "CUDA toolchain configuration is only valid for the CUDA backend" >&2
  exit 2
elif [[ "${#dependency_specs[@]}" -ne 0 ]]; then
  echo "CMake source dependencies are only valid for the CUDA backend" >&2
  exit 2
fi

if [[ "${backend}" == "rocm" ]]; then
  : "${AMDGPU_TARGET:?AMDGPU_TARGET must be selected by Bazel analysis}"
  : "${ROCM_HOME:?ROCM_HOME must be provided by the ROCm toolchain}"
  : "${CXX:?CXX must be provided by the ROCm toolchain}"
  : "${SGL_KERNEL_BUILD_FRONTEND:?SGL_KERNEL_BUILD_FRONTEND must be provided by the ROCm toolchain}"
  if [[ ! -d "${ROCM_HOME}" ]]; then
    echo "ROCm toolchain root does not exist: ${ROCM_HOME}" >&2
    exit 2
  fi
  if ! cxx_bin="$(resolve_executable "${CXX}")"; then
    echo "C++ toolchain executable not found: ${CXX}" >&2
    exit 2
  fi
  export CXX="${cxx_bin}"
  export CUDA_HOME="${ROCM_HOME}"
  export HIP_HOME="${ROCM_HOME}"
  export PATH="${ROCM_HOME}/bin:${PATH}"
  export PYTORCH_ROCM_ARCH="${AMDGPU_TARGET}"
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DFETCHCONTENT_FULLY_DISCONNECTED=ON"
  "${python_bin}" - <<'PY'
import torch

if torch.version.hip is None:
    raise RuntimeError(
        f"ROCm wheel build requires ROCm PyTorch, found torch {torch.__version__}"
    )
print(f"Using torch {torch.__version__} with ROCm {torch.version.hip}")
PY
fi

cd "${stage}/src"
case "${backend}" in
  cpu) cp pyproject_cpu.toml pyproject.toml ;;
  cuda) ;;
  rocm)
    cp pyproject_rocm.toml pyproject.toml
    cp setup_rocm.py setup.py
    ;;
  *)
    echo "unsupported sgl-kernel backend: ${backend}" >&2
    exit 2
    ;;
esac

rm -rf build dist
if [[ -n "${SGL_KERNEL_BUILD_FRONTEND:-}" ]]; then
  if ! build_frontend="$(resolve_executable "${SGL_KERNEL_BUILD_FRONTEND}")"; then
    echo "wheel build frontend not found: ${SGL_KERNEL_BUILD_FRONTEND}" >&2
    exit 2
  fi
  "${build_frontend}" build --wheel -Cbuild-dir=build --no-build-isolation .
elif command -v uv >/dev/null 2>&1; then
  uv build --wheel -Cbuild-dir=build --no-build-isolation .
else
  "${python_bin}" -m build --wheel --no-isolation .
fi

if [[ "${backend}" == "cuda" ]]; then
  PYTHON="${python_bin}" bash ./rename_wheels.sh
fi

shopt -s nullglob
wheels=(dist/*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "expected exactly one wheel, found ${#wheels[@]}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
cp "${wheels[0]}" "${output_dir}/"
