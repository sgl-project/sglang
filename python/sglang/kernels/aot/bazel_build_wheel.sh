#!/usr/bin/env bash
set -euo pipefail

backend="$1"
output_dir="$2"
source_root="$3"
shift 3

case "${output_dir}" in
  /*) ;;
  *) output_dir="${PWD}/${output_dir}" ;;
esac

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
  declare -A dependency_roots=()
  for dependency_spec in "$@"; do
    if [[ "${dependency_spec}" != *=* ]]; then
      echo "invalid CMake source dependency: ${dependency_spec}" >&2
      exit 2
    fi
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
  for dependency in "${required_dependencies[@]}"; do
    cmake_source_args+=(
      "-DFETCHCONTENT_SOURCE_DIR_${dependency}=${dependency_roots[${dependency}]}"
    )
  done
  export CMAKE_ARGS="${CMAKE_ARGS:-} ${cmake_source_args[*]}"
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
if command -v uv >/dev/null 2>&1; then
  uv build --wheel -Cbuild-dir=build --no-build-isolation .
else
  python3 -m build --wheel --no-isolation .
fi

if [[ "${backend}" == "cuda" ]]; then
  PYTHON="$(command -v python3)" bash ./rename_wheels.sh
fi

shopt -s nullglob
wheels=(dist/*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "expected exactly one wheel, found ${#wheels[@]}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
cp "${wheels[0]}" "${output_dir}/"
