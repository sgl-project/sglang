#!/usr/bin/env bash
# Build a pip-installable DeepEP wheel in the current CUDA environment.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GDRCOPY_VERSION="2.5.1"
GDRCOPY_HOME="/usr/src/gdrdrv-${GDRCOPY_VERSION}/"
DEEPEP_REPO="https://github.com/sgl-project/DeepEP.git"
DEEPEP_BUILD_ROOT=""


select_deepep_branch() {
    local arch="$1"
    local cuda_major="$2"

    case "${arch}:${cuda_major}" in
        x86_64:12|x86_64:13)
            echo "sgl-deepep-x86"
            ;;
        aarch64:12)
            echo "sgl-deepep-cu12-arm"
            ;;
        aarch64:13)
            echo "sgl-deepep-arm"
            ;;
        *)
            echo "Unsupported architecture/CUDA combination: ${arch} with CUDA ${cuda_major}" >&2
            return 1
            ;;
    esac
}


parse_cuda_major() {
    local nvcc_output="$1"
    local cuda_major

    if [[ ! "${nvcc_output}" =~ release[[:space:]]+([0-9]+)\. ]]; then
        echo "Could not parse CUDA toolkit version from nvcc output" >&2
        return 1
    fi
    cuda_major="${BASH_REMATCH[1]}"

    case "${cuda_major}" in
        12|13)
            echo "${cuda_major}"
            ;;
        *)
            echo "Unsupported CUDA toolkit major version: ${cuda_major}" >&2
            return 1
            ;;
    esac
}


patch_cuda13_cccl() {
    local setup_py="$1"
    local cuda_home="$2"
    local cccl_dir="${cuda_home}/include/cccl"

    if [[ ! -d "${cccl_dir}" ]]; then
        echo "CCCL include directory not found: ${cccl_dir}" >&2
        return 1
    fi
    if [[ ! -f "${setup_py}" ]]; then
        echo "DeepEP setup.py not found: ${setup_py}" >&2
        return 1
    fi

    "${PYTHON_BIN}" - "${setup_py}" "${cccl_dir}" <<'PY'
import pathlib
import sys

setup_py = pathlib.Path(sys.argv[1])
cccl_dir = sys.argv[2]
text = setup_py.read_text()
anchor = "    include_dirs = ['csrc/']"
addition = f"    include_dirs.append('{cccl_dir}')"

if addition in text:
    raise SystemExit(0)
if anchor not in text:
    print(
        "Could not find DeepEP include_dirs insertion point in setup.py",
        file=sys.stderr,
    )
    raise SystemExit(1)

setup_py.write_text(text.replace(anchor, f"{anchor}\n{addition}", 1))
PY
}


find_single_wheel() {
    local output_dir="$1"
    local nullglob_state
    local -a wheels

    nullglob_state="$(shopt -p nullglob || true)"
    shopt -s nullglob
    wheels=("${output_dir}"/deep_ep-*.whl)
    eval "${nullglob_state}"

    if [[ "${#wheels[@]}" -ne 1 ]]; then
        echo "Expected exactly one DeepEP wheel, found ${#wheels[@]} in ${output_dir}" >&2
        return 1
    fi

    local wheel_dir
    wheel_dir="$(cd "$(dirname "${wheels[0]}")" && pwd -P)"
    echo "${wheel_dir}/$(basename "${wheels[0]}")"
}


build_deepep() {
    local source_dir="$1"
    local output_dir="$2"

    if [[ ! -f "${source_dir}/setup.py" ]]; then
        echo "DeepEP setup.py not found: ${source_dir}/setup.py" >&2
        return 1
    fi

    mkdir -p "${output_dir}"
    output_dir="$(cd "${output_dir}" && pwd -P)"
    rm -f "${output_dir}"/deep_ep-*.whl

    (
        cd "${source_dir}"
        TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' \
            MAX_JOBS="${MAX_JOBS}" \
            "${PYTHON_BIN}" setup.py bdist_wheel -d "${output_dir}"
    )

    find_single_wheel "${output_dir}" >/dev/null
}


build_and_report() {
    local source_dir="$1"
    local output_dir="$2"
    local wheel_path

    build_deepep "${source_dir}" "${output_dir}"
    wheel_path="$(find_single_wheel "${output_dir}")"
    echo "--- Done ---"
    echo "DeepEP wheel: ${wheel_path}"
}


run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "Root privileges are required, but sudo is unavailable" >&2
        return 1
    fi
}


install_apt_packages() {
    local package

    if run_as_root apt-get install -y --no-install-recommends "$@"; then
        return 0
    fi

    echo "apt-get install failed; checking whether every package is already installed" >&2
    for package in "$@"; do
        if ! dpkg -l "${package}" 2>/dev/null | grep -q '^ii'; then
            echo "Required package ${package} is not installed" >&2
            return 1
        fi
    done
    echo "All requested packages are already installed; continuing" >&2
}


install_system_dependencies() {
    local -a system_deps=(
        curl
        wget
        git
        sudo
        rdma-core
        infiniband-diags
        openssh-server
        perftest
        libibumad3
        libibverbs-dev
        libibverbs1
        ibverbs-providers
        ibverbs-utils
        libnl-3-200
        libnl-route-3-200
        librdmacm1
        build-essential
        cmake
    )

    run_as_root apt-get update || \
        echo "apt-get update failed; package availability checks will decide whether to continue" >&2
    install_apt_packages "${system_deps[@]}"
}


install_python_dependencies() {
    "${PYTHON_BIN}" -m pip install setuptools wheel ninja
}


install_gdrcopy() {
    local gdrcopy_dir="/opt/gdrcopy"
    local arch="$1"
    local lib_path="/usr/lib/${arch}-linux-gnu"
    local -a gdrcopy_deps_1=(nvidia-dkms-580)
    local -a gdrcopy_deps_2=(
        build-essential
        devscripts
        debhelper
        fakeroot
        pkg-config
        dkms
    )
    local -a gdrcopy_deps_3=(
        check
        libsubunit0
        libsubunit-dev
        python3-venv
    )

    echo "--- Installing GDRCopy v${GDRCOPY_VERSION} ---"
    run_as_root rm -rf "${gdrcopy_dir}"
    run_as_root mkdir -p "${gdrcopy_dir}"
    run_as_root git clone --depth 1 --branch "v${GDRCOPY_VERSION}" \
        https://github.com/NVIDIA/gdrcopy.git "${gdrcopy_dir}"

    install_apt_packages "${gdrcopy_deps_1[@]}"
    install_apt_packages "${gdrcopy_deps_2[@]}"
    install_apt_packages "${gdrcopy_deps_3[@]}"

    run_as_root env CUDA="${CUDA_HOME}" \
        bash -c 'cd "$1/packages" && ./build-deb-packages.sh' bash "${gdrcopy_dir}"
    run_as_root bash -c '
        set -e
        cd "$1/packages"
        dpkg -i gdrdrv-dkms_*.deb
        dpkg -i libgdrapi_*.deb
        dpkg -i gdrcopy-tests_*.deb
        dpkg -i gdrcopy_*.deb
    ' bash "${gdrcopy_dir}"

    if [[ ! -e "${lib_path}/libmlx5.so" ]]; then
        if [[ ! -e "${lib_path}/libmlx5.so.1" ]]; then
            echo "Required mlx5 library not found: ${lib_path}/libmlx5.so.1" >&2
            return 1
        fi
        run_as_root ln -s "${lib_path}/libmlx5.so.1" "${lib_path}/libmlx5.so"
    fi

    run_as_root apt-get update || \
        echo "apt-get update failed before libfabric-dev installation" >&2
    install_apt_packages libfabric-dev
}


cleanup_build_root() {
    if [[ -z "${DEEPEP_BUILD_ROOT}" || ! -d "${DEEPEP_BUILD_ROOT}" ]]; then
        return 0
    fi
    if [[ "$(basename "${DEEPEP_BUILD_ROOT}")" != sgl-deepep.* ]]; then
        echo "Refusing to remove unexpected DeepEP build path: ${DEEPEP_BUILD_ROOT}" >&2
        return 1
    fi
    rm -rf -- "${DEEPEP_BUILD_ROOT}"
}


main() {
    if [[ "$#" -gt 1 ]]; then
        echo "Usage: build_sgl_deepep.sh [OUTPUT_DIR]" >&2
        return 2
    fi

    local output_dir="${1:-${PWD}/dist}"
    local arch
    local nvcc_bin
    local nvcc_output
    local cuda_major
    local deepep_branch
    local deepep_dir

    arch="$(uname -m)"
    if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
        nvcc_bin="${CUDA_HOME}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc_bin="$(command -v nvcc)"
    else
        echo "nvcc not found under CUDA_HOME=${CUDA_HOME} or PATH" >&2
        return 1
    fi
    nvcc_output="$("${nvcc_bin}" --version)"
    cuda_major="$(parse_cuda_major "${nvcc_output}")"
    deepep_branch="$(select_deepep_branch "${arch}" "${cuda_major}")"

    command -v git >/dev/null 2>&1 || {
        echo "git is required to clone DeepEP" >&2
        return 1
    }
    command -v apt-get >/dev/null 2>&1 || {
        echo "apt-get is required to install DeepEP dependencies" >&2
        return 1
    }
    command -v dpkg >/dev/null 2>&1 || {
        echo "dpkg is required to install GDRCopy packages" >&2
        return 1
    }
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
        echo "Python interpreter not found: ${PYTHON_BIN}" >&2
        return 1
    }
    "${PYTHON_BIN}" -m pip --version >/dev/null
    "${PYTHON_BIN}" -c 'import torch' || {
        echo "PyTorch must be installed in the selected Python environment" >&2
        return 1
    }
    if [[ "${EUID}" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true; then
            echo "Run as root or configure passwordless sudo to install system dependencies" >&2
            return 1
        fi
    fi

    mkdir -p "${output_dir}"
    output_dir="$(cd "${output_dir}" && pwd -P)"
    DEEPEP_BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sgl-deepep.XXXXXX")"
    trap cleanup_build_root EXIT
    deepep_dir="${DEEPEP_BUILD_ROOT}/DeepEP"

    echo "----------------------------------------"
    echo "Architecture:     ${arch}"
    echo "CUDA major:      ${cuda_major}"
    echo "CUDA_HOME:       ${CUDA_HOME}"
    echo "DeepEP branch:   ${deepep_branch}"
    echo "Python:          ${PYTHON_BIN}"
    echo "TORCH arch list: 9.0;10.0;10.3"
    echo "Output directory: ${output_dir}"
    echo "----------------------------------------"

    echo "--- Cloning DeepEP (${deepep_branch}) ---"
    git clone --depth 1 --branch "${deepep_branch}" \
        "${DEEPEP_REPO}" "${deepep_dir}"

    echo "--- Removing an existing DeepEP installation ---"
    "${PYTHON_BIN}" -m pip uninstall -y deep_ep || true

    echo "--- Installing Python build dependencies ---"
    install_python_dependencies

    echo "--- Installing system dependencies ---"
    install_system_dependencies
    install_gdrcopy "${arch}"

    if [[ "${cuda_major}" == "13" ]]; then
        echo "--- Adding CUDA 13 CCCL include path ---"
        patch_cuda13_cccl "${deepep_dir}/setup.py" "${CUDA_HOME}"
    fi

    export CUDA_HOME GDRCOPY_HOME
    echo "--- Building DeepEP wheel ---"
    build_and_report "${deepep_dir}" "${output_dir}"
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
