#!/usr/bin/env bash
set -euo pipefail

# Install framework-specific dependencies into isolated venvs so we can reuse
# the base environment prepared by ci_install_dependency.sh without global
# package conflicts.

FRAMEWORK="${1:-}"
ACTION="${2:-install}"
VENV_ROOT="${COMPARISON_VENV_ROOT:-/tmp/sglang-comparison-venvs}"
USE_UV="${COMPARISON_USE_UV:-auto}"   # auto|always|never
RESET_VENV="${COMPARISON_RESET_VENV:-0}"

if [[ -z "${FRAMEWORK}" ]]; then
    echo "Usage: $0 <vllm-omni|lightx2v> [install|remove]"
    exit 1
fi

case "${FRAMEWORK}" in
    vllm-omni|lightx2v)
        ;;
    *)
        echo "Unsupported framework: ${FRAMEWORK}"
        exit 1
        ;;
esac

case "${ACTION}" in
    install|remove)
        ;;
    *)
        echo "Unsupported action: ${ACTION} (expected: install|remove)"
        exit 1
        ;;
esac

VENV_DIR="${VENV_ROOT}/${FRAMEWORK}"
PYTHON_BIN="${VENV_DIR}/bin/python"
MARKER_FILE="${VENV_DIR}/.framework_installed"
UV_BIN="${VENV_DIR}/bin/uv"

mkdir -p "${VENV_ROOT}"

if [[ "${ACTION}" == "remove" ]]; then
    rm -rf "${VENV_DIR}"
    echo "FRAMEWORK=${FRAMEWORK}"
    echo "ACTION=remove"
    echo "FRAMEWORK_VENV_DIR=${VENV_DIR}"
    echo "FRAMEWORK_BIN_DIR="
    exit 0
fi

if [[ "${RESET_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
    rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" || ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip wheel setuptools

# Best-effort install uv inside the isolated venv. If unavailable, we
# gracefully fall back to pip commands below.
if [[ "${USE_UV}" != "never" && ! -x "${UV_BIN}" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade uv >/dev/null 2>&1 || true
fi

have_uv() {
    [[ -x "${UV_BIN}" ]]
}

if [[ "${USE_UV}" == "always" ]] && ! have_uv; then
    echo "Failed to install uv in ${VENV_DIR} while COMPARISON_USE_UV=always"
    exit 1
fi

install_pkg() {
    local package_spec="$1"
    if have_uv; then
        "${UV_BIN}" pip install --python "${PYTHON_BIN}" "${package_spec}"
    else
        "${PYTHON_BIN}" -m pip install "${package_spec}"
    fi
}

# Re-install framework package if marker is absent.
if [[ ! -f "${MARKER_FILE}" ]]; then
    case "${FRAMEWORK}" in
        vllm-omni)
            # vLLM-Omni GPU install guide:
            #   uv pip install vllm --torch-backend=auto
            #   uv pip install vllm-omni
            # `--torch-backend=auto` is uv-specific and invalid for pip, so we
            # only use it when uv is present.
            # Pin vllm to <0.19 because vllm-omni 0.18.x imports
            # vllm.inputs.data which was removed in vllm 0.19.
            if have_uv; then
                "${UV_BIN}" pip install --python "${PYTHON_BIN}"  \
                    "vllm==0.18.0" --torch-backend=auto
            else
                install_pkg "vllm==0.18.0"
            fi
            install_pkg "vllm-omni"
            ;;
        lightx2v)
            # LightX2V quick start recommends source install (`pip install -e .`)
            # after cloning the repo. For CI, install directly from GitHub.
            install_pkg "git+https://github.com/ModelTC/LightX2V.git"
            # pyzmq is imported unconditionally by lightx2v but not declared
            # as a hard dependency.
            install_pkg "pyzmq"
            # LightX2V's Gemma3 text encoder path expects
            # Gemma3TextConfig.rope_local_base_freq, which is missing in
            # transformers 5.x.
            install_pkg "transformers==4.52.4"
            ;;
    esac

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${MARKER_FILE}"
fi

echo "FRAMEWORK=${FRAMEWORK}"
echo "ACTION=install"
echo "FRAMEWORK_VENV_DIR=${VENV_DIR}"
echo "FRAMEWORK_BIN_DIR=${VENV_DIR}/bin"
