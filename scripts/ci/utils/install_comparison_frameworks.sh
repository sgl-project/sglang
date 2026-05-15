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

# ---------------------------------------------------------------------------
# Pinned framework versions — do not track "latest" on PyPI or branch HEAD on git.
# Bump these intentionally when upgrading; CI/nightly stays reproducible.
# - vLLM / vLLM-Omni: exact PyPI versions (override with VLLM_PIN / VLLM_OMNI_PIN).
# - LightX2V: install only from git at an immutable full commit SHA (override with
#   LIGHTX2V_GIT_COMMIT). Never set LIGHTX2V_GIT_COMMIT to a branch name (e.g. main).
# ---------------------------------------------------------------------------
VLLM_PIN="${VLLM_PIN:-0.18.0}"
VLLM_OMNI_PIN="${VLLM_OMNI_PIN:-0.18.0}"
LIGHTX2V_GIT_COMMIT="${LIGHTX2V_GIT_COMMIT:-cc9087edb71b3b53ed7104f47deaacafee9100d4}"
# flash-attn (import name: flash_attn). Pin for reproducibility, e.g. FLASH_ATTN_PACKAGE='flash-attn==2.7.4.post1'
# Set COMPARISON_SKIP_FLASH_ATTN=1 only to skip (server will likely still fail to start).
FLASH_ATTN_PACKAGE="${FLASH_ATTN_PACKAGE:-flash-attn}"

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

# Upgrade pip/wheel only — upgrading setuptools to the latest breaks vllm 0.18.x
# on Python 3.12+ (requires setuptools>=77.0.3,<81.0.0). Framework blocks pin
# setuptools when needed.
"${PYTHON_BIN}" -m pip install --upgrade pip wheel

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

# Some packages (notably flash-attn) need access to already-installed torch
# during build, but do not declare it correctly as a build dependency.
# Use pip with --no-build-isolation so build backend can see venv packages.
install_pkg_no_build_isolation() {
    local package_spec="$1"
    "${PYTHON_BIN}" -m pip install --no-build-isolation "${package_spec}"
}

case "${FRAMEWORK}" in
    vllm-omni)
        INSTALL_KEY="vllm==${VLLM_PIN}|vllm-omni==${VLLM_OMNI_PIN}|setuptools>=77,<81"
        ;;
    lightx2v)
        INSTALL_KEY="lightx2v@git-${LIGHTX2V_GIT_COMMIT}|pyzmq|transformers==4.52.4|matplotlib|${FLASH_ATTN_PACKAGE}"
        ;;
esac

# Re-install when the marker is absent or the install spec (pins/commits) changed.
NEED_INSTALL=1
if [[ -f "${MARKER_FILE}" ]]; then
    SAVED_KEY=$(head -n 1 "${MARKER_FILE}")
    if [[ "${SAVED_KEY}" == "${INSTALL_KEY}" ]]; then
        NEED_INSTALL=0
    fi
fi

if [[ "${NEED_INSTALL}" -eq 1 ]]; then
    case "${FRAMEWORK}" in
        vllm-omni)
            # vllm 0.18.x declares setuptools>=77.0.3,<81.0.0 on Python 3.12+.
            install_pkg "setuptools>=77.0.3,<81.0.0"
            # vLLM-Omni GPU install guide:
            #   uv pip install vllm --torch-backend=auto
            #   uv pip install vllm-omni
            # `--torch-backend=auto` is uv-specific and invalid for pip, so we
            # only use it when uv is present.
            # VLLM_PIN / VLLM_OMNI_PIN: keep vllm and vllm-omni on matching lines
            # (vllm-omni 0.18.x expects vllm 0.18.x; vllm 0.19+ removed APIs vllm-omni uses).
            if have_uv; then
                "${UV_BIN}" pip install --python "${PYTHON_BIN}"  \
                    "vllm==${VLLM_PIN}" --torch-backend=auto
            else
                install_pkg "vllm==${VLLM_PIN}"
            fi
            install_pkg "vllm-omni==${VLLM_OMNI_PIN}"
            ;;
        lightx2v)
            # PyPI does not publish lightx2v; install from GitHub at fixed full SHA only.
            if [[ ! "${LIGHTX2V_GIT_COMMIT}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
                echo "ERROR: LIGHTX2V_GIT_COMMIT must be a git commit hash (7–40 hex chars), not a branch name." >&2
                exit 1
            fi
            install_pkg "git+https://github.com/ModelTC/LightX2V.git@${LIGHTX2V_GIT_COMMIT}"
            # pyzmq is imported unconditionally by lightx2v but not declared
            # as a hard dependency.
            install_pkg "pyzmq"
            # LightX2V's Gemma3 text encoder path expects
            # Gemma3TextConfig.rope_local_base_freq, which is missing in
            # transformers 5.x.
            install_pkg "transformers==4.52.4"
            # LightX2V imports worldmirror modules at package import time;
            # those modules require matplotlib.
            install_pkg "matplotlib"
            # LightX2V imports worldmirror at package load; it requires flash_attn
            # (PyPI: flash-attn). Without it: ModuleNotFoundError: flash_attn / flash_attn_interface.
            if [[ "${COMPARISON_SKIP_FLASH_ATTN:-0}" == "1" ]]; then
                echo "WARNING: COMPARISON_SKIP_FLASH_ATTN=1 — skipping flash-attn; lightx2v.server will likely fail to start." >&2
            else
                "${PYTHON_BIN}" -m pip install ninja packaging >/dev/null 2>&1 || true
                if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
                    echo "ERROR: torch is not importable in ${VENV_DIR}; cannot build flash-attn." >&2
                    exit 1
                fi
                if ! install_pkg_no_build_isolation "${FLASH_ATTN_PACKAGE}"; then
                    echo "ERROR: flash-attn failed to install (needs CUDA toolchain matching PyTorch; may compile from source)." >&2
                    echo "  See: https://github.com/Dao-AILab/flash-attention/blob/main/README.md" >&2
                    echo "  Hint: MAX_JOBS=4 ${PYTHON_BIN} -m pip install --no-build-isolation ${FLASH_ATTN_PACKAGE}" >&2
                    exit 1
                fi
            fi
            ;;
    esac

    {
        echo "${INSTALL_KEY}"
        echo "INSTALLED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    } > "${MARKER_FILE}"
fi

echo "FRAMEWORK=${FRAMEWORK}"
echo "ACTION=install"
echo "FRAMEWORK_VENV_DIR=${VENV_DIR}"
echo "FRAMEWORK_BIN_DIR=${VENV_DIR}/bin"
