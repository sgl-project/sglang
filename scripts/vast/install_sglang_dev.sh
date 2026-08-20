#!/usr/bin/env bash
# =============================================================================
# install_sglang_dev.sh
#
# Editable source install of SGLang for PP-boundary activation quantization /
# compression research, targeting a Vast.ai GPU instance.
#
# The instance is itself an unprivileged container (no docker-in-docker), and it
# ships a working SGLang release install in system Python plus a supervisor
# service holding the GPUs. So this script installs into an isolated uv venv and
# leaves the image's own environment intact: if anything here goes wrong, the
# box still has its original, known-good stack.
#
# Steps
#   1. preflight            GPU / CUDA / python / uv / cargo / disk
#   2. stop_vast_services   release the GPUs held by the image's own server
#   3. fetch_source         clone or update the checkout to build from
#   4. read_pins            every version comes from python/pyproject.toml and
#                           docker/Dockerfile, so this keeps working as upstream
#                           moves its pins
#   5. make_venv            isolated uv venv
#   6. install_torch_stack  torch/vision/audio/codec from the CUDA-matched index
#   7. install_sglang       editable install of THIS tree (python edits then
#                           need no reinstall — that is the whole point)
#   8. pin_kernel_wheels    sglang-kernel, NCCL, flashinfer cubin + JIT cache
#   9. verify               prove `sglang` resolves to this checkout
#
# Usage
#   bash scripts/vast/install_sglang_dev.sh
#
# Knobs (all optional, env vars)
#   REPO_DIR REPO_URL SGLANG_REF VENV_DIR EXTRAS CU_VERSION SECRETS_FILE
#   SGLANG_BUILD_RUST_EXTS STOP_VAST_SERVICES INSTALL_PRECOMMIT
#   INSTALL_FLASHINFER_CACHE RECREATE_VENV
#
# This script never modifies tracked files in the checkout.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO_URL="${REPO_URL:-https://github.com/shamanez/sglang.git}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/sgl-project/sglang.git}"
# Default to the checkout this script lives in; fall back to a fresh clone.
if REPO_DIR_DETECTED="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_DIR="${REPO_DIR:-$REPO_DIR_DETECTED}"
else
    REPO_DIR="${REPO_DIR:-/workspace/sglang-src}"
fi
SGLANG_REF="${SGLANG_REF:-}"                       # empty = leave checkout alone
VENV_DIR="${VENV_DIR:-/workspace/venv-sgl}"
EXTRAS="${EXTRAS:-dev,tracing}"                    # dev == test tooling
SECRETS_FILE="${SECRETS_FILE:-/workspace/.env}"
SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
STOP_VAST_SERVICES="${STOP_VAST_SERVICES:-1}"
INSTALL_PRECOMMIT="${INSTALL_PRECOMMIT:-1}"
INSTALL_FLASHINFER_CACHE="${INSTALL_FLASHINFER_CACHE:-1}"
RECREATE_VENV="${RECREATE_VENV:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"

# uv copies rather than hardlinks across filesystems; be explicit.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
if [ -t 1 ]; then B=$'\033[1m'; BLUE=$'\033[1;34m'; RED=$'\033[1;31m'; YEL=$'\033[1;33m'; N=$'\033[0m'
else B=""; BLUE=""; RED=""; YEL=""; N=""; fi

STEP_START=$SECONDS
log()  { printf '\n%s==> %s%s\n' "$BLUE" "$*" "$N"; STEP_START=$SECONDS; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %sok%s  %s  (%ss)\n' "$B" "$N" "$*" "$((SECONDS - STEP_START))"; }
warn() { printf '%swarn%s %s\n' "$YEL" "$N" "$*" >&2; }
die()  { printf '%sfail%s %s\n' "$RED" "$N" "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. Preflight
# --------------------------------------------------------------------------- #
preflight() {
    log "Preflight"

    echo probe | grep -qP 'probe' 2>/dev/null || die "GNU grep with -P is required (this script targets the Linux instance, not macOS)"
    command -v git >/dev/null || die "git not found"
    command -v python3 >/dev/null || die "python3 not found"

    PY_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    info "python3         $(python3 -V 2>&1 | awk '{print $2}')  (venv will use python${PY_MINOR})"

    command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this script expects an NVIDIA GPU host"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader \
        | while IFS= read -r line; do info "gpu             ${line}"; done
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    info "gpu count       ${GPU_COUNT}"

    # CU_VERSION drives every wheel index below. Derive it from nvcc when we can.
    if [ -z "${CU_VERSION:-}" ]; then
        if command -v nvcc >/dev/null; then
            local nvcc_ver major
            nvcc_ver="$(nvcc --version | grep -Po 'release \K[0-9]+\.[0-9]+' || true)"
            major="${nvcc_ver%%.*}"
            case "$major" in
                13) CU_VERSION="cu130" ;;
                12) CU_VERSION="cu129" ;;
                *)  die "unsupported CUDA toolkit ${nvcc_ver:-unknown}; set CU_VERSION explicitly" ;;
            esac
            info "nvcc            ${nvcc_ver} -> ${CU_VERSION}"
        else
            CU_VERSION="cu130"
            warn "nvcc not found; assuming ${CU_VERSION} (override with CU_VERSION=...)"
        fi
    else
        info "CU_VERSION      ${CU_VERSION} (from environment)"
    fi
    CU_MAJOR="${CU_VERSION#cu}"; CU_MAJOR="${CU_MAJOR:0:2}"

    if ! command -v uv >/dev/null; then
        info "installing uv"
        python3 -m pip install --quiet --upgrade uv || die "could not install uv"
    fi
    info "uv              $(uv --version 2>&1 | awk '{print $2}')"

    if [ "$SGLANG_BUILD_RUST_EXTS" = "none" ]; then
        info "rust exts       skipped (SGLANG_BUILD_RUST_EXTS=none)"
    elif command -v cargo >/dev/null; then
        info "cargo           $(cargo --version 2>&1 | awk '{print $2}')"
    else
        die "SGLANG_BUILD_RUST_EXTS=${SGLANG_BUILD_RUST_EXTS} needs cargo; install rustup or set it to none"
    fi

    local target_fs free_gb
    target_fs="$(dirname "$VENV_DIR")"
    mkdir -p "$target_fs"
    free_gb="$(df -BG --output=avail "$target_fs" | tail -1 | tr -dc '0-9')"
    info "free space      ${free_gb}G on ${target_fs}"
    [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] || die "need >= ${MIN_FREE_GB}G free on ${target_fs}, found ${free_gb}G"

    # Secrets (HF_TOKEN etc.) are optional; load them if the file is there so a
    # later `sglang serve` of a gated model just works. Never print the values.
    if [ -f "$SECRETS_FILE" ]; then
        set -a
        # shellcheck source=/dev/null
        . "$SECRETS_FILE"
        set +a
        local loaded=""
        for v in HF_TOKEN WANDB_API_KEY VAST_API_KEY R2_ACCESS_KEY_ID; do
            [ -n "${!v:-}" ] && loaded="${loaded}${v} "
        done
        info "secrets         ${SECRETS_FILE} -> ${loaded:-nothing usable} (values masked)"
    else
        info "secrets         none at ${SECRETS_FILE} (fine unless you need gated weights)"
    fi

    ok "preflight"
}

# --------------------------------------------------------------------------- #
# 2. Stop the image's own server so it stops holding GPU memory
# --------------------------------------------------------------------------- #
stop_vast_services() {
    [ "$STOP_VAST_SERVICES" = "1" ] || { info "leaving vast services alone"; return; }
    command -v supervisorctl >/dev/null || return 0

    log "Releasing GPUs held by the image's own services"
    # Only these two. caddy / instance_portal / tunnel_manager are the vast
    # management and auth surface and must keep running.
    for svc in sglang model-ui; do
        if supervisorctl status "$svc" 2>/dev/null | grep -q RUNNING; then
            info "stopping ${svc}"
            supervisorctl stop "$svc" >/dev/null || warn "could not stop ${svc}"
        else
            info "${svc} not running"
        fi
    done
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | while IFS= read -r line; do info "gpu mem now     ${line}"; done
    ok "gpus released"
}

# --------------------------------------------------------------------------- #
# 3. Source tree
# --------------------------------------------------------------------------- #
fetch_source() {
    log "Source tree"
    if [ -d "$REPO_DIR/.git" ]; then
        info "reusing         ${REPO_DIR}"
        git -C "$REPO_DIR" remote get-url upstream >/dev/null 2>&1 \
            || git -C "$REPO_DIR" remote add upstream "$UPSTREAM_URL"
        if [ -n "$SGLANG_REF" ]; then
            info "fetching        ${SGLANG_REF}"
            git -C "$REPO_DIR" fetch --all --tags --quiet
            git -C "$REPO_DIR" checkout --quiet "$SGLANG_REF"
        fi
    else
        info "cloning         ${REPO_URL} -> ${REPO_DIR}"
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone --quiet "$REPO_URL" "$REPO_DIR"
        git -C "$REPO_DIR" remote add upstream "$UPSTREAM_URL" 2>/dev/null || true
        [ -n "$SGLANG_REF" ] && git -C "$REPO_DIR" checkout --quiet "$SGLANG_REF"
    fi
    [ -f "$REPO_DIR/python/pyproject.toml" ] || die "${REPO_DIR} does not look like an sglang checkout"
    info "commit          $(git -C "$REPO_DIR" rev-parse --short HEAD)  ($(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD))"
    ok "source ready"
}

# --------------------------------------------------------------------------- #
# 4. Dependency pins, read from the tree itself
# --------------------------------------------------------------------------- #
read_pins() {
    log "Dependency pins (from the checkout, nothing hardcoded)"
    local pyproject="$REPO_DIR/python/pyproject.toml"
    local dockerfile="$REPO_DIR/docker/Dockerfile"

    # Full requirement strings, environment markers included, like CI does.
    TORCH_SPECS=()
    local package spec
    for package in torch torchaudio torchvision torchcodec; do
        spec="$(grep -Po -m1 "\"${package}([<>=!~ ;][^\"]*)?\"" "$pyproject" | tr -d '"' || true)"
        [ -n "$spec" ] && TORCH_SPECS+=("$spec")
    done
    [ "${#TORCH_SPECS[@]}" -gt 0 ] || die "found no torch pins in ${pyproject}"

    pin() { grep -Po -m1 "\"$1(\[[^]]+\])?==\K[0-9A-Za-z\.\-]+" "$pyproject" || true; }
    SGL_KERNEL_VERSION="$(pin 'sglang-kernel')"
    FLASHINFER_VERSION="$(pin 'flashinfer_python')"
    NCCL_VERSION="$(grep -Po -m1 '^ARG SGL_NCCL_VERSION=\K[0-9A-Za-z\.\-]+' "$dockerfile" || true)"

    for s in "${TORCH_SPECS[@]}"; do info "torch stack     ${s}"; done
    info "sglang-kernel   ${SGL_KERNEL_VERSION:-<unpinned>}"
    info "flashinfer      ${FLASHINFER_VERSION:-<unpinned>}"
    info "nccl (cu13)     ${NCCL_VERSION:-<none>}"
    ok "pins read"
}

# --------------------------------------------------------------------------- #
# 5. Isolated venv
# --------------------------------------------------------------------------- #
make_venv() {
    log "Virtualenv"
    if [ -d "$VENV_DIR" ] && [ "$RECREATE_VENV" = "1" ]; then
        info "removing        ${VENV_DIR} (RECREATE_VENV=1)"
        rm -rf "$VENV_DIR"
    fi
    if [ -d "$VENV_DIR" ]; then
        info "reusing         ${VENV_DIR}"
    else
        info "creating        ${VENV_DIR}"
        uv venv "$VENV_DIR" --python "python${PY_MINOR}" --seed
    fi
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    [ "${VIRTUAL_ENV:-}" = "$VENV_DIR" ] || die "venv activation failed"
    # uv would otherwise prefer the system interpreter for `uv pip`.
    unset UV_SYSTEM_PYTHON || true
    info "python          $(command -v python3)"
    ok "venv ready"
}

# --------------------------------------------------------------------------- #
# 6. torch stack from the CUDA-matched index
# --------------------------------------------------------------------------- #
install_torch_stack() {
    log "PyTorch stack (index: download.pytorch.org/whl/${CU_VERSION})"
    uv pip install "${TORCH_SPECS[@]}" \
        --index-url "https://download.pytorch.org/whl/${CU_VERSION}"
    ok "torch stack"
}

# --------------------------------------------------------------------------- #
# 7. This tree, editable
# --------------------------------------------------------------------------- #
install_sglang() {
    log "SGLang (editable, extras: ${EXTRAS})"
    info "rust exts       SGLANG_BUILD_RUST_EXTS=${SGLANG_BUILD_RUST_EXTS}"
    ( cd "$REPO_DIR" && SGLANG_BUILD_RUST_EXTS="$SGLANG_BUILD_RUST_EXTS" \
        uv pip install -e "python[${EXTRAS}]" --index-strategy unsafe-best-match )
    ok "sglang editable"
}

# --------------------------------------------------------------------------- #
# 8. CUDA-matched kernel wheels
# --------------------------------------------------------------------------- #
installed_version() {  # $1 = distribution name
    python3 - "$1" <<'PY' 2>/dev/null || true
import sys
from importlib.metadata import PackageNotFoundError, version
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("")
PY
}

pin_kernel_wheels() {
    log "Kernel wheels"

    if [ -n "$SGL_KERNEL_VERSION" ]; then
        local have; have="$(installed_version sglang-kernel)"
        if [ "$have" = "$SGL_KERNEL_VERSION" ]; then
            info "sglang-kernel   ${have} already correct"
        else
            info "sglang-kernel   ${have:-none} -> ${SGL_KERNEL_VERSION}"
            # The PyPI wheel tracks one CUDA version (currently cu130); anything
            # else needs the +cuXXX-tagged wheel from the SGLang index.
            if [ "$CU_VERSION" = "cu130" ]; then
                uv pip install "sglang-kernel==${SGL_KERNEL_VERSION}" \
                    --force-reinstall --no-deps --index-strategy unsafe-best-match
            else
                uv pip install "sglang-kernel==${SGL_KERNEL_VERSION}" \
                    --index-url "https://docs.sglang.ai/whl/${CU_VERSION}/" \
                    --force-reinstall --no-deps --index-strategy unsafe-best-match
            fi
        fi
    fi

    if [ "$CU_MAJOR" = "13" ] && [ -n "$NCCL_VERSION" ]; then
        info "nccl            nvidia-nccl-cu13==${NCCL_VERSION}"
        uv pip install "nvidia-nccl-cu13==${NCCL_VERSION}" --force-reinstall --no-deps
    fi

    if [ -n "$FLASHINFER_VERSION" ]; then
        # cubin is CUDA-agnostic, so its index has no cuXXX suffix.
        info "flashinfer      cubin==${FLASHINFER_VERSION}"
        uv pip install "flashinfer-cubin==${FLASHINFER_VERSION}" \
            --index-url https://flashinfer.ai/whl || warn "flashinfer-cubin install failed (kernels will JIT instead)"

        if [ "$INSTALL_FLASHINFER_CACHE" = "1" ]; then
            info "flashinfer      jit-cache==${FLASHINFER_VERSION}+${CU_VERSION} (~1.2G, saves minutes on first serve)"
            uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
                --index-url "https://flashinfer.ai/whl/${CU_VERSION}" \
                || warn "flashinfer-jit-cache unavailable for ${FLASHINFER_VERSION}+${CU_VERSION}; first server start will JIT-compile"
        else
            info "flashinfer      jit-cache skipped (INSTALL_FLASHINFER_CACHE=0)"
        fi
    fi
    ok "kernel wheels"
}

# --------------------------------------------------------------------------- #
# 9. Lint hooks (needed if any of this is going upstream)
# --------------------------------------------------------------------------- #
install_hooks() {
    [ "$INSTALL_PRECOMMIT" = "1" ] || return 0
    [ -f "$REPO_DIR/.pre-commit-config.yaml" ] || return 0
    log "pre-commit hooks"
    uv pip install pre-commit >/dev/null
    ( cd "$REPO_DIR" && pre-commit install >/dev/null ) && info "installed into ${REPO_DIR}/.git/hooks"
    ok "pre-commit"
}

# --------------------------------------------------------------------------- #
# 10. Verify
# --------------------------------------------------------------------------- #
verify() {
    log "Verify"
    SGLANG_EXPECTED_INIT="${REPO_DIR}/python/sglang/__init__.py" python3 - <<'PY'
import importlib.util, os, sys

import torch

want = os.environ["SGLANG_EXPECTED_INIT"]
spec = importlib.util.find_spec("sglang")
if spec is None:
    raise SystemExit("FAIL: sglang is not importable")
if spec.origin != want:
    raise SystemExit(
        f"FAIL: sglang resolves to {spec.origin}, expected {want} — "
        "something in site-packages is shadowing the checkout"
    )

import sglang  # noqa: E402  (after the find_spec check, like CI does)

print(f"    sglang          {sglang.__version__}  <- {spec.origin}")
print(f"    torch           {torch.__version__}  (cuda {torch.version.cuda})")
print(f"    cuda available  {torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"    gpu {i}           {torch.cuda.get_device_name(i)}  sm_{major}{minor}")

for mod in ("sgl_kernel", "flashinfer"):
    try:
        m = __import__(mod)
        print(f"    {mod:<15} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        print(f"    {mod:<15} NOT importable: {type(exc).__name__}: {exc}")
        sys.exit(1)
PY
    info "server args     $(python3 -c 'from sglang.srt.server_args import ServerArgs; print("ServerArgs imports clean")')"
    ok "verified"
}

# --------------------------------------------------------------------------- #
# Next steps
# --------------------------------------------------------------------------- #
summary() {
    local gpus="${GPU_COUNT:-4}"
    cat <<EOF

${B}Done.${N} Editable SGLang from ${REPO_DIR} @ $(git -C "$REPO_DIR" rev-parse --short HEAD)

Activate it:
    source ${VENV_DIR}/bin/activate

Serve (weights land in /workspace/models, reused across runs):
    python3 -m sglang.launch_server \\
        --model-path Qwen/Qwen3.6-35B-A3B \\
        --download-dir /workspace/models \\
        --tp-size ${gpus} --host 0.0.0.0 --port 30000

Pipeline-parallel research loop (edit python, restart, no reinstall):
    python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \\
        --download-dir /workspace/models --pp-size ${gpus} --tp-size 1 \\
        --port 30000 --disable-cuda-graph

    PP boundary hooks:
      python/sglang/srt/managers/scheduler_pp_mixin.py     (typed PP send/recv)
      python/sglang/srt/distributed/parallel_state.py      (send/recv_tensor_dict)
      python/sglang/srt/model_executor/forward_batch_info.py  (PPProxyTensors)

Housekeeping:
    killall_sglang                       # free wedged GPU memory
    supervisorctl start sglang model-ui  # hand the GPUs back to the image server
EOF
}

main() {
    printf '%ssglang dev install%s  %s\n' "$B" "$N" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    preflight
    stop_vast_services
    fetch_source
    read_pins
    make_venv
    install_torch_stack
    install_sglang
    pin_kernel_wheels
    install_hooks
    verify
    summary
    printf '\n%stotal%s %ss\n' "$B" "$N" "$SECONDS"
}

main "$@"
