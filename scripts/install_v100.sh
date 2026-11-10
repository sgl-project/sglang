#!/usr/bin/env bash
# Reproducible host installer for SGLang on NVIDIA V100 (SM70).
# Run this script as a child process; do not source it from an SSH login shell.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[install_v100] Do not source this file; run: bash %q\n' \
    "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local rc=$?
  printf '\n[install_v100] FAILED (exit %d) at line %d: %s\n' \
    "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2
  printf '[install_v100] Your login shell is still active; fix the error and rerun this script.\n' >&2
  exit "$rc"
}
trap on_error ERR

log() { printf '\n\033[1;34m[install_v100]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install_v100] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
FLASHINFER_REV="c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833"
# Pinned source revisions for the temporarily retained TurboMind component.
TURBOMIND_SOURCE_REV="6ada86ed64af6d1a7b3cb0f34df237fd86f06d48"
TURBOMIND_CUTLASS_REV="da5e086dab31d63815acafdac9a9c5893b1c69e2"

[[ -d "$REPO_ROOT/.git" ]] || die "$REPO_ROOT is not an SGLang-V100 checkout."

if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null || die "sudo is required to install system packages."
  SUDO=(sudo)
fi

log "Installing host compiler and CUDA 12.8 prerequisites"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential ca-certificates cmake curl git g++-12 g++-14 libnuma-dev \
  ninja-build pkg-config wget

# SGLang's Rust extensions use edition 2024 (rustc 1.85+); distro cargo is often
# older, so fall back to a user-local rustup toolchain.
[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
rust_minor="$(rustc --version 2>/dev/null | sed -n 's/^rustc 1\.\([0-9]*\).*/\1/p')"
if [[ -z "$rust_minor" ]] || (( rust_minor < 85 )); then
  log "Installing a Rust toolchain with rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal
  . "$HOME/.cargo/env"
fi

if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  CUDA_REPO="ubuntu${VERSION_ID//./}"
  wget -q \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y cuda-toolkit-12-8
fi

if ! command -v conda >/dev/null 2>&1; then
  log "Installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

CONDA_EXE="$(command -v conda || true)"
[[ -n "$CONDA_EXE" ]] || CONDA_EXE="$HOME/miniconda3/bin/conda"
[[ -x "$CONDA_EXE" ]] || die "conda was not found after installation."
# shellcheck disable=SC1090
. "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx sglang-v100; then
  conda create -y -n sglang-v100 python=3.12 pip
fi
conda activate sglang-v100

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0

# Use every CPU only when RAM can sustain that many compiler processes.  The
# previous unconditional nproc (88 jobs on the reference host, with no swap)
# could invoke the global OOM killer.  This is computed, never hard-coded to 8.
CPU_JOBS="$(nproc)"
MEM_AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
MEM_JOBS=$(( (MEM_AVAILABLE_KIB - 16 * 1024 * 1024) / (4 * 1024 * 1024) ))
(( MEM_JOBS < 1 )) && MEM_JOBS=1
SAFE_JOBS="$CPU_JOBS"
(( MEM_JOBS < SAFE_JOBS )) && SAFE_JOBS="$MEM_JOBS"
export MAX_JOBS="${MAX_JOBS:-$SAFE_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
log "Build parallelism: MAX_JOBS=$MAX_JOBS (CPU=$CPU_JOBS, RAM-safe=$SAFE_JOBS), NVCC_THREADS=$NVCC_THREADS"

# The validated V100 set, without resolution: upstream's pyproject pins target
# CUDA 13 and cannot resolve against torch 2.9.1 / cu128.
python -m pip install --upgrade pip
python -m pip install --no-deps -r "$REPO_ROOT/requirements.txt"
# The Rust radix tree needs torch >= 2.11 and is optional (the Python tree is
# the default backend).
SGLANG_BUILD_RUST_EXTS=grpc,multimodal,server \
  python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT/python"

prepare_patched_repo() {
  local name=$1 url=$2 rev=$3 destination=$4
  shift 4
  local backup_destination old_fingerprint patch_file patch_fingerprint stamp_file

  if (( $# > 0 )); then
    patch_fingerprint="$({ sha256sum "$@"; } | sha256sum | awk '{print $1}')"
  else
    patch_fingerprint="$(printf '%s' "$rev" | sha256sum | awk '{print $1}')"
  fi
  stamp_file="$destination/.sglang-v100-patches"

  # An installer-managed checkout is expected to be dirty because patches are
  # applied without creating commits. If those patches change, retain that
  # checkout verbatim and build a clean replacement rather than either
  # rejecting a normal upgrade or discarding possible local edits.
  if [[ -d "$destination/.git" ]] && [[ -f "$stamp_file" ]] && \
      [[ "$(<"$stamp_file")" != "$patch_fingerprint" ]] && \
      ! git -C "$destination" diff --quiet; then
    old_fingerprint="$(<"$stamp_file")"
    backup_destination="${destination}.sglang-v100-backup-${old_fingerprint:0:12}"
    [[ ! -e "$backup_destination" ]] || \
      die "managed dependency backup already exists: $backup_destination"
    log "Preserving previous patched $name checkout at $backup_destination"
    mv "$destination" "$backup_destination"
    stamp_file="$destination/.sglang-v100-patches"
  fi

  if [[ ! -d "$destination/.git" ]]; then
    log "Cloning $name at $rev"
    mkdir -p "$(dirname "$destination")"
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$rev"
  elif [[ "$(git -C "$destination" rev-parse HEAD)" != "$rev" ]]; then
    git -C "$destination" diff --quiet && \
      git -C "$destination" diff --cached --quiet || \
      die "$destination has local changes; move it aside or set SGLANG_V100_DEPS_DIR."
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi

  if [[ -f "$stamp_file" ]] && [[ "$(<"$stamp_file")" == "$patch_fingerprint" ]]; then
    return
  fi
  git -C "$destination" diff --quiet && \
    git -C "$destination" diff --cached --quiet || \
    die "$destination has untracked installer patch state; remove it and rerun."

  for patch_file in "$@"; do
    git -C "$destination" apply --check "$patch_file" || \
      die "$name patch does not apply cleanly: $patch_file"
    git -C "$destination" apply "$patch_file"
  done
  printf '%s\n' "$patch_fingerprint" >"$stamp_file"
}

prepare_sparse_repo() {
  local name=$1 url=$2 rev=$3 destination=$4
  shift 4

  if [[ ! -d "$destination/.git" ]]; then
    log "Fetching the attributed $name source subset at $rev"
    mkdir -p "$(dirname "$destination")"
    git clone --filter=blob:none --sparse --no-checkout "$url" "$destination"
  elif [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    die "$destination has local changes; move it aside or set SGLANG_V100_DEPS_DIR."
  fi

  git -C "$destination" sparse-checkout set "$@"
  if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$rev" ]]; then
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi
}

FLASHINFER_DIR="$DEPS_ROOT/flashinfer-sm70"
prepare_patched_repo \
  FlashInfer https://github.com/haohervchb/flashinfer.git \
  "$FLASHINFER_REV" "$FLASHINFER_DIR" \
  "$REPO_ROOT/patches/flashinfer-sm70.patch"
log "Installing the proven FlashInfer SM70 source"
python -m pip uninstall -y flashinfer-python flashinfer-cubin || true
python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"

# Language attention uses the in-tree TileLang package. Preserve an optional
# flash_attn_v100 installation: the throughput profile uses it for vision.
# See python/sglang/kernels/third_party/flash_v100/README.md for the separately
# validated vision build.

TURBOMIND_SOURCE_DIR="$DEPS_ROOT/turbomind-sm70-source"
prepare_sparse_repo \
  "LMDeploy/1Cat TurboMind" https://github.com/1CatAI/1Cat-vLLM.git \
  "$TURBOMIND_SOURCE_REV" "$TURBOMIND_SOURCE_DIR" \
  LICENSE csrc/core csrc/sm70_turbomind csrc/moe

TURBOMIND_CUTLASS_DIR="$DEPS_ROOT/cutlass-turbomind"
prepare_patched_repo \
  CUTLASS https://github.com/NVIDIA/cutlass.git \
  "$TURBOMIND_CUTLASS_REV" "$TURBOMIND_CUTLASS_DIR"

log "Building the attributed TurboMind SM70 block-FP8 and FP16 MoE backend"
SGLANG_TURBOMIND_SM70_ROOT="$TURBOMIND_SOURCE_DIR" \
SGLANG_TURBOMIND_CUTLASS_ROOT="$TURBOMIND_CUTLASS_DIR" \
  python "$REPO_ROOT/scripts/build_sm70_turbomind.py"

if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi
export CUTLASS_DIR="$HOME/cutlass"

log "Building lean SM70-only sglang-kernel"
# Remove the newer-GPU wheel first so pip cannot leave an orphaned common_ops
# filename beside the locally built ABI3 module.
python -m pip uninstall -y sglang-kernel || true
# A prior in-place build can leave an ignored CPython .so inside the source
# package. pip then silently bundles it beside the fresh ABI3 SM70 extension.
find "$REPO_ROOT/python/sglang/kernels/aot/python/sgl_kernel" \
  -type f -name 'common_ops*.so' -delete
python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    for artifact in (Path(root) / "sgl_kernel").glob("*/common_ops*.so"):
        artifact.unlink()
PY
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=$NVCC_THREADS"
python -m pip install --no-deps --no-build-isolation \
  "$REPO_ROOT/python/sglang/kernels/aot"

log "Restoring the CUDA 12 NCCL required by torch 2.9.1"
python -m pip uninstall -y nvidia-nccl-cu13 || true
python -m pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.5

log "Building V100 Marlin GPTQ/AWQ kernels"
export MARLIN_V100_REPO="${MARLIN_V100_REPO:-$DEPS_ROOT/marlin-v100}"
export MARLIN_V100_REF="${MARLIN_V100_REF:-6d72a49939701d26b15b617a4cd2423174adb2d1}"
bash "$REPO_ROOT/scripts/setup_v100_marlin.sh"

log "Running SM70 smoke checks and precompiling first-chat sampling"
if ! SGLANG_V100_FLASHINFER_DIR="$FLASHINFER_DIR" \
  bash "$REPO_ROOT/scripts/smoke_v100.sh"; then
  printf '\n[install_v100] All expensive builds completed successfully.\n' >&2
  printf '[install_v100] Rerun only validation (no rebuild) with:\n' >&2
  printf '  bash %q\n' "$REPO_ROOT/scripts/smoke_v100.sh" >&2
  exit 1
fi

log "Complete. Run: conda activate sglang-v100"
