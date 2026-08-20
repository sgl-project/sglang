#!/bin/bash
set -euo pipefail
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default
SKIP_TT_DEPS=""
SKIP_SGLANG_BUILD=""
SKIP_AITER_BUILD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-aiter-build) SKIP_AITER_BUILD="1"; shift;;
    --skip-sglang-build) SKIP_SGLANG_BUILD="1"; shift;;
    --skip-test-time-deps) SKIP_TT_DEPS="1"; shift;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [OPTIONAL_DEPS]"
      echo "Options:"
      echo "  --skip-sglang-build         Don't build checkout sglang, use what was shipped with the image"
      echo "  --skip-aiter-build          Don't build aiter, use what was shipped with the image"
      echo "  --skip-test-time-deps       Don't build miscellaneous dependencies"
      exit 0
      ;;
    *) break ;;
  esac
done

OPTIONAL_DEPS="${1:-}"

# Build python extras
EXTRAS="dev_hip,tracing"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev_hip,tracing,${OPTIONAL_DEPS}"
fi

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Identify the Dockerfile stage that built this image. Both the python extras
# below and the AITER pin lookup further down need the flavor, and they have to
# agree, so detect it once here.
#
# Prefer GPU_ARCH stamped into the image (gfx950-rocm724, gfx942, ...).
# Images built before that ENV existed: 724 stages already set
# PIP_CONSTRAINT and HSA_ENABLE_IPC_MODE_LEGACY; remaining HIP 7.2* is
# 720; else 7.0. Do not key off torch 2.11 — 720 may ship that later.
IMAGE_TORCH_VERSION=$(docker exec ci_sglang python3 -c 'import torch; print(torch.__version__)')
IMAGE_HIP_VERSION=$(docker exec ci_sglang python3 -c 'import torch; print(torch.version.hip or "")')
IMAGE_GPU_ARCH=$(docker exec ci_sglang printenv GPU_ARCH 2>/dev/null || true)
if [[ "${IMAGE_GPU_ARCH}" =~ ^(gfx942|gfx950)(-rocm720|-rocm724)?$ ]]; then
    echo "[CI-IMAGE] Image GPU_ARCH=${IMAGE_GPU_ARCH}"
    case "${IMAGE_GPU_ARCH}" in
        *-rocm724) IMAGE_BASE_ARG_SUFFIX="_ROCM724"; IMAGE_STAGE_SUFFIX="-rocm724" ;;
        *-rocm720) IMAGE_BASE_ARG_SUFFIX="_ROCM720"; IMAGE_STAGE_SUFFIX="-rocm720" ;;
        *)         IMAGE_BASE_ARG_SUFFIX=""; IMAGE_STAGE_SUFFIX="" ;;
    esac
    IMAGE_GFX="${IMAGE_GPU_ARCH%-*}"
else
    IMAGE_PIP_CONSTRAINT=$(docker exec ci_sglang printenv PIP_CONSTRAINT 2>/dev/null || true)
    IMAGE_HSA_LEGACY=$(docker exec ci_sglang printenv HSA_ENABLE_IPC_MODE_LEGACY 2>/dev/null || true)
    if [[ -n "${IMAGE_PIP_CONSTRAINT}" || "${IMAGE_HSA_LEGACY}" == "1" ]]; then
        IMAGE_BASE_ARG_SUFFIX="_ROCM724"
        IMAGE_STAGE_SUFFIX="-rocm724"
    elif [[ "${IMAGE_HIP_VERSION}" == 7.2* ]]; then
        IMAGE_BASE_ARG_SUFFIX="_ROCM720"
        IMAGE_STAGE_SUFFIX="-rocm720"
    else
        IMAGE_BASE_ARG_SUFFIX=""
        IMAGE_STAGE_SUFFIX=""
    fi
    if [[ "${GPU_ARCH}" == "mi35x" ]]; then
        IMAGE_GFX="gfx950"
    else
        IMAGE_GFX="gfx942"
    fi
    echo "[CI-IMAGE] Image has no GPU_ARCH stamp; inferred ${IMAGE_GFX}${IMAGE_STAGE_SUFFIX} (PIP_CONSTRAINT='${IMAGE_PIP_CONSTRAINT}', HSA_ENABLE_IPC_MODE_LEGACY='${IMAGE_HSA_LEGACY}', HIP=${IMAGE_HIP_VERSION})"
    unset IMAGE_PIP_CONSTRAINT IMAGE_HSA_LEGACY
fi
unset IMAGE_GPU_ARCH

# Install the required dependencies in CI.
# ROCm 7.2.4 images ship torch 2.11, which srt_hip cannot satisfy (it pins
# compressed-tensors 0.15.0, requiring torch<2.11). Select the rocm724 extras.
if [[ "${IMAGE_STAGE_SUFFIX}" == "-rocm724" ]]; then
  EXTRAS="${EXTRAS/dev_hip/dev_hip_rocm724}"
fi
echo "Image torch ${IMAGE_TORCH_VERSION}, HIP ${IMAGE_HIP_VERSION}; installing python extras: [${EXTRAS}]"

# Fix permissions on pip cache, ignore errors from concurrent access or missing temp files
docker exec ci_sglang chown -R root:root /sgl-data/pip-cache 2>/dev/null || true
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache --upgrade pip

# Helper function to install with retries and fallback PyPI mirror
install_with_retry() {
  local max_attempts=3
  local cmd="$@"

  for attempt in $(seq 1 $max_attempts); do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      echo "Success!"
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Failed, retrying in 5 seconds..."
      sleep 5
      # Try with alternative PyPI index on retry
      if [[ "$cmd" =~ "pip install" ]] && [ $attempt -eq 2 ]; then
        cmd="$cmd --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
        echo "Using fallback PyPI mirror: $cmd"
      fi
    fi
  done

  echo "Failed after $max_attempts attempts"
  return 1
}

# The anthropic SDK passes `socket_options` to httpx.HTTPTransport, which only
# exists in httpx>=0.25.0. The CI image ships an older httpx, and several deps
# installed below (lmms-eval, aiter's requirements.txt, etc.) can pull a stale
# httpx back in, so test_anthropic_server fails with:
#   TypeError: HTTPTransport.__init__() got an unexpected keyword argument 'socket_options'
# Call this as the LAST pip operation so nothing can downgrade httpx afterwards.
ensure_httpx() {
  install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache --upgrade 'httpx>=0.25.0'
}

# Helper function to git clone with retries
git_clone_with_retry() {
  local repo_url="$1"
  local dest_dir="${2:-}"
  local branch_args="${3:-}"
  local max_attempts=3

  for attempt in $(seq 1 $max_attempts); do
    echo "Git clone attempt $attempt/$max_attempts: $repo_url"

    # prevent from partial clone
    if [ -n "$dest_dir" ] && [ -d "$dest_dir" ]; then
      rm -rf "$dest_dir"
    fi

    if git \
      -c http.lowSpeedLimit=1000 \
      -c http.lowSpeedTime=30 \
      clone --depth 1 ${branch_args:+$branch_args} "$repo_url" "$dest_dir"; then
      echo "Git clone succeeded."
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Git clone failed, retrying in 5 seconds..."
      sleep 5
    fi
  done

  echo "Git clone failed after $max_attempts attempts: $repo_url"
  return 1
}

# Install checkout sglang
if [ -n "$SKIP_SGLANG_BUILD" ]; then
  echo "Didn't build checkout SGLang"
else
  docker exec ci_sglang pip uninstall sgl-kernel -y || true
  docker exec ci_sglang pip uninstall sglang-kernel -y || true
  docker exec ci_sglang pip uninstall sglang -y || true
  # Clear Python cache to ensure latest code is used
  docker exec ci_sglang find /opt/venv -name "*.pyc" -delete || true
  docker exec ci_sglang find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true
  # Also clear cache in sglang-checkout
  docker exec ci_sglang find /sglang-checkout -name "*.pyc" -delete || true
  docker exec ci_sglang find /sglang-checkout -name "__pycache__" -type d -exec rm -rf {} + || true
  docker exec -w /sglang-checkout/python/sglang/kernels/aot ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"

  docker exec ci_sglang bash -c 'rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml'
  install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e "python[${EXTRAS}]"
fi

# shellcheck source=scripts/ci/utils/sgl_eval_ref.sh
source "$(dirname "${BASH_SOURCE[0]}")/../utils/sgl_eval_ref.sh"
install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache "$SGL_EVAL_SPEC"

if [[ -n "${SKIP_TT_DEPS}" ]]; then
  echo "Didn't build lmms_eval, human-eval, and others"
else
  # For lmms_evals evaluating MMMU
  # Clone on host (with retry), then copy into the container. The checkout is
  # owned by the runner (non-root); mark it safe so setuptools_scm /
  # vcs_versioning can run `git` introspection during pip install.
  git_clone_with_retry https://github.com/EvolvingLMMs-Lab/lmms-eval.git lmms-eval "--branch v0.4.1"
  docker cp lmms-eval ci_sglang:/
  docker exec ci_sglang git config --global --add safe.directory /lmms-eval
  install_with_retry docker exec -w /lmms-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .

  # lmms-eval v0.4.1 pulls latex2sympy2, which pins antlr4-python3-runtime==4.7.2
  # and uninstalls the 4.9.3 that sgl-eval's latex2sympy2_extended requires, so
  # every `sgl-eval run mmlu` dies with "Unsupported ANTLR version 4.7.2". Pin it
  # back, the same way the CUDA installer does after its own lmms-eval install.
  install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache "antlr4-python3-runtime==4.9.3" --force-reinstall --no-deps

  git_clone_with_retry https://github.com/akao-amd/human-eval.git human-eval
  docker cp human-eval ci_sglang:/
  install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .

  mkdir -p dummy-grok
  cat > dummy-grok/config.json << 'EOF'
  {
    "architectures": [
      "Grok1ModelForCausalLM"
    ],
    "embedding_multiplier_scale": 78.38367176906169,
    "output_multiplier_scale": 0.5773502691896257,
    "vocab_size": 131072,
    "hidden_size": 6144,
    "intermediate_size": 32768,
    "max_position_embeddings": 8192,
    "num_experts_per_tok": 2,
    "num_local_experts": 8,
    "num_attention_heads": 48,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "model_type": "mixtral",
    "torch_dtype": "bfloat16"
  }
EOF
  docker exec -w / ci_sglang mkdir -p /dummy-grok
  docker cp ./dummy-grok/config.json ci_sglang:/dummy-grok/config.json

  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache huggingface_hub[hf_xet]
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache pytest

  # Install cache-dit for qwen_image_t2i_cache_dit_enabled test (added in PR 16204)
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache --upgrade 'cache-dit==1.3.0' || echo "cache-dit installation failed"

  # Install accelerate for distributed training and inference support
  docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache accelerate || echo "accelerate installation failed"
fi

# -----------------------
# MORI
# The CI image bakes MORI at the docker/rocm.Dockerfile-pinned commit; when a PR
# bumps MORI_COMMIT the image is not rebuilt, so reinstall MORI here the same way
# the Dockerfile does. Only ENABLE_MORI=1 images ship /sgl-workspace/mori.
if docker exec ci_sglang test -d /sgl-workspace/mori; then
  MORI_REPO=$(grep -E '^[[:space:]]*ARG[[:space:]]+MORI_REPO=' docker/rocm.Dockerfile | head -n1 | sed 's/.*MORI_REPO="\([^"]*\)".*/\1/')
  MORI_COMMIT=$(grep -E '^[[:space:]]*ARG[[:space:]]+MORI_COMMIT=' docker/rocm.Dockerfile | head -n1 | sed 's/.*MORI_COMMIT="\([^"]*\)".*/\1/')

  if [[ "${GPU_ARCH}" == "mi35x" ]]; then
    MORI_GPU_ARCHS="gfx950"
  else
    MORI_GPU_ARCHS="gfx942"
  fi

  echo "[MORI] Reinstalling MORI ${MORI_COMMIT} (MORI_GPU_ARCHS=${MORI_GPU_ARCHS})"
  docker exec ci_sglang bash -c "
    set -euo pipefail
    export MORI_GPU_ARCHS='${MORI_GPU_ARCHS}'
    rm -rf /sgl-workspace/mori
    git clone '${MORI_REPO}' /sgl-workspace/mori
    cd /sgl-workspace/mori
    git checkout '${MORI_COMMIT}'
    git submodule update --init --recursive
    apt-get update
    apt-get install -y --no-install-recommends libgrpc++-dev 2>/dev/null || true
    python3 setup.py develop
    python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))' > /etc/ld.so.conf.d/torch.conf
    ldconfig
  "
  echo "[MORI] Done."
fi

if [[ -n "${SKIP_AITER_BUILD}" ]]; then
  ensure_httpx
  exit 0
fi

# Detect AITER version
#############################################
# Detect correct AITER_COMMIT for this runner
# + Check mismatch
# + Rebuild AITER if needed
#############################################

echo "[CI-AITER-CHECK] === AITER VERSION CHECK START ==="

DOCKERFILE="docker/rocm.Dockerfile"

# GPU_ARCH
GPU_ARCH="${GPU_ARCH:-mi30x}"
echo "[CI-AITER-CHECK] Runner GPU_ARCH=${GPU_ARCH}"

# Image owns Triton (pinned in docker/rocm.Dockerfile). Rebuild AITER against it.

#############################################
# 1. Extract AITER_COMMIT from the Dockerfile stage that built this image, as
# identified near the top of this script.
#############################################
if [[ "${IMAGE_GFX}" == "gfx950" ]]; then
    _from_line="FROM \$BASE_IMAGE_950${IMAGE_BASE_ARG_SUFFIX} AS gfx950${IMAGE_STAGE_SUFFIX}"
else
    _from_line="FROM \$BASE_IMAGE_942${IMAGE_BASE_ARG_SUFFIX} AS gfx942${IMAGE_STAGE_SUFFIX}"
fi
echo "[CI-AITER-CHECK] Using ${_from_line} from Dockerfile..."
REPO_AITER_COMMIT=$(grep -F -A20 "${_from_line}" docker/rocm.Dockerfile \
                    | grep 'AITER_COMMIT_DEFAULT=' \
                    | head -n1 \
                    | sed 's/.*AITER_COMMIT_DEFAULT="\([^"]*\)".*/\1/')
unset _from_line


if [[ -z "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] ERROR: Failed to extract AITER_COMMIT from Dockerfile."
    exit 1
fi

echo "[CI-AITER-CHECK] Dockerfile expects AITER_COMMIT=${REPO_AITER_COMMIT}"

#############################################
# 2. Check container pre-installed AITER version
#############################################
IMAGE_AITER_VERSION=$(docker exec ci_sglang bash -c "pip show amd-aiter 2>/dev/null | grep '^Version:' | awk '{print \$2}'" || echo "none")
IMAGE_AITER_VERSION="v${IMAGE_AITER_VERSION}"
echo "[CI-AITER-CHECK] AITER version inside CI image: ${IMAGE_AITER_VERSION}"

#############################################
# 3. Decide rebuild
#############################################
NEED_REBUILD="false"

if [[ -n "${AITER_COMMIT_OVERRIDE:-}" ]]; then
    echo "[CI-AITER-CHECK] AITER_COMMIT_OVERRIDE=${AITER_COMMIT_OVERRIDE} → forcing rebuild"
    REPO_AITER_COMMIT="${AITER_COMMIT_OVERRIDE}"
    NEED_REBUILD="true"
elif [[ "${IMAGE_AITER_VERSION}" == "vnone" || "${IMAGE_AITER_VERSION}" == "v" ]]; then
    echo "[CI-AITER-CHECK] No AITER found in image → rebuild needed"
    NEED_REBUILD="true"
elif [[ "${IMAGE_AITER_VERSION}" == "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] AITER version matches"
elif [[ "${IMAGE_AITER_VERSION}" =~ (dev|\+g[0-9a-f]+) ]]; then
    # Dev/patched version (contains 'dev' or git hash) → preserve it
    echo "[CI-AITER-CHECK] Dev/patched version detected: ${IMAGE_AITER_VERSION} → skipping rebuild"
else
    echo "[CI-AITER-CHECK] Version mismatch: image=${IMAGE_AITER_VERSION}, repo=${REPO_AITER_COMMIT}"
    NEED_REBUILD="true"
fi

#############################################
# 4. Rebuild AITER if needed
#############################################
if [[ "${NEED_REBUILD}" == "true" ]]; then
    echo "[CI-AITER-CHECK] === AITER REBUILD START ==="

    # uninstall existing aiter
    docker exec ci_sglang pip uninstall -y amd-aiter || true

    # delete old aiter directory
    docker exec ci_sglang rm -rf /sgl-workspace/aiter

    # clone a fresh copy to /sgl-workspace/aiter
    docker exec ci_sglang git clone https://github.com/ROCm/aiter.git /sgl-workspace/aiter

    # checkout correct version and install requirements
    # Use `checkout -f` so the smudge-filter-induced "dirty" working tree from
    # AITER's .gitattributes (*.csv text eol=lf, added in ROCm/aiter#3370) does
    # not block switching to commits that predate that rule. The working tree
    # was just produced by `rm -rf` + fresh `git clone` above, so there are no
    # real user changes to preserve.
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        git fetch --all && \
        git checkout -f ${REPO_AITER_COMMIT} && \
        git submodule update --init --recursive && \
        pip install -r requirements.txt
    "

    # Re-apply the Dockerfile torch.Stream patch after re-clone (ROCm/aiter#4817).
    if [[ "${IMAGE_STAGE_SUFFIX}" == "-rocm724" ]]; then
        docker exec -i ci_sglang python3 - <<'PY'
from pathlib import Path
p = Path("/sgl-workspace/aiter/csrc/cpp_itfs/torch_utils.py")
s = p.read_text()
old = """        elif isinstance(arg, torch.cuda.Stream):
            c_args.append(ctypes.cast(arg.cuda_stream, ctypes.c_void_p))
"""
new = """        elif isinstance(arg, torch.Stream):
            handle = getattr(arg, "cuda_stream", None)
            if handle is None:
                handle = torch.cuda.Stream(
                    stream_id=arg.stream_id,
                    device_index=arg.device_index,
                    device_type=arg.device_type,
                ).cuda_stream
            c_args.append(ctypes.cast(handle, ctypes.c_void_p))
"""
if old in s:
    p.write_text(s.replace(old, new))
PY
    fi

    if [[ "${GPU_ARCH}" == "mi35x" ]]; then
        GPU_ARCH_LIST="gfx950"
    else
        GPU_ARCH_LIST="gfx942"
    fi
    echo "[CI-AITER-CHECK] GPU_ARCH_LIST=${GPU_ARCH_LIST}"

    # The image already has the Dockerfile-pinned Triton; compile against it.
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=${GPU_ARCH_LIST} python3 setup.py develop
    "

    echo "[CI-AITER-CHECK] === AITER REBUILD COMPLETE ==="
fi

echo "[CI-AITER-CHECK] === AITER VERSION CHECK END ==="

# Must be the final pip operation: force httpx>=0.25.0 so the anthropic SDK can
# construct its httpx transport (see ensure_httpx definition above).
ensure_httpx


# # Clear pre-built AITER kernels from Docker image to avoid segfaults
# # The Docker image may contain pre-compiled kernels incompatible with the current environment
# echo "Clearing pre-built AITER kernels from Docker image..."
# docker exec ci_sglang find /sgl-workspace/aiter/aiter/jit -name "*.so" -delete 2>/dev/null || true
# docker exec ci_sglang ls -la /sgl-workspace/aiter/aiter/jit/ 2>/dev/null || echo "jit dir empty or not found"

# # Pre-build AITER kernels to avoid timeout during tests
# echo "Warming up AITER JIT kernels..."
# docker exec -e SGLANG_USE_AITER=1 ci_sglang python3 /sglang-checkout/scripts/ci/amd/amd_ci_warmup_aiter.py || echo "AITER warmup completed (some kernels may not be available)"
