#!/usr/bin/env bash
set -euo pipefail

# Runs the sampling-mask smoke test in a pinned SGLang Docker image while
# importing SGLang from this local checkout.
#
# The default image is docker.io/lmsysorg/sglang:latest resolved on 2026-06-04.
# Override SGLANG_DOCKER_IMAGE=lmsysorg/sglang:latest to intentionally use the
# moving latest tag instead of this pinned digest.
#
# Normal reviewer run from inside a GPU pod or GPU host with Docker + NVIDIA
# Container Toolkit. The current working tree must be the checkout under review
# and the 35B model must be mounted at the default path below:
#
#   cd /workspace/sglang
#   scripts/docker_smoke_sglang_sampling_mask.sh
#
# Preconditions for the normal run:
#
#   - Run from the local SGLang checkout containing the patch under review.
#     If your checkout is not at /workspace/sglang, cd to its actual path first.
#   - The machine has a visible NVIDIA GPU and `docker run --gpus all` works.
#   - The default model path exists:
#       /mnt/lustre/slime/models/Qwen3.5-35B-A3B-FP8
#     Override MODEL_PATH if the model lives somewhere else.
#
# Reproducible GKE/Docker-socket run from inside an already allocated GPU pod:
#
#   # Pod requirements:
#   #   - nvidia.com/gpu: 1
#   #   - hostPath /var/run/docker.sock mounted at /var/run/docker.sock
#   #   - an exec-capable hostPath workspace mounted at
#   #     /mnt/stateful_partition/sglang-docker-smoke
#   #
#   # Stage the checkout and host NVIDIA driver into that hostPath. This matters
#   # because the Docker daemon resolves bind mounts on the node, not inside the
#   # Kubernetes container.
#   mkdir -p /mnt/stateful_partition/sglang-docker-smoke
#   cp -a /path/to/sglang /mnt/stateful_partition/sglang-docker-smoke/sglang
#   cp -a /usr/local/nvidia /mnt/stateful_partition/sglang-docker-smoke/nvidia
#   mkdir -p /mnt/stateful_partition/sglang-docker-smoke/hf
#
#   cd /mnt/stateful_partition/sglang-docker-smoke/sglang
#   SGLANG_DOCKER_GPU_MODE=devices \
#   SGLANG_DOCKER_COPY_DRIVER=1 \
#   NVIDIA_DRIVER_HOST_DIR=/mnt/stateful_partition/sglang-docker-smoke/nvidia \
#   HF_CACHE=/mnt/stateful_partition/sglang-docker-smoke/hf \
#   scripts/docker_smoke_sglang_sampling_mask.sh
#
# The GKE-specific device mode is only for nested Docker on Kubernetes nodes
# where `docker run --gpus all` is unavailable. Maintainers on ordinary GPU
# hosts should use the normal command above.

DEFAULT_IMAGE="lmsysorg/sglang:latest@sha256:ceaf8b16e02d165143633ac228bbb994a05fe77d7e0526cf035ae4bbf4eacc36"
DEFAULT_MODEL_PATH="/mnt/lustre/slime/models/Qwen3.5-35B-A3B-FP8"

usage() {
  cat <<EOF
Usage: $0 [smoke-test args]

Runs scripts/smoke_sglang_sampling_mask.py inside a pinned SGLang Docker image.
The local checkout is mounted into the container and takes precedence via
PYTHONPATH, so uncommitted local changes are what the smoke test exercises.

Default command:
  $0

Environment:
  SGLANG_DOCKER_IMAGE  Docker image to run.
                       Default: ${DEFAULT_IMAGE}
  MODEL_PATH           HF model id or host model path.
                       Default: ${DEFAULT_MODEL_PATH}
  HF_CACHE             Host Hugging Face cache directory.
                       Default: \$HF_HOME or ~/.cache/huggingface
  GPUS                 Docker --gpus value. Default: all
                       In Kubernetes pods, defaults to device=\$NVIDIA_VISIBLE_DEVICES.
  SGLANG_DOCKER_GPU_MODE
                       gpus: use Docker --gpus. devices: mount /dev/nvidia*
                       manually. none: do not pass GPU flags. Default: gpus
  NVIDIA_DRIVER_HOST_DIR
                       Host-visible NVIDIA driver directory for devices mode.
  SGLANG_DOCKER_COPY_DRIVER
                       In devices mode, copy NVIDIA_DRIVER_HOST_DIR inside the
                       container before loading driver libs. Default: 0
  SGLANG_DOCKER_REMAP_GPU_DEVICE
                       In devices mode, remap a single /dev/nvidiaN to
                       /dev/nvidia0. Use 1, 0, or auto. Default: auto
  PORT                 Server port. Default: 30000
  SHM_SIZE             Docker shared memory size. Default: 32g
  SGLANG_DOCKER_PULL   Pull image before running. Default: 1
  SGLANG_DOCKER_INSTALL_KERNEL
                       Install the sglang-kernel version required by this
                       checkout inside the container. Default: 1
  DISABLE_PIECEWISE_CUDA_GRAPH
                       Pass --disable-piecewise-cuda-graph. Default: 1
  DRY_RUN              Print the docker command without running it. Default: 0

Any extra arguments are forwarded to smoke_sglang_sampling_mask.py and can
override defaults, for example:
  $0 --mem-fraction-static 0.45
  MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct $0 --top-p 0.95 --top-k 32
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_REPO="${CONTAINER_REPO:-/workspace/sglang}"

IMAGE="${SGLANG_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
HF_CACHE="${HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
if [[ -n "${GPUS:-}" ]]; then
  GPUS="${GPUS}"
elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "all" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
  GPUS="device=${NVIDIA_VISIBLE_DEVICES}"
else
  GPUS="all"
fi
PORT="${PORT:-30000}"
SHM_SIZE="${SHM_SIZE:-32g}"
SGLANG_DOCKER_PULL="${SGLANG_DOCKER_PULL:-1}"
PLATFORM="${SGLANG_DOCKER_PLATFORM:-linux/amd64}"
DRY_RUN="${DRY_RUN:-0}"
GPU_MODE="${SGLANG_DOCKER_GPU_MODE:-gpus}"
COPY_DRIVER="${SGLANG_DOCKER_COPY_DRIVER:-0}"
REMAP_GPU_DEVICE="${SGLANG_DOCKER_REMAP_GPU_DEVICE:-auto}"
INSTALL_KERNEL="${SGLANG_DOCKER_INSTALL_KERNEL:-1}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH" >&2
  exit 1
fi

mkdir -p "${HF_CACHE}"
HF_CACHE="$(cd "${HF_CACHE}" && pwd)"

if [[ "${DRY_RUN}" != "1" && "${MODEL_PATH}" == /* && ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is absolute but does not exist on the host: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${SGLANG_DOCKER_PULL}" != "0" && "${DRY_RUN}" != "1" ]]; then
  docker pull --platform "${PLATFORM}" "${IMAGE}"
fi

docker_args=(
  run
  --rm
  --platform "${PLATFORM}"
  --ipc=host
  --network=host
  --shm-size "${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  --workdir "${CONTAINER_REPO}"
  --env "PYTHONPATH=${CONTAINER_REPO}/python"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=/root/.cache/huggingface"
  --env "SGLANG_DOCKER_INSTALL_KERNEL=${INSTALL_KERNEL}"
  --mount "type=bind,source=${REPO_ROOT},target=${CONTAINER_REPO},readonly"
  --mount "type=bind,source=${HF_CACHE},target=/root/.cache/huggingface"
)

case "${GPU_MODE}" in
  gpus)
    docker_args+=(--gpus "${GPUS}")
    ;;
  devices)
    found_nvidia_device=0
    remapped_nvidia_device=0
    while IFS= read -r device_path; do
      device_name="$(basename "${device_path}")"
      if [[ "${device_name}" =~ ^nvidia[0-9]+$ ]] \
        && [[ "${remapped_nvidia_device}" == "0" ]] \
        && { [[ "${REMAP_GPU_DEVICE}" == "1" ]] || { [[ "${REMAP_GPU_DEVICE}" == "auto" ]] && [[ "${device_path}" != "/dev/nvidia0" ]] && [[ ! -e /dev/nvidia0 ]]; }; }; then
        docker_args+=(--device "${device_path}:/dev/nvidia0")
        docker_args+=(--env "CUDA_VISIBLE_DEVICES=0")
        remapped_nvidia_device=1
      else
        docker_args+=(--device "${device_path}")
      fi
      found_nvidia_device=1
    done < <(find /dev -maxdepth 1 -name 'nvidia*' -type c | sort)
    if [[ "${found_nvidia_device}" != "1" ]]; then
      echo "No /dev/nvidia* devices found for SGLANG_DOCKER_GPU_MODE=devices" >&2
      exit 1
    fi
    if [[ -n "${NVIDIA_DRIVER_HOST_DIR:-}" ]]; then
      if [[ "${COPY_DRIVER}" == "1" ]]; then
        docker_args+=(--mount "type=bind,source=${NVIDIA_DRIVER_HOST_DIR},target=/host_nvidia,readonly")
      else
        docker_args+=(
          --mount "type=bind,source=${NVIDIA_DRIVER_HOST_DIR},target=/usr/local/nvidia,readonly"
          --env "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
          --env "PATH=/usr/local/nvidia/bin:${PATH}"
        )
      fi
    fi
    ;;
  none)
    ;;
  *)
    echo "Unsupported SGLANG_DOCKER_GPU_MODE=${GPU_MODE}; expected gpus, devices, or none" >&2
    exit 1
    ;;
esac

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  docker_args+=(--env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
fi

if [[ "${MODEL_PATH}" == /* ]]; then
  docker_args+=(--mount "type=bind,source=${MODEL_PATH},target=${MODEL_PATH},readonly")
fi

smoke_args=(
  --model-path "${MODEL_PATH}"
  --host "0.0.0.0"
  --port "${PORT}"
  --context-length "${CONTEXT_LENGTH:-2048}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}"
  --top-p "${TOP_P:-0.9}"
  --top-k "${TOP_K:-16}"
  --temperature "${TEMPERATURE:-0.7}"
  --max-new-tokens "${MAX_NEW_TOKENS:-4}"
  --top-logprobs-num "${TOP_LOGPROBS_NUM:-32}"
)

if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  smoke_args+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  smoke_args+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi
if [[ -n "${TP_SIZE:-}" ]]; then
  smoke_args+=(--tp-size "${TP_SIZE}")
fi
if [[ -n "${DP_SIZE:-}" ]]; then
  smoke_args+=(--dp-size "${DP_SIZE}")
fi
if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" != "0" ]]; then
  smoke_args+=(--disable-piecewise-cuda-graph)
fi

echo "Running sampling-mask smoke in Docker image: ${IMAGE}"
echo "Mounted checkout: ${REPO_ROOT} -> ${CONTAINER_REPO}"
echo "Model: ${MODEL_PATH}"

inner_cmd=(
  python3 "${CONTAINER_REPO}/scripts/smoke_sglang_sampling_mask.py" \
    "${smoke_args[@]}" "$@"
)

inner_prelude='set -euo pipefail;'
if [[ "${GPU_MODE}" == "devices" && "${COPY_DRIVER}" == "1" ]]; then
  inner_prelude+=' rm -rf /tmp/nvidia-runtime; mkdir -p /tmp/nvidia-runtime; cp -a /host_nvidia/. /tmp/nvidia-runtime/; export LD_LIBRARY_PATH="/tmp/nvidia-runtime/lib64:${LD_LIBRARY_PATH:-}"; export PATH="/tmp/nvidia-runtime/bin:${PATH}";'
fi
inner_prelude+=' if [[ "${SGLANG_DOCKER_INSTALL_KERNEL:-1}" != "0" ]]; then required_kernel="$(python3 -c '\''import tomllib; deps=tomllib.load(open("/workspace/sglang/python/pyproject.toml", "rb"))["project"]["dependencies"]; print(next(dep.split("==", 1)[1] for dep in deps if dep.startswith("sglang-kernel==")))'\'')"; python3 -m pip install "sglang-kernel==${required_kernel}" --force-reinstall --no-deps; fi; exec "$@"'

cmd=(
  docker "${docker_args[@]}" "${IMAGE}" \
    bash -lc "${inner_prelude}" \
    bash "${inner_cmd[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry-run command:\n'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
