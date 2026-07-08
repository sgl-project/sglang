#!/bin/bash
set -uo pipefail

MODE="${NETWORK_DIAG_MODE:-full}"
SOURCE="${NETWORK_DIAG_SOURCE:-dockerhub}"
PHASE="${NETWORK_DIAG_PHASE:-manual}"
ROCM_VERSION="${NETWORK_DIAG_ROCM_VERSION:-rocm700}"
IMAGE_TAG="${NETWORK_DIAG_IMAGE_TAG:-v0.5.14-${ROCM_VERSION}-mi30x-20260705}"
LOCAL_DOCKER_REGISTRY="${NETWORK_DIAG_LOCAL_REGISTRY:-10.44.14.109:5000}"
PULL_TIMEOUT_SECONDS="${NETWORK_DIAG_PULL_TIMEOUT_SECONDS:-7200}"
REMOVE_IMAGE="${NETWORK_DIAG_REMOVE_IMAGE:-1}"
HF_RANGE_END="${NETWORK_DIAG_HF_RANGE_END:-104857599}"
HF_URL="${NETWORK_DIAG_HF_URL:-https://huggingface.co/gpt2/resolve/main/pytorch_model.bin}"
CONTAINER_NAME="${NETWORK_DIAG_CONTAINER:-}"
NETWORK_DIAG_GIT_REPO="${NETWORK_DIAG_GIT_REPO:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
NETWORK_DIAG_GIT_BRANCH="${NETWORK_DIAG_GIT_BRANCH-v0.4.1}"
export NETWORK_DIAG_GIT_REPO NETWORK_DIAG_GIT_BRANCH

case "${MODE}" in
  full|prepull|downloads|high-concurrency) ;;
  *)
    echo "Unsupported NETWORK_DIAG_MODE='${MODE}'. Use full, prepull, downloads, or high-concurrency."
    exit 2
    ;;
esac

safe_job="${GITHUB_JOB:-manual}"
safe_job="$(printf '%s' "${safe_job}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_phase="$(printf '%s' "${PHASE}" | tr -c 'A-Za-z0-9_.-' '_')"
out_dir="${RUNNER_TEMP:-/tmp}"
out_file="${out_dir}/network-diagnostics-${MODE}-${SOURCE}-${ROCM_VERSION}-${safe_phase}-${safe_job}.txt"
mkdir -p "${out_dir}"
exec > >(tee -a "${out_file}") 2>&1

public_image="docker.io/rocm/sgl-dev:${IMAGE_TAG}"
local_image="${LOCAL_DOCKER_REGISTRY}/rocm/sgl-dev:${IMAGE_TAG}"
case "${SOURCE}" in
  dockerhub) pull_image="${public_image}" ;;
  local) pull_image="${local_image}" ;;
  inline|prepull|downloads|high-concurrency|mi300) pull_image="${public_image}" ;;
  *)
    echo "Unsupported NETWORK_DIAG_SOURCE='${SOURCE}'. Use dockerhub, local, inline, or mi300."
    exit 2
    ;;
esac

probe() {
  local key="$1"
  local desc="$2"
  shift 2
  local start end rc

  echo "::group::${desc}"
  echo "${key}_start=$(date -Is)"
  start=$(date +%s)
  "$@"
  rc=$?
  end=$(date +%s)
  echo "${key}_rc=${rc}"
  echo "${key}_elapsed_sec=$((end - start))"
  echo "${key}_end=$(date -Is)"
  echo "::endgroup::"
  return 0
}

curl_probe() {
  local key="$1"
  local url="$2"
  local max_time="${3:-60}"

  probe "${key}" "curl timing: ${url}" \
    curl -L -o /dev/null -sS --max-time "${max_time}" \
      -w "${key}_url_effective=%{url_effective}\n${key}_http_code=%{http_code}\n${key}_time_namelookup=%{time_namelookup}\n${key}_time_connect=%{time_connect}\n${key}_time_appconnect=%{time_appconnect}\n${key}_time_starttransfer=%{time_starttransfer}\n${key}_time_total=%{time_total}\n${key}_speed_download_bytes_per_sec=%{speed_download}\n" \
      "${url}"
}

container_probe() {
  local key="$1"
  local desc="$2"
  local script="$3"

  if [[ -n "${CONTAINER_NAME}" ]] && docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    probe "${key}" "${desc} (container: ${CONTAINER_NAME})" \
      docker exec "${CONTAINER_NAME}" bash -lc "${script}"
  else
    probe "${key}" "${desc} (host fallback)" \
      bash -lc "${script}"
  fi
}

write_header() {
  echo "=== AMD network diagnostics ==="
  echo "date=$(date -Is)"
  echo "mode=${MODE}"
  echo "source=${SOURCE}"
  echo "phase=${PHASE}"
  echo "rocm_version=${ROCM_VERSION}"
  echo "image_tag=${IMAGE_TAG}"
  echo "public_image=${public_image}"
  echo "local_image=${local_image}"
  echo "pull_image=${pull_image}"
  echo "container_name=${CONTAINER_NAME}"
  echo "runner_name=${RUNNER_NAME:-}"
  echo "runner_os=${RUNNER_OS:-}"
  echo "runner_arch=${RUNNER_ARCH:-}"
  echo "github_run_id=${GITHUB_RUN_ID:-}"
  echo "github_job=${GITHUB_JOB:-}"
  echo "github_repository=${GITHUB_REPOSITORY:-}"
  echo "github_sha=${GITHUB_SHA:-}"
  echo "output_file=${out_file}"
}

run_host_context_probes() {
  probe host_identity "host identity and routes" bash -lc '
    set -x
    hostname || true
    uname -a || true
    ip addr || true
    ip route || true
    getent hosts registry-1.docker.io || true
    getent hosts production.cloudflare.docker.com || true
    getent hosts huggingface.co || true
    getent hosts files.pythonhosted.org || true
    getent hosts github.com || true
    getent hosts codeload.github.com || true
  '

  probe disk_and_cache "disk and cache mounts" bash -lc '
    set -x
    df -h || true
    mount | grep -E "sglang-data|sgl-data|docker" || true
    ls -ld /home/runner/sglang-data /home/runner/temp-sglang-data 2>/dev/null || true
    du -sh /home/runner/sglang-data /home/runner/temp-sglang-data 2>/dev/null || true
  '
}

run_registry_probes() {
  probe docker_info "docker info" bash -lc '
    set -x
    docker version || true
    docker info || true
    docker system df || true
  '

  if [[ -n "${DOCKERHUB_AMD_USERNAME:-}" && -n "${DOCKERHUB_AMD_TOKEN:-}" ]]; then
    probe dockerhub_login "Docker Hub login" \
      bash -lc 'echo "${DOCKERHUB_AMD_TOKEN}" | docker login -u "${DOCKERHUB_AMD_USERNAME}" --password-stdin'
  else
    echo "Docker Hub login skipped: DOCKERHUB_AMD_USERNAME/DOCKERHUB_AMD_TOKEN not set."
  fi

  curl_probe dockerhub_registry_v2 https://registry-1.docker.io/v2/ 60
  curl_probe dockerhub_tags_api "https://registry.hub.docker.com/v2/repositories/rocm/sgl-dev/tags/${IMAGE_TAG}" 60
  curl_probe dockerhub_blob_cdn https://production.cloudflare.docker.com/ 60
  curl_probe local_registry_v2 "http://${LOCAL_DOCKER_REGISTRY}/v2/" 60
  curl_probe local_registry_manifest "http://${LOCAL_DOCKER_REGISTRY}/v2/rocm/sgl-dev/manifests/${IMAGE_TAG}" 60
  curl_probe pypi_simple https://pypi.org/simple/pip/ 60
  curl_probe huggingface_home https://huggingface.co/ 60
  curl_probe github_home https://github.com/ 60
  curl_probe github_raw https://raw.githubusercontent.com/sgl-project/sglang/main/README.md 60

  probe docker_manifest_public "public Docker Hub manifest inspect" \
    docker manifest inspect "${public_image}"
}

run_docker_pull_probe() {
  if [[ "${REMOVE_IMAGE}" == "1" ]]; then
    probe docker_remove_target_images "remove target image tags before pull" \
      docker image rm "${pull_image}" "rocm/sgl-dev:${IMAGE_TAG}" "${public_image}" "${local_image}"
  fi

  if command -v timeout >/dev/null 2>&1; then
    probe docker_pull "${SOURCE} docker pull: ${pull_image}" \
      timeout "${PULL_TIMEOUT_SECONDS}" docker pull "${pull_image}"
  else
    probe docker_pull "${SOURCE} docker pull: ${pull_image}" \
      docker pull "${pull_image}"
  fi

  probe docker_image_size "docker image size after pull" \
    docker image inspect "${pull_image}" --format 'image_size_bytes={{.Size}}'
}

run_named_docker_pull_probe() {
  local key_prefix="$1"
  local image="$2"
  local desc="$3"

  if [[ "${REMOVE_IMAGE}" == "1" ]]; then
    probe "${key_prefix}_remove_image" "remove target image tags before ${desc} pull" \
      docker image rm "${image}" "rocm/sgl-dev:${IMAGE_TAG}" "${public_image}" "${local_image}"
  fi

  if command -v timeout >/dev/null 2>&1; then
    probe "${key_prefix}_pull" "${desc} docker pull: ${image}" \
      timeout "${PULL_TIMEOUT_SECONDS}" docker pull "${image}"
  else
    probe "${key_prefix}_pull" "${desc} docker pull: ${image}" \
      docker pull "${image}"
  fi

  probe "${key_prefix}_image_size" "${desc} docker image size after pull" \
    docker image inspect "${image}" --format 'image_size_bytes={{.Size}}'
}

run_download_probes() {
  probe github_clone "GitHub clone speed probe: ${NETWORK_DIAG_GIT_REPO}" bash -lc '
    set -ex
    rm -rf /tmp/network-diag-git
    trap "rm -rf /tmp/network-diag-git" EXIT
    mkdir -p /tmp/network-diag-git
    branch_args=()
    if [[ -n "${NETWORK_DIAG_GIT_BRANCH}" ]]; then
      branch_args=(--branch "${NETWORK_DIAG_GIT_BRANCH}")
    fi
    git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
      clone --depth 1 "${branch_args[@]}" "${NETWORK_DIAG_GIT_REPO}" /tmp/network-diag-git/repo
    du -sh /tmp/network-diag-git/repo
  '

  container_probe pypi_download "PyPI wheel download speed probe" '
    set -ex
    rm -rf /tmp/network-diag-pip
    trap "rm -rf /tmp/network-diag-pip" EXIT
    mkdir -p /tmp/network-diag-pip
    python3 -m pip download --no-cache-dir -d /tmp/network-diag-pip pyarrow==16.1.0
    du -sh /tmp/network-diag-pip
  '

  container_probe hf_range_download "Hugging Face range download speed probe" "
    set -ex
    rm -f /tmp/network-diag-hf.bin
    trap 'rm -f /tmp/network-diag-hf.bin' EXIT
    curl -L --range 0-${HF_RANGE_END} --max-time 900 -o /tmp/network-diag-hf.bin -sS \
      -w 'hf_range_url_effective=%{url_effective}\nhf_range_http_code=%{http_code}\nhf_range_size_download=%{size_download}\nhf_range_time_total=%{time_total}\nhf_range_speed_download_bytes_per_sec=%{speed_download}\n' \
      '${HF_URL}'
    ls -lh /tmp/network-diag-hf.bin
  "
}

run_download_probes_with_temp_container() {
  local image="$1"
  local previous_container="${CONTAINER_NAME}"
  local container="network-diag-${safe_job}-${safe_phase}-$$"
  container="$(printf '%s' "${container}" | tr -c 'A-Za-z0-9_.-' '-' | cut -c1-120)"

  probe temp_container_start "start temporary container for download probes: ${image}" \
    docker run -d --name "${container}" --entrypoint sleep "${image}" 1800

  if docker inspect "${container}" >/dev/null 2>&1; then
    CONTAINER_NAME="${container}"
    run_download_probes
    CONTAINER_NAME="${previous_container}"
    probe temp_container_cleanup "remove temporary container: ${container}" \
      docker rm -f "${container}"
  else
    echo "Temporary container did not start; falling back to host download probes."
    CONTAINER_NAME="${previous_container}"
    run_download_probes
  fi
}

run_high_concurrency_probes() {
  local download_probe_image=""

  run_host_context_probes
  run_registry_probes
  run_named_docker_pull_probe docker_local "${local_image}" "local registry"
  run_named_docker_pull_probe docker_public "${public_image}" "public Docker Hub"

  if docker image inspect "${public_image}" >/dev/null 2>&1; then
    download_probe_image="${public_image}"
  elif docker image inspect "${local_image}" >/dev/null 2>&1; then
    download_probe_image="${local_image}"
  fi

  if [[ -n "${download_probe_image}" ]]; then
    run_download_probes_with_temp_container "${download_probe_image}"
  else
    echo "No pulled CI image available; running download probes on host."
    run_download_probes
  fi
}

write_summary() {
  echo "=== diagnostic summary ==="
  grep -E '(^docker_.*pull_|^docker_.*image_size|^github_clone_|^pypi_download_|^hf_range_|_elapsed_sec=|_rc=|speed_download|time_total|image_size_bytes)' "${out_file}" || true

  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "### AMD network diagnostics (${MODE}, ${SOURCE}, ${ROCM_VERSION})"
      echo ""
      echo "- phase: \`${PHASE}\`"
      echo "- image: \`${pull_image}\`"
      echo "- log file: \`${out_file}\`"
      echo ""
      echo '```text'
      grep -E '(^docker_.*pull_|^docker_.*image_size|^github_clone_|^pypi_download_|^hf_range_|_elapsed_sec=|_rc=|speed_download|time_total|image_size_bytes)' "${out_file}" || true
      echo '```'
    } >> "${GITHUB_STEP_SUMMARY}"
  fi
}

write_header

case "${MODE}" in
  prepull)
    run_host_context_probes
    run_registry_probes
    ;;
  downloads)
    curl_probe pypi_simple https://pypi.org/simple/pip/ 60
    curl_probe huggingface_home https://huggingface.co/ 60
    curl_probe github_home https://github.com/ 60
    curl_probe github_raw https://raw.githubusercontent.com/sgl-project/sglang/main/README.md 60
    run_download_probes
    ;;
  full)
    run_host_context_probes
    run_registry_probes
    run_docker_pull_probe
    run_download_probes
    ;;
  high-concurrency)
    run_high_concurrency_probes
    ;;
esac

write_summary
echo "Diagnostics complete. Probe failures are recorded above but do not fail this evidence job."
exit 0
