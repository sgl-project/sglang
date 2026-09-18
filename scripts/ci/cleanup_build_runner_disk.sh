#!/usr/bin/env bash

set -euo pipefail

required_gb=0
deep_clean=false

usage() {
  cat <<'EOF'
Usage: cleanup_build_runner_disk.sh [--required-gb N] [--deep-clean]

Reclaim disposable Docker and SGLang build caches without deleting volumes.
With --deep-clean, also remove known SGLang and transient buildx builders.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --required-gb)
      required_gb="${2:?--required-gb requires a value}"
      shift 2
      ;;
    --deep-clean)
      deep_clean=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${required_gb}" =~ ^[0-9]+$ ]]; then
  echo "--required-gb must be a non-negative integer" >&2
  exit 2
fi

available_gb() {
  df --output=avail -BG / | tail -1 | tr -dc '0-9'
}

show_disk_state() {
  df -h /
  df -ih /
  docker system df || true
  docker buildx ls || true
}

remove_builder_if_present() {
  local builder="$1"
  if docker buildx inspect "${builder}" >/dev/null 2>&1; then
    echo "Removing disposable buildx builder: ${builder}"
    docker buildx rm "${builder}" || true
  fi
}

echo "Disk state before cleanup"
show_disk_state

if [[ "${deep_clean}" == "true" ]]; then
  remove_builder_if_present "sgl-kernel-builder"
  while IFS= read -r builder; do
    [[ -n "${builder}" ]] || continue
    case "${builder}" in
      builder-* | sgl-kernel-*)
        remove_builder_if_present "${builder}"
        ;;
    esac
  done < <(docker buildx ls --format '{{.Name}}' 2>/dev/null | sed 's/\*$//' | sort -u)
fi

# Do not prune volumes: they can contain state owned by another workload.
docker container prune -f
docker image prune -af
docker builder prune -af

cache_dir="$(realpath -m "${HOME:?HOME must be set}/.cache/sgl-kernel")"
case "${cache_dir}" in
  "${HOME}/.cache/sgl-kernel")
    if [[ -d "${cache_dir}" ]]; then
      echo "Removing disposable SGLang kernel cache: ${cache_dir}"
      sudo rm -rf -- "${cache_dir}"
    fi
    ;;
  *)
    echo "Refusing to remove unexpected cache path: ${cache_dir}" >&2
    exit 1
    ;;
esac

echo "Disk state after cleanup"
show_disk_state

current_available_gb="$(available_gb)"
if ((current_available_gb < required_gb)); then
  echo "::error::Insufficient disk space: ${current_available_gb}GB available; ${required_gb}GB required"
  docker system df -v || true
  exit 1
fi

echo "Disk preflight passed: ${current_available_gb}GB available; ${required_gb}GB required"
