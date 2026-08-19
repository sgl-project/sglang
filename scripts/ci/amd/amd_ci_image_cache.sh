#!/bin/bash
# Cache the CI container image as a tarball on the persistent runner volume.
#
# The AMD runners are docker-in-docker with ephemeral storage, so an image
# pulled by one job is gone before the next job starts and every job re-pulls
# ~23GB from Docker Hub. The volume mounted at /home/runner/sglang-data is the
# only thing that survives between jobs. So the first job to miss pays a pull
# and seeds a tarball there; later jobs `docker load` it instead.
#
# Every operation here is best-effort. A cache that is full, slow, locked or
# corrupt must never fail a test job -- the caller falls back to pulling.
#
# Whether to cache at all is the caller's decision: pass a directory to
# image_cache_init to enable, or the empty string to disable.
#
# Environment:
#   AMD_CI_IMAGE_CACHE_MIN_FREE_GB   don't seed below this much free space (default 120)
#   AMD_CI_IMAGE_CACHE_MAX_AGE_DAYS  prune tarballs older than this (default 1)

IMAGE_CACHE_DIR=""
IMAGE_CACHE_EXT=""
IMAGE_CACHE_MIN_FREE_GB="${AMD_CI_IMAGE_CACHE_MIN_FREE_GB:-120}"
IMAGE_CACHE_MAX_AGE_DAYS="${AMD_CI_IMAGE_CACHE_MAX_AGE_DAYS:-1}"

# image_cache_init <cache_host_dir>
# Leaves IMAGE_CACHE_DIR empty when caching is unavailable or switched off.
image_cache_init() {
  local cache_host="${1:-}"
  if [[ -z "${cache_host}" ]]; then
    return 0
  fi
  if [[ ! -d "${cache_host}" ]]; then
    echo "Image tarball cache unavailable: no persistent volume at '${cache_host}'" >&2
    return 0
  fi
  # zstd keeps a tarball near the image's compressed size instead of its
  # on-disk size, which is what makes this affordable next to the weight cache.
  if command -v zstd >/dev/null 2>&1; then
    IMAGE_CACHE_EXT=".zst"
  else
    IMAGE_CACHE_EXT=""
    echo "Note: zstd not found; image tarballs will be stored uncompressed." >&2
  fi
  if ! mkdir -p "${cache_host}/docker-images" 2>/dev/null; then
    echo "Image tarball cache unavailable: cannot create ${cache_host}/docker-images" >&2
    return 0
  fi
  IMAGE_CACHE_DIR="${cache_host}/docker-images"
  echo "Image tarball cache: ${IMAGE_CACHE_DIR} (suffix '.tar${IMAGE_CACHE_EXT}')"
}

# image_cache_path <image_ref>
image_cache_path() {
  local safe="${1//[^A-Za-z0-9._-]/_}"
  echo "${IMAGE_CACHE_DIR}/${safe}.tar${IMAGE_CACHE_EXT}"
}

# image_cache_load <image_ref> -> 0 when the image is now in the local store
image_cache_load() {
  local image="$1" path
  [[ -n "${IMAGE_CACHE_DIR}" ]] || return 1
  path=$(image_cache_path "${image}")
  # A concurrent seed writes to <path>.tmp.* and renames, so seeing <path> at
  # all means it is complete.
  [[ -f "${path}" ]] || return 1
  echo "Loading image from tarball cache: ${path}"
  if [[ -n "${IMAGE_CACHE_EXT}" ]]; then
    zstd -dc "${path}" 2>/dev/null | docker load >/dev/null 2>&1 || true
  else
    docker load -i "${path}" >/dev/null 2>&1 || true
  fi
  if [[ -n "$(docker images -q "${image}" 2>/dev/null)" ]]; then
    return 0
  fi
  echo "Warning: tarball cache did not yield ${image}; falling back to a pull" >&2
  return 1
}

# image_cache_prune -- bound the footprint; tags are date-stamped so anything
# older than the lookback window is dead weight.
image_cache_prune() {
  [[ -n "${IMAGE_CACHE_DIR}" ]] || return 0
  find "${IMAGE_CACHE_DIR}" -maxdepth 1 -name '*.tar*' \
    -mtime "+${IMAGE_CACHE_MAX_AGE_DAYS}" -print -delete 2>/dev/null \
    | sed 's/^/  pruned stale tarball: /' || true
  return 0
}

# image_cache_save <image_ref> -- always returns 0
image_cache_save() {
  local image="$1" path lock tmp avail
  [[ -n "${IMAGE_CACHE_DIR}" ]] || return 0
  path=$(image_cache_path "${image}")
  [[ -f "${path}" ]] && return 0

  image_cache_prune

  avail=$(df -BG --output=avail "${IMAGE_CACHE_DIR}" 2>/dev/null | tail -n 1 | tr -dc '0-9')
  if [[ -z "${avail}" ]]; then
    echo "Warning: cannot determine free space on ${IMAGE_CACHE_DIR}; not seeding" >&2
    return 0
  fi
  if (( avail < IMAGE_CACHE_MIN_FREE_GB )); then
    echo "Not seeding image tarball: ${avail}G free, need ${IMAGE_CACHE_MIN_FREE_GB}G" >&2
    return 0
  fi

  # One seeder at a time: a save streams tens of GB, and dozens of jobs start
  # together. mkdir is the atomic primitive that works on a shared volume.
  lock="${path}.lock"
  if ! mkdir "${lock}" 2>/dev/null; then
    echo "Another job is seeding ${path}; skipping"
    return 0
  fi
  # shellcheck disable=SC2064
  trap "rmdir '${lock}' 2>/dev/null || true" RETURN

  tmp="${path}.tmp.$$"
  echo "Seeding image tarball cache: ${path}"
  local ok=1
  if [[ -n "${IMAGE_CACHE_EXT}" ]]; then
    docker save "${image}" 2>/dev/null | zstd -T0 -3 -q > "${tmp}" 2>/dev/null || ok=0
  else
    docker save "${image}" > "${tmp}" 2>/dev/null || ok=0
  fi
  if (( ok )) && [[ -s "${tmp}" ]] && mv -f "${tmp}" "${path}" 2>/dev/null; then
    echo "Seeded $(du -h "${path}" 2>/dev/null | cut -f1) tarball for ${image}"
  else
    echo "Warning: failed to seed tarball for ${image}; continuing" >&2
    rm -f "${tmp}" 2>/dev/null || true
  fi
  return 0
}
