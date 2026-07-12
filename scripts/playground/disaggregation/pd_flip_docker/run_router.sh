#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"

router_admin_key="${PD_FLIP_ROUTER_ADMIN_API_KEY:-${ADMIN_API_KEY:-}}"
case "${ADMIN_API_KEY:-}" in
  ""|replace-with-*|changeme|CHANGE_ME)
    echo "ADMIN_API_KEY must be set to a non-placeholder secret" >&2
    exit 2
    ;;
esac
case "${router_admin_key}" in
  ""|replace-with-*|changeme|CHANGE_ME)
    echo "PD_FLIP_ROUTER_ADMIN_API_KEY must be set to a non-placeholder secret" >&2
    exit 2
    ;;
esac
if [[ "${router_admin_key}" != "${ADMIN_API_KEY}" ]]; then
  echo "PD_FLIP_ROUTER_ADMIN_API_KEY must match ADMIN_API_KEY because the controller uses one credential" >&2
  exit 2
fi
export PD_FLIP_ROUTER_ADMIN_API_KEY="${router_admin_key}"

mounts=(-v "${SGLANG_REPO}:/sgl-workspace/sglang")
if [[ -d "${MODEL_PATH}" ]]; then
  mounts+=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
fi

args=(
  --host 0.0.0.0
  --port "${ROUTER_PORT}"
  --model-id "${MODEL_ID}"
)
router_tokenizer_path="${TOKENIZER_PATH:-}"
if [[ -d "${router_tokenizer_path}" && -f "${router_tokenizer_path}/tokenizer.json" ]]; then
  router_tokenizer_path="${router_tokenizer_path}/tokenizer.json"
elif [[ -z "${router_tokenizer_path}" && -f "${MODEL_PATH}/tokenizer.json" ]]; then
  router_tokenizer_path="${MODEL_PATH}/tokenizer.json"
fi
if [[ -n "${router_tokenizer_path}" ]]; then
  args+=(--tokenizer-path "${router_tokenizer_path}")
fi
args+=(--worker-urls "${NODE0}" "${NODE1}" "${NODE2}" "${NODE3}")

if [[ -n "${EXTRA_ROUTER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_router_args=(${EXTRA_ROUTER_ARGS})
  args+=("${extra_router_args[@]}")
fi

# shellcheck disable=SC2206
extra_docker_args=(${EXTRA_DOCKER_ARGS:-})

exec docker run --rm \
  -i \
  --network host \
  -e PD_FLIP_ROUTER_ADMIN_API_KEY \
  "${extra_docker_args[@]}" \
  "${mounts[@]}" \
  "${IMAGE}" \
  bash -s -- "${args[@]}" <<'INNER'
set -euo pipefail

cd /sgl-workspace/sglang/experimental/sgl-router

if [[ -x target/release/sgl-router ]]; then
  exec target/release/sgl-router "$@"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo 'sgl-router binary not found and cargo is unavailable. Build experimental/sgl-router/target/release/sgl-router first or use an image with cargo.' >&2
  exit 1
fi

export RUSTUP_TOOLCHAIN="${ROUTER_RUST_TOOLCHAIN:-stable}"
export CARGO_HOME="${ROUTER_CARGO_HOME:-/sgl-workspace/sglang/.pd_flip_cargo}"
export CARGO_NET_GIT_FETCH_WITH_CLI="${CARGO_NET_GIT_FETCH_WITH_CLI:-true}"
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-10}"
export CARGO_HTTP_TIMEOUT="${CARGO_HTTP_TIMEOUT:-600}"
export CARGO_HTTP_LOW_SPEED_LIMIT="${CARGO_HTTP_LOW_SPEED_LIMIT:-0}"
export CARGO_HTTP_MULTIPLEXING="${CARGO_HTTP_MULTIPLEXING:-false}"
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL="${CARGO_REGISTRIES_CRATES_IO_PROTOCOL:-sparse}"

git config --global http.version "${ROUTER_GIT_HTTP_VERSION:-HTTP/1.1}"
git config --global http.lowSpeedLimit "${ROUTER_GIT_LOW_SPEED_LIMIT:-0}"
git config --global http.postBuffer "${ROUTER_GIT_POST_BUFFER:-524288000}"
git config --global http.maxRequests "${ROUTER_GIT_MAX_REQUESTS:-1}"

crates_mirror="${ROUTER_CRATES_MIRROR_URL:-sparse+https://rsproxy.cn/index/}"
if [[ -n "${crates_mirror}" ]]; then
  mkdir -p "${CARGO_HOME}"
  cat > "${CARGO_HOME}/config.toml" <<CONFIG
[source.crates-io]
replace-with = "pd-flip-mirror"

[source.pd-flip-mirror]
registry = "${crates_mirror}"

[registries.pd-flip-mirror]
index = "${crates_mirror}"

[net]
git-fetch-with-cli = true
retry = ${CARGO_NET_RETRY}

[http]
timeout = ${CARGO_HTTP_TIMEOUT}
low-speed-limit = ${CARGO_HTTP_LOW_SPEED_LIMIT}
multiplexing = false
CONFIG
fi

if [[ "${ROUTER_DYNAMO_TARBALL_FALLBACK:-1}" == "1" ]]; then
  dynamo_rev="${ROUTER_DYNAMO_REV:-1efdd4dcb901caeae636131321094090d252c8d6}"
  dynamo_dir="${ROUTER_DYNAMO_DIR:-/sgl-workspace/sglang/.pd_flip_deps/dynamo-${dynamo_rev}}"
  dynamo_cargo_path="${ROUTER_DYNAMO_CARGO_PATH:-../../.pd_flip_deps/dynamo-${dynamo_rev}}"
  dynamo_url="${ROUTER_DYNAMO_TARBALL_URL:-https://codeload.github.com/ai-dynamo/dynamo/tar.gz/${dynamo_rev}}"
  if [[ ! -f "${dynamo_dir}/Cargo.toml" ]]; then
    tmp_tar="/tmp/dynamo-${dynamo_rev}.tar.gz"
    tmp_dir="${dynamo_dir}.tmp"
    rm -rf "${tmp_dir}" "${tmp_tar}"
    mkdir -p "$(dirname "${dynamo_dir}")" "${tmp_dir}"
    echo "router dynamo prefetch: ${dynamo_url}"
    curl --fail --location --retry 10 --retry-delay 5 --connect-timeout 30 \
      --output "${tmp_tar}" "${dynamo_url}"
    tar -xzf "${tmp_tar}" -C "${tmp_dir}" --strip-components=1
    rm -rf "${dynamo_dir}"
    mv "${tmp_dir}" "${dynamo_dir}"
  fi
  tmp_cargo_toml="$(mktemp)"
  awk '
    /^# pd-flip local dynamo patch$/ { skip = 1; next }
    skip && /^\[patch\."https:\/\/github.com\/ai-dynamo\/dynamo"\]$/ { next }
    skip && /^dynamo-(protocols|tokenizers|parsers) = / { next }
    skip && /^$/ { skip = 0; next }
    { print }
  ' Cargo.toml > "${tmp_cargo_toml}"
  mv "${tmp_cargo_toml}" Cargo.toml
  sed -i \
    -e "s#^dynamo-protocols = .*#dynamo-protocols = { path = \"${dynamo_cargo_path}/lib/protocols\" }#" \
    -e "s#^dynamo-tokenizers = .*#dynamo-tokenizers = { path = \"${dynamo_cargo_path}/lib/tokenizers\" }#" \
    -e "s#^dynamo-parsers = .*#dynamo-parsers = { path = \"${dynamo_cargo_path}/lib/parsers\" }#" \
    Cargo.toml
fi

cargo_offline="${ROUTER_CARGO_OFFLINE:-auto}"
if [[ "${cargo_offline}" == "auto" ]]; then
  if [[ -f Cargo.lock && -d "${CARGO_HOME}/registry/cache" ]]; then
    cargo_offline=1
  else
    cargo_offline=0
  fi
fi

cargo_fetch_args=(fetch -vv --locked)
cargo_run_args=(run --release --locked)
if [[ "${cargo_offline}" == "1" || "${cargo_offline}" == "true" ]]; then
  export CARGO_NET_OFFLINE=true
  cargo_fetch_args+=(--offline)
  cargo_run_args+=(--offline)
fi

echo "router cargo fetch: CARGO_HOME=${CARGO_HOME} git_http=$(git config --global http.version) crates_mirror=${crates_mirror:-default} offline=${cargo_offline}"
cargo "${cargo_fetch_args[@]}"
exec cargo "${cargo_run_args[@]}" -- "$@"
INNER
