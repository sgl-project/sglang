#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ROUTER_DIR="${REPO_ROOT}/experimental/sgl-router"

DYNAMO_REV="${ROUTER_DYNAMO_REV:-1efdd4dcb901caeae636131321094090d252c8d6}"
DYNAMO_DIR="${REPO_ROOT}/.pd_flip_deps/dynamo-${DYNAMO_REV}"
DYNAMO_CARGO_PATH="${ROUTER_DYNAMO_CARGO_PATH:-../../.pd_flip_deps/dynamo-${DYNAMO_REV}}"
DYNAMO_URL="${ROUTER_DYNAMO_TARBALL_URL:-https://codeload.github.com/ai-dynamo/dynamo/tar.gz/${DYNAMO_REV}}"
CARGO_HOME="${ROUTER_CARGO_HOME:-${REPO_ROOT}/.pd_flip_cargo}"
CRATES_MIRROR="${ROUTER_CRATES_MIRROR_URL:-sparse+https://rsproxy.cn/index/}"
ARCHIVE="${ROUTER_DEPS_ARCHIVE:-${REPO_ROOT}/router_deps.tgz}"

mkdir -p "${REPO_ROOT}/.pd_flip_deps" "${CARGO_HOME}"

if [[ ! -f "${DYNAMO_DIR}/Cargo.toml" ]]; then
  tmp_tar="/tmp/dynamo-${DYNAMO_REV}.tar.gz"
  tmp_dir="${DYNAMO_DIR}.tmp"
  rm -rf "${tmp_dir}" "${tmp_tar}"
  mkdir -p "${tmp_dir}"
  echo "[router-deps] downloading ${DYNAMO_URL}"
  curl --fail --location --retry 10 --retry-delay 5 --connect-timeout 30 \
    --output "${tmp_tar}" "${DYNAMO_URL}"
  tar -xzf "${tmp_tar}" -C "${tmp_dir}" --strip-components=1
  rm -rf "${DYNAMO_DIR}"
  mv "${tmp_dir}" "${DYNAMO_DIR}"
fi

cat > "${CARGO_HOME}/config.toml" <<CONFIG
[source.crates-io]
replace-with = "pd-flip-mirror"

[source.pd-flip-mirror]
registry = "${CRATES_MIRROR}"

[registries.pd-flip-mirror]
index = "${CRATES_MIRROR}"

[net]
git-fetch-with-cli = true
retry = 10

[http]
timeout = 600
low-speed-limit = 0
multiplexing = false
CONFIG

cd "${ROUTER_DIR}"
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
  -e "s#^dynamo-protocols = .*#dynamo-protocols = { path = \"${DYNAMO_CARGO_PATH}/lib/protocols\" }#" \
  -e "s#^dynamo-tokenizers = .*#dynamo-tokenizers = { path = \"${DYNAMO_CARGO_PATH}/lib/tokenizers\" }#" \
  -e "s#^dynamo-parsers = .*#dynamo-parsers = { path = \"${DYNAMO_CARGO_PATH}/lib/parsers\" }#" \
  Cargo.toml

echo "[router-deps] cargo fetch with CARGO_HOME=${CARGO_HOME}"
CARGO_HOME="${CARGO_HOME}" \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
CARGO_NET_RETRY=10 \
CARGO_HTTP_TIMEOUT=600 \
CARGO_HTTP_LOW_SPEED_LIMIT=0 \
CARGO_HTTP_MULTIPLEXING=false \
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
cargo fetch -vv

cd "${REPO_ROOT}"
echo "[router-deps] packing ${ARCHIVE}"
tar -czf "${ARCHIVE}" \
  .pd_flip_cargo \
  .pd_flip_deps \
  experimental/sgl-router/Cargo.toml \
  experimental/sgl-router/Cargo.lock
echo "[router-deps] wrote ${ARCHIVE}"
