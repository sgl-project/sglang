#!/bin/bash
# Entrypoint for the DeepSWE eval container (amd64 pod).
#
# Runs next to a `docker:dind` sidecar: install the docker CLI + compose plugin
# (pier shells out to both), wait for the daemon, then drive evalscope via
# run_deepswe_eval.py. Config comes from env vars injected by
# test_npu_accuracy_utils.run_evalscope_on_x86_pod.

set -euo pipefail

export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

echo "===== DeepSWE eval container ====="
echo "arch=$(uname -m) python=$(python3 -V 2>&1) DOCKER_HOST=${DOCKER_HOST}"
echo "api_url=${DEEPSWE_API_URL:-} datasets=${DEEPSWE_DATASETS:-} limit=${DEEPSWE_LIMIT:-} timeout=${DEEPSWE_TIMEOUT:-} metrics_dir=${METRICS_DATA_FILE:-}"
echo "sglang commit: $(git -C /root/sglang rev-parse --short HEAD 2>/dev/null || echo unknown)"

download_file() {
    # download_file <url> <dest>; non-zero on failure.
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --connect-timeout 15 -o "${dest}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${dest}" "${url}"
    else
        echo "Neither curl nor wget is available to download ${url}" >&2
        return 1
    fi
}

install_compose_v2() {
    # Fallback when the distro has no docker-compose-v2 package: install the
    # compose v2 CLI plugin. pier shells out to `docker compose`, so the v2
    # plugin (not the legacy docker-compose v1 binary) is what is required.
    if docker compose version >/dev/null 2>&1; then
        return 0
    fi
    local compose_arch url tmp
    case "$(uname -m)" in
        x86_64|amd64) compose_arch=x86_64 ;;
        aarch64|arm64) compose_arch=aarch64 ;;
        *) echo "ERROR: unsupported arch $(uname -m) for compose v2" >&2; return 1 ;;
    esac
    tmp="/tmp/docker-compose-plugin"
    local urls=()
    if [ -n "${DEEPSWE_COMPOSE_URL:-}" ]; then
        urls+=("${DEEPSWE_COMPOSE_URL}")
    fi
    urls+=("https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${compose_arch}")
    for url in "${urls[@]}"; do
        echo "Installing docker compose v2 plugin from ${url}"
        if download_file "${url}" "${tmp}"; then
            chmod +x "${tmp}"
            for plugin_dir in /usr/local/lib/docker/cli-plugins "${HOME:-/root}/.docker/cli-plugins"; do
                mkdir -p "${plugin_dir}"
                cp "${tmp}" "${plugin_dir}/docker-compose"
                chmod +x "${plugin_dir}/docker-compose"
            done
            rm -f "${tmp}"
            if docker compose version >/dev/null 2>&1; then
                echo "docker compose v2 ready: $(docker compose version)"
                return 0
            fi
        fi
        echo "WARNING: could not install compose v2 from ${url}" >&2
        rm -f "${tmp}"
    done
    return 1
}

negotiate_docker_api() {
    # Older dind daemons reject the newest docker CLI ("client version X is too
    # new"). Read the daemon's advertised API version from /_ping and pin the
    # client to it. Harmless when client and daemon already agree.
    local api=""
    if command -v curl >/dev/null 2>&1; then
        api="$(curl -fsS --max-time 3 -D - -o /dev/null "http://127.0.0.1:2375/_ping" 2>/dev/null \
            | tr -d '\r' | awk 'tolower($1) == "api-version:" {print $2}' | tail -n1)"
    fi
    if [ -n "${api}" ]; then
        if [ "${DOCKER_API_VERSION:-}" != "${api}" ]; then
            echo "dind advertises API ${api}; pinning DOCKER_API_VERSION=${api}"
            export DOCKER_API_VERSION="${api}"
        fi
        return 0
    fi
    return 1
}

if ! command -v docker >/dev/null 2>&1; then
    echo "Installing docker CLI and compose plugin ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    # docker-compose-v2 provides the `docker compose` subcommand pier calls;
    # fall back to the plain CLI when the package is unavailable so the eval
    # still starts and reports the real error instead of failing at install.
    apt-get install -y --no-install-recommends \
        docker.io docker-compose-v2 ca-certificates curl git \
        || apt-get install -y --no-install-recommends \
            docker.io ca-certificates curl git
fi

echo "Waiting for the Docker daemon at ${DOCKER_HOST} ..."
docker_ready=0
for _ in $(seq 1 150); do
    # Pin the client to the daemon's advertised API version; otherwise an old
    # dind (e.g. API 1.38) rejects the newer docker CLI and 'docker version'
    # keeps failing even though the daemon is up.
    if [ -z "${DOCKER_API_VERSION:-}" ]; then
        negotiate_docker_api || true
    fi
    if docker version >/dev/null 2>&1; then
        docker_ready=1
        break
    fi
    sleep 2
done
if [ "${docker_ready}" != "1" ]; then
    echo "ERROR: Docker daemon is not reachable at ${DOCKER_HOST}"
    docker version || true
    exit 1
fi
docker version
if ! docker compose version >/dev/null 2>&1; then
    install_compose_v2 || echo "WARNING: docker compose v2 unavailable; pier will likely fail"
fi
docker compose version || echo "WARNING: docker compose plugin is still unavailable"

# The official ubuntu:24.04 image lacks the Ubuntu archive signing key in this
# environment, so pier's egress-proxy build fails at `apt-get update` with
# "NO_PUBKEY 871920D1991BC93C". This eval image (Ubuntu 22.04) ships a keyring
# that has the key, so build a patched local image and retag it as
# ubuntu:24.04; compose's `FROM ubuntu:24.04` then uses it instead of pulling.
seed_ubuntu_base() {
    local keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg
    if [ ! -f "${keyring}" ]; then
        echo "WARNING: ${keyring} not found; skip patching ubuntu:24.04"
        return 0
    fi
    local work
    work="$(mktemp -d)"
    cp "${keyring}" "${work}/ubuntu-archive-keyring.gpg"
    cat > "${work}/Dockerfile" <<'EOF'
FROM ubuntu:24.04
COPY ubuntu-archive-keyring.gpg /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    echo "Patching ubuntu:24.04 with the archive signing key ..."
    if docker build -t ubuntu:24.04 "${work}"; then
        echo "Patched ubuntu:24.04 base ready."
    else
        echo "WARNING: failed to patch ubuntu:24.04; apt in the build may fail"
    fi
    rm -rf "${work}"
}
seed_ubuntu_base

python3 /root/sglang/python/sglang/test/ascend/e2e/run_deepswe_eval.py
