#!/usr/bin/env bash
# Install a GitHub Actions self-hosted runner on an AMD MI35x Conductor node.
#
# Run as the user account that holds the Conductor reservation (no sudo needed).
# The runner is installed under ~/actions-runner and started via the
# user-mode `./run.sh` (or nohup) — NOT installed as a systemd service,
# because we don't have root on Conductor nodes.
#
# REQUIRED env vars:
#   GH_REPO   - e.g. "sgl-project/sglang" (or your fork)
#   GH_TOKEN  - short-lived registration token from
#               https://github.com/<repo>/settings/actions/runners/new
#               (Settings → Actions → Runners → New self-hosted runner)
#   RUNNER_NAME    - e.g. "amd-mi300x-e06u43"
#   RUNNER_LABELS  - comma-separated, e.g. "self-hosted,linux,x64,mi300x,pd-cluster"
#
# Optional:
#   RUNNER_VERSION - default 2.321.0
#   RUNNER_DIR     - default $HOME/actions-runner

set -euo pipefail

: "${GH_REPO:?set GH_REPO=owner/repo}"
: "${GH_TOKEN:?set GH_TOKEN=<registration token from GitHub UI>}"
: "${RUNNER_NAME:?set RUNNER_NAME}"
: "${RUNNER_LABELS:?set RUNNER_LABELS}"

RUNNER_VERSION="${RUNNER_VERSION:-2.321.0}"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  RUNNER_ARCH=x64 ;;
    aarch64) RUNNER_ARCH=arm64 ;;
    *) echo "unsupported arch: $ARCH"; exit 1 ;;
esac

TARBALL="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"

if [ -d "$RUNNER_DIR" ] && [ -f "$RUNNER_DIR/.runner" ]; then
    echo "Runner already configured at $RUNNER_DIR — aborting (remove first if you want to re-register)"
    exit 1
fi

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [ ! -f "$TARBALL" ]; then
    echo "Downloading $URL"
    curl -fsSLO "$URL"
fi
tar xzf "$TARBALL"

./config.sh \
    --url "https://github.com/${GH_REPO}" \
    --token "$GH_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work _work \
    --unattended \
    --replace

echo
echo "Runner configured. Start it with one of:"
echo "  cd $RUNNER_DIR && ./run.sh                      # foreground"
echo "  cd $RUNNER_DIR && nohup ./run.sh > runner.log 2>&1 &   # background, dies on logout"
echo "  cd $RUNNER_DIR && tmux new -d -s gh-runner ./run.sh    # survives logout"
echo
echo "If you have sudo, prefer: sudo ./svc.sh install \$USER && sudo ./svc.sh start"
