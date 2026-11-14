#!/bin/bash
# Finish putting sgl-router at sgl-router-bin/sgl-router in the checkout: build it
# when no download left a working one there, then export SGLANG_TEST_PD_ROUTER_BINARY.
set -euo pipefail

workspace="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
binary="${workspace}/sgl-router-bin/sgl-router"

# An artifact download drops the executable bit, and a failed one can leave a
# truncated file.
chmod +x "${binary}" 2>/dev/null || true
if ! "${binary}" --help >/dev/null 2>&1; then
    rm -f "${binary}"
    bash "${workspace}/scripts/ci/utils/install_rustup.sh"
    bash "${workspace}/scripts/ci/utils/build_sgl_router.sh" "${workspace}/sgl-router-bin"
    "${binary}" --help >/dev/null
fi
echo "SGLANG_TEST_PD_ROUTER_BINARY=${binary}" >> "${GITHUB_ENV}"
