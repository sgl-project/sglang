#!/bin/bash
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
PIP_INSTALL="python3 -m pip install --no-cache-dir"

# SAIL SDK v2.1.1 PyPI source (replaces the old vendor-hosted index)
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

git config --global --add safe.directory "${REPO_ROOT}"

# Only pip/wheel. Deliberately NOT setuptools or dill: the v2.1.1 image pins them
# to satisfy constraints that an --upgrade breaks (observed in run 31663500015):
#   setuptools 84.0.0 -> torch 2.11.0 requires setuptools<82
#   dill 0.4.1        -> datasets 3.1.0 requires dill<0.3.9 (image ships 0.3.6)
${PIP_INSTALL} --upgrade pip wheel

# ==================== PPU Dependencies (SAIL SDK v2.1.1) ==================== #
# The v2.1.1 base image already ships the whole PPU stack, and at versions NEWER
# than the v0.5.13 user guide's PyPI section lists. Verified in-image 2026-08-13:
#
#   torch              2.11.0+v0.1.0.ppu2.1.1
#   sglang             0.5.13+v0.1.0.ppu2.1.1
#   sglang-kernel      0.4.3+v0.1.0.ppu2.1.1     <- has the `fwd` op
#   flashinfer-python  0.6.12+v0.1.0.ppu2.1.1
#   triton             3.6.0+v0.2.0.ppu2.1.1
#   apache-tvm-ffi     0.1.9
#   transformers       5.8.1
#
# So we deliberately do NOT reinstall them. Two reasons this matters:
#
#  1. The guide's PyPI list pins flashinfer_python==0.6.8.post1, which would
#     DOWNGRADE the image's 0.6.12 — and contradicts the guide's own stated
#     requirement of flashinfer>=0.6.11.post1 (section 2). The image is right.
#  2. Those wheels carry a +v0.1.0.ppu2.1.1 local version. A bare `==0.4.3`
#     --force-reinstall could pull a differently-built artifact for the same
#     public version. Leaving the image's stack alone avoids the whole question.
#
# Only install what the image genuinely lacks. Verified in-image 2026-08-13 that
# these are already present and need no action:
#   z3-solver     4.13.0.0                  (note: `==4.13.0` would force a reinstall)
#   sglang-router 0.3.2+v0.1.0.ppu2.1.1     (PPU build; the guide's bare ==0.3.2
#                                            could replace it with a generic one)
#
# uvicorn is the one real gap: image ships 0.29.0, sglang's multi-worker path needs
# timeout_worker_healthcheck (added in 0.37.0).
${PIP_INSTALL} /nas_aisw/datasets/packages/uvicorn-0.37.0-py3-none-any.whl --force-reinstall --no-deps

# dill: image ships 0.3.6, but Python 3.12's ABC implementation makes
# _abc._abc_data unpicklable with dill<0.3.8, crashing custom_logit_processor
# tests. Upgrade to 0.3.8 (still satisfies datasets' dill<0.3.9 constraint).
${PIP_INSTALL} "dill>=0.3.8,<0.3.9"

# ==================== Install SGLang from source ==================== #
# The version this install reports for itself, settled before pyproject.toml is
# replaced. pyproject_other.toml declares `dynamic = ["version"]` -- upstream's
# arrangement, which expects setuptools-scm to supply the number -- while this
# fork's own pyproject.toml pins it statically. Swapping the files therefore
# hands the question to a setuptools-scm that --no-build-isolation never
# installs, and setuptools falls back to its own default: every run before this
# recorded `sglang 0.0.0` in provenance, having first uninstalled the image's
# perfectly good 0.5.13+v0.1.0.ppu2.1.1 (observed, run 34548547824). The static
# version read here is 0.5.18, so the composed number becomes 0.5.18+g<sha>.
#
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_<DIST> is setuptools-scm's own answer for
# builds whose git metadata cannot answer the question -- its "Usage from
# Docker" section prescribes exactly this for `pip install -e`, and prefers the
# per-distribution form over the bare one. It also short-circuits the SCM
# lookup, so no tag has to be reachable: these checkouts are --depth=2 without
# tags, and git_describe_command would exit 1 here.
#
# --no-deps on setuptools-scm is deliberate. Its requirements (setuptools>=61,
# packaging>=20) are already satisfied by the image, and the header above
# records what moving setuptools costs: torch 2.11.0 requires setuptools<82.
# --no-deps makes it impossible for this line to move the pinned stack.
SGLANG_STATIC_VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' "${REPO_ROOT}/python/pyproject.toml" | head -1)
SGLANG_SHORT_SHA=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "")
if [[ -n "${SGLANG_STATIC_VERSION}" && -n "${SGLANG_SHORT_SHA}" ]]; then
    ${PIP_INSTALL} --no-deps "setuptools-scm>=8,<10"
    # The local segment names the commit rather than a distance from a tag: the
    # commit is what provenance already treats as authoritative, and unlike a
    # tag distance it is derivable on a shallow checkout.
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_STATIC_VERSION}+g${SGLANG_SHORT_SHA}"
    echo "sglang version for this install: ${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG}"
else
    # Not fatal. A missing version leaves the install as it was before this
    # block existed, and the report collector records the field as absent
    # rather than as 0.0.0.
    echo "::warning::could not compose an sglang version (static='${SGLANG_STATIC_VERSION}' sha='${SGLANG_SHORT_SHA}'); the install will report a placeholder"
fi

rm -f "${REPO_ROOT}/python/pyproject.toml"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"

# Extras must exist, or pip merely warns `does not provide the extra '<name>'`
# and installs nothing from it. v0.5.13 asked for `all_ppu`, which its
# pyproject_other.toml defined; v0.5.18 moved the PPU deps to pyproject_ppu.toml
# and dropped the extra, so the request silently became a no-op and every runtime
# dependency stopped being installed. That left the image's v0.5.13-era wheels in
# place -- including an xgrammar predating AnyTokensFormat, which
# srt/function_call/kimik3_structural_tag.py imports at module scope by way of
# server_args.py, so every sglang import died (run 34859223040).
#
# runtime_common is the portion v0.5.13 actually relied on pip to install
# (xgrammar==0.2.1, llguidance>=1.7.6, transformers==5.12.1, mistral_common,
# outlines, timm, compressed-tensors). The platform wheels that `all_ppu` also
# named -- torch, deep_ep, deep_gemm, flash-attn-3 -- are deliberately left out:
# the image's +v0.1.0.ppu2.1.1 builds are the ones we want, per the note above.
for _extra in runtime_common tracing; do
    if ! grep -qE "^${_extra} *= *\[" "${REPO_ROOT}/python/pyproject.toml"; then
        echo "::error::pyproject.toml declares no '${_extra}' extra; the dependency install would silently no-op"
        exit 1
    fi
done

# tracing: the v2.1.1 image dropped opentelemetry (v2.1.0 shipped it), and
# runtime_common doesn't pull it in. test_tracing needs it to exercise the OTLP path.
cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[runtime_common,tracing]" --no-build-isolation

# ==================== sgl-kernel: build from source and override ==================== #
# On PPU we build the repo's own kernel sources (python/sglang/kernels/aot,
# dist name sglang-kernel 0.4.6.post1) and force-reinstall over the image's
# 0.4.3+v0.1.0.ppu2.1.1, so the kernel under test matches the tested commit
# (tested == code). See scripts/ci/ppu/ppu_build_kernel.sh for the build/override
# details and the measured 0.4.6-vs-0.4.3 op-surface caveat (the source tree
# registers 5 fewer sgl_kernel:: ops, all deliberately retired in 0.4.6).
#
# Note on DSA (FlashMLA): ppu_flashmla_hooks.py REPLACEs sgl_kernel.flash_mla.*
# at runtime with a thin wrapper over the standalone flash_mla package (from the
# image), so this override does not change DSA behavior either way.
bash "${REPO_ROOT}/scripts/ci/ppu/ppu_build_kernel.sh"

# ==================== EIC SDK + mooncake-barex (for disaggregation tests) ==================== #
EIC_PKG_DIR=/nas_aisw/datasets/packages/eic-sdk
if [[ -d "${EIC_PKG_DIR}" ]]; then
    echo "Installing EIC SDK components for PPU disaggregation..."
    dpkg -i --force-overwrite "${EIC_PKG_DIR}"/ali-rdma-core_*.deb || echo "WARNING: ali-rdma-core install failed"
    dpkg -i "${EIC_PKG_DIR}"/accl-barex-cuda13-*.deb || echo "WARNING: accl-barex install failed"
    dpkg -i "${EIC_PKG_DIR}"/u2mm.deb || echo "WARNING: u2mm install failed"
    dpkg -i "${EIC_PKG_DIR}"/eic-tools_*.deb 2>/dev/null || echo "WARNING: eic-tools install failed (non-critical)"
    if [[ -f "${EIC_PKG_DIR}/libunicm.so" ]]; then
        cp "${EIC_PKG_DIR}/libunicm.so" /usr/lib64/
    fi
    if [[ -d /usr/lib64 ]] && ! grep -q '/usr/lib64' /etc/ld.so.conf /etc/ld.so.conf.d/*.conf 2>/dev/null; then
        echo "/usr/lib64" > /etc/ld.so.conf.d/usr-lib64.conf
    fi
    ldconfig
    ${PIP_INSTALL} --no-deps --force-reinstall \
        "${EIC_PKG_DIR}"/mooncake_transfer_engine-0.3.6.post1-*.whl
    echo "EIC SDK installation complete."
else
    echo "WARNING: EIC SDK not found at ${EIC_PKG_DIR}, disaggregation tests may fail."
fi
