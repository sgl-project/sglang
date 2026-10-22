# AWS EFA variant of a published CUDA 13 SGLang image: adds the EFA userspace
# stack, the EFA Mooncake wheel (amd64 only), and EFA defaults for Mooncake,
# NIXL and NCCL. Published by .github/workflows/release-docker-efa.yml.
#
#   docker build -f docker/efa.Dockerfile \
#       --build-arg BASE_IMAGE=lmsysorg/sglang:latest-runtime -t sglang:efa docker/
#
# The host still provides the EFA kernel module and memlock limits.
ARG BASE_IMAGE=lmsysorg/sglang:latest-runtime
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_SHA256=fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841

# Install the AWS EFA userspace stack. The distro libfabric1 stays (libopenmpi
# depends on it); the ENV below makes the AWS copy win at runtime.
RUN set -eux; \
    EFA_INSTALLER_ARCHIVE="/tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
    curl -fsSL --retry 3 --retry-all-errors \
        "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
        -o "${EFA_INSTALLER_ARCHIVE}"; \
    echo "${EFA_INSTALLER_SHA256}  ${EFA_INSTALLER_ARCHIVE}" | sha256sum -c -; \
    tar -xzf "${EFA_INSTALLER_ARCHIVE}" -C /tmp; \
    # The installer apt-installs dependencies; the base image purged the lists.
    apt-get update -y; \
    # Not --minimal: it skips libfabric, which the EFA data path needs.
    (cd /tmp/aws-efa-installer && \
        ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify); \
    test -x /opt/amazon/efa/bin/fi_info; \
    /opt/amazon/efa/bin/fi_info --version; \
    # `-l` lists providers without EFA hardware, so this works in a build.
    /opt/amazon/efa/bin/fi_info -l | grep -q '^efa:'; \
    ldconfig -p | grep -m1 'libfabric\.so\.1' | grep -q '/opt/amazon/efa/lib/'; \
    # NIXL's plugin needs FABRIC_1.7, which the distro libfabric 1.17.0 lacks.
    NIXL_LIBFABRIC_PLUGINS="$(find "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
        -name libplugin_LIBFABRIC.so)"; \
    test -n "${NIXL_LIBFABRIC_PLUGINS}"; \
    for plugin in ${NIXL_LIBFABRIC_PLUGINS}; do \
        ldd "${plugin}" | grep -E '^[[:space:]]*libfabric\.so\.1 => /opt/amazon/efa/lib/'; \
    done; \
    # Plugins that NCCL_NET_PLUGIN=ofi and NCCL_TUNER_PLUGIN=ofi resolve to.
    ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-net-ofi\.so$'; \
    ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi\.so$'; \
    apt-get clean; \
    rm -rf "${EFA_INSTALLER_ARCHIVE}" /tmp/aws-efa-installer \
        /var/lib/apt/lists/*

# LD_LIBRARY_PATH is required: under `--gpus` the NVIDIA toolkit reruns
# ldconfig, which puts the distro libfabric first in the cache.
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/efa/bin:${PATH}" \
    NCCL_NET_PLUGIN=ofi \
    NCCL_TUNER_PLUGIN=ofi

# Swap in the EFA Mooncake wheel at the base image's Mooncake version. Both
# wheels own the `mooncake` module; it is published for x86_64 only.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    if [ "${TARGETARCH}" = "amd64" ]; then \
        pkg_version() { python3 -c 'import importlib.metadata as m, sys; print(m.version(sys.argv[1]))' "$1"; }; \
        MOONCAKE_VERSION="$(pkg_version mooncake-transfer-engine-cuda13)"; \
        test -n "${MOONCAKE_VERSION}"; \
        python3 -m pip uninstall -y mooncake-transfer-engine-cuda13; \
        python3 -m pip install --no-deps \
            "mooncake-transfer-engine-efa-cuda13==${MOONCAKE_VERSION}"; \
        test "$(pkg_version mooncake-transfer-engine-efa-cuda13)" = "${MOONCAKE_VERSION}"; \
        # `! cmd` is exempt from `set -e`, so check explicitly.
        if pkg_version mooncake-transfer-engine-cuda13 >/dev/null 2>&1; then \
            echo "mooncake-transfer-engine-cuda13 is still installed" >&2; \
            exit 1; \
        fi; \
    else \
        echo "TARGETARCH=${TARGETARCH}: mooncake-transfer-engine-efa-cuda13 is \
x86_64-only, keeping the standard wheel (Mooncake EFA protocol unavailable)"; \
    fi

# Only amd64 has the EFA Mooncake wheel; other arches use tcp. ENV cannot be
# conditional, so this is "tcp if TARGETARCH is not amd64, else efa".
ARG ARCH_IF_NOT_AMD64=${TARGETARCH#amd64}
ARG PROTOCOL_IF_NOT_AMD64=${ARCH_IF_NOT_AMD64:+tcp}
ENV MOONCAKE_PROTOCOL=${PROTOCOL_IF_NOT_AMD64:-efa} \
    SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC

# Replace the inherited image tag with this image's. The ARG has its own name
# because the inherited SGLANG_IMAGE_TAG ENV would shadow it.
ARG BASE_IMAGE
ARG EFA_IMAGE_TAG=
ENV SGLANG_IMAGE_TAG=${EFA_IMAGE_TAG:-${SGLANG_IMAGE_TAG}}
LABEL org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}" \
      ai.sglang.efa.base-image="${BASE_IMAGE}" \
      ai.sglang.efa.installer-version="${EFA_INSTALLER_VERSION}"
