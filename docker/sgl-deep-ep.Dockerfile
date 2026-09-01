ARG BASE_IMAGE=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=13.0

FROM ${BASE_IMAGE}:cuda${CUDA_VERSION}

ARG ARCHITECTURE=x86_64
ARG CUDA_TAG=cu130
ARG CUDA_VERSION=13.0
ARG GDRCOPY_VERSION=2.5.1
ARG NCCL_VERSION=2.30.7
ARG PYTHON_TAG=cp312-cp312
ARG TORCH_VERSION=2.13.0

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/opt/python/${PYTHON_TAG}/bin:${PATH}
ENV PYTHON_BIN=/opt/python/${PYTHON_TAG}/bin/python

# These mirror the build and RDMA dependencies used by ci_install_deepep.sh.
RUN yum install -y --nogpgcheck --enablerepo=powertools \
        cmake \
        curl \
        gcc \
        gcc-c++ \
        git \
        infiniband-diags \
        libfabric \
        libfabric-devel \
        libibverbs \
        libibverbs-devel \
        libibverbs-utils \
        librdmacm \
        librdmacm-devel \
        make \
        openssh-server \
        patchelf \
        perftest \
        pkgconfig \
        rdma-core \
        wget \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN set -eux; \
    if [ "${ARCHITECTURE}" = aarch64 ]; then cuda_target=sbsa; else cuda_target="${ARCHITECTURE}"; fi; \
    cuda_stub="/usr/local/cuda-${CUDA_VERSION}/targets/${cuda_target}-linux/lib/stubs/libcuda.so"; \
    test -f "${cuda_stub}"; \
    mkdir -p /usr/lib64 "/usr/lib/${ARCHITECTURE}-linux-gnu"; \
    ln -sf "${cuda_stub}" /usr/lib64/libcuda.so; \
    ln -sf "${cuda_stub}" "/usr/lib/${ARCHITECTURE}-linux-gnu/libcuda.so"

# DeepEP v2 uses NCCL Gin on CUDA 13. Keep GDRCopy only for the CUDA 12
# legacy NVSHMEM/IBGDA build.
RUN set -eux; \
    if [ "${CUDA_TAG}" = cu129 ]; then \
        git clone --depth 1 --branch "v${GDRCOPY_VERSION}" \
            https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy; \
        make -C /opt/gdrcopy CUDA="${CUDA_HOME}" prefix=/usr/local lib_install; \
        printf '%s\n' /usr/local/lib > /etc/ld.so.conf.d/gdrcopy.conf; \
        ldconfig; \
        test -f /usr/local/include/gdrapi.h; \
        ldconfig -p | grep -q libgdrapi; \
    fi

RUN --mount=type=cache,id=sgl-deep-ep-pip-${CUDA_TAG}-${PYTHON_TAG}-${ARCHITECTURE},target=/root/.cache/pip \
    set -eux; \
    "${PYTHON_BIN}" -m pip uninstall -y deep-ep sgl-deep-ep || true; \
    "${PYTHON_BIN}" -m pip install --upgrade pip; \
    "${PYTHON_BIN}" -m pip install --force-reinstall \
        "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"; \
    if [ "${CUDA_TAG}" = cu130 ]; then \
        "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps \
            "nvidia-nccl-cu13==${NCCL_VERSION}"; \
    fi; \
    "${PYTHON_BIN}" -m pip install \
        "auditwheel>=6.0" \
        build \
        ninja \
        packaging \
        setuptools \
        wheel; \
    TORCH_VERSION="${TORCH_VERSION}" "${PYTHON_BIN}" -c \
        'import os, torch; assert torch.__version__.startswith(os.environ["TORCH_VERSION"]); print(torch.__version__, torch.version.cuda)'
