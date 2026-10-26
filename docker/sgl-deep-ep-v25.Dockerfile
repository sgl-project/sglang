ARG BASE_IMAGE=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=13.4

FROM ${BASE_IMAGE}:cuda${CUDA_VERSION}

ARG ARCHITECTURE=x86_64
ARG CUDA_TAG=cu134
ARG CUDA_VERSION=13.4
ARG NCCL_VERSION=2.32.3
ARG PYTHON_TAG=cp312-cp312
ARG TORCH_CUDA_TAG=cu130
ARG TORCH_VERSION=2.14.1

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/opt/python/${PYTHON_TAG}/bin:${PATH}
ENV PYTHON_BIN=/opt/python/${PYTHON_TAG}/bin/python

# DeepEP 2.5 has no NVSHMEM or RDMA build dependency. DeepJIT needs C++20 with <format>
# (the builder images put gcc-toolset-13 first on PATH) and the elfutils headers; libdw
# itself is only dlopen'ed at runtime.
RUN yum install -y --nogpgcheck \
        curl \
        elfutils-devel \
        git \
        make \
        patchelf \
        pkgconfig \
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

# PyTorch pins an older nvidia-nccl-cu13, so install the NCCL that DeepEP 2.5 needs over it.
RUN --mount=type=cache,id=sgl-deep-ep-v25-pip-${CUDA_TAG}-${PYTHON_TAG}-${ARCHITECTURE},target=/root/.cache/pip \
    set -eux; \
    "${PYTHON_BIN}" -m pip uninstall -y deep-ep sgl-deep-ep sgl-deep-ep-v25 || true; \
    "${PYTHON_BIN}" -m pip install --upgrade pip; \
    "${PYTHON_BIN}" -m pip install --force-reinstall \
        "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"; \
    "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps \
        "nvidia-nccl-cu13==${NCCL_VERSION}"; \
    "${PYTHON_BIN}" -m pip install \
        "auditwheel>=6.0" \
        build \
        ninja \
        packaging \
        setuptools \
        wheel; \
    TORCH_VERSION="${TORCH_VERSION}" "${PYTHON_BIN}" -c \
        'import os, torch; assert torch.__version__.startswith(os.environ["TORCH_VERSION"]); print(torch.__version__, torch.version.cuda)'; \
    g++ --version | head -1; \
    nvcc --version | tail -1
