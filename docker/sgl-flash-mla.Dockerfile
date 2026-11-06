ARG BASE_IMG=pytorch/manylinux2_28-builder
# FlashMLA's kernels use `cvt` with `.scaled::n2::ue8m0`, which needs nvcc >= 13.1,
# and older toolkits emit suboptimal instructions for them. Build with 13.4.
ARG CUDA_VERSION=13.4

FROM ${BASE_IMG}:cuda${CUDA_VERSION}

ARG ARCH=x86_64
ARG CUDA_VERSION=13.4
ARG PYTHON_VERSION=3.12
ARG PYTHON_TAG=cp312-cp312
ARG TORCH_VER=2.14.1
ARG PIP_DEFAULT_INDEX=https://pypi.python.org/simple
ARG PYTORCH_MIRROR=download.pytorch.org

ENV PYTHON_ROOT_PATH=/opt/python/${PYTHON_TAG}
ENV PATH=${PYTHON_ROOT_PATH}/bin:${PATH}

RUN yum install -y --nogpgcheck git wget tar gcc gcc-c++ make \
 && yum clean all && rm -rf /var/cache/yum

# The build links -lcuda, which on a machine with no driver resolves against the
# toolkit's stub.
RUN set -eux; \
    if [ "${ARCH}" = "aarch64" ]; then _LIB=sbsa; else _LIB="${ARCH}"; fi; \
    mkdir -p /usr/lib/${ARCH}-linux-gnu/; \
    ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${_LIB}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

# torch is needed at build time (setup.py imports it) and is what the wheel pins.
# The CUDA minor version of this torch does not have to match CUDA_VERSION above:
# the extensions link libcudart.so.13, whose SONAME carries only the major, and
# use no API newer than 13.0.
RUN --mount=type=cache,id=sgl-flash-mla-pip,target=/root/.cache/pip \
    set -eux; \
    ${PYTHON_ROOT_PATH}/bin/pip install torch==${TORCH_VER} --index-url https://${PYTORCH_MIRROR}/whl/cu130; \
    ${PYTHON_ROOT_PATH}/bin/pip install --index-url ${PIP_DEFAULT_INDEX} \
        ninja setuptools wheel build numpy
