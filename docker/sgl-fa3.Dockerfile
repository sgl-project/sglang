# Build environment for the sgl-fa3 wheel. sgl_fa3/scripts/build.sh in
# sgl-project/sgl-flash-attn builds this image, then runs the wheel build inside
# it with the sgl-flash-attn checkout mounted at /src.
ARG BASE_IMG=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=13.0

FROM ${BASE_IMG}:cuda${CUDA_VERSION}

ARG ARCH=x86_64
ARG CUDA_VERSION=13.0
ARG PYTHON_TAG=cp310-cp310
ARG TORCH_VER=2.14.1
ARG CU_TAG=cu130
ARG PIP_DEFAULT_INDEX=https://pypi.python.org/simple
ARG PYTORCH_INDEX_BASE=https://download.pytorch.org/whl

ENV PYTHON_ROOT_PATH=/opt/python/${PYTHON_TAG}
ENV PATH=${PYTHON_ROOT_PATH}/bin:${PATH}
ENV NINJA_STATUS="[%f/%t %es] "
# CUDA 13 moved the CCCL headers (cub/thrust/libcu++) under include/cccl.
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

RUN yum install -y --nogpgcheck git wget tar gcc gcc-c++ make \
 && yum clean all && rm -rf /var/cache/yum

# The extension links -lcuda, which on a machine with no driver resolves against
# the toolkit's stub.
RUN set -eux; \
    mkdir -p /usr/lib/${ARCH}-linux-gnu/; \
    ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

# torch is needed at build time (CMake finds it, and the build is not isolated)
# and is what the wheel pins. Keep TORCH_VER in step with sgl_fa3/pyproject.toml
# and TORCH_VER in .github/workflows/release-whl-fa3.yml.
RUN --mount=type=cache,id=sgl-fa3-pip,target=/root/.cache/pip \
    set -eux; \
    pip install torch==${TORCH_VER} --index-url ${PYTORCH_INDEX_BASE}/${CU_TAG}; \
    pip install --index-url ${PIP_DEFAULT_INDEX} \
        "cmake>=3.26" ninja "scikit-build-core>=0.10" wheel numpy
