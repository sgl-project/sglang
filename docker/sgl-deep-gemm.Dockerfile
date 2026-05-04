ARG BASE_IMG=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=13.0

FROM ${BASE_IMG}:cuda${CUDA_VERSION}

ARG ARCH=x86_64
ARG CUDA_VERSION=13.0
ARG PYTHON_VERSION=3.10
ARG PYTHON_TAG=cp310-cp310
ARG TORCH_VER=2.11.0
ARG TVM_FFI_VER=0.1.9
ARG PIP_DEFAULT_INDEX=https://pypi.python.org/simple
ARG PYTORCH_MIRROR=download.pytorch.org

ENV PYTHON_ROOT_PATH=/opt/python/${PYTHON_TAG}
ENV PATH=${PYTHON_ROOT_PATH}/bin:${PATH}

RUN yum install -y --nogpgcheck git wget tar gcc gcc-c++ make \
 && yum clean all && rm -rf /var/cache/yum

RUN set -eux; \
    if [ "${ARCH}" = "aarch64" ]; then _LIB=sbsa; else _LIB="${ARCH}"; fi; \
    mkdir -p /usr/lib/${ARCH}-linux-gnu/; \
    ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${_LIB}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

RUN --mount=type=cache,id=sgl-deep-gemm-pip,target=/root/.cache/pip \
    set -eux; \
    case "${CUDA_VERSION}" in \
      13.0) CU_TAG=cu130 ;; \
      12.9) CU_TAG=cu129 ;; \
      *)    CU_TAG=cu130 ;; \
    esac; \
    ${PYTHON_ROOT_PATH}/bin/pip install torch==${TORCH_VER} --index-url https://${PYTORCH_MIRROR}/whl/${CU_TAG}; \
    ${PYTHON_ROOT_PATH}/bin/pip install --index-url ${PIP_DEFAULT_INDEX} \
        ninja setuptools wheel build numpy apache-tvm-ffi==${TVM_FFI_VER}
