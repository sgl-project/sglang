FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base

ARG BRANCH_TYPE=remote

# Python tools
RUN python3 -m pip install --no-cache-dir \
    datamodel_code_generator \
    mooncake-transfer-engine==0.3.7.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    nixl \
    py-spy

FROM scratch AS local_src
COPY . /src

FROM base AS build-image
WORKDIR /sgl-workspace
ARG BRANCH_TYPE
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    else \
        git clone --depth=1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    fi \
 && rm -rf /tmp/local_src

# Modify source code to use existing torch
# Remove after the next torch release
RUN sed -i "/torch/d" sglang/sgl-kernel/pyproject.toml && \
    sed -i -e "/torchaudio/d" \
        -e "s/torch==2.8.0/torch==2.8.0a0+34c6371d24.nv25.8/" \
        -e "s/torchao==0.9.0/torchao==0.12.0+git/" \
        sglang/python/pyproject.toml

# Necessary for cuda 13
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# Make fa_4 run on B300
ENV CUTE_DSL_ARCH=sm_100f

RUN cd sglang/sgl-kernel/ && \
    make build && \
    cd .. && \
    python3 -m pip install -e "python[all]"

# Modify Triton source file to support cuda 13
ENV TRITON_DIR=/usr/local/lib/python3.12/dist-packages/triton
RUN grep -q 'if major >= 13:' ${TRITON_DIR}/backends/nvidia/compiler.py || bash -lc $'sed -i \'/^def ptx_get_version(cuda_version) -> int:/,/^[[:space:]]*raise RuntimeError/s/^\\([[:space:]]*\\)raise RuntimeError.*/\\1if major >= 13:\\n\\1    base_ptx = 90\\n\\1    return base_ptx + (major - 13) * 10 + minor\\n\\n\\1raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)/\' ${TRITON_DIR}/backends/nvidia/compiler.py'
