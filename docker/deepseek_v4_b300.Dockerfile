FROM lmsysorg/sglang:v0.5.7-cu130-runtime

# need: cu13, x86 docker

RUN mkdir -p /workspace && cd /workspace && rm -rf sglang && \
    git clone -b deepseek_v4 https://github.com/sgl-project/sglang.git

RUN pip install cuda-python --upgrade

RUN pip install flashinfer-jit-cache==0.6.8 --index-url https://flashinfer.ai/whl/cu130

RUN pip uninstall -y deep-gemm deep_gemm 2>/dev/null; \
    cd /tmp && rm -rf DeepGEMM && \
    git clone https://github.com/sgl-project/DeepGEMM.git -b release && \
    cd DeepGEMM && git checkout 7f2a70 && \
    git submodule update --init --recursive && \
    ln -sf $(pwd)/third-party/cutlass/include/cutlass $(pwd)/deep_gemm/include/cutlass && \
    ln -sf $(pwd)/third-party/cutlass/include/cute $(pwd)/deep_gemm/include/cute && \
    bash install.sh

RUN pip install https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl

RUN pip install -e /workspace/sglang/python/

# TileLang for B300 (sm_100a / sm_103, needs CUDA 13+)
# Nightly 0.1.9 (2026-04-22) has sm_100a/sm_103 support
RUN pip install tilelang==0.1.9 -f https://tile-ai.github.io/whl/nightly
