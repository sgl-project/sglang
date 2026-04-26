FROM lmsysorg/sglang:v0.5.7-cu130-runtime

# need: cu13, arm docker
RUN mkdir -p /workspace && cd /workspace && rm -rf sglang && \
    git clone -b deepseek_v4 https://github.com/sgl-project/sglang.git

RUN pip install https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl
RUN pip install cuda-python --upgrade
RUN cd /tmp && rm -rf flash-mla && git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && cd flash-mla && ln -s /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda && git submodule update --init --recursive && pip install --no-build-isolation -v .

RUN pip install flashinfer-jit-cache==0.6.8 --index-url https://flashinfer.ai/whl/cu130
RUN pip uninstall -y deep-gemm deep_gemm 2>/dev/null; \
    cd /tmp && rm -rf DeepGEMM && git clone https://github.com/sgl-project/DeepGEMM.git -b release && \
    cd DeepGEMM && git checkout 7f2a70 && \
    git submodule update --init --recursive && \
    ln -sf $(pwd)/third-party/cutlass/include/cutlass $(pwd)/deep_gemm/include/cutlass &&  \
    ln -sf $(pwd)/third-party/cutlass/include/cute $(pwd)/deep_gemm/include/cute && \
    bash install.sh
RUN pip install -e /workspace/sglang/python/

# Install TileLang for arm
RUN pip install https://github.com/tile-ai/tilelang/releases/download/v0.1.8/tilelang-0.1.8-cp38-abi3-manylinux_2_34_aarch64.whl

# Install hadamard transform
RUN pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git

# Install mooncake
RUN pip install mooncake-transfer-engine-cuda13
