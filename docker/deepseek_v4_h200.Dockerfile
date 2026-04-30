FROM lmsysorg/sglang:v0.5.7

# need: cu12.9, x86_64 docker

RUN mkdir -p /workspace && cd /workspace && rm -rf sglang && \
    git clone -b deepseek_v4 https://github.com/sgl-project/sglang.git

# tilelang 0.1.8 pinned: mhc.py uses T.gemm(wg_wait=0), removed in 0.1.9.
RUN pip install tilelang==0.1.8

RUN pip install flashinfer-jit-cache==0.6.8 --index-url https://flashinfer.ai/whl/cu129

RUN cd /tmp && rm -rf flash-mla && \
    git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && \
    cd flash-mla && git submodule update --init --recursive && \
    pip install --no-build-isolation -v . && \
    cd /tmp && rm -rf flash-mla

RUN pip install -e /workspace/sglang/python/

# Build kernel for w4a16 marlin
RUN cd /workspace/sglang/sgl-kernel && make build

# DeepGEMM must come after sglang install: sglang pyproject pulls
# cuda-python / sgl-kernel / quack-kernels / nvidia-cutlass-dsl, which
# DeepGEMM depends on at the resolved versions.
RUN pip uninstall -y deep-gemm deep_gemm 2>/dev/null; \
    cd /tmp && rm -rf DeepGEMM && \
    git clone https://github.com/sgl-project/DeepGEMM.git -b release && \
    cd DeepGEMM && git checkout 7f2a70 && \
    git submodule update --init --recursive && \
    bash install.sh

# DeepGEMM install.sh bumps apache-tvm-ffi to 0.1.10, which breaks tilelang
# 0.1.8 ABI. Re-pin to 0.1.9 (--no-deps so pip does not touch deep-gemm).
RUN pip install --no-deps apache-tvm-ffi==0.1.9
