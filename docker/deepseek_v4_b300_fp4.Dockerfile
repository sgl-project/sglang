FROM lmsysorg/sglang:v0.5.7-cu130-runtime

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# tilelang's bundled libtvm.so depends on libz3.so (no version suffix).
# Base image ships nothing matching, and apt's libz3-4 only installs libz3.so.4.
RUN apt-get update && apt-get install -y --no-install-recommends libz3-4 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libz3.so.4 /usr/lib/x86_64-linux-gnu/libz3.so && \
    ldconfig && rm -rf /var/lib/apt/lists/*

# sglang: our fork branch with FP4-acts opt-in + deep_gemm.mega_moe_pre_dispatch wiring.
RUN mkdir -p /workspace && cd /workspace && rm -rf sglang && \
    git clone -b fp4-acts-pre-dispatch https://github.com/pranjalssh/sglang.git

RUN pip install cuda-python --upgrade
RUN pip install flashinfer-jit-cache==0.6.8 --index-url https://flashinfer.ai/whl/cu130


RUN pip install https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl

# DeepGEMM: our local tree (subtree of pranjalssh/kernels) with FP4 acts +
# MXF4 kind + the new fused mega_moe_pre_dispatch kernel.
# DG_USE_LOCAL_VERSION=0 tells setup.py to skip the working-tree-clean check
# (the install runs inside the parent kernels repo, not a clean DeepGEMM clone).
RUN pip uninstall -y deep-gemm deep_gemm 2>/dev/null; \
    cd /tmp && rm -rf kernels && \
    git clone --depth 1 -b worktree-hazy-noodling-giraffe \
        https://github.com/pranjalssh/kernels.git && \
    cd kernels/DeepGEMM && \
    ln -sf $(pwd)/third-party/cutlass/include/cutlass $(pwd)/deep_gemm/include/cutlass && \
    ln -sf $(pwd)/third-party/cutlass/include/cute $(pwd)/deep_gemm/include/cute && \
    DG_USE_LOCAL_VERSION=0 bash install.sh

RUN pip install -e /workspace/sglang/python/


RUN pip install --force-reinstall --no-deps tilelang==0.1.8

RUN pip install nvidia-cuda-cccl && \
    CCCL_INC=$(find /usr/local/lib -path "*/include/cccl/cuda/std" -type d 2>/dev/null | head -1 | sed 's|/cuda/std$||') && \
    ln -sf $CCCL_INC/cuda /usr/local/cuda/include/cuda && \
    mv /usr/local/cuda/targets/x86_64-linux/include/cccl /usr/local/cuda/targets/x86_64-linux/include/cccl.bak && \
    ln -sf $CCCL_INC /usr/local/cuda/targets/x86_64-linux/include/cccl
# FlashMLA — required by deepseek_v4_backend_radix.py
RUN cd /tmp && rm -rf flash-mla && \
    git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && \
    cd flash-mla && git submodule update --init --recursive && \
    pip install --no-build-isolation .
# fast_hadamard_transform — sgl_kernel 0.3.21 lacks hadamard_transform on B300
RUN pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git

# Install mooncake
RUN pip install mooncake-transfer-engine-cuda13
