FROM lmsysorg/sglang:v0.5.7

# need: cu12.9, x86_64 docker
# Same dependency set as H200 (preset.py treats H200/B200 as one flavor).

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

# Mooncake MC_PKEY_INDEX support (upstream PR #1985):
#   The pip-released mooncake-transfer-engine wheels (<= v0.3.10.post2) hardcode
#   attr.pkey_index = 0 when transitioning RDMA QPs into INIT state. On clusters
#   where the routed InfiniBand partition lives at pkey index != 0 (e.g. some
#   Verda B200 clusters expose 0xf282 only at pkey index 1), cross-node KV
#   transfer silently fails with "transport retry counter exceeded" / "Context
#   is not active: <hca>". Mooncake commit ae292ee adds MC_PKEY_INDEX env var
#   to override this. Pin to that commit until a release tag containing it is
#   available, then this whole RUN block can be removed (base image will ship
#   the fixed wheel).
RUN pip uninstall -y mooncake-transfer-engine 2>/dev/null; \
    cd /tmp && rm -rf Mooncake && \
    git clone https://github.com/kvcache-ai/Mooncake.git && \
    cd Mooncake && \
    git checkout ae292ee837e344839eecba943363a8cd84ba49b3 && \
    git submodule update --init --recursive && \
    bash dependencies.sh -y && \
    mkdir -p build && cd build && cmake .. && make -j"$(nproc)" && cd .. && \
    bash scripts/build_wheel.sh "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    pip install mooncake-wheel/dist/*.whl && \
    cd /tmp && rm -rf Mooncake
