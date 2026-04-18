# ===== Base Image（已包含 PyTorch + vLLM + Triton）=====
FROM rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0

# ===== 环境变量：强制支持 gfx1201 =====
ENV HSA_OVERRIDE_GFX_VERSION=12.0.1
ENV ROCM_TARGETS="gfx1201"
ENV AMDGPU_TARGETS="gfx1201"
ENV PYTORCH_ROCM_ARCH="gfx1201"
ENV TRITON_CODEGEN_ARCH="gfx1201"

# 避免某些 kernel 编译失败
ENV TORCH_CUDA_ARCH_LIST="gfx1201"

# ===== 基础工具 =====
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== 设置工作目录 =====
WORKDIR /workspace

# ===== 拉取 sglang（你可以换成你的 fork）=====
RUN git clone https://github.com/sgl-project/sglang.git

WORKDIR /workspace/sglang

# ===== 可选：切换分支（如果你有 fork）=====
# RUN git checkout rocm-gfx1201

# ===== 安装 Python 依赖 =====
RUN pip install --upgrade pip setuptools wheel

# sglang 依赖（避免重新装 torch/vllm）
RUN pip install -e ".[all]" --no-build-isolation

# ===== 编译 sgl-kernel（关键）=====
WORKDIR /workspace/sglang/sgl-kernel

# 强制 rocm 编译环境
ENV FORCE_ROCM=1

RUN python setup.py develop

# ===== 回到主目录 =====
WORKDIR /workspace/sglang

# ===== 默认命令 =====
CMD ["/bin/bash"]
