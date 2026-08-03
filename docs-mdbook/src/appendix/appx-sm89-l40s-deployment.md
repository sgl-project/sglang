# 附录 C：SM89/L40S 生产部署实录

> 本文档为 2026-08-04 修订版，修正了 sm89/sm90 架构表述、CUDA 13 运行时归属、stubs 根因实证三处硬伤，并补充了恢复后验证与回滚方案。

## 1. 环境概要

### 1.1 旧机器（首次验证）

| 项目 | 值 |
|------|-----|
| GPU | 4x NVIDIA L40S (SM89, 46GB VRAM) |
| CUDA Driver | 535.129.03 (CUDA 12.2) |
| CUDA Toolkit | 12.1 (nvcc) |
| Python | 3.12 |
| SGLang | 0.5.16 (源码编译, editable install) |
| sglang-kernel | 0.4.5 (pip安装, 兼容CUDA 12.2) |
| PyTorch | 2.7.0+cu121 |
| flashinfer | 0.6.7.post3 (python+cubin版本一致) |
| 模型 | Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8 |

### 1.2 新机器（镜像固化）

| 项目 | 值 |
|------|-----|
| GPU | 1x NVIDIA L40S (SM89, 46GB VRAM) |
| CUDA Driver | 535.x (CUDA 12.2) |
| CUDA Toolkit | 12.1 (nvcc) |
| Python | 3.12 |
| SGLang | 0.5.17.dev459 (源码复制到site-packages) |
| sglang-kernel | 0.4.5 (源码编译, sm_89 only) |
| PyTorch | 2.7.0+cu121 |
| flashinfer | 0.6.13 (python+cubin版本对齐) |
| 模型 | Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8 |

### 1.3 新旧机器关键差异

| 差异点 | 旧机器 | 新机器 |
|--------|--------|--------|
| sglang-kernel安装 | pip安装（pip版兼容CUDA 12.2） | **必须源码编译**（pip预编译版依赖 CUDA 13 运行时库 libcudart.so.13，系统只有 CUDA 12.1） |
| flashinfer版本 | 0.6.7.post3（天然一致） | 0.6.16 vs 0.6.7.post3（**需对齐到0.6.13**） |
| SGLang安装 | editable install (egg-link) | **复制到site-packages**（非editable，镜像可保存） |
| stubs.so编译 | 不带RPATH | **带RPATH**（解决libc10.so找不到的问题） |

---

## 2. 源码/环境修改清单

### 2.1 sglang-kernel load_utils.py 路由修复

**文件**: `/usr/local/lib/python3.12/site-packages/sgl_kernel/load_utils.py`
**行号**: 第60行
**修改前**:
```python
if compute_capability == 90:
```
**修改后**:
```python
if compute_capability in (89, 90):
```
**原因**: sglang-kernel 0.4.5 的 load_utils.py 仅将 cc=90 路由到 sm90 子目录，cc=89 会落到 sm100 目录导致加载失败。sgl-kernel 构建时（`ENABLE_BELOW_SM90` 开启的 pip 产物，或新机器以 `TORCH_CUDA_ARCH_LIST=8.9` 编译的源码产物）会在 **sm90 目录的产物中编入 sm_89 可用的 gencode**，因此 cc=89 应路由到 sm90 目录。

> 注意：不是"sm89 是 sm90 的子集"——两者是不同 major 架构（Ada 8.x vs Hopper 9.x），sm90 的 cubin 不能在 sm89 上运行。修复能成立是因为 sm90 目录产物里包含 sm89 的二进制代码。

> **新旧机器一致**，此修改两台机器都需要。

### 2.2 sm90 目录软链接

**文件**: `/usr/local/lib/python3.12/site-packages/sgl_kernel/sm90/common_ops.abi3.so`
**操作**:
```bash
ln -sf common_ops_sm90_build.abi3.so common_ops.abi3.so
```
**原因**: 0.4.5 的 sm90 目录下只有 `common_ops_sm90_build.abi3.so`，而 load_utils.py 的 glob 模式是 `common_ops.*`（不含 .so 后缀），匹配不到带 `_sm90_build` 的文件名。软链接使文件名匹配 glob 模式。

> 注：较新的 sgl-kernel 源码已在 CMakeLists 中设置 `OUTPUT_NAME=common_ops`，产物名不再是 `*_sm90_build.*`；升级 sglang-kernel 后此软链接需复查是否还需要。

> **新旧机器一致**，此修改两台机器都需要。

### 2.3 LD_PRELOAD stubs 方案（核心修复）

**问题**: `common_ops_sm90_build.abi3.so` 在动态加载时报 4 个专家特化函数符号的 undefined symbol（`fp8_blockwise_scaled_grouped_mm`、`es_fp8_blockwise_scaled_grouped_mm`、`es_sm100_mxfp8_blockscaled_grouped_mm`、`es_sm100_mxfp8_blockscaled_grouped_quant`）。这些符号对应 SM90/SM100 特化实现，在 SM89 + CUDA 12.1 环境下不可用。

**解决方案**: 创建 stubs 共享库提供这些符号，运行时通过 LD_PRELOAD 注入。

**stubs 源码** (`/usr1/project/sglang/python/sglang/kernels/aot/csrc/expert_specialization/stubs.cc`):
```cpp
// Stubs for expert specialization functions not available on SM89/CUDA 12.1
#include <torch/all.h>

void fp8_blockwise_scaled_grouped_mm(
    at::Tensor& p1, at::Tensor& p2, at::Tensor& p3,
    at::Tensor& p4, at::Tensor& p5, at::Tensor& p6,
    const at::Tensor& p7,  const at::Tensor& p8,  const at::Tensor& p9,
    const at::Tensor& p10, const at::Tensor& p11, const at::Tensor& p12,
    const at::Tensor& p13, const at::Tensor& p14, const at::Tensor& p15,
    const at::Tensor& p16, const at::Tensor& p17, const at::Tensor& p18) {
  TORCH_CHECK(false, "fp8_blockwise_scaled_grouped_mm not available on SM89");
}

void es_fp8_blockwise_scaled_grouped_mm(
    at::Tensor& p1,
    const at::Tensor& p2,  const at::Tensor& p3,  const at::Tensor& p4,
    const at::Tensor& p5,  const at::Tensor& p6,  const at::Tensor& p7,
    const at::Tensor& p8,  const at::Tensor& p9,  const at::Tensor& p10,
    const at::Tensor& p11) {
  TORCH_CHECK(false, "es_fp8_blockwise_scaled_grouped_mm not available on SM89");
}

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const at::Tensor& p1, const at::Tensor& p2,
    const at::Tensor& p3, const at::Tensor& p4,
    at::Tensor& p5,
    const at::Tensor& p6, const at::Tensor& p7, const at::Tensor& p8) {
  TORCH_CHECK(false, "es_sm100_mxfp8_blockscaled_grouped_mm not available on SM89");
}

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const at::Tensor& p1, const at::Tensor& p2,
    const at::Tensor& p3, const at::Tensor& p4,
    at::Tensor& p5, at::Tensor& p6) {
  TORCH_CHECK(false, "es_sm100_mxfp8_blockscaled_grouped_quant not available on SM89");
}
```

**旧机器编译命令**（不带RPATH，LD_PRELOAD时可能找不到libc10.so）:
```bash
g++ -shared -fPIC -o /usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so \
    /usr1/project/sglang/python/sglang/kernels/aot/csrc/expert_specialization/stubs.cc \
    -I/usr/local/lib/python3.12/site-packages/torch/include \
    -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -L/usr/local/lib/python3.12/site-packages/torch/lib \
    -lc10 -ltorch -ltorch_cpu -ltorch_python
```

**新机器编译命令**（带RPATH，运行时自动定位torch/lib下的.so）:
```bash
g++ -shared -fPIC -o /usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so \
    /usr1/project/sglang/python/sglang/kernels/aot/csrc/expert_specialization/stubs.cc \
    -I/usr/local/lib/python3.12/site-packages/torch/include \
    -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -L/usr/local/lib/python3.12/site-packages/torch/lib \
    -Wl,-rpath,/usr/local/lib/python3.12/site-packages/torch/lib \
    -lc10 -ltorch -ltorch_cpu -ltorch_python
```

**安装位置**:
```
/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
```

**使用方式**:
```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
```

**实证记录（建议固化到报告）**: undefined symbol 的根因请以实际输出为准，建议在两台机器上各跑一次并存档：
```bash
nm -D --undefined-only /usr/local/lib/python3.12/site-packages/sgl_kernel/sm90/common_ops_sm90_build.abi3.so | grep -E 'fp8_blockwise|es_sm100'
```
如果该命令没有任何输出（符号在 .so 内已定义），说明报错另有来源，需再用完整报错堆栈核对；如果输出了这 4 个符号，则本方案就是正确的修复。

**安全性**: 这些函数在当前部署中不会被调用（Qwen3.6 走标准 triton MoE 路径，不调用 SM90/SM100 专家特化内核），stubs 仅提供符号解决链接问题。**前提是 MoE 路径保持 triton**——如果以后启用 `--moe-backend es` 或内核路由变化导致这些函数被调用，stub 会直接 `TORCH_CHECK` 报错而非 crash（这也算安全网，但需要知道有这个前提）。如果意外调用，TORCH_CHECK 会明确报错而非 crash。

> **新旧机器都需要**，但新机器编译时必须加 `-Wl,-rpath` 参数。

### 2.4 sglang-kernel 源码编译（仅新机器需要）

**问题**: 新机器上 pip 安装的 sglang-kernel 0.4.5 预编译产物依赖 CUDA 13 运行时库（`libcudart.so.13`、`libcublas.so.13` 等——注意 `libcudart.so.13` 的 SONAME 对应 **CUDA 13.x**，不是 CUDA 12.8；12.x 的 SONAME 是 `libcudart.so.12`），而系统只有 CUDA 12.1。即使通过 pip 侧装 CUDA 13 运行时，也会与 torch cu121、nvcc 12.1 混用，逐个软链接不可维护。

**解决方案**: 从源码编译 sglang-kernel 0.4.5，仅编译 sm_89 gencode。

**编译步骤**:
```bash
cd /usr1/project/sglang/python/sglang/kernels/aot

# 安装编译依赖
pip install scikit-build-core "cmake>=3.26" setuptools-rust wheel setuptools-scm

# 设置环境变量
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BUILD_RUST_EXTS=none
export MAX_JOBS=40

# 编译安装
pip install . --no-build-isolation
```

**编译产出验证**:
```bash
python3.12 -c "import sgl_kernel; print(sgl_kernel.__version__)"  # 0.4.5
```

**为什么只编 sm_89**: 仅编译 L40S 目标架构，避免 sm100 特化内核在 CUDA 12.1 下编译/加载失败（这些源文件在当前版本中会无条件编入产物），同时大幅缩短编译时间。这不是为了"避免符号冲突"——同一 .so 内多 arch 是 fatbin 共存，不存在符号冲突。

> **旧机器不需要此步骤**，旧机器 CUDA 12.2 driver 兼容 pip 版 sglang-kernel。

### 2.5 flashinfer 版本对齐（仅新机器需要）

**问题**: 新机器上 flashinfer-python 0.6.16 与 flashinfer-cubin 0.6.7.post3 版本不匹配，import 报错：
```
RuntimeError: flashinfer-cubin version (0.6.7.post3) does not match flashinfer version (0.6.16)
```

**解决方案**: 将两者统一到 0.6.13（cubin 最高可用版本）:
```bash
pip install flashinfer-cubin==0.6.13 flashinfer-python==0.6.13
```

**副作用**: pip 会自动安装 cuda-toolkit-13.3.1、nvidia-nvjitlink-13.3.33 等依赖，增加约 500MB 磁盘空间。

> 注意：装完这一步后，机器上 site-packages/nvidia/ 下其实已有 CUDA 13 运行时（libcudart.so.13）。这与 2.4 的结论不冲突——sglang-kernel 源码编译仍以系统 CUDA 12.1 为准，避免运行时库版本混杂；后续排查问题时别被"机器上存在 .so.13"误导。

> **旧机器不需要此步骤**，旧机器 flashinfer 版本天然一致。

### 2.6 SGLang 安装方式变更

#### 旧机器做法: editable install

```bash
cd /usr1/project/sglang/python
SGLANG_BUILD_RUST_EXTS=none pip install -e . --no-build-isolation
```

- 优点: 简单快速，源码修改即时生效
- 缺点: sglang代码在`/usr1/project/sglang/python/`（JuiceFS挂载），**镜像保存时不包含**
- 实际效果: 镜像中只有egg-link和.pth文件，运行时需要`/usr1/project/sglang`可访问

#### 新机器做法: 复制到site-packages

由于`pip install .`（非editable）反复在build_wheel阶段卡住（目录结构无限嵌套bug），采用替代方案：

**Step 1**: 先做 editable install（确保依赖都安装好）:
```bash
cd /usr1/project/sglang/python
SGLANG_BUILD_RUST_EXTS=none pip install -e . --no-build-isolation
```

**Step 2**: 将源码同步到site-packages（整目录复制，排除缓存即可；原命令中的 `--include='*.py'` 在缺少 `--exclude='*'` 时实际上不筛选任何文件，属于无效写法，已去掉）:
```bash
rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    /usr1/project/sglang/python/sglang/ \
    /usr/local/lib/python3.12/site-packages/sglang/
```

**Step 3**: 删除editable机制，让Python直接用site-packages:
```bash
rm -f /usr/local/lib/python3.12/site-packages/__editable__.sglang-*.pth
rm -f /usr/local/lib/python3.12/site-packages/__editable___sglang_*_finder.py
rm -rf /usr/local/lib/python3.12/site-packages/sglang-*-py3.12.egg-info
# 如仍有 sglang-*.dist-info 残留，仅影响 pip show 元数据，不影响 import，可一并清理
```

**Step 4**: 验证:
```bash
python3.12 -c "import sglang; print(sglang.__file__)"
# 期望: /usr/local/lib/python3.12/site-packages/sglang/__init__.py
```

> **注意**: pip install . 非editable安装理论上更规范，但SGLang 0.5.17的build_wheel存在bug（`build/lib/build/lib/build/lib/...`无限嵌套），在多台机器上均复现。rsync+删除editable是目前可用的workaround（无限嵌套与源码树内残留 `python/build/` 有关，4.3 的清理步骤不能省）。

---

## 3. SGLang 启动参数适配

### 3.1 L40S 限制对应的参数

| 参数 | 值 | 原因 |
|------|-----|------|
| `--enforce-disable-flashinfer-allreduce-fusion` | 必须 | FlashInfer allreduce fusion 需要 SM90+；SM89 下本就不会启用，此 flag 属防御性保险，保留无害 |
| `--disable-cuda-graph` | 必须 | 实测 L40S（SM89）下 mamba triton + fp8 KV 组合的 CUDA graph capture 失败，关闭以保证稳定启动（与 driver 535 无关，驱动本身支持 CUDA graph；具体报错建议后续补充存档） |
| `--mamba-radix-cache-strategy extra_buffer` | 必须 | Qwen3.6 混合 SSM 模型需要（替代已废弃的--mamba-scheduler-strategy） |
| `--mamba-backend triton` | 必须 | Qwen3.6 混合 SSM 模型需要（该值也是默认值；SM100 限制只作用于 stochastic rounding 路径，本部署未涉及） |
| `--kv-cache-dtype fp8_e5m2` | 推荐 | 减少显存占用，支持 96K 上下文 |
| `--skip-server-warmup` | 推荐 | 加快启动，避免 warmup OOM；代价是首个请求延迟略高（无预编译预热） |
| `--tool-call-parser qwen3_coder` | 必须 | Qwen3.6 tool call 支持 |

### 3.2 96K DP 内存优化参数

| 参数 | 64K TP2 (DP=1) | 96K DP≥2 | 原因 |
|------|---------|---------|------|
| `--mem-fraction-static` | 0.85 | 0.78 | 96K+DP KV cache 更大，需降低预留 |
| `--max-running-requests` | 16 | 8/worker | per-worker值；DP=3时8×3=24总并发 |
| `--context-length` | 65536 | 96256 | 96K 上下文窗口 |

> **重要**: `--max-running-requests` 是 **per-worker** 参数，不是全局值。DP=3 + 8/worker = 24总并发，与3个独立TP2实例（各8）效果一致。

> 注：65536=64×1024，而 96256=94×1024（=98304−2048，即 96K 规格窗口留 2K 余量）。如果业务规格要求字面意义的 96K，请用 `--context-length 98304`；命令中的数值与表格口径保持一致。

### 3.3 统一启动脚本

使用 `sglang_start.sh`，根据GPU数量自动推断TP/DP/上下文/并发等参数：

```bash
# 单卡验证
bash /usr1/sglang_scripts/sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8

# 6卡生产
bash /usr1/sglang_scripts/sglang_start.sh --model-path /usr1/project/models/Qwen3.6-35B-A3B-FP8
```

| GPU数 | 自动推断 | 上下文 | mem | 并发(总) |
|--------|---------|--------|-----|------|
| 1卡 | TP=1 DP=1 | 64K | 0.85 | 16 |
| 2卡 | TP=2 DP=1 | 64K | 0.85 | 16 |
| 4卡 | TP=2 DP=2 | 96K | 0.78 | 16 (8×2) |
| 6卡 | TP=2 DP=3 | 96K | 0.78 | 24 (8×3) |

### 3.4 完整启动参数示例（6卡生产）

```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3.12 -m sglang.launch_server \
    --model-path /usr1/project/models/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --tp-size 2 --dp-size 3 \
    --mem-fraction-static 0.78 \
    --context-length 96256 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8_e5m2 \
    --chunked-prefill-size 4096 \
    --max-running-requests 8 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-backend triton \
    --enable-flashinfer \
    --attention-backend flashinfer \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-cuda-graph \
    --enable-cache-report \
    --skip-server-warmup \
    --enable-metrics \
    --log-level info
```

> 注：示例与 3.3 的 6 卡口径统一为 35B-A3B-FP8；如需在 6 卡上跑 27B-FP8（例如压测），仅需改 model-path 与 served-model-name，其余参数不变。

---

## 4. 镜像持久化方案

### 4.1 目录持久化规则

| 路径 | 是否保存到镜像 | 说明 |
|------|---------------|------|
| `/usr1/project/` | **否** | JuiceFS挂载，不进镜像 |
| `/usr1/其他目录/` | **是** | 如 `/usr1/sglang_scripts/` 会保存 |
| `/usr/local/` | **是** | site-packages等 |
| `/root/` | **是** | pip cache等（需清理） |
| `/tmp/` | **是** | 临时文件（需清理） |

### 4.2 需要持久化的修改

| 修改 | 位置 | 是否在 site-packages | 镜像是否保留 |
|------|------|---------------------|-------------|
| sgl_stubs.so | `/usr/local/lib/.../sgl_kernel/` | 是 | 是 |
| load_utils.py 修改 | `/usr/local/lib/.../sgl_kernel/` | 是 | 是 |
| sm90 软链接 | `/usr/local/lib/.../sgl_kernel/sm90/` | 是 | 是 |
| sglang-kernel 源码编译 | `/usr/local/lib/.../sgl_kernel/` | 是 | 是 |
| flashinfer 版本对齐 | `/usr/local/lib/.../flashinfer*/` | 是 | 是 |
| sglang 包 (site-packages) | `/usr/local/lib/.../sglang/` | 是 | 是 |
| 启动脚本 | `/usr1/sglang_scripts/` | 否 | **是**（/usr1非project目录） |
| sglang 源码 (JuiceFS) | `/usr1/project/sglang/` | 否 | **否** |

### 4.3 镜像保存前清理

```bash
# 清理pip cache（~7GB）
rm -rf /root/.cache/pip /root/.cache/torch /root/.cache/nv

# 清理临时文件
rm -rf /tmp/build_sglang* /tmp/pip-* /tmp/tmp*

# 清理sglang源码build artifact（JuiceFS上，不进镜像，但节省空间；也是2.6中build_wheel无限嵌套的诱因之一）
rm -rf /usr1/project/sglang/python/build /usr1/project/sglang/python/sglang.egg-info

# 停止测试服务
pkill -9 -f sglang.launch_server
```

### 4.4 镜像体积估算

| 组件 | 大小 |
|------|------|
| site-packages总计 | ~14GB |
| - nvidia/ | 4.9GB |
| - torch/ | 1.8GB |
| - flashinfer-cubin/ | 1.2GB |
| - triton/ | 597MB |
| - sglang/ | 109MB |
| - sgl_kernel/ | 24MB |
| - 其他 | ~5.4GB |
| /usr1/sglang_scripts/ | <10KB |
| /root/ (清理后) | ~40MB |
| /tmp/ (清理后) | ~20MB |
| **总计** | **~14GB** |

### 4.5 镜像保存检查清单

- [x] sglang 复制到 site-packages（非editable）
- [x] sgl_stubs.so 在 site-packages/sgl_kernel/ 下（带RPATH编译）
- [x] load_utils.py 第60行为 `if compute_capability in (89, 90):`
- [x] sm90/common_ops.abi3.so 软链接存在
- [x] flashinfer-python 与 flashinfer-cubin 版本一致（0.6.13）
- [x] 启动脚本中 LD_PRELOAD 指向 site-packages 内路径
- [x] 启动脚本固化在 /usr1/sglang_scripts/
- [x] pip cache 已清理
- [x] 单卡 Qwen3.6-27B-FP8 推理验证通过
- [x] RadixTree 正常初始化（日志可见 `Init Unified RadixTree`）
- [ ] nm -D undefined symbol 实证输出已存档（见 2.3，建议补）
- [ ] 镜像保存/恢复命令已记录（见 4.6，待补充）
- [ ] 恢复后验证脚本已执行（见 4.7，待补充）

### 4.6 镜像保存与恢复命令（待补充）

> 本节取决于镜像平台（docker commit / VM 快照 / 平台镜像固化），保存命令需按实际平台填写，例如：
> ```bash
> # docker 示例（按实际平台替换）
> docker commit <container> sglang-l40s-sm89:0.5.17.dev459
> ```
> 恢复后必须执行 4.7 的验证，不能只依赖 4.5 的保存前清单。

### 4.7 恢复后验证步骤

```bash
# 1) sglang 包指向 site-packages（非editable）
python3.12 -c "import sglang; print(sglang.__file__)"
# 期望: /usr/local/lib/python3.12/site-packages/sglang/__init__.py

# 2) sglang-kernel 版本与路由修改
python3.12 -c "import sgl_kernel; print(sgl_kernel.__version__)"   # 0.4.5
sed -n '60p' /usr/local/lib/python3.12/site-packages/sgl_kernel/load_utils.py
# 期望: if compute_capability in (89, 90):

# 3) sm90 软链接
ls -l /usr/local/lib/python3.12/site-packages/sgl_kernel/sm90/common_ops.abi3.so

# 4) stubs 存在且带 RPATH
ls -l /usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
readelf -d /usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so | grep -i rpath

# 5) flashinfer 版本一致
python3.12 -c "from importlib.metadata import version; print(version('flashinfer-python'), version('flashinfer-cubin'))"
# 期望: 0.6.13 0.6.13

# 6) 单卡冒烟：启动后 curl 可用、日志含 RadixTree 初始化
bash /usr1/sglang_scripts/sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8
curl -s http://127.0.0.1:8000/v1/models
# 启动日志应含: Init Unified RadixTree with components (<ComponentType.FULL: 0>, <ComponentType.MAMBA: 2>)
```

### 4.8 回滚方案

| 故障场景 | 回滚操作 |
|---------|---------|
| stubs 疑似引起问题 | 去掉 LD_PRELOAD 重启，对比报错是否变化；确认后删除/替换 sgl_stubs.so |
| sglang-kernel 源码编译产物异常 | 回到 pip 安装 0.4.5 wheel（需先补齐 CUDA 13 运行时依赖或换 CUDA 12.2 兼容 wheel） |
| site-packages 的 sglang 包异常 | 重新执行 2.6 的 rsync + 删 editable 步骤 |
| 上述均无法恢复 | 回退到保存前的镜像快照 |

---

## 5. 架构说明

### 5.1 TP=2 DP=3 生产架构（6卡）

```
Client
  │
  ▼
SGLang Server (port 8000)
  │
  ├─ DP Router (Prefix-Aware)
  │    │
  │    ├─ Worker 0: GPU 0,1 (TP=2)
  │    │    └─ RadixTree KV Cache (fp8_e5m2, 96K)
  │    │
  │    ├─ Worker 1: GPU 2,3 (TP=2)
  │    │    └─ RadixTree KV Cache (fp8_e5m2, 96K)
  │    │
  │    └─ Worker 2: GPU 4,5 (TP=2)
  │         └─ RadixTree KV Cache (fp8_e5m2, 96K)
  │
  └─ 并发: 8/worker × 3 = 24总
```

- DP Router 自动将相同 prefix 的请求路由到同一 worker，复用 RadixTree KV cache
- 不同 prefix 的请求负载均衡到各 worker
- 每个 worker 独立持有 96K 上下文的 KV cache
- max-running-requests=8 是 per-worker 值，总并发 = 8 × DP_SIZE

### 5.2 RadixTree 说明

- SGLang 内置 RadixTree，用于 KV cache 前缀复用
- C++ 实现（`cpp_radix_tree/`），JIT 编译，首次运行时自动构建
- Qwen3.6 混合 SSM 模型使用 UnifiedRadixCache（FULL + MAMBA 组件）
- 启动参数 `--mamba-radix-cache-strategy extra_buffer` + `--enable-cache-report`
- 日志验证: `Init Unified RadixTree with components (<ComponentType.FULL: 0>, <ComponentType.MAMBA: 2>)`

---

## 6. 版本兼容性注意事项

| 组件 | 版本约束 | 原因 |
|------|---------|------|
| CUDA Driver | >= 535 | L40S 最低支持 |
| CUDA Toolkit | 12.1 | sglang-kernel 编译时使用的版本 |
| sglang-kernel | 0.4.5 | 必须源码编译；pip 预编译版依赖 CUDA 13 运行时（libcudart.so.13），与系统 CUDA 12.1 不兼容 |
| flashinfer | 0.6.13 | python与cubin版本必须一致，0.6.13是cubin最高可用版 |
| TORCH_CUDA_ARCH_LIST | 8.9 | 仅编译 L40S 目标架构，避免 sm100 特化内核在 CUDA 12.1 下编译/加载失败，并缩短编译时间 |
| LD_PRELOAD | sgl_stubs.so | 必须在 sglang 启动前设置 |
| SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK | 1 | 跳过内核版本检查（源码编译版版本号可能不匹配） |

---

## 7. 常见启动日志报错说明

SGLang启动时会尝试加载所有已知模型处理器，部分加载失败是**正常行为**，不影响Qwen3.6运行：

### 7.1 `deep_gemm/_C.so libcudart.so.13` — 可忽略

```
Ignore import error when loading sglang.srt.models.deepseek_v4: 
  Failed to load dynamic shared library deep_gemm/_C.so libcudart.so.13: cannot open shared object file
```

- 原因: DeepSeek V3/V4等MoE模型依赖deep_gemm库，该库需要较新的 CUDA 运行时（`libcudart.so.13` 对应 **CUDA 13** 运行时；若实际安装的是 CUDA 12.8 版 deep_gemm，依赖应为 `libcudart.so.12`，请以 ldd 输出为准）
- 影响范围: 仅DeepSeek、GLM4-MoE、Kimi K25等特定MoE架构
- **Qwen3.6-35B-A3B不受影响**: 虽然也是MoE模型，但使用SGLang内置标准MoE Runner（fused_moe/triton），不依赖deep_gemm

### 7.2 `libtorchcodec` / `libavutil` — 可忽略

```
Ignore import error when loading sglang.srt.multimodal.processors.mimo_audio: 
  Could not load libtorchcodec... libavutil.so.60: cannot open shared object file
```

- 原因: 视频/音频多模态模型需要FFmpeg库
- 影响: 仅多模态模型，纯文本推理无关
