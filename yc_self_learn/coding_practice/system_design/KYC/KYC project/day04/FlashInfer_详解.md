# FlashInfer 详解

## 📋 目录
1. [FlashInfer 是什么？](#flashinfer-是什么)
2. [FlashInfer 的核心功能](#flashinfer-的核心功能)
3. [FlashInfer 在 SGLang 中的使用](#flashinfer-在-sglang-中的使用)
4. [FlashInfer CuteDSL 和 MoE](#flashinfer-cutedsl-和-moe)
5. [FlashInfer vs 其他 Backend](#flashinfer-vs-其他-backend)
6. [常见问题和解决方案](#常见问题和解决方案)

---

## FlashInfer 是什么？

### 定义

**FlashInfer** 是一个**高性能的 CUDA attention kernel 库**，专门为 LLM（Large Language Model）推理优化。

### 核心特点

1. **高性能**：
   - 针对 GPU 优化的 attention 计算
   - 比标准 PyTorch 实现快很多
   - 支持多种 attention 模式（prefill、decode、extend）

2. **功能丰富**：
   - 支持 Paged KV Cache（分页 KV 缓存）
   - 支持 Speculative Decoding（推测解码）
   - 支持 MLA（Multi-head Latent Attention）
   - 支持 Sliding Window Attention
   - 支持 MultiModal（多模态）

3. **硬件要求**：
   - 只支持 **sm75 及以上**的 GPU
   - 支持的 GPU：T4, A10, A100, L4, L40S, H100 等
   - 不支持较老的 GPU（如 V100）

### 类比

- **FlashInfer** = **高性能的 attention 计算引擎**
- 就像 **NVIDIA cuDNN** 是深度学习的基础库一样，**FlashInfer** 是 LLM 推理的加速库

---

## FlashInfer 的核心功能

### 1. Attention 计算

#### 1.1 Prefill（预填充）

**作用**：处理输入序列，计算初始的 KV Cache

```python
# FlashInfer 的 Prefill 操作
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

# 处理一批输入序列，填充 KV Cache
prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(...)
prefill_wrapper.begin_forward(...)
```

#### 1.2 Decode（解码）

**作用**：逐个 token 生成，使用已缓存的 KV

```python
# FlashInfer 的 Decode 操作
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

# 使用缓存的 KV，生成下一个 token
decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(...)
decode_wrapper.begin_forward(...)
```

#### 1.3 Extend（扩展）

**作用**：在已有 KV Cache 的基础上，添加新的 KV

```python
# FlashInfer 的 Extend 操作（prefill with cached prefix）
# 在已有前缀的基础上，继续填充新的 KV
```

### 2. Paged KV Cache

**作用**：高效管理 KV Cache 内存

- **传统方式**：为每个序列分配固定大小的内存（浪费空间）
- **Paged 方式**：按页分配内存，动态管理（节省空间）

### 3. Speculative Decoding

**作用**：使用小模型推测，大模型验证，加速生成

- FlashInfer 支持高效的 speculative decoding
- 可以显著提升生成速度

---

## FlashInfer 在 SGLang 中的使用

### 1. 作为默认 Attention Backend

**SGLang 的配置**：

```python
# SGLang 默认使用 FlashInfer 作为 attention backend
# 如果 FlashInfer 不可用，会 fallback 到 Triton

from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    # 使用 FlashInfer backend
    backend = FlashInferAttnBackend(...)
else:
    # Fallback 到 Triton
    backend = TritonAttnBackend(...)
```

### 2. 支持的功能矩阵

根据 SGLang 文档，FlashInfer 支持：

| 功能 | FlashInfer | 说明 |
|------|------------|------|
| **Page Size > 1** | ❌ | 不支持（但可以通过转换支持） |
| **Spec Decoding** | ✅ | 支持推测解码 |
| **MLA** | ✅ | 支持 Multi-head Latent Attention |
| **Sliding Window** | ✅ | 支持滑动窗口 attention |
| **MultiModal** | ✅ | 支持多模态输入 |

### 3. 代码示例

```python
# python/sglang/srt/layers/attention/flashinfer_backend.py

class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""
    
    def __init__(self, model_runner, ...):
        # 初始化 FlashInfer wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(...)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(...)
    
    def forward(self, ...):
        # 使用 FlashInfer 进行 attention 计算
        if mode == ForwardMode.PREFILL:
            return self.prefill_wrapper.begin_forward(...)
        elif mode == ForwardMode.DECODE:
            return self.decode_wrapper.begin_forward(...)
```

---

## FlashInfer CuteDSL 和 MoE

### 1. CuteDSL 是什么？

**CuteDSL** 是 FlashInfer 的一个**扩展模块**，专门用于**高性能的 GEMM（矩阵乘法）计算**。

### 2. FlashInfer CuteDSL 在 MoE 中的应用

#### 2.1 MoE（Mixture of Experts）

**MoE** 是一种模型架构，其中：
- 模型包含多个"专家"（expert）网络
- 每个 token 只激活部分专家（如 top-2）
- 可以大幅减少计算量

#### 2.2 `flashinfer_cutedsl` 的作用

**`flashinfer_cutedsl`** 是 FlashInfer 的 CuteDSL 模块，专门用于 **MoE 计算**：

```python
# python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py

from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

def flashinfer_cutedsl_moe_masked(
    hidden_states,
    w1, w1_blockscale, w1_alpha,  # FP4 量化的权重
    w2, w2_blockscale, w2_alpha,
    masked_m,
    ...
):
    """
    使用 FlashInfer CuteDSL 进行 MoE 计算
    
    特点：
    1. 支持 FP4 量化（uint8 格式）
    2. 支持 Block-scaled GEMM（块级缩放）
    3. 支持 Masked 操作（只计算激活的专家）
    4. 支持 TBO/SBO（Two-Batch Overlap / Single-Batch Overlap）
    """
    # 使用 CuteDSL 的 grouped_gemm_nt_masked 进行高效计算
    grouped_gemm_nt_masked(...)
```

### 3. 为什么需要 `flashinfer_cutedsl`？

#### 3.1 支持 FP4 量化

- **FP4 量化**：将模型权重压缩到 4 位（uint8 格式）
- **Block-scaled**：每个块有独立的缩放因子（float8_e4m3fn）
- **优势**：大幅减少显存占用，提升推理速度

#### 3.2 支持 TBO/SBO（Overlap 优化）

- **TBO（Two-Batch Overlap）**：两个 batch 的计算重叠，提升 GPU 利用率
- **SBO（Single-Batch Overlap）**：单个 batch 内部的计算重叠
- **优势**：减少等待时间，提升吞吐量

#### 3.3 与 `forward_deepgemm_masked` 的区别

| 特性 | `flashinfer_cutedsl` | `forward_deepgemm_masked` |
|------|---------------------|--------------------------|
| **FP4 支持** | ✅ 支持 | ❌ 不支持（只支持 FP8） |
| **TBO/SBO 支持** | ✅ 支持 | ❌ 不支持 |
| **状态** | ✅ 活跃维护 | ⚠️ 已废弃（deprecated） |

---

## FlashInfer vs 其他 Backend

### 1. SGLang 支持的 Backend 对比

| Backend | Page Size > 1 | Spec Decoding | MLA | Sliding Window | MultiModal |
|---------|---------------|---------------|-----|----------------|------------|
| **FlashInfer** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **FA3** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Triton** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Torch Native** | ❌ | ❌ | ✅ | ❌ | ❌ |

### 2. 为什么 SGLang 默认使用 FlashInfer？

1. **性能优秀**：FlashInfer 在大多数场景下性能最好
2. **功能完整**：支持 Spec Decoding、MLA、Sliding Window、MultiModal
3. **社区活跃**：FlashInfer 是开源项目，持续优化

### 3. 什么时候切换到其他 Backend？

- **FlashInfer 不可用**：GPU 不支持（sm75 以下）或安装失败
- **特定需求**：需要 Page Size > 1 时使用 FA3
- **调试需求**：使用 Triton 更容易调试和定制

---

## 常见问题和解决方案

### 1. FlashInfer 安装问题

**问题**：`ImportError: No module named 'flashinfer'`

**解决方案**：

```bash
# 重新安装 FlashInfer
pip3 install --upgrade flashinfer-python --force-reinstall --no-deps

# 清除缓存
rm -rf ~/.cache/flashinfer
```

### 2. GPU 不支持问题

**问题**：在 sm75 以下的 GPU 上使用 FlashInfer

**解决方案**：

```bash
# 切换到 Triton backend
python3 -m sglang.launch_server \
    --model-path <model> \
    --attention-backend triton \
    --sampling-backend pytorch
```

### 3. MoE + TBO/SBO 问题

**问题**：使用 MoE + TBO/SBO 时出错

**解决方案**：

```bash
# 确保使用 flashinfer_cutedsl backend
python3 -m sglang.launch_server \
    --model-path <model> \
    --moe-runner-backend flashinfer_cutedsl \
    --enable-two-batch-overlap  # 或 --enable-single-batch-overlap
```

**注意**：
- `flashinfer_cutedsl` 只支持 **FP4/FP8 量化的模型**
- 如果是 bfloat16 模型，需要切换到 FP8 checkpoint 或禁用 TBO/SBO

---

## 总结

### 核心要点

1. **FlashInfer 是什么**：
   - 高性能的 CUDA attention kernel 库
   - SGLang 的默认 attention backend
   - 支持 sm75+ GPU

2. **FlashInfer 的核心功能**：
   - Prefill、Decode、Extend
   - Paged KV Cache
   - Speculative Decoding
   - MLA、Sliding Window、MultiModal

3. **FlashInfer CuteDSL**：
   - FlashInfer 的扩展模块
   - 专门用于 MoE 计算
   - 支持 FP4 量化和 TBO/SBO

4. **使用建议**：
   - 默认使用 FlashInfer（性能最好）
   - 遇到问题时切换到 Triton
   - MoE + TBO/SBO 必须使用 `flashinfer_cutedsl`

---

## 相关资源

- **FlashInfer GitHub**: https://github.com/flashinfer-ai/flashinfer
- **SGLang 文档**: https://docs.sglang.ai/
- **SGLang Attention Backend 文档**: `docs/advanced_features/attention_backend.md`

---

## 相关文档

- [KYC_Day04_A1_B1_Feature_Flag实现详解.md](./KYC_Day04_A1_B1_Feature_Flag实现详解.md) - Feature Flag 实现
- [bug_16952_analysis/A01_B02_original_solutions.md](../../../../bug_16952_analysis/A01_B02_original_solutions.md) - Bug #16952 相关（MoE + TBO/SBO）
