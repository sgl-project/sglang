# AMD MI355X (gfx950) Custom Kernel Development Guide

> **Target**: BF16 GEMM for N=32, K=6144 (MoE expert down projection)
> **Hardware**: AMD Instinct MI355X (gfx950, CDNA4/UDNA)
> **Software**: ROCm 7.2.0, HIP 7.2.26015, Triton 3.6.0, flydsl, CK

---

## 一、为什么 N=32 没有 flydsl kernel

### 硬件约束

gfx950 的 MFMA (Matrix Fused Multiply-Add) 指令使用 `WmmaHalf_m16n16k32`:
- **WMMA_M = 16** (M tile 最小粒度)
- **WMMA_N = 16** (N tile 最小粒度)
- **WMMA_K = 32** (K tile 最小粒度)

每个 warp 一次计算 16×16 的输出 tile。

### flydsl kernel 的 N 约束

| Kernel 路径 | TILE_N 最小值 | N 约束 | N=32 可用 |
|-------------|-------------|--------|----------|
| **Generic HGEMM** | 64 | `n % TILE_N == 0` 且 `n >= TILE_N` | ❌ (32 < 64) |
| **Small-M HGEMM** | 32 | `n % TILE_N == 0` 且 `n >= TILE_N` | ✅ 但被注释禁用 |
| **ASM** | — | `N % 64 == 0` | ❌ |
| **Skinny** | — | 仅 M≤4 | ❌ (M 动态) |

### 关键发现

`small_m_hgemm.py` 支持 `TILE_N=32`，但：
1. 限制 `M < 17` (SMALL_M_KERNEL_MAX=17)，decode 时 M 通常 > 17
2. 在 `gemm_kernels.py` 中被**注释禁用**（`# from .kernels.small_m_hgemm import ...`）
3. `get_flydsl_splitk_hgemm_kernels()` 中 small_m 枚举也被注释

---

## 二、Kernel 编写方案（从底层到高层）

### 方案 A: 启用 small_m kernel 路径（最快，改动最小）

**原理**: small_m kernel 已支持 TILE_N=32，只需取消注释并放宽 M 限制

**改动**:
1. `gemm_kernels.py`: 取消注释 `from .kernels.small_m_hgemm import ...`
2. `gemm_kernels.py`: 取消注释 `get_flydsl_splitk_hgemm_kernels()` 中的 small_m 枚举
3. `small_m_hgemm.py`: 放宽 `SMALL_M_KERNEL_MAX` 从 17 到 8192（或更大）
4. 运行 AITER tuning 找到 N=32 K=6144 的最优配置
5. 将结果写入 `glm5_bf16_tuned_gemm.csv`

**风险**: small_m kernel 的 wide-N 优化可能在大 M 时不如 generic HGEMM
**预期收益**: N=32 从 torch 回退提升到 flydsl，约 2-5x GEMM 加速

### 方案 B: N padding 32→64（简单，有额外开销）

**原理**: 将权重矩阵 B 从 [32, 6144] pad 到 [64, 6144]，用 TILE_N=64 的 flydsl kernel

**改动**:
1. 在 SGLang 的 MoE expert 层，对 N=32 的权重做 zero-padding 到 N=64
2. GEMM 后截取前 32 行结果
3. 使用现有 TILE_N=64 的 flydsl kernel

**开销**: 2x 计算量（但 flydsl 远快于 torch，净收益仍正）
**预期收益**: ~1.5-3x GEMM 加速（扣除 padding 开销）

### 方案 C: Triton 自定义 kernel（中等难度）

**原理**: 用 Triton 3.6.0 编写 N=32 专用 GEMM kernel

```python
import triton
import triton.language as tl

@triton.jit
def gemm_n32_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # N=32 固定，一个 block 处理全部 N
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Load A: [BLOCK_M, BLOCK_K]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

    # Load B: [32, BLOCK_K] (N=32 fixed)
    offs_n = tl.arange(0, 32)
    b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=offs_k[None, :] < K, other=0.0)

    # GEMM: [BLOCK_M, 32] = [BLOCK_M, BLOCK_K] @ [BLOCK_K, 32]
    c = tl.dot(a, tl.trans(b))

    # Store
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c, mask=offs_m[:, None] < M)
```

**优势**: Triton 自动处理寄存器分配和指令调度
**劣势**: Triton 在 gfx950 上可能不如 flydsl 手写 MFMA 指令高效

### 方案 D: HIP C++ + MFMA 内联汇编（最高性能，最大难度）

**原理**: 直接用 HIP C++ 编写 kernel，内联 MFMA 指令

```cpp
#include <hip/hip_runtime.h>

// gfx950 MFMA instruction: mfma_f32_16x16x32_bf16
// Computes: C[16x16] += A[16x32] * B[32x16] (bf16 inputs, f32 accumulator)
__device__ void mfma_16x16x32_bf16(
    float* d, const __bf16* a, const __bf16* b) {
    // MFMA: 16x16x32, 4 CUs per wave, 64 threads per wave
    asm volatile(
        "v_mfma_f32_16x16x32_bf16 "
        "v[0:3], v[4:5], v[6:7]"
        : : : "memory"
    );
}

__global__ void gemm_n32_bf16_kernel(
    const __bf16* __restrict__ A,  // [M, K]
    const __bf16* __restrict__ B,  // [32, K]
    __bf16* __restrict__ C,        // [M, 32]
    int M, int K) {
    // One block per M tile, N=32 fits in one MFMA tile (2x WMMA_N=16)
    // ...
}
```

**优势**: 完全控制指令调度、寄存器分配、LDS 布局
**劣势**: 开发周期长，需要深入理解 gfx950 ISA

### 方案 E: Composable Kernel (CK) 模板（AMD 官方推荐）

**原理**: 用 AMD CK 库的模板化 GEMM，自动生成优化的 kernel

```cpp
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"

// CK 会自动枚举 tile/warp 配置，生成最优 kernel
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm<
    // RowMajor A, ColMajor B, RowMajor C
    ck::tensor_operation::device::GemmASRowMajor,
    ck::tensor_operation::device::GemmBSColMajor,
    ck::tensor_operation::device::GemmCSRowMajor,
    bf16, bf16, bf16,
    ck::tensor_operation::device::GemmAccDataType::F32>;
```

**优势**: AMD 官方维护，自动适配架构
**劣势**: CK 模板复杂，编译时间长

---

## 三、推荐实施路径

| 优先级 | 方案 | 难度 | 预期收益 | 开发时间 |
|--------|------|------|---------|---------|
| **P0** | A: 启用 small_m | 低 | 2-5x GEMM | 1-2 天 |
| **P1** | B: N padding 32→64 | 低 | 1.5-3x GEMM | 0.5 天 |
| **P2** | C: Triton kernel | 中 | 2-4x GEMM | 2-3 天 |
| **P3** | E: CK 模板 | 高 | 3-5x GEMM | 3-5 天 |
| **P4** | D: HIP+MFMA 汇编 | 极高 | 4-6x GEMM | 5-10 天 |

### 建议路线

1. **先试方案 A**（启用 small_m kernel）— 改动最小，flydsl 已有 TILE_N=32 支持
2. **如果 M > 17 限制无法放宽**，用方案 B（N padding）作为快速 workaround
3. **长期方案**用方案 C（Triton）或 E（CK），编写 N=32 专用 kernel

---

## 四、gfx950 硬件参数

| 参数 | 值 |
|------|-----|
| 架构 | CDNA4 / UDNA |
| CUs | 256 |
| VRAM | 256GB HBM3e |
| WMMA 指令 | m16n16k32 (bf16) |
| WMMA_M | 16 |
| WMMA_N | 16 |
| WMMA_K | 32 |
| Warp size | 64 |
| Max warps/block | 16 |
| LDS (shared mem) | 64KB/block (gfx950) |
| DMA bytes | 16 (async copy) |

### N=32 的 MFMA 映射

N=32 = 2 × WMMA_N(16)，所以一个 warp 可以用 2 次 MFMA 覆盖整个 N=32:
- MFMA 1: C[0:16, 0:16] += A[0:16, k:k+32] × B[0:16, k:k+32]
- MFMA 2: C[0:16, 16:32] += A[0:16, k:k+32] × B[16:32, k:k+32]

这意味着 N=32 在硬件层面是完全可行的，只是 flydsl 的 generic HGEMM 路径没有覆盖。
