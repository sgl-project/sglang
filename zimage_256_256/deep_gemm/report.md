# SGLang DeepGEMM FP8 量化逻辑调研报告

> 调研日期: 2026-03-22
> 调研范围: `python/sglang/srt/layers/` + `python/sglang/multimodal_gen/`
> 硬件目标: NVIDIA H20 (SM90) / Blackwell (SM100)

---

## 1. DeepGEMM 集成入口

### 1.1 全局配置开关

**文件**: `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`

```python
ENABLE_JIT_DEEPGEMM = (
    sm_version >= 90 and            # SM90+ (H100/H20/Hopper+)
    deep_gemm package available     # pip install deep_gemm
)

DEEPGEMM_BLACKWELL = (
    ENABLE_JIT_DEEPGEMM and
    sm_version >= 100               # Blackwell (B100, GB200 等)
)

DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL  # Blackwell 特定 scale 格式
```

环境变量控制:
- `SGLANG_ENABLE_JIT_DEEPGEMM=1`: 强制启用
- `SGLANG_USE_AITER=1`: ROCm/HIP 平台改用 aiter kernel

**文件**: `python/sglang/compile_deep_gemm.py`
- 预先编译常用 shape 的 DeepGEMM kernel，避免推理时 JIT 延迟

---

### 1.2 Kernel 分派机制

**文件**: `python/sglang/srt/layers/quantization/fp8_utils.py`

```python
def dispatch_w8a8_block_fp8_linear():
    """在模块加载时决定用哪个 kernel，返回函数指针"""
    if get_bool_env_var("SGLANG_USE_AITER") and _is_hip:
        # ROCm/HIP 平台
        return aiter_w8a8_block_fp8_linear_with_fallback
    elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        # CUDA + Hopper/Blackwell: DeepGEMM
        return deepgemm_w8a8_block_fp8_linear_with_fallback
    else:
        # Fallback: Pure Triton kernel
        return triton_w8a8_block_fp8_linear
```

---

## 2. FP8 量化流程（完整数据流）

### 2.1 权重 (W8) 量化

**文件**: `python/sglang/srt/layers/quantization/fp8.py` (`Fp8LinearMethod`)

```
权重加载流程:

Step 1: create_weights()
  ├─ FP8 序列化检查点:   创建 torch.float8_e4m3fn 张量
  └─ BF16 检查点:        创建 params_dtype (BF16/FP16) 张量

Step 2: process_weights_after_loading()
  ├─ [Block Quant 路径]
  │    ├─ ROCm:           normalize_e4m3fn_to_e4m3fnuz()
  │    ├─ CPU (AMX):      _amx_process_weight_after_loading()
  │    └─ CUDA/Blackwell:
  │         if should_deepgemm_weight_requant_ue8m0():
  │             requant_weight_ue8m0_inplace(weight, scale, block_size=[128,128])
  │             # FP32 scale → UE8M0 uint8，打包为 int32
  │
  └─ [Per-Tensor 路径]
       ├─ Cutlass/Marlin:  per_token_group_quant_fp8()
       └─ 标准:            input_to_float8()
```

**关键函数**:
- `input_to_float8(x)`: BF16/FP16 → (FP8_E4M3FN, per-tensor scale)
- `requant_weight_ue8m0_inplace(weight, scale, block_size)`: 就地将 FP32 scale 转换为 UE8M0 格式

### 2.2 激活 (A8) 量化 — Dynamic

**文件**: `python/sglang/srt/layers/quantization/fp8_kernel.py`

```python
# 在每次 forward 时，对激活动态量化
q_x, x_scale = sglang_per_token_group_quant_fp8(
    x,                               # 输入: [M, K] BF16
    group_size=block_size[1],        # = 128
    column_major_scales=True,        # DeepGEMM 要求列主序 scale
    scale_tma_aligned=True,          # 分配 TMA 对齐的 buffer
    scale_ue8m0=DEEPGEMM_SCALE_UE8M0 # Blackwell: UE8M0 格式
)
# q_x:     [M, K]       FP8_E4M3FN
# x_scale: [M, K//128]  FP32 或 UE8M0 (uint8 打包为 int32)
```

**Scale 计算原理**:
```
per_group_max = max(abs(x[:, g*128 : (g+1)*128]))
scale_fp32    = per_group_max / 448.0   # FP8_E4M3FN 动态范围最大值

# Blackwell (UE8M0):
scale_ue8m0 = extract_exponent(scale_fp32)   # 仅保留指数部分 (bits 23-30)
# 4 个 uint8 打包成 1 个 int32 → 存储体积缩小 4x
```

**Scale Layout 转换**:
```python
if scale_ue8m0:
    from deep_gemm import transform_sf_into_required_layout
    x_scale = transform_sf_into_required_layout(
        x_scale,
        mn=M, k=K,
        recipe=(1, 128, 128),   # DeepGEMM scale 打包格式
        is_sfa=True,
    )
    # 形状: [M, K//128] → [M, K//128//4]（int32 打包）
```

---

## 3. DeepGEMM 调用接口

### 3.1 主要函数签名

**文件**: `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`

```python
def gemm_nt_f8f8bf16(
    lhs: Tuple[Tensor, Tensor],   # (A_fp8 [M,K],  A_scale [M,K//128])
    rhs: Tuple[Tensor, Tensor],   # (B_fp8 [N,K],  B_scale [N,K//128])
    out: Tensor,                  # 输出 [M,N] BF16
) -> None:
    """标准 W8A8 non-transposed GEMM，输出 BF16"""
    deep_gemm.fp8_gemm_nt(lhs, rhs, out)
```

### 3.2 Block-wise W8A8 完整调用链

**文件**: `python/sglang/srt/layers/quantization/fp8_kernel.py`

```python
def w8a8_block_fp8_matmul_deepgemm(
    A:  Tensor,        # FP8 激活   [M, K]
    B:  Tensor,        # FP8 权重   [N, K]
    As: Tensor,        # 激活 scale [M, K//128]  (UE8M0/FP32)
    Bs: Tensor,        # 权重 scale [N, K//128]  (UE8M0/FP32)
    block_size: List[int],     # [128, 128]
    output_dtype: torch.dtype, # torch.bfloat16
) -> Tensor:           # 输出 [M, N] BF16

    M, N, K, C = prepare_block_fp8_matmul_inputs(
        A, B, As, Bs, block_size, output_dtype
    )
    deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)
    return C
```

**Fallback 策略** (`deepgemm_w8a8_block_fp8_linear_with_fallback`):
```python
# 检查 DeepGEMM 是否支持当前 shape
if (N % 64 == 0 and K % 128 == 0 and output_dtype == torch.bfloat16
        and not use_triton_fallback):
    # → DeepGEMM kernel
    return w8a8_block_fp8_matmul_deepgemm(...)
else:
    # → Triton kernel (解包 UE8M0 → FP32 后再计算)
    return triton_w8a8_block_fp8_linear(...)
```

---

## 4. Linear 层实现

### 4.1 FP8 Linear Forward

**文件**: `python/sglang/srt/layers/quantization/fp8.py` (`Fp8LinearMethod.apply`)

```python
def apply(self, layer, x: Tensor, bias=None) -> Tensor:
    if self.block_quant:
        # ─── Block-wise W8A8 路径 ───
        if isinstance(x, tuple):
            # 预量化激活 (上游已量化): (x_q, x_scale)
            out = self.w8a8_block_fp8_linear(
                input=x[0], weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=x[1],   # 直接使用预计算 scale
                bias=bias,
            )
        else:
            # 动态量化激活 (本层内量化)
            out = self.w8a8_block_fp8_linear(
                input=x, weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,   # kernel 内部动态计算
                bias=bias,
            )
    else:
        # ─── Per-Tensor 路径 ───
        out = apply_fp8_linear(
            input=x, weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )
    return out
```

### 4.2 DeepGEMM 内部反量化（隐式）

DeepGEMM kernel 内部自动完成反量化，无需显式调用：

```
对每个 Block [i*128:(i+1)*128] × [j*128:(j+1)*128]:
  1. 加载 FP8 activations & weights
  2. 解码 UE8M0 scale → FP32 scale
  3. FP8 → FP32 转换
  4. 应用 scale: result = (A_fp32 * scale_a) × (B_fp32 * scale_b)
  5. 四舍五入到 BF16
  6. 累加到输出张量
```

---

## 5. ZImage DiT 集成

### 5.1 量化配置传递

**文件**: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`

```python
class ZImageDiT(nn.Module):
    def __init__(self, config, quant_config):
        # QKV 投影层 — FP8 量化
        self.qkv_proj = MergedColumnParallelLinear(
            dim, [head_dim * num_heads * 3, ...],
            quant_config=quant_config,   # ← Fp8Config 传入
        )
        # Output 投影层 — FP8 量化
        self.out_proj = RowParallelLinear(
            dim, dim,
            quant_config=quant_config,
        )
        # FFN Gate+Up — FP8 量化
        self.mlp = FeedForward(
            dim, hidden_dim,
            quant_config=quant_config,   # ← 两个 linear 均量化
        )
```

**被量化的层**:

| 层名称 | 维度 | 量化 |
|--------|------|------|
| QKV Proj | `d_model → 3×d_head×n_head` | ✅ W8A8 |
| Output Proj | `d_model → d_model` | ✅ W8A8 |
| FFN Gate+Up | `d_model → 4×d_model` | ✅ W8A8 |
| FFN Down | `4×d_model → d_model` | ✅ W8A8 |

### 5.2 权重加载

**文件**: `python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py`

```
加载流程:
1. 从 checkpoint 读取 FP8 权重张量 (float8_e4m3fn)
2. 调用 weight_loader() 分配到对应层
3. 全部权重加载完毕后，调用 process_weights_after_loading()
4. Blackwell 设备:
   should_deepgemm_weight_requant_ue8m0() == True
       → requant_weight_ue8m0_inplace(weight, weight_scale_inv, [128, 128])
       → weight_scale_inv 就地从 FP32 转换为 UE8M0 int32 格式
```

### 5.3 Forward 推理流程

```python
def forward(self, x: Tensor) -> Tensor:
    # x: [B, N, D] BF16

    # ─── Self-Attention ───
    # QKV 投影 (W8A8 Block GEMM)
    qkv, _ = self.qkv_proj(x)
    # 内部:
    #   x BF16 → per_token_group_quant_fp8() → (q_x, x_scale)
    #   → deepgemm_w8a8_block_fp8_linear()
    #   → qkv BF16

    q, k, v = split(qkv)
    # Attention 计算保持 BF16 (无量化)
    attn_out = flash_attention(q, k, v)

    # Output 投影 (W8A8 Block GEMM)
    out, _ = self.out_proj(attn_out)

    # ─── FFN ───
    # Gate + Up 投影 (W8A8 Block GEMM)
    gate, up = self.mlp.gate_up(x)
    # Activation function (保持 BF16)
    hidden = silu(gate) * up
    # Down 投影 (W8A8 Block GEMM)
    ffn_out, _ = self.mlp.down(hidden)

    return ffn_out
```

---

## 6. 完整数据流图

```
═══════════════════════════════════════════════════════════
                    推理前（离线权重处理）
═══════════════════════════════════════════════════════════

  Checkpoint (FP8)
       │
       ▼
  加载 layer.weight: float8_e4m3fn [N, K]
  加载 layer.weight_scale_inv: float32 [N//128, K//128]
       │
       ▼  (Blackwell only)
  requant_weight_ue8m0_inplace()
  ├─ FP32 scale → 提取指数 → uint8
  └─ 4 × uint8 → 1 × int32 (打包)
  layer.weight_scale_inv: int32 [N//128, K//128//4]
       │
       ▼
  权重就绪 (保持在显存中)

═══════════════════════════════════════════════════════════
                    推理中（每次 Forward）
═══════════════════════════════════════════════════════════

  输入 x: BF16 [M, K]
       │
       ▼
  sglang_per_token_group_quant_fp8()
  ├─ 按 group_size=128 分组
  ├─ max(|x|) / 448.0 → FP32 scale
  ├─ x / scale → FP8_E4M3FN
  └─ transform_sf_into_required_layout() → UE8M0
       │
       ▼
  q_x:     FP8_E4M3FN [M, K]
  x_scale: int32      [M, K//128//4]  (UE8M0 packed)
       │
       ▼
  w8a8_block_fp8_matmul_deepgemm()
  ├─ A  = q_x          [M, K]
  ├─ As = x_scale      [M, K//128//4]
  ├─ B  = layer.weight [N, K]
  └─ Bs = layer.weight_scale_inv [N//128, K//128//4]
       │
       ▼
  deep_gemm.fp8_gemm_nt()  ← CUDA kernel on H20/Blackwell
  ├─ 解码 UE8M0 → FP32 scale
  ├─ FP8 → FP32 转换
  ├─ Block-wise FP32 GEMM with scale fusion
  └─ 输出四舍五入到 BF16
       │
       ▼
  输出: BF16 [M, N]
  (可选) + bias
```

---

## 7. 关键文件总结

| 文件路径 | 功能 | 关键符号 |
|---------|------|---------|
| `srt/layers/deep_gemm_wrapper/configurer.py` | 全局配置 | `ENABLE_JIT_DEEPGEMM`, `DEEPGEMM_SCALE_UE8M0` |
| `srt/layers/deep_gemm_wrapper/entrypoint.py` | DeepGEMM 调用接口 | `gemm_nt_f8f8bf16()`, `grouped_gemm_nt_f8f8bf16_masked()` |
| `srt/layers/quantization/fp8.py` | FP8 量化方法 | `Fp8Config`, `Fp8LinearMethod` |
| `srt/layers/quantization/fp8_utils.py` | Kernel 分派 | `dispatch_w8a8_block_fp8_linear()` |
| `srt/layers/quantization/fp8_kernel.py` | FP8 量化 kernel | `sglang_per_token_group_quant_fp8()`, `w8a8_block_fp8_matmul_deepgemm()` |
| `srt/model_loader/utils.py` | 模型加载工具 | `should_deepgemm_weight_requant_ue8m0()`, `requant_weight_ue8m0_inplace()` |
| `srt/layers/moe/moe_runner/deep_gemm.py` | MoE + DeepGEMM | `DeepGemmRunnerCore` |
| `multimodal_gen/runtime/layers/linear.py` | 通用 Linear | `LinearBase`, `ColumnParallelLinear` |
| `multimodal_gen/runtime/models/dits/zimage.py` | ZImage 模型 | `ZImageDiT`, `ZImageAttention`, `FeedForward` |
| `multimodal_gen/runtime/loader/.../transformer_loader.py` | 权重加载 | `process_weights_after_loading()` |

---

## 8. 关键配置参数速查

```python
# ─── 量化 Config ───
quant_config = Fp8Config(
    is_checkpoint_fp8_serialized = True,   # 检查点已是 FP8 格式
    activation_scheme = "dynamic",         # 激活动态量化（推理时实时计算）
    weight_block_size = [128, 128],        # Block-wise 量化粒度
)

# ─── Activation 量化参数 ───
group_size          = 128     # 每 128 个元素共享一个 scale
column_major_scales = True    # DeepGEMM 要求列主序存储 scale
scale_tma_aligned   = True    # 对齐到 TMA 访问边界（H100/H20 硬件加速）
scale_ue8m0         = True    # Blackwell: 仅保留指数位（uint8），4个打包

# ─── DeepGEMM 触发条件 ───
# N % 64 == 0 and K % 128 == 0 and dtype == bfloat16
```

---

## 9. 性能优化要点

| 优化项 | 原理 | 收益 |
|--------|------|------|
| **UE8M0 Scale 格式** | 只存指数位，4 个 uint8 → 1 个 int32 | Scale 显存减少 4× |
| **TMA 对齐 Buffer** | 启用 H100/H20 硬件 Tensor Memory Access | 显存带宽利用率提升 |
| **Block-wise 量化** | 128×128 粒度，比 per-tensor 精度更高 | 精度/速度平衡 |
| **动态激活量化** | 推理时 online 计算 scale，无需离线校准 | 部署便捷性 |
| **Fused Dequant** | DeepGEMM kernel 内融合 scale 应用 | 减少额外 kernel launch |
| **预编译 Kernels** | `compile_deep_gemm.py` 预热常用 shape | 消除首次推理的 JIT 延迟 |
| **列主序 Scale** | 与 DeepGEMM 内存访问模式匹配 | 减少 cache miss |

---

## 10. 与标准 LLM 推理的差异

ZImage DiT FP8 与标准 LLM（如 DeepSeek-V3）的 FP8 方案在 SGLang 中共用同一套基础设施，主要差异：

| 特性 | 标准 LLM (Transformer) | ZImage DiT |
|------|----------------------|------------|
| 模型路径 | `srt/models/` | `multimodal_gen/runtime/models/dits/` |
| Linear 实现 | `srt/layers/` | `multimodal_gen/runtime/layers/` |
| 量化触发 | `quant_config` in model config | 同左 |
| Attention 是否量化 | QKV/O projections only | 同左 |
| FFN 是否量化 | gate/up/down projections | 同左 |
| Attention 计算本身 | BF16 (Flash Attention) | BF16 (自定义 Attention) |
