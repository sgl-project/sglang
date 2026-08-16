# sm86 (RTX 3090) MXFP4 KV Cache 设计方案

状态:设计 v3(最终),全部验收通过
# 2026-08-16 最终结果:
# - block16 + RNE(round-to-nearest-even)量化:GSM8K 200 = 90.0%(达标线 90%)
# - M2 融合 decode kernel(sm86, 读 fp4 直接计算):TPOT +0.5%、TTFT +4%(达标线 +20%)
# - KV 容量 2x(93,591 -> 187,182 tokens)
日期:2026-08-15
基线:容器 v0.5.2-cu126(torch 2.8.0+cu126, flashinfer 0.3.1),单卡 GPU 3
模型:Qwen3-4B(GQA:32 Q heads / 8 KV heads, head_dim=128)
验收:GSM8K 200 题 ≥90%(BF16 baseline 94%);TTFT/TPOT 增幅 ≤20%

## 0. 调研结论(为什么现有方案不适用 sm86)

| 方案 | 存储/量化 | decode 计算 | sm86 可用性 |
|---|---|---|---|
| main 分支通用 FP4(FP4MXBlock16KVQuantizeUtil) | torch.compile,依赖 `torch.float4_e2m1fn_x2`(torch≥2.8)与 flashinfer `fp4_quantize` | flashinfer 融合 kernel(SM90/SM100 专属) | ✗ flashinfer fp4 kernel 无 sm86 分支,`raise ValueError` |
| NVFP4 PR #31269(DSV4/GLM-5.2) | sgl-kernel AOT CUTLASS | FlashMLA 三阶段(scheduler + persistent WGMMA + combine) | ✗ WGMMA/TMA 是 Hopper 专属 |
| DSV4 MXFP4 PR #32741(用户自己的) | Triton codec,block-32 E8M0 | FlashMLA 移植版(load_jit JIT CUDA) | ✗ 同样依赖 WGMMA;但**布局/dequant 技巧/上层接入/验证方法论全部可借鉴** |

**结论**:sm86 无 fp4 量化/反量化基础设施,必须自研 CUDA kernel;但布局、量化公式、dequant 技巧(LUT+PRMT、E8M0 bit-shift)、pool/backend 接入架构、验证方法均可从上述方案移植。

## 1. KV Cache 布局(MXFP4 标准, block_size=32)

```
每层(每个 layer)4 个 buffer:
  k_data  [size+page, kv_heads, head_dim/2]  uint8   ← 2×fp4 打包
  k_scale [size+page, kv_heads, head_dim/32] uint8   ← E8M0 (exponent-only)
  v_data  [size+page, kv_heads, head_dim/2]  uint8
  v_scale [size+page, kv_heads, head_dim/32] uint8
```

- **数据**:E2M1(1 符号位 + 2 指数位 + 1 尾数位),16 个取值 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6};每 2 个元素打包进 1 字节(低 4 位为偶索引)。
- **scale**:E8M0(8-bit 纯指数),block_size=32(遵循 MXFP4 规范),**沿 head_dim 方向**;每个 uint8 值 v 表示 2^(v-127)。反量化寄存器内 `bits << 23` 得到 fp32 scale(用户 PR #32741 的技巧)。
- **量化公式**(MXFP4 规范,与 main 分支 FP4MXBlock16 一致):
  - `exp = ceil(log2(block_max / 6.0))`,`scale_e8m0 = exp + 127`
  - `x_scaled = x / 2^exp`,然后 round-to-nearest 到 E2M1
- **内存收益**(每 token 每层):
  - BF16:2 × 8 × 128 × 2B = 4096 B
  - MXFP4:data 2×8×64B + scale 2×8×4B = **1088 B → 省 73.4%**
  - 对比 FP8:2048 B(省 50%)。KV 容量 93591 tokens → ~35 万 tokens(理论上限,受显存约束)
- 为什么 block 沿 head_dim:decode 读 KV 按"单 token 全 head"访问,块内连续;跨 token 共享 scale 会破坏读取局部性。
- 为什么不量化 RoPE 分量:Qwen3 的 K 已含 RoPE,head_dim=128 无分离 rope 部分(与 DSV4 的 368B row 不同,MHA 布局更简单)。

## 2. CUDA 算子设计(sm86)

### 2.1 KV 量化写入 kernel(`kv_quantize_store`)
- **触发点**:prefill/extend 的 save_kv_cache 阶段(model 算完 cache_k/cache_v bf16 之后,pool 写入之前)。
- **签名**:`quantize_store(cache_kv_bf16 [T, heads, 128], loc [T], data_ptr, scale_ptr, ...)`(K/V 各调一次或一次处理两个)。
- **并行划分**:每 CTA 处理 `kTokens × kHeads`;线程沿 head_dim=128 展开,每线程 8 元素(128-bit 向量)。
- **sm86 特性利用**:
  - `ld.global.nc.v4.b32` / 128-bit 访问:bf16 8 元素/次,block16 = 2 次;fp4 写入 32 元素/次
  - block 求 abs-max:16 元素在 1 warp 内完成(`__reduce_max_sync` 或 shfl)
  - scale 计算:`__log2f` + ceil,或 `frexp` 直接取指数(避免除法:scaled = x * (2^-exp),2^-exp 可用 `__frcp` 预计算一次)
  - E2M1 round-to-nearest:LUT 查表(16 项 uint8 数组)或 7 次比较 → PRMT 打包(用户 PR 技巧:4 指令解 16 元素)
  - 打包:lo | hi<<4,128-bit 写回

### 2.2 decode 读 KV(两条路径,分阶段)

**阶段 A(dequant workspace,先保证正确性)**:
- 每步 decode 前,把本步需要的 KV slot 从 fp4 dequant 成 bf16 临时 workspace,再喂给现有 flashinfer decode。
- 带宽反而更差(fp4 读 + bf16 写 + bf16 读),性能必然超 20% 预算 → 仅用于精度验证。

**阶段 B(融合 decode attention kernel,性能达标路径,主要工作量)**:
- 自研 sm86 flash-decoding 风格 kernel,完全替换 fp4 模式下的 flashinfer decode:
  - 每 CTA 处理 1 个 query 的 1-2 个头,按 key 长度分块(split-KV,flashattention-3 式分块 + 原子写或 PDL)
  - 读 fp4 KV:128-bit 读 32 元素 + 对应 scale(1 字节/块)→ **寄存器/SMEM 内 dequant 成 fp16**
  - `mma.sync.aligned.m16n8k16.f32.f16.f16.f32`(Ampere 最高效 mma,对比 Hopper 的 WGMMA)
  - 在线 softmax + 最终合并(与 flashinfer decode kernel 同构,参考其 sm86 分支的调度/分块)
  - dequant 在寄存器内完成,不产生额外全局带宽 —— 这是性能达标的关键
- 工作量最大,参考:flashinfer `decode` kernel(0.3.1 源码,sm86 路径)+ 用户 PR #32741 的 dequant 技巧 + sglang jit_kernels 现有 decode 实现。

## 3. 上层接入(基于容器 v0.5.2 源码)

1. **server_args.py**:`kv_cache_dtype` choices 增加 `fp4_mx_block16`;启动检查允许 sm86(与 main 分支相反的断言)。
2. **memory_pool.py `MHATokenToKVPool`**:`_create_buffers` 加 fp4 分支——创建 data+scale 4 个 buffer(store_dtype=uint8);`set_kv_buffer` 走量化写入 kernel;`get_key_buffer`/`get_value_buffer` 返回 data+scale(或按调用方需要)。
3. **flashinfer_backend.py**:
   - `forward`:save_kv_cache 处换量化写入 kernel;
   - decode:阶段 A 走 dequant workspace + 原 flashinfer;阶段 B 走融合 kernel(仅 fp4 模式激活,不动 bf16 路径)。
4. **radix cache**:prefix cache 也存 KV 原数据(fp8 时代已处理过 uint8 存储,0.5.2 的 radix 需要同步适配)。**先 `--disable-radix-cache` 简化**(GSM8K 每题独立,无前缀复用,不影响评测),后续再适配。
5. **只支持 Qwen3(MHA/GQA)**:fp4 模式在模型非 MHA 时 fail-fast;不动 MLA 路径。

## 4. 验证与验收

| 层级 | 方法 |
|---|---|
| kernel 单测 | 量化往返误差 vs BF16(随机 + 真实激活);与 FP4MXBlock16 的 torch 参考实现逐元素对比 |
| 精度 e2e | GSM8K 200 题(sgl_eval_grouped.py 既有脚本),验收 ≥90% |
| 性能 e2e | bench_serving 100 请求(既有方法),TPOT/TTFT vs BF16 baseline,增幅 ≤20% |

## 5. 里程碑

| 里程碑 | 内容 | 产出 |
|---|---|---|
| M1 | 布局 + 量化/反量化 CUDA kernel + pool/backend 接入 + workspace decode | 端到端跑通,GSM8K 精度 ≥90% |
| M2 | 融合 decode kernel(阶段 B) | TPOT/TTFT ≤ +20% |
| M3 | 收尾:radix cache 适配(可选)、benchmark 报告、代码推送 fork | 毕设交付 |

## 6. 风险

- M1 的性能必然超预算(workspace 方案带宽翻倍)→ M2 融合 kernel 是必经之路,不能省
- 融合 kernel 工作量最大(预估 1-3 周),是毕设核心增量
- flashinfer 0.3.1 的 decode 调度结构(sm86)需精读源码才能保证融合 kernel 性能不输
- 0.5.2 的 CUDA-graph 捕获路径(fp4 模式涉及新 kernel,需验证 graph 兼容)
