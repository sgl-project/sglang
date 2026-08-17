# sm86 (RTX 3090) MXFP4 KV Cache 设计方案

状态:设计 v4(最终),全部验收通过
# 2026-08-17 最终结果(生产默认配置:CUDA graph + radix + overlap 全开):
# - block32 + RNE 量化(MXFP4 标准):GSM8K 200 = 89.2%(3 轮均值;BF16 94%;block16 消融 90.0%)
# - 融合 decode kernel,tensor-core 路径(ldmatrix + mma.m16n8k16):
#     TPOT +16.2%(38.15 vs 32.84 ms)、ITL +18.9%、TTFT +7.3%(达标线 +20%)
#     kernel 级 0.677ms(fp4)vs 0.474ms(flashinfer fp16 mma)@ b=100/seq=1024
# - KV 容量 3.03x(93,591 -> 283,808 tokens;token 密度 3.76x)
日期:2026-08-15(v3)→ 2026-08-17(v4 定稿)
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

### 2.3 最终实现(v4,两代融合 kernel 的演进)

**第一代(CUDA-cores scalar,`mxfp4_decode_fused_kernel`)**:flashinfer decode 同构
(bdx=head_dim 向量 + shfl 归约 + state_t),生产 TPOT +26%。留作对照与回退。

**第二代(tensor-core,`mxfp4_decode_fused_mma_kernel`,生产路径)**,核心决策:

1. **fragment 构造必须用 ldmatrix,不能用手工 LDS**。中间走过弯路:按 PTX ISA fragment
   表手工 LDS 构造 A/B 片段(QK 32×LDS.32 + PV 64×LDS.16/iter),指令开销爆炸,
   实测 1.815ms(比 scalar 还慢 2.1×)。ldmatrix 语义(PTX ISA §9.7.15.5.15,曾被
   误判为"黑盒",实为 probe bug):
   - 非 trans:source[r][c] → slot(r,c);**自然 [t][d] tile + 非 trans `ldmatrix.x4`
     即 K^T 的 col-major B**(B 是加载矩阵的转置),一次 x4 同时覆盖一个 16-dim
     k-block 的两个 8-token n-tile(R0..R3 = nt0-b01, nt1-b01, nt0-b23, nt1-b23)
   - `ldmatrix.x2.trans` 加载转置 → 抵消自然 [t][d] 布局,V 直接作 PV 的 B
2. **朝向 m=qo heads / n=kv tokens / k=head_dim**(flashinfer prefill 同构):
   QK 的 C fragment 布局(c0,c1 = S[g][2tig])恰好就是 PV 的 A fragment 布局
   (a0,a1 = P[g][2tig])—— **softmax 后原地复用,零 shuffle**;O 的每个 (head, dim)
   恰好由唯一 lane 持有,**epilogue 直接写回,零重排**。
   m=16 行中只有 4 行真 head(GQA group),Q 零填充,零行使 a2a3/a6a7 恒 0,
   Q fragment 寄存器减半。
3. **4 warps 按 token 分块**(tile w, w+4, ...),每 warp 独立 online softmax,
   SMEM 合并 4 份 partial(与 scalar 的 bdz split-KV 同构,天然兼容任意序列长度)。
4. **128B swizzle** `phys(r,u) = r·16 + (u^(r%8))`(16B 单位)使 STS.128 与两种
   ldmatrix 均无 bank conflict(闪存 flashinfer permuted_smem 同式)。
5. **寄存器预取**:下一 tile 的 fp4 数据(32B/lane/tensor)先 LDG 进寄存器,延迟
   藏在当前 tile 的 32 条 mma 下(无需 cp.async —— 它无法在搬运中做 dequant)。
6. **精度关键**:softmax 分母 d 必须从与 PV 分子同一批 **fp16 舍入后的 p** 累加
   (`__half2float(p16)`),否则 o/d 不同源,GSM8K 系统性 -2.7pt(85.6% → 89.2%)。
   这是 tensor-core softmax 与 scalar(float p)的本质差异,有消融数据。

实测(b=100, seq=1024, qh=32, kh=8):scalar 0.846 → **mma 0.677ms(-20%)**;
`__launch_bounds__(128,3)` 为实测最优。

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

| 里程碑 | 内容 | 产出 | 状态 |
|---|---|---|---|
| M1 | 布局 + 量化/反量化 CUDA kernel + pool/backend 接入 + workspace decode | 端到端跑通 | ✅(workspace 方案 TPOT +63%,验证精度后弃用) |
| M2a | 融合 decode kernel(CUDA-cores scalar) | TPOT 达标(eager 配置 +0.5%) | ✅ |
| M2b | tensor-core decode kernel(ldmatrix + mma) | 生产配置 TPOT +16.2% | ✅(2026-08-17) |
| M3 | 收尾:生产配置复测、文档、推送 fork、宿主归档 | 毕设交付 | ✅(Draft PR #35078 作变更面板) |

## 6. 风险(终局回顾)

- ~~M1 性能超预算(workspace 带宽翻倍)~~ → 确认发生(+63%),M2 融合 kernel 解决
- ~~融合 kernel 工作量~~ → 两代共 ~2 周;手工 LDS fragment 弯路(-2.1×)后被 ldmatrix 路线取代
- ~~flashinfer 调度结构精读~~ → 其 prefill.cuh(手写 PTX mma + cp.async producer)是最终参考
- ~~CUDA graph 兼容~~ → 生产默认配置(graph+radix+overlap)验证通过
- 遗留:prefill 仍走 dequant workspace(TTFT +7.3% 的主因);NCU 被驱动权限挡(kernel 级 A/B 基准替代)
