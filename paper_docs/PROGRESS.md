# MXFP4 KV Cache on sm86 — 进度同步文件

> 分支:`paper/mxfp4-kv-sm86`(fork TobyMint/sglang;Draft PR #35078 用作变更观察面板)
> 更新:2026-08-17(生产配置 + tensor-core decode kernel 定稿)
> 用途:进度同步(结果、命令、实现要点、坑)

---

## 1. 项目概况

在 **RTX 3090(sm86)** 上为 sglang 实现 **MXFP4 KV Cache**:

- **存储**:K/V 以 packed E2M1(fp4)+ E8M0 scale 存储,block32 粒度(MXFP4 标准)
- **量化**:block32 + round-to-nearest-even(RNE,MXFP4 规范舍入)
- **decode**:自研融合 attention kernel,读 fp4 直接 dequant 计算,零额外带宽;
  **tensor-core 版**(ldmatrix + mma.m16n8k16)为生产路径
- **目标模型**:仅 Qwen3-4B(GQA:32 Q / 8 KV heads, head_dim=128, group=4)
- **验收(2026-08-17,生产默认配置,CUDA graph + radix + overlap 全开,全部达成)**:

| 指标 | 线 | 实测 |
|---|---|---|
| GSM8K 200 题 | ≥90%(block16 达 90.0%;block32 接受 ~89%) | **89.2%**(mma 3 轮均值) |
| Median TPOT 增幅 vs BF16 | ≤+20% | **+16.2%**(38.15 vs 32.84 ms) |
| Median ITL 增幅 | ≤+20% | **+18.9%**(34.88 vs 29.34 ms) |
| Median TTFT 增幅 | ≤+20% | **+7.3%**(106.3 vs 99.1 ms) |
| KV 容量 | — | **3.03×**(283,808 vs 93,591 tokens) |

## 2. 环境

| 项 | 值 |
|---|---|
| 主机 | server3090-poweredge-t640,单卡 GPU 3(最后一张卡) |
| GPU | RTX 3090(sm86),驱动 570.133.07(CUDA ≤12.8) |
| 容器 | `sglang-qwen3`:`lmsysorg/sglang:v0.5.2-cu126`,bind-mount 开发 |
| 容器内 | torch 2.8.0+cu126, flashinfer 0.3.1, sglang v0.5.2 源码(editable,`/sgl-workspace/sglang`) |
| 客户端 | 宿主机 conda `/data/xbw/conda_envs/sglang`(sglang 0.5.10.post1 + sgl-eval) |
| 模型 | `/data/models/Qwen3-4B` |
| 数据 | GSM8K 缓存于 `/data/xbw/datasets/hf_cache`(hf-mirror) |

启动开发容器(bind-mount,宿主改代码即容器内生效,git push 从容器内):

```bash
docker run -d --name sglang-qwen3 --gpus '"device=3"' -p 30000:30000 \
  -v /data/models:/data/models \
  -v /data/xbw/datasets:/data/datasets \
  -v /home/xubowen/.ssh:/root/.ssh:ro \
  -v /home/xubowen/mxfp4/sglang:/sgl-workspace/sglang \
  lmsysorg/sglang:v0.5.2-cu126 sleep infinity
```

> 注:容器重建会丢失 pip 安装的额外包(sgl-eval 装在宿主客户端,不受影响)。
> 仓库在宿主 `~/mxfp4/sglang`(分支 `paper/mxfp4-kv-sm86`,remote `fork` = `git@github.com:TobyMint/sglang.git`)。
> 旧的开发目录(`~/gsm8k`、`~/design`、`~/sglang-dev`)已归档至 `~/mxfp4-archive/`(内容均已入库,可删)。

## 3. 启动服务(生产默认配置)

**切换服务前必须杀干净残留进程**(否则 OOM):`docker exec sglang-qwen3 bash -c "pkill -9 -f sglang"` 后等 ~6s。

### BF16 服务(基准)

```bash
docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && \
  nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-4B \
    --port 30000 --host 0.0.0.0 \
    > /tmp/server_bf16.log 2>&1 &"
```

### MXFP4 服务(本方案)

```bash
docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && \
  nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-4B \
    --port 30000 --host 0.0.0.0 \
    --kv-cache-dtype fp4_mx_block32 \
    > /tmp/server_fp4.log 2>&1 &"
```

两者均为生产默认配置(CUDA graph + radix cache + overlap schedule 全开),同配置对比。等待就绪(约 30-60s):`docker exec sglang-qwen3 bash -c "grep -q 'fired up' /tmp/server_fp4.log && echo READY"`。

## 4. 评测

### GSM8K 200 题(精度)

脚本在 repo `paper_docs/eval/sgl_eval_grouped.py`(基于 sgl-eval,按 finish_reason 分组统计)。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/data/xbw/datasets/hf_cache \
  /data/xbw/conda_envs/sglang/bin/python \
  /home/xubowen/mxfp4/sglang/paper_docs/eval/sgl_eval_grouped.py \
  --num 200 --max-tokens 1024 --threads 32 \
  --out /home/xubowen/mxfp4/sglang/paper_docs/results/result_<tag>.jsonl
```

关键:`enable_thinking: False` 必须显式传(sglang 0.5.2 的 Qwen3 模板认 `enable_thinking`,传 `thinking` 无效)。
注意:temperature=0 下结果仍随 batch 组成有 ±1.5pt 波动(kernel 归约顺序变化),结论取多轮均值。

### 性能(bench_serving)

```bash
/data/xbw/conda_envs/sglang/bin/python \
  /data/xbw/conda_envs/sglang/lib/python3.12/site-packages/sglang/bench_serving.py \
  --backend sglang --model /data/models/Qwen3-4B \
  --dataset-name random --num-prompts 100 --request-rate 10
```

注意:seed 默认 1(同 prompt 复跑会热 radix 缓存,轮次间系统性变快)——跨服务对比必须取相同轮次序号,且每服务先跑 1 轮预热。

### Kernel 级基准(容器内)

```bash
docker exec sglang-qwen3 python /sgl-workspace/sglang/paper_docs/tests/kernel_bench.py
# fp4 mma 0.677 / fp4 CUDA-cores 0.846 / fi fp16 scalar 0.556 / fi fp16 mma 0.474 ms
# (batch=100, seq=1024, qh=32, kh=8;绕过 staging 层直调 ext)
```

## 5. 测试结果(2026-08-17 定稿,生产默认配置)

### 精度(GSM8K 200 题)

| 配置 | 准确率 | 备注 |
|---|---|---|
| BF16 baseline | 94.0%(188/200) | 多轮 186-188 |
| block32 + round-half-up(旧舍入) | 89.5% | 消融数据 |
| block16 + round-half-up | 88.5% | 消融数据 |
| block16 + RNE | 90.0% | 消融数据(eager 配置) |
| block32 + RNE,scalar kernel,生产配置 | 86.5-90.0%(5 轮) | 均值 88.3% |
| **block32 + RNE,mma kernel,生产配置** | **87.0/85.5/86.0/84.0(修复前)** | o/d 不一致 bug |
| **block32 + RNE,mma kernel + o/d 一致性修复** | **90.0/90.5/87.0**(均值 **89.2%**) | ✅ 定稿 |

RNE(+1.5pt)与 o/d 一致性(+2.7pt)是两个关键精度修复,均有消融数据。

### 性能(bench_serving 100 req rate 10,背靠背 3 轮取热态 run3)

| 指标 | BF16 | MXFP4(mma) | Δ |
|---|---|---|---|
| Median TPOT | 32.84 ms | 38.15 ms | **+16.2%** ✅ |
| Median ITL | 29.34 ms | 34.88 ms | **+18.9%** ✅ |
| Median TTFT | 99.1 ms | 106.3 ms | **+7.3%** ✅ |

里程碑演进(scalar kernel 时代,生产配置):TPOT +26% → mma kernel 后 +16.2%。
更早(eager 配置,三个 disable,上层开销掩盖):+0.5%。

### Kernel 级(batch=100, seq=1024, qh=32, kh=8)

| kernel | 耗时 |
|---|---|
| fp4 fused **mma/ldmatrix(生产)** | **0.677 ms** |
| fp4 fused CUDA-cores(scalar) | 0.846 ms |
| flashinfer fp16 scalar | 0.556 ms |
| flashinfer fp16 mma(BF16 基线实际用) | 0.474 ms |

TPOT +16.2% 的差距构成:kernel 级 mma 路径比 fi fp16 mma 慢 0.20ms @seq1024(按生产序列长度缩放后 ~0.1ms/层 × 36 层 ≈ 3.5-4.5ms)+ quantize 写入开销;上层调度零回归(同栈同配置,输出正确性已验证)。

### 内存

| 项 | BF16 | MXFP4 |
|---|---|---|
| KV 容量 | 93,591 tokens | **283,808 tokens(3.03×)** |
| K/V 占用 | 各 6.43 GB | 各 5.18 GB |

token 密度 3.76×(39,168 vs 147,456 B/token);容量比 3.03× 是因为 fp4 池更密后为激活预留了 2.5GB(`model_runner.py` profile 特判,否则高负载 OOM)。

## 6. 实现要点

### 代码位置

```
python/sglang/srt/layers/jit_kernels/
├── mxfp4_kv.py                 # Python 封装:JIT 编译(load_inline)+ 调用 + stage 缓冲
└── cuda_kernels/
    ├── mxfp4_kv.cu             # 量化写入 / dequant / dequant_indices kernel
    ├── mxfp4_decode_fused.cu   # 融合 decode attention:scalar kernel + mma kernel(mma_v2)
    └── flashinfer_vendored/    # flashinfer 0.3.1 decode.cuh 依赖闭包(CUDA 12.6 补丁)
```

接入点(相对 v0.5.2):
- `server_args.py`:`--kv-cache-dtype` choices 加 `fp4_mx_block32`
- `model_runner.py`:dtype 解析 + pool 创建 + fp4 容量公式(cell_size 特判)+ 激活预留 2.5GB
- `memory_pool.py` `MHATokenToKVPool`:fp4 4-buffer 布局 + `set_kv_buffer` 量化写 + `get_kv_fp4_buffers`
- `flashinfer_backend.py`:prefill 量化写;**decode 走 `decode_fused_mma`**;extend o2 走 dequant workspace

### 存储布局

```
每层 4 个 buffer:
  k_data  [S, H, 64]  uint8   ← 2×E2M1 打包(lo=偶索引)
  k_scale [S, H, 4]   uint8   ← E8M0, block32 粒度, 值 = 2^(bits-127)
  v_data / v_scale 同理
```

量化:`exp = ceil(log2(block_max / 6.0))`,scale_bits = exp+127;`x·2^-exp` RNE 到 E2M1。
反量化:`__int_as_float(bits << 23)` 位构造 E8M0 scale,乘 LUT 幅值。

### mma decode kernel(生产路径,mma_v2 namespace)

- **朝向**:m = qo heads(4 真 + 12 零行),n = kv tokens,k = head_dim
  → **QK 的 C fragment 原地复用为 PV 的 A fragment(零 shuffle)**,O 从 C fragment 直接写回(零重排)
- **ldmatrix 语义**(PTX ISA 确认;此前"黑盒"结论是 probe bug):
  自然 `[t][d]` tile + **非 trans** `ldmatrix.x4` = K^T 的 col-major B(一次 x4 同时覆盖一个 16-dim k-block 的两个 8-token n-tile);`ldmatrix.x2.trans` 抵消自然布局使 V 直接作 B
- CTA = (request, kv_head),block(32,4):**4 warps 按 token 分块**(tile w, w+4, ...),每 warp 独立 online softmax,SMEM 合并 4 份 partial state(同 scalar 的 split-KV 结构)
- SMEM 45,184B:Q[16][128] fp16 + 每 warp K/V [16][128] fp16 + o_part/md_part
- 128B swizzle `phys(r,u)=r·16+(u^(r%8))`(16B 单位)→ STS.128 与两种 ldmatrix 均无 bank conflict
- 寄存器预取下一 tile 的 LDG(32B/lane/tensor),延迟藏在当前 tile 的 32 条 mma 下
- `__launch_bounds__(128,3)` 实测最优((128,2)=0.73ms、(128,4)=0.75ms、(128,3)=0.677ms)
- **精度关键**:softmax 分母 d 必须从与 PV 分子**同一批 fp16 舍入的 p** 累加(`__half2float(p16)`);不一致会系统性掉 ~2.7pt(GSM8K 85.6% → 89.2%)

### scalar decode kernel(保留,CUDA cores)

- 128 线程 = (16,4,2):bdx=head_dim 向量、bdy=GQA、bdz=split-KV,2 级双缓冲,flashinfer state_t 风格
- 作为 mma 版的对照与回退(`decode_fused` 仍可用)

### 关键坑(调试血泪史)

1. **torch 2.8 小对象池**(cudaMemMap 低地址)被自定义 kernel 读会崩(570 驱动/CUDA 12.6)→ 输入统一 `_stage()` 到预分配大 buffer(plain cudaMalloc);sanitizer 不跟踪 cudaMemMap 会误报
2. **async copy_ 悬垂** → keepalive ring(每 buffer 名 8 槽)
3. **extend 的 kv_indices 带 +256 padding** → o2 分支 dequant 前 `indices[:n]`
4. flashinfer 头文件 CUDA 13 语法(half 隐式转换)→ CUDA 12.6 编译补丁
5. **残留进程 OOM**:切换服务前 `pkill -9 -f sglang` + 等 6s
6. **overlap/graph 兼容**:kernel 全部走 `current_stream`,staging 固定地址 + replay_prepare 路径,graph 捕获期无 sync(两个旧 `torch.cuda.synchronize()` 已删)
7. **mma fragment 手工 LDS 是死路**(2.1× 慢):必须走 ldmatrix;Q 零行使 a2a3/a6a7 恒 0,省一半 Q fragment 寄存器
8. **epilogue 越界**:C fragment 的 g∈[0,8),只有 g<4 是真 head——写 o_part/md_part 必须 `if (g < kBdy)`
9. **d_state 双缩放**:p 已在 m_new 尺度,更新时不可再乘 `exp2(m_tile-m_new)`(单 tile 时恰好=1,多 tile 才暴露)

## 7. 已知限制与后续工作

- [x] ~~CUDA graph / radix / overlap 兼容~~(生产默认配置已验证)
- [x] ~~tensor-core decode kernel~~(mma/ldmatrix,0.677ms)
- [ ] **prefill 融合**:fp4 prefill 仍走 flashinfer + dequant workspace(TTFT +7.3% 主要来自此;长上下文场景收益更大)
- [ ] kernel 再优化空间:短序列的固定开销摊销、dequant LUT 位技巧、cp.async 双缓冲(SMEM 需 ~77KB,opt-in dynamic)
- [ ] NCU profile 被驱动挡(`RmProfilingAdminOnly=1`,需宿主机 root 改 `NVreg_RestrictProfilingToAdminUsers=0`);kernel 级 A/B 基准已替代其指导作用
- [ ] 论文素材:ldmatrix 语义推导、o/d 一致性消融、kernel 级四数据点、block16/32 消融均已就绪

## 8. 评测脚本与资产位置

| 文件 | 位置(repo 内,挂载进容器) | 用途 |
|---|---|---|
| `sgl_eval_grouped.py` | `paper_docs/eval/` | GSM8K 评测(finish_reason 分组) |
| `test_mxfp4_kv.py` | `paper_docs/tests/` | 量化 kernel 单测(bit-exact) |
| `test_fused_decode.py` | `paper_docs/tests/` | scalar kernel 数值验证 |
| `test_fused_decode_mma.py` | `paper_docs/tests/` | mma kernel 数值验证(vs torch 参考 + scalar 对照) |
| `kernel_bench.py` | `paper_docs/tests/` | kernel 级四方案基准 |
| `ncu_profile.py` | `paper_docs/tests/` | NCU profiling 入口(当前被驱动权限挡) |
| 结果文件 | `paper_docs/results/*.jsonl`(gitignore,宿主可见) | 各配置评测明细 |
| 设计文档 | `paper_docs/design.md` | 完整设计方案 |
