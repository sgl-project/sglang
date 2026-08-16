# MXFP4 KV Cache on sm86 — 开发日志与操作手册

> 分支:`paper/mxfp4-kv-sm86`
> 更新:2026-08-16(验收通过)
> 用途:开发日记(给人看)+ AI 指路手册(给后续接手的模型看)

---

## 1. 项目概况

在 **RTX 3090(sm86)** 上为 sglang 实现 **MXFP4 KV Cache**:

- **存储**:K/V 以 packed E2M1(fp4)+ E8M0 exponent-only scale 存储,省 73% 内存(KV 容量 2 倍)
- **量化**:block16 粒度 + round-to-nearest-even(RNE)舍入
- **decode**:自研融合 attention kernel,读 fp4 直接在寄存器 dequant 后计算,**零额外带宽**
- **目标模型**:仅 Qwen3-4B(GQA:32 Q / 8 KV heads, head_dim=128)
- **验收**(全部达成):GSM8K 200 题 ≥90%;TTFT/TPOT 增幅 ≤20%

## 2. 环境

| 项 | 值 |
|---|---|
| 主机 | server3090-poweredge-t640,单卡 GPU 3(最后一张卡) |
| GPU | RTX 3090(sm86),驱动 570.133.07(CUDA ≤12.8) |
| 容器 | `lmsysorg/sglang:v0.5.2-cu126`(镜像已在本地) |
| 容器内 | torch 2.8.0+cu126, flashinfer 0.3.1, sglang v0.5.2(源码在 `/sgl-workspace/sglang`) |
| 客户端 | 宿主机 conda 环境 `/data/xbw/conda_envs/sglang`(sglang 0.5.10.post1 + sgl-eval) |
| 模型 | `/data/models/Qwen3-4B`(挂载进容器) |

启动开发容器:

```bash
docker run -d --name sglang-qwen3 --gpus '"device=3"' -p 30000:30000 \
  -v /data/models:/data/models \
  -v /home/xubowen/.ssh:/root/.ssh:ro \
  lmsysorg/sglang:v0.5.2-cu126 sleep infinity
```

容器内代码即 sglang 源码 git repo(remote `fork` = `git@github.com:TobyMint/sglang.git`),开发在 `paper/mxfp4-kv-sm86` 分支,直接 push。

## 3. 启动服务

### BF16 服务(基准)

```bash
docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && \
  nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-4B \
    --port 30000 --host 0.0.0.0 \
    --disable-cuda-graph --disable-radix-cache --disable-overlap-schedule \
    > /tmp/server_bf16.log 2>&1 &"
```

### MXFP4 服务(本方案)

```bash
docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && \
  nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-4B \
    --port 30000 --host 0.0.0.0 \
    --kv-cache-dtype fp4_mx_block32 \
    --disable-cuda-graph --disable-radix-cache --disable-overlap-schedule \
    > /tmp/server_fp4.log 2>&1 &"
```

> 注:三个 `--disable-*` 是当前已知限制(见 §7)。对比基准时 BF16 也必须用同样的 disable 参数,保证同配置。

等待就绪(约 30-60s):

```bash
docker exec sglang-qwen3 bash -c "grep -q 'fired up' /tmp/server_fp4.log && echo READY"
```

## 4. 评测(GSM8K 200 题)

评测脚本在宿主机 `/home/xubowen/gsm8k/`(`sgl_eval_grouped.py`,基于 sgl-eval,按 finish_reason 分组统计)。数据已缓存(hf-mirror 下载)。

```bash
cd /home/xubowen/gsm8k
HF_ENDPOINT=https://hf-mirror.com /data/xbw/conda_envs/sglang/bin/python \
  sgl_eval_grouped.py --num 200 --max-tokens 1024 --threads 32 \
  --out /home/xubowen/gsm8k/result_<tag>.jsonl
```

输出:按 stop/length(error 分组)统计准确率。关键:`chat_template_kwargs={"enable_thinking": False}` 必须显式关闭 thinking(sglang 0.5.2 的 Qwen3 模板认 `enable_thinking` 这个 key,传 `thinking` 无效)。

### 性能对比(bench_serving)

```bash
/data/xbw/conda_envs/sglang/bin/python \
  /data/xbw/conda_envs/sglang/lib/python3.12/site-packages/sglang/bench_serving.py \
  --backend sglang --model /data/models/Qwen3-4B \
  --dataset-name random --num-prompts 100 --request-rate 10
```

## 5. 测试结果(2026-08-16)

### 精度(GSM8K 200 题,同配置)

| 配置 | 准确率 | 备注 |
|---|---|---|
| BF16 baseline | 94.0%(188/200) | 1 题截断 |
| block32 + round-half-up | 89.5% | 未达标 |
| block16 + round-half-up | 88.5% | 未达标 |
| **block16 + RNE** | **90.0%(180/200)** | ✅ 0 error,1 截断(与 BF16 同) |

RNE(round-to-nearest-even,MXFP4 规范舍入)是精度达标的关键(+1.5pt)。

### 性能(bench_serving 100 请求, rate 10, 同配置)

| 指标 | BF16 | MXFP4(fused) | Δ |
|---|---|---|---|
| Median TPOT | 54.70 ms | 54.99 ms | **+0.5%** ✅ |
| Median TTFT | 201 ms | 209 ms | **+4%** ✅ |
| Output throughput | 979 tok/s | 981 tok/s | 持平 |

中间里程碑:workspace 方案(先 dequant 到 bf16 再算)TPOT +63%(57.1ms)——**融合 kernel 把它拉回 +0.5%**。

### 内存

| 项 | BF16 | MXFP4 |
|---|---|---|
| KV 容量 | 93,591 tokens | 187,182 tokens(**2x**) |
| K+V 占用 | 各 6.43 GB | 各 3.41 GB |

## 6. 实现要点(给 AI 指路)

### 代码位置

```
python/sglang/srt/layers/jit_kernels/
├── mxfp4_kv.py                 # Python 封装:JIT 编译(load_inline)+ 调用 + stage 缓冲
└── cuda_kernels/
    ├── mxfp4_kv.cu             # 量化写入 / dequant / dequant_indices kernel
    ├── mxfp4_decode_fused.cu   # 融合 decode attention kernel(M2)
    └── flashinfer_vendored/    # flashinfer 0.3.1 decode.cuh 依赖闭包(10 个头文件)
                                # 注意:已打 CUDA 12.6 补丁((float)half → __half2float)
```

接入点(相对 v0.5.2):
- `server_args.py`:`--kv-cache-dtype` choices 加 `fp4_mx_block32`
- `model_runner.py`:kv_cache_dtype 解析 + pool 创建传 recipe
- `memory_pool.py` `MHATokenToKVPool`:fp4 布局(k_data/k_scale/v_data/v_scale 4 buffer)+ `set_kv_buffer` 量化写 + `get_kv_fp4_buffers`
- `flashinfer_backend.py`:prefill 量化写;decode 走 fused kernel;extend o2(prefix)走 dequant workspace

### 布局

```
每层 4 个 buffer:
  k_data  [S, H, 64]  uint8   ← 2×E2M1 打包(lo=偶索引)
  k_scale [S, H, 8]   uint8   ← E8M0, block16 粒度, 值 = 2^(bits-127)
  v_data / v_scale 同理
```

量化公式:`exp = ceil(log2(block_max / 6.0))`,scale_bits = exp+127;
`x_scaled = x * 2^-exp`,RNE 到 E2M1{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}。
反量化:`__int_as_float(bits << 23)`(E8M0 位操作)。

### fused decode kernel 结构

- 1 CTA = 1 (request, kv_head);128 线程 = (16, 4, 2):bdx=16(head_dim), bdy=4(GQA group), bdz=2(split-KV,最后 sync_states 合并)
- KV 加载:LDG 读 fp4(4B/线程)+ scale(2 线程共享 1B)→ 寄存器 dequant → fp16 SMEM(2 级双缓冲)
- 计算:flashinfer 风格 CUDA-cores 点积 + shfl reduce + online softmax(state_t)
- 关键陷阱:flashinfer 的 `tile_size` 参数 = bdy×tile(4 行,编码 ty);bdz 是 split-KV 不是流水线深度

### 关键坑(调试血泪史)

1. **torch 2.8 小对象池**(cudaMemMap 低地址,<1MB 张量)被自定义 kernel 读会崩(CUDA 12.6/570 驱动兼容问题)→ 所有 kernel 输入统一 `_stage()` 到预分配大 buffer(plain cudaMalloc)。sanitizer 会误报(不跟踪 cudaMemMap)。
2. **async copy_ 悬垂**:`_stage` 的拷贝源(contiguous()/to() 临时)在函数返回后释放 → keepalive ring(每 buffer 名 8 槽)。
3. **extend 的 kv_indices 带 +256 padding**:o2 分支 dequant 前必须 `indices[:n]`。
4. **overlap schedule 竞争**:共享 stage buffer 在并发 forward 下竞争 → 当前 `--disable-overlap-schedule`。
5. flashinfer 头文件是 CUDA 13 语法(half 隐式转换),CUDA 12.6 编译需补丁。
6. server 内跑评测时**残留进程会 OOM**(22GB 被占)——杀干净再测。

## 7. 已知限制与后续工作

- [ ] **CUDA graph 兼容**(当前禁用;对比基准同配置)。生产默认配置(graph 开)下 BF16 更快,需 graph 兼容后按生产配置复测
- [ ] **cp.async 双缓冲流水线**:当前 fused kernel 加载/计算串行,长序列可再优化
- [ ] **radix cache / prefix cache 适配**(当前 `--disable-radix-cache`)
- [ ] **prefill 融合**(fp4 prefill 仍走 flashinfer + dequant workspace;GSM8K 无前缀影响,长上下文场景需要)
- [ ] block32 与 block16 的精度对比数据已在,可写入论文讨论

## 8. 评测脚本位置汇总

| 文件 | 位置 | 用途 |
|---|---|---|
| `sgl_eval_grouped.py` | 宿主机 `/home/xubowen/gsm8k/` | GSM8K 评测(finish_reason 分组) |
| `run_sgl_eval_gsm8k.py` | 宿主机 `/home/xubowen/gsm8k/` | sgl-eval CLI 包装 |
| `test_mxfp4_kv.py` | 宿主机 `/home/xubowen/sglang-dev/` | 量化 kernel 单测(bit-exact vs torch 参考) |
| `test_fused_decode.py` | 宿主机 `/home/xubowen/sglang-dev/` | fused decode kernel 数值验证(vs dequant 参考) |
| 结果文件 | 宿主机 `/home/xubowen/gsm8k/result_*.jsonl` | 各配置评测明细 |
| 设计文档 | 宿主机 `/home/xubowen/design/mxfp4-kv-sm86-design.md` | 完整设计方案 v3 |
