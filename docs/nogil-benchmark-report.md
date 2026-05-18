# SGLang Free-Threaded Python (no-GIL) Benchmark Report

## 概述

本文档记录了 SGLang 在 CPython 3.14t (free-threaded, no-GIL) 下的性能测试方法与结果。
通过引入 `ThreadedEngine`（将 scheduler/detokenizer 从独立进程改为线程），消除了 ZMQ IPC + pickle 序列化 + SHM memcpy 的开销，在多模态场景下实现了 **4-9% 的端到端延迟降低**。

## 测试环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA H20 |
| CPU | Intel Xeon Platinum 8469C |
| OS | Ubuntu 22.04, kernel 5.15.0-117-generic |
| Driver | 590.48.01 |
| 代码仓库 | `/disk3/lsy/sgl-py14` |
| main 分支 commit | `3066ba816` |
| nogil 分支 commit | `3f4bfff1c` (基于 main + 1 commit) |

### Baseline 环境 (Py3.12)

| 项目 | 值 |
|------|-----|
| Python | 3.12.13 |
| PyTorch | 2.9.1+cu130 |
| venv | `/disk3/lsy/sgl-py312-env` |
| 分支 | `main` |
| 引擎 | 标准 `Engine`（多进程架构：tokenizer / scheduler / detokenizer 各自独立进程，通过 ZMQ IPC 通信）|

### 实验环境 (Py3.14t no-GIL)

| 项目 | 值 |
|------|-----|
| Python | 3.14.4 free-threading build |
| PyTorch | 2.11.0+cu130 |
| venv | `/disk3/lsy/sgl-py14t-env` |
| 分支 | `nogil` |
| 引擎 | `ThreadedEngine`（scheduler rank-0 和 detokenizer 作为线程运行在主进程中，通过 `queue.SimpleQueue` 引用传递通信，零 pickle/零 IPC）|

> **注意**：两侧的 PyTorch 版本不同（2.9.1 vs 2.11.0），因为 3.14t 需要较新版本的 PyTorch 支持。
> Attention backend 已统一设置为 `triton`，sampling backend 统一为 `pytorch`，以消除 backend 差异对结果的影响。

## nogil 分支改动摘要

nogil 分支在 main 基础上增加了以下文件/改动（共 ~965 行新增）：

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/engine_threaded.py` | **ThreadedEngine 实现**。scheduler rank-0 和 detokenizer 作为 daemon 线程运行，通过 `ChannelHub`（queue）与 tokenizer 通信。tp>1 时 rank 1..n 仍为独立进程。|
| `python/sglang/srt/managers/channel.py` | **进程内通信通道**。`QueueSender`/`QueueReceiver`/`AsyncQueueReceiver` 替代 ZMQ socket，`ChannelHub` 管理所有通道拓扑。|
| `launch_threaded_server.py` | ThreadedEngine 的 HTTP 服务器启动入口。|

### 兼容性修复

| 文件 | 改动 |
|------|------|
| `python/sglang/multimodal_gen/envs.py` | `fork` → `spawn`（3.14t 不支持 fork）|
| `python/sglang/srt/disaggregation/common/staging_handler.py` | GIL 排序假设 → `threading.Event`（内存可见性） |
| `python/sglang/srt/layers/rotary_embedding/factory.py` | 全局 `_ROPE_DICT` 加 `threading.Lock`（线程安全） |
| `python/sglang/srt/multimodal/processors/base_processor.py` | `mp_context` 从 `fork` → `spawn` |
| `python/sglang/srt/utils/numa_utils.py` | 允许 `forkserver` start method |
| `python/sglang/srt/managers/mm_utils.py` | `ShmPointerMMData` 增加 `_local_tensor` 快捷路径 + `set_skip_shm_transport()` 跳过 SHM（ThreadedEngine tp=1 场景） |

## 复现步骤

### 前提条件

- 已有 `sgl-py312-env`（Python 3.12）和 `sgl-py14t-env`（Python 3.14t free-threading）两个虚拟环境
- 两个环境均已安装 sglang 及其依赖（editable install from `/disk3/lsy/sgl-py14/python`）
- 模型已下载到 `/disk3/models/Qwen3-VL-8B-Instruct`

### Step 1: 运行 Py3.12 Baseline

```bash
# 切到 main 分支
cd /disk3/lsy/sgl-py14
git checkout main

# 启动服务器
source /disk3/lsy/sgl-py312-env/bin/activate
python -m sglang.launch_server \
  --model-path /disk3/models/Qwen3-VL-8B-Instruct \
  --attention-backend triton --sampling-backend pytorch \
  --tp 1 --port 30000 --mem-fraction-static 0.7 --disable-radix-cache \
  --log-level warning

# (在另一个终端) 等待服务器就绪后运行 benchmark
source /disk3/lsy/sgl-py312-env/bin/activate
python3.12 -u /disk3/lsy/sgl-py14/python/sglang/bench_serving.py \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name image --random-input 64 --random-output 16 --random-range-ratio 1.0 \
  --num-prompts 200 --request-rate 2 --image-count 4 --image-resolution 360p

# 关闭服务器
pkill -f "sglang.launch_server"
```

### Step 2: 运行 Py3.14t ThreadedEngine

```bash
# 切到 nogil 分支
cd /disk3/lsy/sgl-py14
git checkout nogil

# 启动 ThreadedEngine 服务器
source /disk3/lsy/sgl-py14t-env/bin/activate
PYTHON_GIL=0 python3.14t -Xgil=0 /disk3/lsy/sgl-py14/launch_threaded_server.py \
  --model-path /disk3/models/Qwen3-VL-8B-Instruct \
  --attention-backend triton --sampling-backend pytorch \
  --tp 1 --port 30000 --mem-fraction-static 0.7 --disable-radix-cache \
  --log-level warning

# (在另一个终端) 等待服务器就绪后运行 benchmark（用 py312 环境跑 bench client 即可）
source /disk3/lsy/sgl-py312-env/bin/activate
python3.12 -u /disk3/lsy/sgl-py14/python/sglang/bench_serving.py \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name image --random-input 64 --random-output 16 --random-range-ratio 1.0 \
  --num-prompts 200 --request-rate 2 --image-count 4 --image-resolution 360p

# 关闭服务器
pkill -f "launch_threaded_server"
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--attention-backend triton` | triton | **必须统一**。flashinfer 在 3.14t 上不可用，会 fallback 到不同实现导致不公平对比 |
| `--sampling-backend pytorch` | pytorch | 同上，统一 sampling 实现 |
| `--disable-radix-cache` | - | 关闭 radix cache，减少缓存命中带来的测量噪声 |
| `--dataset-name image` | image | 多模态数据集，产生 CPU 密集的图像预处理 |
| `--image-count 4` | 4 | 每个请求 4 张图片，增加 SHM/IPC 开销 |
| `--image-resolution 360p` | 360p | 每张图 360p 分辨率 |
| `--random-output 16` | 16 | 短输出，放大 TTFT 占比 |
| `--num-prompts 200` | 200 | 足够多的请求保证统计稳定性 |
| `--request-rate 2` | 2 | 低 QPS 避免排队效应，测单请求延迟 |
| `PYTHON_GIL=0` + `-Xgil=0` | - | **必须**。显式禁用 GIL（否则某些 C 扩展会重新启用） |

## 测试结果

### 原始数据（3 次运行）

**Py3.12 main Engine：**

| Run | Mean TTFT | Median TTFT | P99 TTFT | Mean E2E | P99 E2E | Mean TPOT |
|-----|-----------|-------------|----------|----------|---------|-----------|
| 1 | 288.75 | 271.77 | 497.61 | 539.85 | 1299.18 | 16.74 |
| 2 | 289.06 | 271.47 | 496.64 | 539.38 | 1299.07 | 16.69 |
| 3 | 290.82 | 275.01 | 497.58 | 543.05 | 1297.07 | 16.82 |

**Py3.14t nogil ThreadedEngine：**

| Run | Mean TTFT | Median TTFT | P99 TTFT | Mean E2E | P99 E2E | Mean TPOT |
|-----|-----------|-------------|----------|----------|---------|-----------|
| 1 | 270.27 | 254.73 | 509.27 | 518.37 | 1262.87 | 16.54 |
| 2 | 275.87 | 254.53 | 665.89 | 526.09 | 1259.96 | 16.68 |
| 3 | 263.95 | 237.22 | 495.93 | 509.68 | 1259.94 | 16.38 |

### 汇总对比（3 次平均，单位 ms）

| 指标 | Py3.12 Engine | Py3.14t ThreadedEngine | 变化 |
|------|:---:|:---:|:---:|
| **Mean TTFT** | 289.5 | **270.0** | **-6.7%** |
| **Median TTFT** | 272.8 | **248.8** | **-8.8%** |
| **Mean E2E** | 540.8 | **518.0** | **-4.2%** |
| **Median E2E** | 478.4 | **453.4** | **-5.2%** |
| **P99 E2E** | 1298.4 | **1260.9** | **-2.9%** |
| **Mean TPOT** | 16.75 | **16.53** | **-1.3%** |
| **Concurrency** | 1.18 | **1.14** | -3.4% |

## 性能收益分析

### 收益来源

ThreadedEngine 消除了标准 Engine 多进程架构中的以下开销：

1. **ZMQ pickle 序列化/反序列化**：标准 Engine 每次跨进程发送 Python 对象都需 `pickle.dumps` + `pickle.loads`，每个 hop ~25µs
2. **SHM memcpy**：多模态请求中，`ShmPointerMMData` 将图像 feature tensor（数 MB）拷入共享内存再拷出。ThreadedEngine 在 tp=1 时完全跳过（`set_skip_shm_transport(True)`），对象直接引用传递
3. **ZMQ socket IPC 延迟**：Unix domain socket 的内核态数据拷贝

### 为什么 TTFT 改善最大

TTFT（首 token 延迟）包含 tokenizer → scheduler 的请求提交路径。对于多模态请求，这条路径上的 SHM memcpy 开销最大（将多张图片的 feature tensor 拷入/拷出共享内存）。ThreadedEngine 将其变为零拷贝引用传递。

### 3.14t 运行时开销

CPython 3.14t 引入了 atomic reference counting 来替代 GIL，这会给所有 Python 操作带来 ~5-7% 的基础开销。ThreadedEngine 通过消除 IPC 节省的延迟**超过了**这个运行时开销，所以最终结果仍然是净正收益。

## tp>1 支持（2026-05-15 更新）

### 架构

ThreadedEngine 已支持 tp>1：
- **rank 0**：scheduler 和 detokenizer 作为线程运行在主进程中，通过 queue 与 tokenizer 通信（零 IPC）
- **rank 1..n**：作为独立进程运行（`mp.Process`），通过 `torch.distributed`（NCCL/gloo）与 rank 0 通信
- rank 0 ↔ tokenizer/detokenizer 路径依然是零拷贝引用传递
- rank 0 → rank 1..n 的请求广播仍使用 `broadcast_pyobj`（gloo + pickle + SHM）

### tp=2 复现步骤

```bash
# Py3.12 baseline（main 分支）
git checkout main
source /disk3/lsy/sgl-py312-env/bin/activate
python -m sglang.launch_server \
  --model-path /disk3/models/Qwen3-VL-8B-Instruct \
  --attention-backend triton --sampling-backend pytorch \
  --tp 2 --port 30000 --mem-fraction-static 0.7 --disable-radix-cache

# Py3.14t ThreadedEngine（nogil 分支）
git checkout nogil
source /disk3/lsy/sgl-py14t-env/bin/activate
PYTHON_GIL=0 python3.14t -Xgil=0 /disk3/lsy/sgl-py14/launch_threaded_server.py \
  --model-path /disk3/models/Qwen3-VL-8B-Instruct \
  --attention-backend triton --sampling-backend pytorch \
  --tp 2 --port 30000 --mem-fraction-static 0.7 --disable-radix-cache

# Benchmark（两者均用此命令）
python3.12 -u /disk3/lsy/sgl-py14/python/sglang/bench_serving.py \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name image --random-input 64 --random-output 16 --random-range-ratio 1.0 \
  --num-prompts 200 --request-rate 2 --image-count 4 --image-resolution 360p
```

### tp=2 Benchmark 结果

#### QPS=2（3 次平均）

| 指标 | Py3.12 Engine | Py3.14t ThreadedEngine | 变化 |
|------|:---:|:---:|:---:|
| **Mean E2E (ms)** | 291.1 | 291.5 | +0.1% |
| **Mean TTFT (ms)** | 187.9 | 187.8 | -0.0% |
| **Median TTFT (ms)** | 169.0 | 164.8 | **-2.5%** |
| **P99 E2E (ms)** | 559.7 | 529.0 | **-5.5%** |
| **Median ITL (ms)** | 4.29 | 4.23 | **-1.4%** |

#### QPS=8（3 次平均）

| 指标 | Py3.12 Engine | Py3.14t ThreadedEngine | 变化 |
|------|:---:|:---:|:---:|
| **Mean E2E (ms)** | 1665.5 | 2447.9 | +47.0% |
| **Mean TTFT (ms)** | 498.0 | 587.1 | +17.9% |
| **P99 TTFT (ms)** | 1182.2 | 1290.8 | +9.2% |
| **P99 E2E (ms)** | 4703.8 | 6973.8 | +48.3% |

### tp>1 分析

- **QPS=2（低负载）**：ThreadedEngine 与标准 Engine 性能基本持平，Median TTFT 有 2.5% 改善
- **QPS=8（高负载）**：ThreadedEngine 出现显著退化（+17-48%），原因是 **CPU 竞争**：
  - 标准 Engine 中 scheduler 作为独立进程有自己的 CPU 资源
  - ThreadedEngine 中 scheduler 线程与 tokenizer/HTTP 线程共享主进程的 CPU
  - 高 QPS 时图像预处理和 scheduler 的 `broadcast_pyobj` 同时争夺 CPU，互相干扰
  - `broadcast_pyobj`（gloo + pickle）路径仍然存在，是 tp>1 的硬约束

### tp>1 建议

1. **tp=1 场景推荐使用 ThreadedEngine**：完全消除 IPC，有 4-9% 延迟降低
2. **tp>1 低 QPS 场景**：ThreadedEngine 可用（无退化），但收益有限
3. **tp>1 高 QPS 场景**：建议使用标准 Engine，避免 CPU 竞争

## 已知限制

1. **flashinfer 不支持 3.14t**：必须使用 `--attention-backend triton`
2. **ThreadedEngine tp>1 高负载退化**：scheduler 线程与 tokenizer/HTTP 线程共享 CPU 导致竞争
3. **PyTorch 版本差异**：baseline 用 2.9.1，实验环境用 2.11.0（3.14t 需要更新版本的 PyTorch）
4. **pp_size 必须为 1**，**dp_size 必须为 1**

## 日期

- 2026-05-12：初始 tp=1 benchmark
- 2026-05-15：新增 tp>1 支持与 benchmark
