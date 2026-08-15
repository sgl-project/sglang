# 6×L40S 生产部署 Qwen3.6-27B-FP8：SGLang 从 E2E 89s 到 32s 的调优实录

> 本文基于 2026-08 的线上实践。环境：SGLang 0.5.17 + sglang-kernel 0.4.5（源码编译，sm_89）+ flashinfer 0.6.13 + PyTorch 2.7.0+cu121，6× NVIDIA L40S（46GB，SM89）。版本信息见文末免责说明。

## TL;DR

我们用 6 张 L40S（TP=2 DP=3）生产部署了 Qwen3.6-27B-FP8，端到端延迟（E2E）从初期的 **89 秒降到约 32 秒**，abort 率稳定在 **1.56%**，1.5 天 23,324 个请求的稳态数据全部达标。

性能不是靠某一项"神参数"，而是把五个独立的问题在五个关卡上逐个填平：

1. **显存地基**：`--mem-fraction-static 0.85` + FP8 KV + mamba `extra_buffer`
2. **单请求提速**：MTP（NEXTN 投机解码）
3. **流量阀门**：网关侧控制 `max_tokens` + 限并发 + priority
4. **路由均衡**：DP 路由用 round_robin，避免前缀粘滞热点
5. **稳定性兜底**：预热 + keep-alive + 关闭 CUDA graph

其中任何一环断掉，其他优化都会被拖下水。另外，我们在 L40S（SM89）上踩了一整套环境适配的坑，从 `load_utils` 路由修复到 `LD_PRELOAD` stubs，都在第 3 节给出了可复现的解法。

---

## English Abstract

This post is a production case study of serving **Qwen3.6-27B-FP8** (a 27B dense hybrid GDN model) with **SGLang 0.5.17** on **6× NVIDIA L40S (46GB, SM89)**, TP=2 DP=3, 98K context, FP8 KV cache, and MTP (NEXTN) speculative decoding.

**Results (1.5-day steady state, 23,324 requests):** E2E latency dropped from ~89s to ~32s; abort rate 1.56%; streaming TTFT 5.69s; cache hit rate 45.8%; per-worker load imbalance <2%.

**Key takeaways:**

1. **Five-layer defense**: KV memory budget (`--mem-fraction-static 0.85` + FP8 KV + `extra_buffer` for Mamba state), MTP speculative decoding, gateway-side `max_tokens` gating, DP load balancing (`round_robin` vs. prefix-aware hotspots), and warmup/keep-alive for cold-start stability.
2. **Reproducible SM89 workarounds**: `load_utils.py` routing fix, sm90 directory symlink, `LD_PRELOAD` stubs for four expert-specialization symbols, source-built `sglang-kernel` for sm_89, and flashinfer version pinning.
3. **Measurement pitfall**: computing "effective ITL" as `ITL / accept_len` double-counts MTP gains; the real speedup was ~1.8× ITL and ~2.2× E2E.

Version note: 2026-08, SGLang 0.5.17 / sglang-kernel 0.4.5. The SM89 workarounds are version-specific and should be re-verified against newer releases.

---

## 1. 背景

Qwen3.6-27B 是 Qwen3.6 系列的 **dense 27B 变体**，采用 GDN（Gated Delta Networks，混合线性注意力/SSM + Transformer）架构，原生支持 262,144 token 上下文，官方定位就是"单卡友好"。我们的负载有两类：

- **代码检视**（thinking 开启）：长输出，单请求可达数千 token；
- **工具调用**（thinking 关闭）：短输出，要求低首 token 延迟（TTFT）。

两类请求共享同一个 system prompt 前缀，这既是前缀缓存的机会，也是后面 DP 路由热点的根源。

## 2. 环境与基线

| 项目 | 配置 |
|---|---|
| GPU | 6× NVIDIA L40S（46GB，SM89 / Ada Lovelace） |
| 驱动 / Toolkit | CUDA 12.2 driver / CUDA 12.1 (nvcc) |
| SGLang | 0.5.17（源码同步到 site-packages） |
| sglang-kernel | 0.4.5（源码编译，仅 sm_89 gencode） |
| flashinfer | 0.6.13（python 与 cubin 对齐） |
| PyTorch | 2.7.0+cu121 |
| 模型 | Qwen3.6-27B-FP8（含 MTP 权重 `mtp.safetensors`） |
| 并行架构 | TP=2 DP=3 |
| 上下文 | 98,304 |
| KV cache dtype | fp8_e5m2 |

最初线上配置是 `--mem-fraction-static 0.78` + `--max-running-requests 8`，表现是平均 TTFT 30 秒以上、E2E 60 秒以上，最差时 abort 13.2%、TTFT 45.4 秒。下面先讲环境适配，再讲性能调优主线。

## 3. 第一关：让 SGLang 在 L40S（SM89）上跑起来

L40S 的算力是 SM89（Ada），而 SGLang 生态里的预编译内核大多面向 SM90（Hopper）和 SM100（Blackwell）。我们踩了五个坑，前四个需要改环境，第五个是版本对齐。

### 3.1 问题清单

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | cc=89 加载内核失败 | `load_utils.py` 只把 cc=90 路由到 sm90 目录 | 修改路由条件 |
| 2 | 内核文件匹配不到 | glob 模式与产物文件名不一致 | 软链接 |
| 3 | 动态加载 undefined symbol | 4 个 SM90/SM100 特化符号在 SM89 上不存在 | `LD_PRELOAD` stubs |
| 4 | pip 安装的 wheel 起不来 | 预编译产物依赖 CUDA 13 运行时（`libcudart.so.13`） | 源码编译，仅 sm_89 |
| 5 | flashinfer import 报版本不匹配 | python 包与 cubin 包版本不一致 | 统一版本 |

### 3.2 修复一：`load_utils.py` 路由修复

sglang-kernel 0.4.5 的 `load_utils.py` 第 60 行只把 `compute_capability == 90` 路由到 sm90 子目录，cc=89 会错误地落到 sm100 目录导致加载失败：

```diff
-if compute_capability == 90:
+if compute_capability in (89, 90):
```

这个修复成立的前提是：sm90 目录的产物里包含 sm_89 的 gencode（用 `ENABLE_BELOW_SM90` 构建的 pip 产物，或 `TORCH_CUDA_ARCH_LIST=8.9` 编译的源码产物）。注意 **SM89 不是 SM90 的子集**（Ada 8.x vs Hopper 9.x），sm90 的 cubin 不能直接在 sm89 上跑，能修复是因为产物里同时编入了 sm89 的二进制。

### 3.3 修复二：sm90 目录软链接

0.4.5 的 sm90 目录下产物名是 `common_ops_sm90_build.abi3.so`，而 `load_utils.py` 的 glob 模式是 `common_ops.*`，匹配不到：

```bash
cd /usr/local/lib/python3.12/site-packages/sgl_kernel/sm90
ln -sf common_ops_sm90_build.abi3.so common_ops.abi3.so
```

较新的 sgl-kernel 源码已在 CMakeLists 中设置 `OUTPUT_NAME=common_ops`，升级后此软链接需要复查是否还需要。

### 3.4 修复三：`LD_PRELOAD` stubs（核心修复）

动态加载时报 4 个专家特化函数的 undefined symbol：

```
fp8_blockwise_scaled_grouped_mm
es_fp8_blockwise_scaled_grouped_mm
es_sm100_mxfp8_blockscaled_grouped_mm
es_sm100_mxfp8_blockscaled_grouped_quant
```

**为什么缺符号、为什么用 stub**

这 4 个符号来自 sglang-kernel 里的专家特化（expert specialization）内核。库内的注册/分发代码会**无条件引用**它们，但**实现**只在 SM90/SM100 上编译——要么需要更新的 CUDA，要么根本不支持 sm89。于是在"sm89 专用构建 + CUDA 12.1"的环境里，这个 .so 只有"引用"没有"定义"。

Linux 动态加载器在 `dlopen` 时会把所有未定义符号解析一遍，找不到就直接失败，导致 `import sgl_kernel` 整体报错——**哪怕这 4 个函数永远没人调用**。也就是说，报错的是"符号不存在"，不是"功能不能用"：Qwen3.6 走标准 triton MoE 路径，这些特化内核在 L40S 上不会被触发。

为什么不把实现编译出来？这些内核本身面向 SM90/SM100 的指令特性（如 Blackwell 的 MXFP8），sm89 + CUDA 12.1 编不出来；直接改 .so 二进制又太脆弱，换版本就失效。用 `LD_PRELOAD` 注入一个提供同名符号的共享库，让加载器"找到"这些符号，是最轻量、完全可逆（去掉 `LD_PRELOAD` 即可）的 workaround。

解法：编译一个 stubs 共享库提供这 4 个符号，运行时用 `LD_PRELOAD` 注入：

```cpp
// stubs.cc：为 SM89/CUDA 12.1 环境补齐缺失符号
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

```bash
g++ -shared -fPIC -o /usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so \
    stubs.cc \
    -I/usr/local/lib/python3.12/site-packages/torch/include \
    -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -L/usr/local/lib/python3.12/site-packages/torch/lib \
    -Wl,-rpath,/usr/local/lib/python3.12/site-packages/torch/lib \
    -lc10 -ltorch -ltorch_cpu -ltorch_python

export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
```

安全性说明：Qwen3.6 走标准 triton MoE 路径，不会调用这些特化内核，stubs 只负责解决链接问题；如果真的被调用，`TORCH_CHECK` 会明确报错而不是崩溃。**前提是 MoE 路径保持 triton**，如果以后启用 expert specialization 内核（如 `--moe-backend es`），需要重新评估。

实证方法（建议存档）：用 `nm -D --undefined-only` 确认这 4 个符号确实未定义，再决定是否采用本方案。

### 3.5 修复四：源码编译 sglang-kernel

pip 安装的 0.4.5 预编译产物依赖 CUDA 13 运行时（`libcudart.so.13`，注意 SONAME 对应 CUDA 13.x，不是 12.8），而系统只有 CUDA 12.1。逐个软链接不可维护，直接源码编译，只编 sm_89：

```bash
cd python/sglang/kernels/aot
pip install scikit-build-core "cmake>=3.26" setuptools-rust wheel setuptools-scm
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_BUILD_RUST_EXTS=none
export MAX_JOBS=40
pip install . --no-build-isolation

python3.12 -c "import sgl_kernel; print(sgl_kernel.__version__)"  # 0.4.5
```

只编 sm_89 是为了避免 sm100 特化内核在 CUDA 12.1 下编译/加载失败，同时大幅缩短编译时间。

### 3.6 修复五：flashinfer 版本对齐

新机器上 flashinfer-python 0.6.16 与 flashinfer-cubin 0.6.7.post3 不匹配，import 直接报错：

```
RuntimeError: flashinfer-cubin version (0.6.7.post3) does not match flashinfer version (0.6.16)
```

统一到 0.6.13（cubin 最高可用版本）：

```bash
pip install flashinfer-cubin==0.6.13 flashinfer-python==0.6.13
```

### 3.7 必须的启动参数

| 参数 | 为什么必须 |
|---|---|
| `--disable-cuda-graph` | L40S（SM89）下 mamba triton + FP8 KV 组合的 CUDA graph capture 失败，关闭才能稳定启动 |
| `--enforce-disable-flashinfer-allreduce-fusion` | allreduce fusion 需要 SM90+，SM89 下防御性关闭 |
| `--mamba-radix-cache-strategy extra_buffer` | Qwen3.6 混合 SSM 模型需要（替代已废弃的 `--mamba-scheduler-strategy`） |
| `--mamba-backend triton` | Qwen3.6 混合 SSM 模型需要 |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` | Qwen3.6 的 thinking / tool call 解析 |

### 3.8 给读者的建议

这些 workaround 都是**版本相关的**（2026-08，SGLang 0.5.17 / sgl-kernel 0.4.5），新版本可能已修复，落地前用 `python3.12 -m sglang.launch_server --help | grep <参数名>` 复核。`load_utils` 路由修复看起来是一个真正的 bug，建议直接给上游提 PR，而不是长期依赖本地补丁。

## 4. 性能调优主线：五层防护框架

环境通了之后，性能问题的本质不是"引擎不行"，而是**单模型、单硬件、特定负载下，五个环节各有一个短板**。下面按请求生命周期讲。

### 4.0 背景知识：推理的时间花在哪（先建立心智模型）

LLM 推理只有两个阶段，瓶颈完全不同：

- **Prefill**：一次性并行处理整个 prompt。attention 本质是矩阵乘，GPU 擅长，瓶颈在算力和批次大小；
- **Decode**：逐 token 生成。每生成一个 token，都要把模型权重从显存读一遍，attention 还要读全部历史 KV——这是**显存带宽瓶颈**，不是算力瓶颈。

我们的数据正好印证了这一点：E2E 89 秒里 decode 占 95%，所以"调参数"救不了 E2E——大头不在调度，而在每生成一个 token 要付的物理成本。

decode 的每 token 成本可以拆成两项：

```text
每 token 成本 ≈ 权重读取（固定，所有模型都有） + 上下文读取（随上下文增长，attention 特有）
```

- **权重读取**是带宽下限，与架构无关（27B dense FP8 每步约 6~7GB）；
- **上下文读取**随序列变长越来越大，长上下文下 KV 读取会吃掉越来越多的带宽，decode 越跑越慢。

**线性注意力/Mamba 把第二项干掉了**：KV cache 换成固定大小的状态，上下文读取变成 O(1)；顺带省显存（mamba state 约 146MB/req，同长度 KV 是它的几十倍）。但第一项权重读取还在，decode 的带宽下限没变；而且 prefill 从可并行的矩阵乘变成串行 scan，反而更难算。所以没有纯 Mamba 的生产模型，全是 hybrid——Qwen3.6 部分层用 GDN、部分层保留 attention。

架构解决"上下文读取"之后，decode 只剩两个杠杆：

- **投机解码（MTP）**：每步猜多个 token 一起验证，"读一次权重、出多个 token"，摊薄带宽成本——与架构正交，KV 还是状态都有效；
- **CUDA graph / kernel 优化**：减每步固定开销，量级 10~20%。

一句话收束：**架构决定每 token 的下限和上下文成本；投机解码突破"每步只能出 1 个 token"的低效；调度和显存决定能不能把这些能力用满；业务侧决定要走多远。** 下面五层框架里的①（显存地基）和②（MTP）解决的就是"容量"和"带宽"这两个物理约束，③④⑤解决"有资源但用不好"的调度层问题。

### 4.1 先做延迟构成分析

任何调优之前，先把 E2E 拆开：

```
E2E = TTFT + decode = (排队 + prefill) + (输出 token 数 × ITL)
```

我们的第一个基线（无 MTP）：TTFT 3.75s，E2E 89.2s，其中 **decode 占约 95%**（935 token/req × 92ms）。结论很直接：E2E 的大头是输出长度和单 token 延迟，而不是 prefill。这决定了后面优化的优先级。

### 4.2 层① 显存地基：KV 池要够大

**坑**：`--mem-fraction-static 0.78` 时每卡留了 9GB 闲置显存，KV 池只有 685K token，请求一多就排队、OOM、abort：

| mem-fraction | KV/card | 剩余显存 | Abort | TTFT (stream) | Cache hit |
|---|---|---|---|---|---|
| 0.78 | 685,264 | 9.15 GB | **13.2%** | **45.4s** | 29.2% |
| 0.85 | 746,595~793,189 | 6.1~7.6 GB | **1.56%**（稳态） | **5.69s**（稳态） | 45.8% |

0.85 比 0.78 多出的约 3GB/卡全部给了 KV 池。**注意**：0.78 的慢同时叠了并发低（8/worker）的问题，我们后来拆变量验证过（见第 8 节方法论）——但最终定版就是 0.85 + 12/worker。

对 Qwen3.6 这类混合 SSM 模型，还有第二个地基问题：**Mamba state 和 Attention KV 争显存**。`--mamba-radix-cache-strategy extra_buffer` 为 Mamba state 预留额外 buffer，不挤占 KV 池，代价是约 10% 的额外显存。单卡 TP=1 + MTP 时这个问题最严重。

### 4.3 层② 单请求提速：MTP

decode 92ms/token 意味着一条 thinking 请求要 89 秒。MTP（Multi-Token Prediction，SGLang 里的 `--speculative-algorithm NEXTN`）是唯一能把有效 decode 步数压下来的系统侧手段。

效果（实测）：ITL 从 92.2ms 降到 40.9ms（稳态均值），E2E 从 89.2s 降到约 32s。MTP 的收益与代价、以及一个常见的口径误区，见第 5 节。

### 4.4 层③ 流量阀门：控输出、限并发

**坑**：thinking 请求可能输出几千 token，长时间占住 KV 不释放，新请求全部排队，TTFT 爆炸、abort 升高。

解法是网关侧按场景控制输出上限：

| 场景 | enable_thinking | max_tokens | temperature | repetition_penalty |
|---|---|---|---|---|
| Thinking（代码检视） | true | 8192 | 0.1 | 1.05 |
| Non-Thinking（工具调用） | false | 2048 | 0.0 | 1.0 |

效果：Gen/req 从 935 降到 645，请求更快完成、KV 更快释放，abort 从 5.4% 降到 1.56%。

**注意**：Qwen3.6-27B 的模板不支持 `thinking_budget`（实测无此字段，设置会被静默忽略），控制输出长度的唯一手段就是 `max_tokens` 硬截断。截太短会截在 thinking 中途、答案丢失（2048 已实测复现）——所以这是业务侧的取舍，不是纯性能参数。

另外可以加一层前置代理做"按类型限并发 + 注入 priority + 过载 429 兜底"：工具调用请求注入 `priority=10` 插队，thinking 封顶并发，高峰突发时快速失败而不是越来越慢。收益不在总量，而在**混合流量下的资源分配**——纯单类型或低负载场景收益有限。

### 4.5 层④ 路由均衡：DP Router 热点

**坑**：prefix-aware 的 DP Router 会把相同 system prompt 的请求**全部路由到同一个 worker**（前缀粘滞）。我们的工具调用场景所有请求共享前缀，于是出现"Worker 0 空转（0 running），Worker 1 满载（12 queued）"——**4 卡当 2 卡用**，TTFT 9.95s 几乎全是排队。

修复参数是 `--load-balance-method round_robin`（注意不是 `--dp-load-balancing`）。round_robin 下每个 worker 首请求后各自缓存 system prompt，命中率不受影响，负载被摊平。

**但这不是绝对的**：如果生产流量前缀足够多样，prefix-aware 长时间运行后也会自然均衡（我们的 1.5 天稳态 3 个 worker 请求差 <2%，前缀命中率反而更高）。结论：**默认 prefix-aware，单一前缀的压测/特殊场景切 round_robin**。我们最终选择 round_robin 作为保险。

### 4.6 层⑤ 稳定性兜底：预热 + keep-alive

**坑**：mamba triton 内核按 shape 逐个 JIT，flashinfer 按 shape 逐个 autotune。`--skip-server-warmup` 会把冷启动成本摊进线上前几百个请求的 TTFT，MTP 开启后 draft/verify 内核更多，冷启动更贵。

解法三件套：

1. **上线前预热**：等服务 ready 后，按负载形状发 3 类请求（短非 thinking、长 prompt、thinking），间隔 5~10s，直到 TTFT 收敛、`~/.cache/flashinfer` 不再增长；
2. **常驻 keep-alive**：每 45s 发一个轻量请求（短 prompt、非 thinking、`max_tokens 4`），防"空闲后首请求慢"，兼做健康检查；它不能替代预热；
3. **shape 稳定后加 `--disable-flashinfer-autotune`**：autotune 缓存落盘并覆盖主要 shape 后重启生效，避免偶发卡顿。

### 4.7 层间依赖

这五层不是平行的：

1. 层①是②③的前提——MTP 要占约 46K/卡的 draft KV，长输出要占 KV，池子不够大后面全白搭；
2. 层③是②的刹车——MTP 把 decode 提速后，verify 会在高并发时和 prefill 抢 GPU，不加阀门，"快引擎"会自己把 TTFT 拉爆（4081 请求时 TTFT 14.36s）；
3. 层②是③的放大器——同样限 `max_tokens=2048`，无 MTP 时一条要 17.6s，有 MTP 只要 9.4s，阀门放行的每个请求引擎都能更快处理完。

## 5. MTP 实测：收益与代价（含口径修正）

### 5.1 配置

```bash
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

参数名纠错：正确写法是 `--speculative-algorithm`，网上很多资料写成 `--speculative-algo`。

### 5.2 不同并发下的表现

| 指标 | 无 MTP | MTP 低并发 (692 req) | MTP 中并发 (2,162 req) | MTP 高并发 (4,081 req) |
|---|---|---|---|---|
| ITL avg | 92.2ms | 34.8ms | 50.6ms | 52.2ms |
| Accept Rate | - | 86.1% | 78.8% | 82.1% |
| Accept Len | - | 3.58 | 3.36 | 3.46 |
| TTFT (stream) | 3.75s | **1.35s** | 8.76s | **14.36s** |
| E2E | 89.2s | ~8s | ~41s | ~24.6s |
| Gen throughput | 112 tok/s | 335 tok/s | 723 tok/s | 497 tok/s |
| Abort | 2.2% | 0.3% | 3.3% | 5.4% |

**核心 tradeoff**：低并发时 MTP 全面赢（TTFT 1.35s、E2E ~8s）；高并发时 verify 挤占 prefill，TTFT 急剧恶化（14.36s），吞吐反而让位。**MTP 与 prefill 是同一批 GPU 上的零和博弈**，不是 bug。

### 5.3 口径修正：别把 ITL 除以 accept_len

一个常见误区是"有效 ITL = ITL avg ÷ accept_len"。**这是重复计算**：`ITL avg` 已经是每个输出 token 的实际流式延迟，MTP 的收益已经包含在里面（一轮 verify ~170ms 出 3.36 个 token，平均到每个 token 就是 ~50ms）。再除一次 accept_len 会把收益虚报 3 倍以上。

验证（高并发 2,162 req 数据）：

```text
648 tok/req × 50.6ms + TTFT 8.76s ≈ 41.5s  ← 与实测 E2E 41.1s 吻合
648 tok/req × 15.1ms + TTFT 8.76s ≈ 18.5s  ← 与实测不符
723 tok/s ÷ 36 并发 ≈ 20 tok/s/req ≈ 50ms/token  ← 一致
```

修正后的真实收益：**ITL 92.2 → 50.6ms（约 1.8x），E2E 89.2 → 41.1s（约 2.2x）**。聚合吞吐 112→723 tok/s 里还混着负载/并发差异，不能全部归因 MTP。

### 5.4 稳态验证

最终配置（MTP + 网关控）跑了 1.5 天、23,324 请求：

```text
E2E = TTFT 5.69s + 645 tok × 40.9ms ≈ 32.1s   ← 与 Dashboard 实测 32.9s 吻合
```

| 指标 | 短期混合 (2,275 req) | 1.5 天稳态 (23,324 req) |
|---|---|---|
| Aborted | 1.3% | **1.56%** |
| TTFT (stream) | 6.18s | **5.69s** |
| ITL avg | 50.5ms | **40.9ms** |
| Cache hit | 36.3% | **45.8%** |
| 负载均衡 | Worker1 热点 | **3 worker 差 <2%** |

长时间运行后 RadixTree 积累了更多前缀，缓存命中率自然提升约 10pp；短期采样里的热点和高峰值在长周期下被摊平。

## 6. 双架构拆分的失败教训：为什么回到 6 卡统一

我们曾尝试按场景拆成两个实例：工具调用 4 卡（TP2 DP2）+ 代码检视 2 卡（TP2 DP1）。实测结果：

| 指标 | 工具调用实例（4卡） | 代码检视实例（2卡） |
|---|---|---|
| 请求占比 | 278 (80%) | 69 (20%) |
| TTFT (stream) | **9.95s** | 2.17s |
| ITL avg | 69.7ms | 32.2ms |
| Running / Queue | Worker0 空转 / Worker1 满载 | 1 / 0 |

两个问题：

1. **DP Router 热点**：同前缀请求全部粘到 Worker 1，4 卡当 2 卡用，TTFT 9.95s 几乎全是排队（round_robin 修复）；
2. **容量错配**：80% 的请求只分到 4 卡，thinking 长输出在 2 卡上排队，利用率约 25% vs 统一 6 卡约 90%。

结论：**没有按类型分流能力时，拆分实例不如统一部署 + 优先级调度 + 限并发**。最终回到 6 卡统一 TP2 DP3 + round_robin。

## 7. 最终生产配置（可直接抄）

### 7.1 服务端

```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3.12 -m sglang.launch_server \
    --model-path /path/to/Qwen3.6-27B-FP8 \
    --served-model-name Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --tp-size 2 --dp-size 3 \
    --load-balance-method round_robin \
    --mem-fraction-static 0.85 \
    --context-length 98304 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8_e5m2 \
    --chunked-prefill-size 4096 \
    --max-running-requests 12 \
    --schedule-policy priority \
    --enable-priority-scheduling --default-priority-value 0 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-backend triton \
    --enable-flashinfer \
    --attention-backend flashinfer \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-cuda-graph \
    --enable-cache-report \
    --enable-metrics \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --log-level info
```

要点：

- `--max-running-requests` 是 **per-worker** 值，DP=3 时 12/worker = 36 总并发；
- 启动日志若出现 `max_running_requests was reduced from the requested ...`，说明并发被 KV 容量钳制，加大参数无效，先收紧 context 或调池子；
- 实测每卡留 6~7.6GB 余量，**峰值 soak 时每卡保持 2~3GB 余量即安全**，不稳就退回 0.82；
- 我们把它封装成了启动脚本，默认值即上面的配置，另带 keep-alive、预热、代理（按类型限并发 + priority）、自适应限流、CSV 监控采集等开关。

### 7.2 网关侧

见 4.4 节的场景参数表。工具调用追求 TTFT，代码检视追求质量，两套参数按请求类型下发。

### 7.3 监控与告警

SGLang 原生暴露 Prometheus 格式的 `/metrics`（需 `--enable-metrics`），无需额外 exporter。核心指标：

| 指标 | 含义 | 参考告警 |
|---|---|---|
| `sglang:num_queue_reqs` | 排队请求数 | >10 持续 5m |
| `sglang:token_usage` | KV 池占用率 | ≥0.92 持续 5m |
| `sglang:time_to_first_token_seconds` | TTFT（分 stream/non-stream） | p90 >10s 持续 10m |
| `sglang:inter_token_latency_seconds` | ITL | 配合 `spec_accept_length` 看 |
| `sglang:cache_hit_rate` | 前缀命中率 | 基线 ~45%，明显下降查缓存被挤爆 |
| `sglang:spec_accept_rate` | MTP 接受率 | <0.65 持续 15m |
| `sglang:num_aborted_requests_total` | abort 数 | abort 率 >5% 持续 10m |

四个推荐面板：容量水位（queue / token_usage / kv_available）、TTFT 分位数（叠 queue_time 区分"排队慢"vs"执行慢"）、MTP 健康（accept rate / accept len / verify calls）、吞吐与缓存。阈值先跑 1~2 天基线再校准。

### 7.4 预热与 keep-alive

```bash
# 预热：等 /v1/models 就绪后，按顺序发（间隔 5~10s）
# 1) 2~3 个短 prompt 非 thinking（tool call 形状，max_tokens 32）
# 2) 2~3 个长 prompt（覆盖线上最长 prompt 的 50% 与 100%，max_tokens 16）
# 3) 1~2 个 thinking 请求（生产采样参数：temperature 0.1 / top_p 0.95 / repetition_penalty 1.05）
# 验证：TTFT 收敛、~/.cache/flashinfer 不再增长、日志不再出现编译/autotune 输出

# keep-alive：每 45s 一个轻量请求（短 prompt、非 thinking、max_tokens 4），兼做健康检查
```

## 8. 可复用的方法论

### 8.1 延迟构成分析先行

先回答"时间花在哪"，再决定优化什么。我们的实际路径：E2E 大头是 decode → 上 MTP；MTP 后瓶颈迁移到 prefill/KV → 控输出、限并发；再往后是负载均衡和冷启动。每一步都有明确的数据支撑。

### 8.2 指标决策树

| 现象 | 判定 | 修法 |
|---|---|---|
| `queue_time_seconds` 占 TTFT 大头 | 排队问题 | 加 `--max-running-requests` / 控到达节奏 |
| `queue_time_seconds` 一直为 0 | 并发已够 | 加参数无意义 |
| 日志出现 "reduced from the requested" | 并发被钳制 | 收紧 context 或调大池子 |
| `cache_hit_rate` 低且前缀重复 | 缓存被挤爆 | `--schedule-policy lpm`、加大池子、收紧 context |
| TTFT 里 prefill 段长 | prefill 慢 | `--chunked-prefill-size 8192`、预热、收紧 context |
| E2E 里 decode 段长 | decode 慢 | CUDA graph、投机解码、业务侧砍输出 |

### 8.3 一次只动一个变量

我们早期同时改了 `mem-fraction-static` 和 `max-running-requests`，导致无法判断各自贡献。后来拆开验证（0.78/8 → 0.78/12 → 0.85/12），才确认主因是并发/排队，缓存容量是次要因素。**每次只改一个参数，用 `/metrics` 判断，再动下一个**。

### 8.4 对 hybrid 模型的特别提醒

- 混合 SSM 模型的 Mamba state 与 KV 争显存，必须用 `extra_buffer` 策略并核算额外开销；
- 并发会被"KV 池 ÷ 2"和"mamba 缓存槽位 ÷ 5"等公式自动钳制，别只看用户填的参数；
- 长输出截断（`max_tokens`）是唯一可靠的输出控制手段，`thinking_budget` 可能不存在或静默忽略；
- 单 token 延迟受 active 参数带宽上限约束，配置层优化做完后，剩下的杠杆只有投机解码和业务侧砍输出。

## 9. 版本与免责说明

- 本文所有数字来自 2026-08 的线上生产（1.5 天稳态 23,324 请求），配置为 SGLang 0.5.17；
- L40S 补丁均为版本相关 workaround，新版本可能已修复，落地前用 `--help` 复核；
- `--disable-cuda-graph` 意味着放弃 decode 段约 10~20% 的 CUDA graph 收益，驱动升级后值得重新评估；
- 不同硬件/负载下结论可能反转，请以自己环境的 A/B 数据为准。

**Qwen3.6 版本支持提醒**：Qwen3.6 的官方支持需要 SGLang ≥ 0.5.10（0.5.11 起进入 Day-0 新模型名单）。在旧版本（如 0.5.1）上跑 Qwen3.6-27B 会输出乱码——这不是模型或 FP8 的问题，而是 27B 的混合线性注意力架构当时完全没有实现，只能走通用 fallback。同一旧版本跑 35B-A3B（经典 MoE 路线）正常，很容易误导排查方向。

## 附录 A：常见口径误区速查

| 误区 | 修正 |
|---|---|
| "有效 ITL = ITL ÷ accept_len" | ITL 已含 MTP 收益，再除是重复计算，会虚报 3 倍以上 |
| "Evict/Prompt 比值高 = 缓存有问题" | 淘汰次数与 prompt token 对比无意义；看命中率 |
| "KV 池满载 = 危险" | 缓存装满才可能命中，重点看命中率而非满不满 |
| "0.78 慢是因为显存不够" | 实测主因是并发/排队（8/worker），拆变量后才确认 |

## 附录 B：E2E 优化杠杆优先级（按收益排序）

| 方向 | 预期收益 | 性质 |
|---|---|---|
| 业务侧控制输出（`max_tokens` / 关 thinking） | E2E 随输出 token 线性下降 | 最快，最便宜 |
| MTP/NEXTN 投机解码 | decode 2x 左右 | 系统侧最大单项 |
| CUDA graph（驱动升级后） | ITL 10~20% | 减固定开销 |
| PD 分离 / 扩容 | 解耦 prefill 与 decode | 结构性，适合更大规模 |

> 一个容易忽视的点：MTP 是"系统侧最大优化"，但 E2E<10s 的物理前提依然是限制输出长度，两者必须一起做。
