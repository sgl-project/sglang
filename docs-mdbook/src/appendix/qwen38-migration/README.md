# 附录 H：Qwen3.8-27B 迁移预研

> 状态：**最终判定已出（2026-08-15）：0.5.17 免重编、免升级，直接部署**（见第 11 节）。Qwen3.8-27B 与 Qwen3.6-27B 的 `architectures` 声明完全相同（`Qwen3_5ForConditionalGeneration`），执行路径逐字节一致；生产路径首选 `Qwen/Qwen3.8-27B-FP8`——该变体含 `mtp.safetensors`（MTP 可用），显存账与 Qwen3.6-27B-FP8 相当；BF16 全量变体（≈55.6GB）无独立 MTP 文件，仅作对照。

## 1. 目标与核心风险

**目标**：评估 Qwen3.8-27B 能否在现有 6 卡 L40S + SGLang 0.5.17 环境上直接上线，避免重走附录 C 的自编译补丁流程。

**核心风险**：Qwen3.8-27B 的架构未知。两种可能：

| 分支 | 架构 | 对现有环境的影响 |
|------|------|-----------------|
| A | 沿用 Qwen3Next / Qwen3.6 混合 SSM 架构 | 0.5.17 模型层大概率直接支持，**近乎无缝** |
| B | 类似 Qwen3.5-35B-A3B（线性注意力 + 稀疏 MoE + 原生视觉） | **需要 SGLang main 分支 → 自编译 + 补丁栈重打** |
| **C（实际）** | **Qwen3.5 dense 家族**（线性注意力 + dense 27B + 原生视觉） | **0.5.17 模型层已支持**（v0.5.10 起），无需升级 |

> 为什么分支 B 会触发自编译：Qwen3.5-35B-A3B 官方 README 明确要求 "SGLang from the main branch of the open-source repository is required"——稳定版（含 0.5.17）不支持其架构，必须用主线源码。而主线 SGLang 通常配套新 sgl-kernel（预编译为 CUDA 12.8，缺 libcudart.so.13），在 L40S/CUDA 12.1 上必须源码编译，并重打附录 C 的 load_utils 路由、sm90 软链接、stubs、flashinfer 对齐等补丁。

## 2. 发布信息追踪

| 项目 | 信息 |
|------|------|
| 正式发布 | 2026-08-02（API 已上线千问平台） |
| 权重开源 | 预计 2026-08 中旬（下周） |
| 开源版本 | Qwen3.8-Max（2.4T 参数 / 95B 激活 / 1M 上下文）+ Qwen3.8-27B |
| 主打能力 | 编程（Coding）+ 办公（Cowork），Agentic 推理 1.5x |
| 对 L40S 的适配 | Max 跑不了（2.4T）；27B 是唯一候选 |

## 3. 架构判定决策树（权重开放后第一步）

**Step 0：只看 config.json，不用下全套权重**

```bash
# 权重开放后，先拉 config.json（几十 KB）
curl -sL https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json | python3 -m json.tool
```

**判定依据**（对照 [qwen3_next.py](../../../python/sglang/srt/models/qwen3_next.py) 的 config 字段）：

| config 字段 | 分支 A（Qwen3Next 沿用） | 分支 B（Qwen3.5-A3B 类新架构） |
|------------|------------------------|-------------------------------|
| `architectures` / `model_type` | 含 `Qwen3Next` / `qwen3_next` | 新名字（如含 `LinearAttention` / `DSA` 之类） |
| 注意力层 | `linear_attn` / mamba 混合字段 | 新线性注意力实现字段 |
| 视觉塔 | 无（纯文本） | 可能含 vision tower（原生多模态） |
| MoE | 无或标准 | 稀疏 MoE（A3B 类路由） |

**决策**：

- 判定为分支 A → 直接进入第 4 节快速验证；
- 判定为分支 B（或不确定）→ 进入第 5 节升级风险评估，**不要在现有生产环境直接试**。

> **Step 0 实测结果（2026-08-14）**：Qwen3.8-27B 判定为**分支 C（Qwen3.5 dense 家族）**，支持已合入 v0.5.10，0.5.17 自带模型层。按决策规则应**直接进入第 4 节快速验证清单**（详细分析见第 8 节）。

## 4. 分支 A：快速验证清单（预计 1 天）

1. **单卡冒烟**（新容器/pod，不动生产）：附录 C 的镜像起单卡，`sglang_start.sh --no-speculative --no-proxy` 加载 Qwen3.8-27B；
2. **启动日志检查**：有无 `Ignore import error` / Unknown field / "unsupported architecture" 警告（对照附录 C 7.1）；
3. **一条真实代码检视请求**：对比 Qwen3.6 vs 3.8 的输出质量、token 消耗、是否符合预期格式；
4. **MTP 验证**：`--speculative-algorithm NEXTN` 能否识别其 mtp 权重（对照 Qwen3.6 的验证流程）；
5. **tool call / reasoning parser**：`--tool-call-parser qwen3_coder`、`--reasoning-parser qwen3` 是否仍适用；
6. **性能对比**：同一批压测请求，对比 TTFT / E2E / abort / 吞吐，参照附录 G 监控面板。

全部通过 → 生产迁移；有任一不通过 → 记录并进入第 5 节。

## 5. 分支 B：升级与自编译风险评估

如果 Qwen3.8-27B 需要 SGLang main 分支，现有环境升级成本：

| 项 | 风险 | 应对 |
|----|------|------|
| SGLang 版本 | ~~main 分支每日变化~~ → **已消除**：Qwen3.5 架构 v0.5.10 起支持（PR #18489），0.5.17 含 qwen3_5 / qwen3_5_text / qwen3_5_mtp | 无需为模型层升级；强制文本加载用 #22867 的 config 字段路径（2026-08-15 修正，见 8.3） |
| sgl-kernel | 新版本预编译为 CUDA 12.8，缺 libcudart.so.13 | 源码编译 sm_89 only（附录 C 2.4 流程重走） |
| load_utils 路由 | main 分支可能重构目录/命名 | 重查 `compute_capability` 路由逻辑（附录 C 2.1/2.2） |
| stubs | 新内核符号集变化 | 重查 undefined symbol 清单（附录 C 2.3） |
| flashinfer | 版本可能升级 | 重查 python/cubin 版本对齐（附录 C 2.5） |
| 安装方式 | pip 安装 build_wheel 嵌套 bug | 沿用 rsync + 删 editable（附录 C 2.6） |
| 多模态权重 | config `language_model_only=false`，含 vision tower；0.5.17 无 #22867，无法跳过视觉塔 | 预研环境实测 vision 显存增量；或在 ≥08-10 的源码（含 08-13 Muse-Glimmer 镜像）上用 `--json-model-override-args '{"language_model_only": true}'` 强制文本加载（见 8.3）；或确认官方是否另发 text-only checkpoint |
| MTP / FP8 | 量化转档可能丢弃 `mtp.safetensors`（Qwen3.6 已有先例） | 权重目录内确认 MTP 权重存在；缺失则需带 MTP 的 FP8 转档流程 |

**决策规则**：

- 分支 B 且 SGLang 支持 PR 已合入 → 在**独立预研环境**（新 pod + 新镜像 tag）做完整验证，验证通过才考虑生产；
- 分支 B 且支持未就绪 → **不迁移**，等官方支持稳定（参照 Qwen3Next 滞后约 1 个月、Qwen3.6 更久的先例）；
- 任何情况下**生产环境不动**：现有 Qwen3.6-27B 服务保持运行，直到新模型在预研环境全绿。

## 6. 并行策略（推荐）

- 生产：Qwen3.6-27B 维持现状，按附录 D 11.11 观察；
- 预研：独立 pod（独立镜像 tag `sglang-l40s-sm89:qwen38-pre`），权重开放后并行验证；
- **第 7 卡：长上下文专用实例（2026-08-14 定）**——原"热备/扩容预留"改为 **Qwen3.8-27B-FP8 + YaRN 的专用长上下文服务**（单卡、独立端口、低并发），同时充当 Qwen3.8 预研环境：冒烟/质量/性能验证通过后直接转生产长上下文实例，不动 6 卡 Qwen3.6 服务；
- 决策点：质量对比（Qwen3.8 在代码检视/tool call 上是否明显优于 3.6）+ 性能（TTFT/E2E 不劣化）+ 部署成本（分支 A 低 / 分支 B 高），三者综合后才动生产。

**第 7 卡启动要点**（命令见附录 D 思路 + 本节）：

```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

CUDA_VISIBLE_DEVICES=6 python3.12 -m sglang.launch_server \
    --model-path /path/to/Qwen3.8-27B-FP8 \
    --served-model-name qwen38-longctx \
    --host 0.0.0.0 --port 8001 \
    --tp-size 1 --dp-size 1 \
    --context-length 524288 \
    --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    --mem-fraction-static 0.85 \
    --max-running-requests 2 \
    --chunked-prefill-size 8192 \
    --kv-cache-dtype fp8_e5m2 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm NEXTN --speculative-num-steps 2 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    --attention-backend flashinfer \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-cuda-graph \
    --enable-metrics --log-level info
```

约束：① `--context-length` ~~先 512K~~ **改为 320K**（2026-08-15 精算修正：512K 的 KV 超单卡容量，见 9.2；1M 时再调，prefill 5~15 分钟）；② **不要挂在 10s 超时代理后**，客户端直连 8001 并放宽超时（prefill 分钟级）；③ 预热必须含代表性长 prompt（100K+），否则首个真实请求 JIT/autotune + 可能 OOM；④ 定位为异步批量任务服务（小时级吞吐），`--max-running-requests 2` 起步。

## 7. 执行记录（权重开放后填写）

| 日期 | 事项 | 结果 |
|------|------|------|
| 2026-08-05 | 预研计划建立 | 待权重开放 |
| 2026-08-14 | config.json 架构判定 | **分支 C：Qwen3.5 dense 家族**（详见第 8 节） |
| 2026-08-14 | 权重目录核对 | **无独立 `mtp.safetensors`**；BF16 多模态 18 分片 ≈55.6GB；待查 `model.safetensors.index.json` 是否内嵌 `mtp.*` 键 |
| 2026-08-14 | FP8 变体核对 | **`Qwen/Qwen3.8-27B-FP8` 含 `mtp.safetensors`**（用户核对）→ MTP 可用，显存账与 Qwen3.6-27B-FP8 相当 |
| （待填） | 单卡冒烟 | |
| （待填） | MTP / parser 验证 | |
| （待填） | 质量与性能对比 | |
| （待填） | 迁移决策 | |

## 8. Step 0 结果：config 架构判定（2026-08-14）

### 8.1 数据来源

官方 `Qwen/Qwen3.8-27B` 的 `config.json`。本次分析环境网络受限（DNS 不通），以用户提供的官方 config 为准；建议权重下载后在目录内用 `python3 -c "import json,pathlib; print(json.load(open(pathlib.Path('config.json')))['architectures'])"` 复核一次。

### 8.2 关键字段与判定

| config 字段 | 值 | 含义 |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | **Qwen3.5 架构家族**（多模态入口），不是 Qwen3Next/Qwen3.6，也不是全新架构 |
| `model_type` | `qwen3_5` / `qwen3_5_text` | 同上；SGLang 按 Qwen3.5 处理 |
| `layer_types` | 64 层 = 3×`linear_attention` + 1×`full_attention` 循环 16 次 | GatedDeltaNet 混合线性注意力，`full_attention_interval=4` |
| MoE 字段 | 无（dense：hidden 5120 / 64 层 / intermediate 17408） | **dense 27B**，不是 35B-A3B 的稀疏 MoE |
| 视觉塔 | `vision_config` + image/video token + mrope（section [11,11,10]） | **原生多模态**，`language_model_only=false` |
| MTP | `mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false` | 与 SGLang `Qwen3_5ForCausalLMMTP` draft 路径匹配 |
| 上下文 | `max_position_embeddings=262144` | 与 Qwen3.6 相同，现有 98304 配置可直接沿用 |

**判定：分支 C——Qwen3.5 dense 家族（27B 类）**。预研原分支 B 描述的"线性注意力 + 稀疏 MoE + 原生视觉"中，**MoE 部分不成立**（Qwen3.8-27B 是 dense），其余特征全部命中。

### 8.3 版本线核对（关键修正）

| 事项 | 结论 |
|---|---|
| Qwen3.5 架构支持 | v0.5.10 起（PR #18489 `model: support Qwen3.5`）→ **0.5.17 自带 qwen3_5.py / qwen3_5_text.py / qwen3_5_mtp.py** |
| 文本单模态 checkpoint | v0.5.17 已支持（PR #32401） |
| 强制文本加载（跳过视觉塔） | **#22867（2026-08-10 合入 main，不在 0.5.17）**：读 HF config 的 `language_model_only` 字段，用法 `--json-model-override-args '{"language_model_only": true}'`。⚠️ 区别于 CLI 旗标 `--language-model-only`——后者由 Muse Glimmer 支持（#34262，08-11）引入，白名单仅 `MuseGlimmerForConditionalGeneration`，对 Qwen3_5 直接 ValueError（2026-08-15 核实，当前 main 仍如此） |
| 原预研假设"0.5.17 不支持 Qwen3.5、必须 main + 重打补丁" | **不成立**（模型层 0.5.17 已支持） |

### 8.4 迁移影响

- **模型层：0.5.17 可直接加载**，无需升级 SGLang，也无需为 Qwen3.5 重打 L40S 补丁栈（Qwen3.6 与 Qwen3.5 同为混合线性注意力，kernel 路径一致）；
- **多模态权重**：`language_model_only=false` 意味着完整 checkpoint 带 vision tower；0.5.17 的 qwen3_5 不会跳过视觉塔，显存会增大（vision 27 层 / hidden 1152 / patch 16 / temporal 2，估算 1~3GB 级）。若只想跑文本：用 ≥08-10 的源码 + `--json-model-override-args '{"language_model_only": true}'`（#22867，详见 8.3；CLI 旗标 `--language-model-only` 对 Qwen3_5 无效），或确认官方是否另发 text-only checkpoint；
- **MTP**：config 声明 1 层 MTP，SGLang draft 路径匹配（`Qwen3_5ForConditionalGeneration` → `Qwen3_5ForCausalLMMTP`，`num_nextn_predict_layers=1`）。**生产路径用 FP8 变体（含 `mtp.safetensors`），MTP 直接可用**；BF16 变体无独立 MTP 文件，若需用 BF16 再查 index 是否内嵌 `mtp.*` 键：
  ```bash
  # 仅 BF16 变体需要：有输出 = MTP 权重内嵌在主分片，0.5.17 可直接验证
  rg -o '"mtp\.[^"]+"' model.safetensors.index.json | head
  ```
  若 index 中无任何 `mtp.` 键 → BF16 变体无 MTP（Qwen3.6 实测 MTP 带来 E2E 约 2.2x 收益，长输出场景影响明显）；
- **权重精度与显存**：**生产首选 FP8 变体**（含 MTP，显存账与 Qwen3.6-27B-FP8 相当：TP2 下每卡权重 ~14GB，96K + DP3 + mem 0.85 可直接沿用）。BF16 全量（≈55.6GB，约 27.8B 参数含 vision）在 TP2 下每卡权重约 **27.8GB**，6×L40S 上会显著挤压 KV/state 容量，仅作对照或需收紧 context/并发；
- **为什么 BF16 不适合生产（L40S TP2 评估）**：① 显存——BF16 每卡权重 27.8GB vs FP8 ~14GB，mem 0.85 预算下 KV/state 池大约腰斩，96K + 12 并发/worker 的稳态配置保不住；② 带宽——L40S 是带宽受限卡（864GB/s），decode 每步读取字节翻倍（55.6GB vs 27.8GB），ITL 量级翻倍（FP8 实测 92ms，BF16 预计 130ms 以上）；③ MTP——BF16 无独立 mtp 权重，即使 index 内嵌，verify 同样吃双倍带宽。结论：**BF16 只用于质量对照（低并发、不追求延迟），生产用 FP8**；
- **上下文扩展到 1M（官方 README YaRN 方案）**：`--json-model-override-args` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` + `rope_type: yarn / factor 4.0` 在 **0.5.17 全部支持**（server_args/environ/rotary factory 均已核对，含 mrope+yarn 的 `YaRNScalingMRotaryEmbedding`）。6 卡 L40S 上 1M 是另一个运行体系：单序列 KV（fp8）≈2GB、prefill 分钟级、并发个位数、一次性长文档 radix 命中趋近 0。**单卡 L40S 评估**：FP8 可装（权重 ~28GB），BF16 装不下（55.6GB）；但单卡 1M prefill 约 5~15 分钟、decode 92ms/步——**只适合离线批量长文档分析，不适合交互**。（2026-08-15 修正：此处"池子 ~4~5M token、1M 序列可同时放 2~5 条"及上文"单序列 KV(fp8)≈2GB"系估算错误，精算见 9.2——1M 序列 fp8 KV 实为 ~32GB，单卡纯显存放不下。）仅当确有 >100K 需求时，开独立低并发服务实测（100K/500K/1M 的 prefill 耗时、YaRN 质量衰减、池子并发上限），不要动现有 98K 生产配置；
- **L40S 组合**：沿用 `--mamba-radix-cache-strategy extra_buffer` / `--mamba-backend triton` / `--disable-cuda-graph`；建议显式 `--mamba-ssm-dtype bfloat16`（config 默认 float32，测试与 Qwen3.6 经验均用 bfloat16，需验证精度）；
- **Parser**：`--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` 大概率沿用，冒烟时验证（Qwen3.8 主打 coding/cowork，工具调用格式可能微调）。

### 8.5 下一步（进入第 4 节快速验证清单）

1. 预研 pod（镜像 tag `sglang-l40s-sm89:qwen38-pre`，**不动生产**）：0.5.17 单卡加载 Qwen3.8-27B，`--no-speculative --no-proxy` 冒烟；
2. 启动日志查 `Ignore import error` / Unknown field / 架构警告；
3. 权重目录核对：FP8 变体目录内确认 `mtp.safetensors` 存在（生产路径前提）；BF16 变体仅对照，如需用再查 index 内嵌 `mtp.*` 键；
4. FP8 变体单卡冒烟，实测显存占用，确认 96K + DP3 + mem 0.85 沿用无压力；
5. 按清单 3~6 完成质量、MTP、parser、性能对比后填第 7 节执行记录。

## 9. 上游 sync 盘点与单卡显存精算修正（2026-08-15）

### 9.1 上游同步盘点

fork main 已合并 sgl-project/sglang 最新 main（behind 0）。Qwen3.8-27B 发布前后，上游针对该模型的提交**只有文档，无模型层定制**：

| 提交 | 内容 | 对我们的意义 |
|------|------|-------------|
| #34860 | 新增 Qwen3.8-27B cookbook | 官方部署配方可参考；明确"serving 相关架构与 Qwen3.6-27B 一致"，印证第 8 节判定 |
| #34863 | GB300 benchmark 数据 | 与我们硬件无关 |
| **#34560** | **修复 Qwen3.5 架构家族 MTP + HiCache 同开启动失败** | 9.4 的 HiCache 路线或生产开分层缓存的**前提**，必须确认代码含此提交 |

官方 cookbook 补充信息：checkpoint 共三档——BF16 / FP8（blockwise）/ NVFP4（RadixArk 出，**Blackwell 专属，SM89 不可用**）；官方定位单卡可跑（H200 / RTX PRO 6000 级）；唯一需要重新核算的 sizing 参数是 `--mamba-full-memory-ratio`（cookbook 附计算器）。

### 9.2 单卡显存精算（修正第 6 节 512K 与第 8.4 节池子估算）

hybrid GDN 架构中，48 层 GDN 的状态定长（~146MB/req，不占 KV 池），**只有 16 层 full attention 产生随上下文线性增长的 KV**：

```
每 token KV = 16 层 × 2(K+V) × 4 KV头 × 256 head_dim = 32,768 元素
  fp8 KV → 32 KB/token；bf16 KV → 64 KB/token
单卡 L40S（46GB）：静态池（mem-fraction 0.85）≈ 39GB − FP8 权重 ~28GB ≈ 11GB
  → KV 池上限 ≈ 34 万 token
```

| 目标上下文 | FP8 KV 占用 | + 权重 28GB | 单卡 L40S 判定 |
|---|---|---|---|
| 1,000,000 | 32 GB | 60 GB | **不可行**（纯显存路线） |
| 524,288（512K） | 16 GB | 44 GB | **不可行**（修正第 6 节命令） |
| 327,680（320K） | 10 GB | 38 GB | **可行（推荐验证目标）** |
| 262,144（原生） | 8 GB | 36 GB | 可行，但走不到 YaRN 扩展路径 |

**修正两处既有估算**：① 第 6 节命令的 `--context-length 524288` 超单卡容量，已改为 320K；② 第 8.4 节"池子 ~4~5M token、1M 序列可同时放 2~5 条"与"单序列 KV(fp8)≈2GB"均系估算错误——1M 序列 fp8 KV 实为 ~32GB，单卡纯显存连一条都放不下。

### 9.3 长上下文验证设计（大海捞针对照实验）

验证 YaRN 扩展是否真实生效，做对照实验，各发一次 ~30 万 token 的"大海捞针"请求（拼接长文档，埋入随机事实后提问）：

1. **针必须埋在 262K 之后**——埋在原生训练范围内则实验无区分度；
2. 带 YaRN override 启动 → 答对，扩展路径工作正常；
3. （对照）只加 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`、不加 `--json-model-override-args` → RoPE 直接外推到未训练区域，预期答错或乱码，反向证明第 2 步不是蒙的。

预期管理：GDN 层 prefill 是串行 scan，30 万 token 输入单卡 TTFT 为分钟级，属架构特性而非故障；decode 有 MTP，体验与生产一致。

### 9.4 单卡挑战 1M 的路径（2026-08-15 源码核实后重写）

**先纠正本节此前的错误结论**：原稿认为 HiCache 分层缓存是单卡 1M 的路径，源码核实后**不成立**。HiCache 是**前缀缓存而非虚拟内存**：命中主机内存的前缀会在请求准入前整体载回 GPU（`schedule_policy.py` 的 `needs_host_load_back`），且存在硬上限 `max_req_len = min(context_len-1, 池子总 token 数-1)`（`tp_worker.py:411-416`，池子不够直接 assert 拒绝）——**decode 要求完整 KV 驻留 GPU，源码中不存在部分驻留的 decode 路径**。运行中请求持有 tree lock，KV 也不会被中途降级。HiCache 只对"后续请求共享长前缀"的多轮场景有用，对单条 1M 请求无能为力。

源码核实后单卡 1M 的**唯一路径是权重卸载** `--cpu-offload-gb`：

- 原理：把 ~20GB FP8 权重卸载到主机内存，腾出显存给 32GB 的 1M KV 池；无模型家族限制（`utils/offloader.py` OffloaderV1）；
- 代价：**每个 decode step 和每个 prefill chunk 都要把卸载的权重经 PCIe 重新上传**——decode 约 0.7~1s/token（PCIe gen4），1M prefill 累计 ~2.5TB 主机流量；只适合离线批量分析，不可交互；
- 风险：OffloaderV1 × FP8 量化权重 × GDN triton 内核的组合**无任何测试覆盖**，CUDA graph 捕获大概率失败，需 `--disable-cuda-graph`；
- 配套：`--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.90 --chunked-prefill-size 8192 --max-running-requests 1 --max-mamba-cache-size 4`。

务实路线不变：先按 9.3 跑通 320K 验证；**生产级 1M 用 TP=2 起步**（权重分摊后 KV 池自然够），不在单卡上硬扛。

## 10. 同步后源码适配与参数复用结论（2026-08-15 源码核实）

### 10.1 同步 main 后的 L40S（SM89）重编译评估

| 0.5.17 时代的 workaround | 当前 main 上的状态 |
|---|---|
| `load_utils.py` sm89 路由修复 | bug 仍在（cc=89 仍路由到 sm100 目录），但两个架构目录现在用同一源码+同一 gencode 列表构建，**源码编译含 sm_89 cubin，补丁可不再打**；装预编译 wheel 才需要 |
| `sm90` 目录软链接 | **已废弃**——上游 CMake 现在统一 `OUTPUT_NAME "common_ops"`，命名错配不存在了 |
| 4 个 es_* 符号的 `LD_PRELOAD` stubs | **已废弃**——4 个符号全部改为无条件定义 + `TORCH_CHECK` 兜底体，sm89 构建可直接 `import sgl_kernel`；且旧 stubs.cc 签名已过期（`fp8_blockwise_scaled_grouped_mm` 加了 layout 参数），直接丢弃 |
| 源码编译 sglang-kernel | **必须重编**：版本钉到 `0.4.6.post1`，源码已移到 `python/sglang/kernels/aot`；CMake 仍接受 CUDA 12.1，`ENABLE_BELOW_SM90` 默认 ON（自动带 sm_89 gencode） |
| flashinfer 0.6.13 | 运行期断言要求 **≥0.6.17**（仅 flashinfer backend 路径）；可用 skip 环境变量绕过，但 0.6.13 与新 srt 的 API 漂移是真实风险；0.6.17 以 `[cu13]` 分发，cu121 兼容性未知 |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | 仍需保留（除非同时升级 kernel 0.4.6.post1 + flashinfer 0.6.17） |
| `--disable-cuda-graph` | 不再被架构强制；上游已为 Qwen3.5/Qwen3Next GDN 默认开启 piecewise CUDA graph——首发保留旧参数，A/B 验证后再考虑打开 |

**最大风险是基线整体上移**：当前 main 的开发基线为 `torch==2.13.0` / `flashinfer 0.6.17[cu13]` / `cuda-python>=13.0` / `transformers==5.12.1` / Docker 基座 CUDA 13（上游构建目标已不含 12.1）。现有 torch 2.7.0+cu121 没有硬性 import 阻断，但完全在上游 CI 覆盖之外。若冒烟出现 torch API 相关报错，现实选项是升级到 torch ≥2.8 + CUDA 12.8/12.9（还能直接用官方 wheel，省掉大部分补丁），而不是在 cu121 上硬扛。

### 10.2 Qwen3.6-27B 参数复用审计（对 Qwen3.8-27B）

逐参数对当前源码核实，旧生产命令**无失效/改名参数**，但有几个必须知道的差异：

| 参数 | 判定 | 说明 |
|---|---|---|
| `--mamba-backend triton` / `--mamba-radix-cache-strategy extra_buffer` | 直接复用 | extra_buffer 是投机解码下唯一兼容策略；Qwen3_5 在白名单内 |
| `--speculative-algorithm NEXTN` + 3/1/4 | 直接复用 | NEXTN 仍是 EAGLE 的合法别名；MTP 映射 `Qwen3_5ForConditionalGeneration → Qwen3_5ForCausalLMMTP` 已确认。**注意**：MTP + flashinfer backend 需要 flashinfer > 0.6.15.post1（prefill plan 支持 `uniform_q_len`），否则 spec 路径改走 `--attention-backend triton` |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` | 直接复用 | 两个 detector 均在，官方 cookbook 确认 qwen3_coder |
| `--kv-cache-dtype fp8_e5m2` | 可用但偏离官方配方 | 官方 3.8-27B 配方全部用 auto（bf16 KV），FP8 KV 只文档化 e4m3；e5m2 仍合法，建议按 e4m3 重新标定 |
| `--mem-fraction-static 0.85` | **需要调整** | 3.8-27B 是多模态权重，vision tower 默认常驻加载，且 VLM 会触发 mem-fraction **自动下调**（`server_args.py` `adjust_mem_fraction_for_vlm`），显式 0.85 会被静默改低 |
| `--language-model-only` | **不可用** | 旗标存在，但 `_handle_language_model_only` 白名单不含 Qwen3_5 架构，会直接 ValueError；跳过视觉塔的通用化还在上游未合并分支上。当前唯一跳过的办法是改 checkpoint config 声明 `language_model_only: true`（代价：多模态请求被拒） |
| `--context-length 98304` | 直接复用 | 3.8 原生 262,144 相同 |
| `--disable-cuda-graph` | 复用（首发） | 官方配方已 graphs-on（该架构支持 breakable prefill graph），可作为后续 A/B 项 |
| TP=2 DP=3 布局 | 直接复用 | 官方只出单卡配方，多卡布局是自己的加法 |
| `--mamba-full-memory-ratio` | **新增必算** | 默认 0.9 会过度预留 KV、悄悄钳住并发；按官方公式 `ratio = (S+D)×state_bytes / (L×kv_per_token)` 用自有负载重算（S=5 extra_buffer 槽位、D=draft token 数、state ≈154MB fp32 / 78MB bf16） |
| `--chunked-prefill-size 2048` | 新增建议 | 混合负载下 8192 chunk 会让 decode 卡顿 ~600ms |

另外明确一点：3.8-27B 是 dense，**没有 MoE 路径**，3.6 时代关于 triton MoE / expert-specialization 符号的顾虑对这个 checkpoint 不适用。FP8 blockwise 的最低算力要求是 sm80，L40S 无问题。

### 10.3 单卡 1M 结论（修正 9.4 后的最终版）

- HiCache **不能**让单条 1M 请求跑起来（前缀缓存 ≠ 虚拟内存，decode 要求完整 KV 驻留 GPU，见 9.4）；
- 单卡唯一路径是 `--cpu-offload-gb ~20` 权重卸载换 KV 空间，代价是 decode ~1s/token 级，仅限离线批量；
- 无卸载的单卡上限约 320K~350K token；
- 生产级 1M：TP=2 起步（两张卡分摊权重后 KV 池自然够 1M），这是结构性正解。

## 11. 最终判定：0.5.17 免重编直接部署（2026-08-15）

**结论：不重新编译、不升级 SGLang，现有 0.5.17 生产环境可直接部署 Qwen3.8-27B。CUDA 12.1 不可变不构成障碍。**

### 11.1 判定依据

1. **执行路径逐字节一致**：Qwen3.8-27B 与 Qwen3.6-27B 的 `config.json` 声明完全相同的 `architectures: ["Qwen3_5ForConditionalGeneration"]`，且文本/视觉结构参数一致——跑 3.8 就是跑 3.6 的同一份代码：同一个模型类、同一套 triton GDN 内核（现有 sm_89 编译的 sglang-kernel 0.4.5 已覆盖）、同一条 FP8 blockwise 量化路径（最低算力 sm80）、同一个 MTP 映射（`Qwen3_5ForCausalLMMTP`）。
2. **0.5.17 tag 核实**：`v0.5.17` 源码树中 `qwen3_5.py` / `qwen3_5_text.py` / `qwen3_5_mtp.py` 均在位（架构支持自 v0.5.10 起）。
3. **dense 无 MoE**：expert-specialization 符号 / stubs 的顾虑对该 checkpoint 不会触发（且现有环境 stubs 已就位）。
4. **上游同步盘点**：3.8-27B 发布前后上游仅有文档提交，无模型代码——新 main 没有部署 3.8-27B 必需的任何东西（第 10.1 节的重编译与基线上移风险，只在"要升级 main"时才需要面对）。

### 11.2 部署行动项（仅三项）

1. **权重用 `Qwen/Qwen3.8-27B-FP8`**：已确认含 `mtp.safetensors`，MTP 直接可用；
2. **vision tower 常驻无需额外处理**：3.6 生产 config 同样 `language_model_only: false`，现有环境已在承受该开销，显存账沿用（0.5.17 无跳过视觉塔的旗标）；
3. **按第 4 节清单单卡冒烟**：启动日志检查 → 一条真实请求 → `qwen3` / `qwen3_coder` parser 验证 → MTP 生效确认（看 accept length）。预计当天出结果。

冒烟全绿后，生产切换只是改 `--model-path` 和 `--served-model-name`，TP=2 DP=3 及其余参数全部沿用（复用审计明细见 10.2）。

### 11.3 备选部署载体：08-13 Muse-Glimmer 镜像

08-13 基于新源码构建的 Muse-Glimmer 镜像**同样可以直接部署 Qwen3.8-27B**，且比 08-04 生产镜像多一个"纯文本模式"选项：

- 该镜像源码晚于 08-10 → 含 #22867，可用 `--json-model-override-args '{"language_model_only": true}'` 跳过视觉塔、回收 1~3GB 显存（8.3 已修正：CLI 旗标 `--language-model-only` 白名单仅限 MuseGlimmer，对 Qwen3_5 无效，勿用）；
- 该镜像同样含 qwen3_5 全套模型实现，3.8-27B 加载无障碍；
- **前提**：确认该镜像的 sglang-kernel / flashinfer 是按 L40S（sm_89、CUDA 12.1）环境编译的——若 Muse-Glimmer 镜像面向其他硬件构建，则不可用，仍走 08-04 生产镜像；
- 注意新源码基线风险（10.1）：kernel 0.4.6.post1 / flashinfer ≥0.6.17 断言，MTP + flashinfer 需 > 0.6.15.post1，否则 spec 路径走 `--attention-backend triton`。

选择建议：**生产迁移仍以 08-04 镜像（0.5.17）为首选**——补丁栈已验证、风险最小；08-13 镜像作为"需要纯文本模式回收显存"时的备选，启用前先按 10.1 核对基线。

## 12. 生产拓扑（TP2 DP3）挑战方案：更长上下文 / 更高并发（2026-08-17）

### 12.1 预算前提

Qwen3.8-27B-FP8 与 3.6-27B 的显存账**完全相同**（同 `architectures`、同层数/头数/维度、FP8 权重同 ~28GB、fp8 KV 同 32KB/token)，6 卡 TP2 DP3 下每 worker（2 卡）KV+mamba 池约 **130 万~140 万 token**。**精确值以启动日志 `max_total_num_tokens` 为准**，配置按 `context × 并发 ≤ 实测值 × 0.9` 收口（10% 留给 mamba extra_buffer 与波动）。

### 12.2 挑战矩阵（每 worker）

| 挑战目标 | `--context-length` × `--max-running-requests` | 额外要求 | 风险 |
|---|---|---|---|
| 现状基线 | 98,304 × 12 | — | — |
| **原生上限** | **262,144 × 5** | 无——262K 是原生训练长度，**不加 YaRN、不加环境变量** | 低，仅并发 12→5 |
| 均衡档 | 131,072 × 10 | 无 | 低 |
| **超原生 512K** | 524,288 × 2 | YaRN 全套（§6：环境变量 + rope override，factor 4.0） | 中——>262K 靠 YaRN 外推，质量需抽测 |
| **提并发** | 98,304 × 14~16 | 建议配代理分桶限流（Qwen3.6-27B.md 8.7） | 中——3.6 生产数据：并发>12 TTFT 急剧恶化 |

另有一个独立 A/B 项：`--mem-fraction-static 0.85 → 0.88`，每 worker 多约 8 万 token 余量。**单独验证，不与上表改动合并。**

### 12.3 验证路径（与冒烟合并，不浪费机器时间）

1. **第 7 卡单卡先验**：按 §4 清单冒烟时把单卡拉到 `--context-length 262144`，发一条 20 万 token 真实请求，确认 3.8 长上下文输出正常；
2. **单 worker 灰度**：生产集群摘一个 worker（TP2）换 3.8 + 目标配置，打真实流量，对比 3.6 同形状的 TTFT / E2E / abort / accept length；
3. **全量定稿**：长上下文优先 → 262,144 × 5；吞吐优先 → 维持 98,304、代理配合把并发放到 14~16；
4. **512K 档最后**：确认有 >262K 真实需求才上，且先做大海捞针抽测（针埋 262K 之后，设计见 9.3）。

### 12.4 操作细节

- **核对 sizing 参数**：新代码有 `--mamba-full-memory-ratio`（默认 0.9，按 mamba state 与 KV 比例钳并发，公式见官方 cookbook）；0.5.17 镜像先 `--help` 确认旗标是否存在，没有则忽略；
- **提并发看尾部**：恶化信号是 `num_queue_reqs` 和 TTFT p99，不是平均吞吐——平均吞吐会好看，尾巴会烂；
- **结果回填**：各档实测数据填 §7 执行记录表。
