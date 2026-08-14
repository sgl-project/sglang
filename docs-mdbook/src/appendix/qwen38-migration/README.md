# 附录 H：Qwen3.8-27B 迁移预研

> 状态：**config 架构判定完成 + 权重核对完成**（2026-08-14）。Qwen3.8-27B 判定为 **Qwen3.5 dense 家族架构**（详见第 8 节）。核心结论：SGLang **v0.5.10 起已支持该架构，生产 0.5.17 无需为模型层升级**；**生产路径首选 `Qwen/Qwen3.8-27B-FP8`**——该变体含 `mtp.safetensors`（MTP 可用），显存账与 Qwen3.6-27B-FP8 相当；BF16 全量变体（≈55.6GB）无独立 MTP 文件，仅作对照。

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
| SGLang 版本 | ~~main 分支每日变化~~ → **已消除**：Qwen3.5 架构 v0.5.10 起支持（PR #18489），0.5.17 含 qwen3_5 / qwen3_5_text / qwen3_5_mtp | 无需为模型层升级；如需 `--language-model-only` 强制文本加载多模态权重，需等含 #22867 的版本 |
| sgl-kernel | 新版本预编译为 CUDA 12.8，缺 libcudart.so.13 | 源码编译 sm_89 only（附录 C 2.4 流程重走） |
| load_utils 路由 | main 分支可能重构目录/命名 | 重查 `compute_capability` 路由逻辑（附录 C 2.1/2.2） |
| stubs | 新内核符号集变化 | 重查 undefined symbol 清单（附录 C 2.3） |
| flashinfer | 版本可能升级 | 重查 python/cubin 版本对齐（附录 C 2.5） |
| 安装方式 | pip 安装 build_wheel 嵌套 bug | 沿用 rsync + 删 editable（附录 C 2.6） |
| 多模态权重 | config `language_model_only=false`，含 vision tower；**0.5.17 的 qwen3_5 不识别 `--language-model-only`**（#22867 在 main，2026-08-10） | 预研环境实测 vision 显存增量；或等 #22867 版本后强制文本加载；或确认官方是否另发 text-only checkpoint |
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

约束：① `--context-length` 先 512K（1M 时再调，prefill 5~15 分钟）；② **不要挂在 10s 超时代理后**，客户端直连 8001 并放宽超时（prefill 分钟级）；③ 预热必须含代表性长 prompt（100K+），否则首个真实请求 JIT/autotune + 可能 OOM；④ 定位为异步批量任务服务（小时级吞吐），`--max-running-requests 2` 起步。

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
| `--language-model-only` 对 Qwen3.5 生效 | 2026-08-10 合入 main（PR #22867），**不在 0.5.17** |
| 原预研假设"0.5.17 不支持 Qwen3.5、必须 main + 重打补丁" | **不成立**（模型层 0.5.17 已支持） |

### 8.4 迁移影响

- **模型层：0.5.17 可直接加载**，无需升级 SGLang，也无需为 Qwen3.5 重打 L40S 补丁栈（Qwen3.6 与 Qwen3.5 同为混合线性注意力，kernel 路径一致）；
- **多模态权重**：`language_model_only=false` 意味着完整 checkpoint 带 vision tower；0.5.17 的 qwen3_5 不会跳过视觉塔，显存会增大（vision 27 层 / hidden 1152 / patch 16 / temporal 2，估算 1~3GB 级）。若只想跑文本：等含 #22867 的版本用 `--language-model-only`，或确认官方是否另发 text-only checkpoint；
- **MTP**：config 声明 1 层 MTP，SGLang draft 路径匹配（`Qwen3_5ForConditionalGeneration` → `Qwen3_5ForCausalLMMTP`，`num_nextn_predict_layers=1`）。**生产路径用 FP8 变体（含 `mtp.safetensors`），MTP 直接可用**；BF16 变体无独立 MTP 文件，若需用 BF16 再查 index 是否内嵌 `mtp.*` 键：
  ```bash
  # 仅 BF16 变体需要：有输出 = MTP 权重内嵌在主分片，0.5.17 可直接验证
  rg -o '"mtp\.[^"]+"' model.safetensors.index.json | head
  ```
  若 index 中无任何 `mtp.` 键 → BF16 变体无 MTP（Qwen3.6 实测 MTP 带来 E2E 约 2.2x 收益，长输出场景影响明显）；
- **权重精度与显存**：**生产首选 FP8 变体**（含 MTP，显存账与 Qwen3.6-27B-FP8 相当：TP2 下每卡权重 ~14GB，96K + DP3 + mem 0.85 可直接沿用）。BF16 全量（≈55.6GB，约 27.8B 参数含 vision）在 TP2 下每卡权重约 **27.8GB**，6×L40S 上会显著挤压 KV/state 容量，仅作对照或需收紧 context/并发；
- **为什么 BF16 不适合生产（L40S TP2 评估）**：① 显存——BF16 每卡权重 27.8GB vs FP8 ~14GB，mem 0.85 预算下 KV/state 池大约腰斩，96K + 12 并发/worker 的稳态配置保不住；② 带宽——L40S 是带宽受限卡（864GB/s），decode 每步读取字节翻倍（55.6GB vs 27.8GB），ITL 量级翻倍（FP8 实测 92ms，BF16 预计 130ms 以上）；③ MTP——BF16 无独立 mtp 权重，即使 index 内嵌，verify 同样吃双倍带宽。结论：**BF16 只用于质量对照（低并发、不追求延迟），生产用 FP8**；
- **上下文扩展到 1M（官方 README YaRN 方案）**：`--json-model-override-args` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` + `rope_type: yarn / factor 4.0` 在 **0.5.17 全部支持**（server_args/environ/rotary factory 均已核对，含 mrope+yarn 的 `YaRNScalingMRotaryEmbedding`）。6 卡 L40S 上 1M 是另一个运行体系：单序列 KV（fp8）≈2GB、prefill 分钟级、并发个位数、一次性长文档 radix 命中趋近 0。**单卡 L40S 评估**：FP8 可装（权重 ~28GB，池子 ~4~5M token，1M 序列可同时放 2~5 条），BF16 装不下（55.6GB）；但单卡 1M prefill 约 5~15 分钟、decode 92ms/步——**只适合离线批量长文档分析，不适合交互**。仅当确有 >100K 需求时，开独立低并发服务实测（100K/500K/1M 的 prefill 耗时、YaRN 质量衰减、池子并发上限），不要动现有 98K 生产配置；
- **L40S 组合**：沿用 `--mamba-radix-cache-strategy extra_buffer` / `--mamba-backend triton` / `--disable-cuda-graph`；建议显式 `--mamba-ssm-dtype bfloat16`（config 默认 float32，测试与 Qwen3.6 经验均用 bfloat16，需验证精度）；
- **Parser**：`--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` 大概率沿用，冒烟时验证（Qwen3.8 主打 coding/cowork，工具调用格式可能微调）。

### 8.5 下一步（进入第 4 节快速验证清单）

1. 预研 pod（镜像 tag `sglang-l40s-sm89:qwen38-pre`，**不动生产**）：0.5.17 单卡加载 Qwen3.8-27B，`--no-speculative --no-proxy` 冒烟；
2. 启动日志查 `Ignore import error` / Unknown field / 架构警告；
3. 权重目录核对：FP8 变体目录内确认 `mtp.safetensors` 存在（生产路径前提）；BF16 变体仅对照，如需用再查 index 内嵌 `mtp.*` 键；
4. FP8 变体单卡冒烟，实测显存占用，确认 96K + DP3 + mem 0.85 沿用无压力；
5. 按清单 3~6 完成质量、MTP、parser、性能对比后填第 7 节执行记录。
