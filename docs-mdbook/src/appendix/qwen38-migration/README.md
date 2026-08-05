# 附录 H：Qwen3.8-27B 迁移预研

> 状态：**预研中**（2026-08-05）。Qwen3.8 已发布（API 上线），Qwen3.8-Max / Qwen3.8-27B 权重预计 **2026-08 中旬开源**（ModelScope / Hugging Face）。本文是独立预研目录，权重开放后按清单执行并记录结果。

## 1. 目标与核心风险

**目标**：评估 Qwen3.8-27B 能否在现有 6 卡 L40S + SGLang 0.5.17 环境上直接上线，避免重走附录 C 的自编译补丁流程。

**核心风险**：Qwen3.8-27B 的架构未知。两种可能：

| 分支 | 架构 | 对现有环境的影响 |
|------|------|-----------------|
| A | 沿用 Qwen3Next / Qwen3.6 混合 SSM 架构 | 0.5.17 模型层大概率直接支持，**近乎无缝** |
| B | 类似 Qwen3.5-35B-A3B（线性注意力 + 稀疏 MoE + 原生视觉） | **需要 SGLang main 分支 → 自编译 + 补丁栈重打** |

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
| SGLang 版本 | main 分支每日变化，API/参数可能变 | 固定到支持 Qwen3.8 的首个 release（或首个稳定 commit），不要追最新 |
| sgl-kernel | 新版本预编译为 CUDA 12.8，缺 libcudart.so.13 | 源码编译 sm_89 only（附录 C 2.4 流程重走） |
| load_utils 路由 | main 分支可能重构目录/命名 | 重查 `compute_capability` 路由逻辑（附录 C 2.1/2.2） |
| stubs | 新内核符号集变化 | 重查 undefined symbol 清单（附录 C 2.3） |
| flashinfer | 版本可能升级 | 重查 python/cubin 版本对齐（附录 C 2.5） |
| 安装方式 | pip 安装 build_wheel 嵌套 bug | 沿用 rsync + 删 editable（附录 C 2.6） |

**决策规则**：

- 分支 B 且 SGLang 支持 PR 已合入 → 在**独立预研环境**（新 pod + 新镜像 tag）做完整验证，验证通过才考虑生产；
- 分支 B 且支持未就绪 → **不迁移**，等官方支持稳定（参照 Qwen3Next 滞后约 1 个月、Qwen3.6 更久的先例）；
- 任何情况下**生产环境不动**：现有 Qwen3.6-27B 服务保持运行，直到新模型在预研环境全绿。

## 6. 并行策略（推荐）

- 生产：Qwen3.6-27B 维持现状，按附录 D 11.11 观察；
- 预研：独立 pod（独立镜像 tag `sglang-l40s-sm89:qwen38-pre`），权重开放后并行验证；
- 决策点：质量对比（Qwen3.8 在代码检视/tool call 上是否明显优于 3.6）+ 性能（TTFT/E2E 不劣化）+ 部署成本（分支 A 低 / 分支 B 高），三者综合后才动生产。

## 7. 执行记录（权重开放后填写）

| 日期 | 事项 | 结果 |
|------|------|------|
| 2026-08-05 | 预研计划建立 | 待权重开放 |
| （待填） | config.json 架构判定 | 分支 A / B |
| （待填） | 单卡冒烟 | |
| （待填） | MTP / parser 验证 | |
| （待填） | 质量与性能对比 | |
| （待填） | 迁移决策 | |
