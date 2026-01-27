# RadixArk工程师路径 - 8周详细计划

## 📚 文档位置

**本文档位于**: `yc_self_learn/llm study sglang_yc01252026/learn path way md/sglang study plan/`

---

## 🎯 目标

**成为RadixArk工程师** - 通过8周的系统学习和贡献，获得RadixArk的工作机会。

---

## 📖 目录

1. [RadixArk用人画像](#1-radixark用人画像)
2. [核心信号](#2-核心信号)
3. [8周学习计划](#3-8周学习计划)
4. [四个方向的细化路径](#4-四个方向的细化路径)
5. [PR清单](#5-pr清单)
6. [每日可执行Checklist](#6-每日可执行checklist)

---

## 1. RadixArk用人画像

### 1.1 业务理解

**RadixArk的核心业务**：

```
SGLang = 推理/Serving引擎
├── 性能优化
├── 可观测性
├── Router
└── Cache

slime = 轻量RL post-training框架
└── rollout↔训练的胶水

Miles = RadixArk的企业版RL框架
├── 从slime fork/共同演化
├── 面向MoE
├── 大规模
└── 生产稳定性
```

### 1.2 岗位类型

#### Solutions Engineer（解决方案工程师）
- **职责**: 帮客户部署/调优SGLang & Miles
- **能力要求**: 
  - 性能调优
  - 故障排查
  - 客户支持
  - Benchmark和Profiling

#### AI Infra Resident（AI基础设施轮岗）
- **职责**: 推理/训练/内核/集群轮岗
- **能力要求**:
  - Inference优化
  - Training pipeline
  - Kernels开发
  - Cluster管理

#### Data Infra（数据基础设施）
- **职责**: SGLang内建统计/数据智能
- **能力要求**:
  - 可观测性
  - Metrics收集
  - 数据分析
  - Usage statistics

#### DX/文档（开发者体验）
- **职责**: 文档、工具、开发者体验
- **能力要求**:
  - 技术写作
  - 工具开发
  - 用户体验

---

## 2. 核心信号

### 2.1 成为RadixArk工程师的核心信号

**两个关键信号**：

1. **可复现的benchmark+profiling证据**
   - TTFT/tokens/s/p95/p99
   - Trace分析
   - 回归对比
   - 性能诊断报告

2. **可合并的PR**
   - 带测试
   - 带文档
   - 带benchmark
   - 按贡献流程走完

### 2.2 评估标准

**RadixArk会看什么**：
- ✅ 你的PR质量（测试、文档、benchmark）
- ✅ 你的性能分析能力（profiling、trace、诊断）
- ✅ 你的工程化思维（可复现、可诊断、可维护）
- ✅ 你对业务的理解（SGLang、slime、Miles）

---

## 3. 8周学习计划

### Week 1-2: SGLang工程化读懂

#### 目标
把SGLang从"会用"升级到"工程化理解"

#### 必做任务

**任务1: 完成Benchmark & Profiling报告**

```bash
# 1. 运行官方benchmark
python -m sglang.bench_one_batch_server \
  --model <small_model> \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024 \
  --profile \
  --profile-steps 10 \
  --show-report

# 2. 使用PyTorch Profiler
SGLANG_TORCH_PROFILER_DIR="./profiler_output" \
python -m sglang.bench_one_batch_server \
  --model <small_model> \
  --profile \
  --profile-steps 10

# 3. 使用Nsight Systems（如果有GPU）
nsys profile --trace=cuda,nvtx \
  python -m sglang.bench_one_batch_server \
  --model <small_model>
```

**产出**: `perf_notes.md`报告，包含：
- TTFT、tokens/s、p95/p99指标
- Profiler/trace截图和结论
- 瓶颈分析（copy/cast? kernel launch? attention backend?）
- 性能优化建议

**任务2: 跑通贡献流程**

```bash
# 1. Fork和Clone
git clone https://github.com/<your_username>/sglang.git
cd sglang

# 2. 设置pre-commit
pip3 install pre-commit
pre-commit install
pre-commit run --all-files

# 3. 运行测试
cd test
python -m pytest test_srt/test_xxx.py -v

# 4. 理解CI流程
# 查看.github/workflows/了解CI配置
```

**产出**: 
- 本地开发环境设置完成
- 理解PR提交流程
- 理解CI触发方式

#### 学习资源

- [SGLang Benchmark & Profiling](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) ⭐⭐⭐
- [Contribution Guide](https://docs.sglang.ai/developer_guide/contribution_guide.html) ⭐⭐⭐
- [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) ⭐⭐⭐

---

### Week 3-4: 选择方向并开始贡献

#### 目标
选择一个RadixArk最在意的方向，开始第一个实质性贡献

#### 三个推荐方向

**方向A: 可复现RL（Deterministic Inference）** ⭐⭐⭐

**背景**: 
- LMSYS专门写过：SGLang正在推进deterministic inference
- 已经和slime集成用于可复现RL
- 温度>0的确定性采样、配置NCCL/FA kernel等

**可贡献点**：

1. **补单测/回归**
   - 确保deterministic模式在更多backend/配置下不silently break
   - 添加回归测试

2. **性能差距定位**
   - 他们也承认deterministic慢
   - 目标是把差距压到<20%
   - 做性能分析和优化建议

**第一个PR建议**:
- 为某个backend添加deterministic模式的测试
- 或做性能差距分析报告

**方向B: Radix Cache / 前缀复用** ⭐⭐

**背景**:
- 为了公平benchmark甚至暂时关闭radix cache
- FlashInfer/Triton的radix cache支持仍在推进
- "增强radix tree兼容更多attention kernel"列为未来工作

**可贡献点**：

1. **兼容层/测试**
   - 为某个backend补齐radix cache的兼容层
   - 添加测试用例

2. **性能评测协议**
   - 什么时候该开radix cache
   - 怎么报告命中率/收益
   - 创建benchmark脚本

**第一个PR建议**:
- 为某个backend添加radix cache支持
- 或创建radix cache性能评测脚本

**方向C: Speculative Decoding / 在线Draft训练** ⭐⭐⭐

**背景**:
- Miles博客把"spec decoding + 在线训练draft model"当成重点能力
- RL场景里frozen draft会导致accept length下降

**可贡献点**：

1. **最小可运行demo**
   - 在slime里做一个最小可运行demo/脚本
   - 指标：accept length、speedup、质量回归

2. **工具化能力**
   - 权重同步
   - Rollout metrics
   - Debug utilities

**第一个PR建议**:
- 在slime中添加spec decoding demo
- 或添加rollout metrics收集工具

#### 选题原则

✅ **优先选择**:
- 有明确指标能证明收益/不回归的任务
- PR最容易被accept
- 不需要顶级GPU就能完成

❌ **避免选择**:
- 需要大量GPU资源
- 没有明确验收标准
- 与当前路线图不相关

---

### Week 5-6: slime/Miles端到端小闭环

#### 目标
像RadixArk工程师那样把链路跑通并可诊断

#### slime学习抓手

**slime的定位**:
- 连接Megatron（训练）与SGLang（推理/rollout）
- 提供灵活的数据生成接口

**四个接口边界**:
1. **Algorithm**: 算法接口
2. **Data**: 数据接口
3. **Rollout**: 推理/rollout接口
4. **Eval**: 评估接口

#### 必做任务

**任务1: 创建minimal_grpo_or_sft_recipe/**

```bash
# 1. 理解slime结构
cd slime
# 阅读README和examples

# 2. 创建最小可运行recipe
mkdir minimal_grpo_or_sft_recipe
# 包含：
# - 配置文件
# - 训练脚本
# - 评估脚本
# - README说明
```

**产出**:
- 能跑、能打点、能定位失败的recipe
- 包含完整的配置和脚本

**任务2: 创建debug_playbook.md**

**内容**:
- OOM时看哪些指标
- 卡住时跑哪些命令
- 吞吐下降时如何诊断
- 故障排查流程

**产出**:
- 完整的debug playbook
- 可复用的诊断工具/脚本

#### 学习资源

- [slime GitHub](https://github.com/lm-sys/slime) ⭐⭐⭐
- [Miles博客](https://radixark.com/blog) ⭐⭐⭐
- [Deterministic Inference博客](https://lmsys.org/blog/2024-12-19-deterministic-inference/) ⭐⭐

---

### Week 7-8: 可合并PR + 可复现证据

#### 目标
把贡献升级成可合并的PR，包含完整的验证和文档

#### PR标准（RadixArk风格）

**PR必须包含**:

1. **Why（问题/背景）**
   - 清晰描述问题
   - 说明为什么需要这个改动
   - 链接到相关issue/讨论

2. **What（改动点）**
   - 详细的改动说明
   - 代码变更摘要
   - 架构变更（如果有）

3. **How to validate（bench/test/trace）**
   - Benchmark结果
   - 测试用例
   - Trace分析（如果涉及性能）

4. **Regression risk（边界/开关/默认值）**
   - 回归风险分析
   - 边界情况处理
   - 开关/配置选项
   - 默认值选择

#### PR提交流程

```bash
# 1. 创建分支
git checkout -b feature/your-feature

# 2. 实现改动
# ... 编写代码 ...

# 3. 运行pre-commit
pre-commit run --all-files

# 4. 运行测试
python -m pytest test/...

# 5. 运行benchmark（如果涉及性能）
python -m sglang.bench_one_batch_server ...

# 6. 提交PR
git push origin feature/your-feature
# 在GitHub创建PR，链接到issue
```

#### PR检查清单

- [ ] 代码通过pre-commit检查
- [ ] 添加了单元测试
- [ ] 更新了文档
- [ ] 提供了benchmark结果（如果涉及性能）
- [ ] 提供了trace分析（如果涉及性能）
- [ ] 说明了回归风险
- [ ] 链接到相关issue
- [ ] 请求了codeowner review
- [ ] 触发了CI（添加"run-ci"标签）

---

## 4. 四个方向的细化路径

### 方向1: Inference/Serving（SGLang）

#### Week 1-2: 基础
- [ ] 理解SGLang架构
- [ ] 完成benchmark报告
- [ ] 理解router和cache机制

#### Week 3-4: 性能优化
- [ ] 选择一个性能优化方向
- [ ] 完成性能分析和优化
- [ ] 提交第一个PR

#### Week 5-6: 可观测性
- [ ] 理解metrics系统
- [ ] 添加新的metrics
- [ ] 改进可观测性

#### Week 7-8: 生产级特性
- [ ] 故障排查工具
- [ ] 性能诊断工具
- [ ] 提交完整的PR

**推荐PR**:
- Benchmark脚本增强
- 性能诊断工具
- Router优化
- Cache优化

---

### 方向2: RL/Miles/slime

#### Week 1-2: 基础
- [ ] 理解slime架构
- [ ] 理解Miles架构
- [ ] 理解RL pipeline

#### Week 3-4: Deterministic Inference
- [ ] 理解deterministic模式
- [ ] 添加测试/回归
- [ ] 性能差距分析

#### Week 5-6: 端到端闭环
- [ ] 创建minimal recipe
- [ ] 创建debug playbook
- [ ] 工具化能力

#### Week 7-8: 生产级特性
- [ ] Spec decoding集成
- [ ] Rollout metrics
- [ ] Debug utilities

**推荐PR**:
- Deterministic模式测试
- Spec decoding demo
- Rollout metrics工具
- Debug utilities

---

### 方向3: Kernels/Numerics（FP8/Quant/Spec）

#### Week 1-2: 基础
- [ ] 理解FP8/FP4量化
- [ ] 理解Speculative Decoding
- [ ] 理解Kernel开发

#### Week 3-4: 量化优化
- [ ] FP8量化优化
- [ ] FP4量化优化
- [ ] KV Cache量化

#### Week 5-6: Spec Decoding
- [ ] 理解Spec Decoding原理
- [ ] 实现最小demo
- [ ] 性能分析

#### Week 7-8: 生产级特性
- [ ] Kernel优化
- [ ] 量化工具
- [ ] 性能benchmark

**推荐PR**:
- 量化kernel优化
- Spec decoding实现
- 性能benchmark脚本
- 量化工具

---

### 方向4: Data Infra/Observability

#### Week 1-2: 基础
- [ ] 理解metrics系统
- [ ] 理解logging系统
- [ ] 理解可观测性架构

#### Week 3-4: Metrics增强
- [ ] 添加新metrics
- [ ] 改进metrics收集
- [ ] 创建metrics dashboard

#### Week 5-6: Usage Statistics
- [ ] 理解usage统计需求
- [ ] 实现usage统计
- [ ] 数据分析工具

#### Week 7-8: 生产级特性
- [ ] 完整的可观测性系统
- [ ] 数据分析工具
- [ ] 报告生成工具

**推荐PR**:
- Metrics增强
- Usage statistics
- 数据分析工具
- 可观测性改进

---

## 5. PR清单

### 第一批PR（不依赖强GPU）

#### 1. Benchmark/Profiling脚本增强 + 文档

**目标**: 让别人更容易复现TTFT/p95/吞吐

**内容**:
- 改进benchmark脚本
- 添加更多profiling选项
- 创建benchmark文档
- 添加结果分析工具

**验收标准**:
- 脚本可以复现结果
- 文档清晰完整
- 包含示例输出

#### 2. Deterministic模式的测试/回归

**目标**: 确保deterministic模式不break

**内容**:
- 为更多backend添加测试
- 添加回归测试
- 创建测试文档

**验收标准**:
- 所有测试通过
- 覆盖主要backend
- 文档完整

#### 3. Radix Cache兼容性/开关/测试

**目标**: 推进radix cache支持

**内容**:
- 为某个backend添加radix cache支持
- 添加开关选项
- 添加测试用例
- 创建性能评测脚本

**验收标准**:
- Radix cache正常工作
- 测试通过
- 性能评测脚本可用

#### 4. slime的Debug Utilities / Metrics / Post-hoc Analyzer

**目标**: 提升slime的可诊断性

**内容**:
- Debug工具
- Metrics收集
- Post-hoc分析工具
- 文档

**验收标准**:
- 工具可用
- Metrics准确
- 文档完整

---

## 6. 每日可执行Checklist

### 选择方向后，按方向细化

#### 方向1: Inference/Serving（SGLang）

**Day 1-2: 环境设置和基础理解**
- [ ] Fork SGLang仓库
- [ ] 设置开发环境
- [ ] 阅读Contribution Guide
- [ ] 运行第一个benchmark

**Day 3-4: 性能分析**
- [ ] 运行profiler
- [ ] 分析trace结果
- [ ] 识别性能瓶颈
- [ ] 创建perf_notes.md

**Day 5-7: 选择优化方向**
- [ ] 阅读相关代码
- [ ] 理解优化点
- [ ] 设计优化方案
- [ ] 开始实现

**Week 2: 实现和测试**
- [ ] 完成实现
- [ ] 添加测试
- [ ] 运行benchmark
- [ ] 准备PR

#### 方向2: RL/Miles/slime

**Day 1-2: slime基础**
- [ ] Clone slime仓库
- [ ] 阅读README
- [ ] 理解四个接口边界
- [ ] 运行第一个example

**Day 3-4: Deterministic Inference**
- [ ] 阅读deterministic博客
- [ ] 理解实现原理
- [ ] 查看相关代码
- [ ] 设计测试方案

**Day 5-7: 实现测试**
- [ ] 实现测试用例
- [ ] 运行测试
- [ ] 修复问题
- [ ] 准备PR

**Week 2: 端到端闭环**
- [ ] 创建minimal recipe
- [ ] 创建debug playbook
- [ ] 测试完整流程
- [ ] 准备PR

#### 方向3: Kernels/Numerics

**Day 1-2: 量化基础**
- [ ] 阅读FP8/FP4文档
- [ ] 理解量化原理
- [ ] 查看量化代码
- [ ] 运行量化示例

**Day 3-4: Spec Decoding**
- [ ] 阅读Spec Decoding论文
- [ ] 理解实现原理
- [ ] 查看相关代码
- [ ] 设计demo方案

**Day 5-7: 实现demo**
- [ ] 实现最小demo
- [ ] 添加metrics
- [ ] 测试功能
- [ ] 准备PR

**Week 2: 优化和工具**
- [ ] 性能优化
- [ ] 添加工具
- [ ] 完善文档
- [ ] 准备PR

#### 方向4: Data Infra/Observability

**Day 1-2: Metrics系统**
- [ ] 理解metrics架构
- [ ] 查看metrics代码
- [ ] 理解收集流程
- [ ] 运行metrics示例

**Day 3-4: 添加新Metrics**
- [ ] 设计metrics方案
- [ ] 实现metrics收集
- [ ] 测试metrics
- [ ] 准备PR

**Day 5-7: Usage Statistics**
- [ ] 理解usage需求
- [ ] 设计统计方案
- [ ] 实现统计功能
- [ ] 测试功能

**Week 2: 工具和文档**
- [ ] 创建分析工具
- [ ] 创建dashboard
- [ ] 完善文档
- [ ] 准备PR

---

## ✅ 8周结束检查清单

### 必须完成

- [ ] 完成benchmark报告（perf_notes.md）
- [ ] 跑通贡献流程
- [ ] 选择一个方向并完成第一个PR
- [ ] 创建minimal recipe（如果选择RL方向）
- [ ] 创建debug playbook
- [ ] 提交至少2个可合并的PR

### 推荐完成

- [ ] 参与社区讨论
- [ ] 帮助其他贡献者
- [ ] 改进文档
- [ ] 创建工具/脚本

---

## 🎯 下一步行动

1. **立即选择方向**: 从4个方向中选择1个
2. **开始Week 1任务**: 完成benchmark报告
3. **准备第一个PR**: 根据选择的方向准备
4. **持续贡献**: 保持每周至少1个PR的节奏

---

## 💡 关键建议

1. **质量>数量**: 1个高质量的PR比10个低质量的PR更有价值
2. **可复现**: 所有benchmark和测试都要可复现
3. **文档完整**: PR必须包含完整的文档和测试
4. **持续学习**: 技术栈在快速发展，保持学习
5. **积极参与**: 参与社区讨论，建立声誉

---

**开始你的RadixArk工程师之旅！** 🚀

**联系方式**: 
- SGLang: [Slack](https://slack.sglang.ai)
- RadixArk: [Website](https://radixark.com)
