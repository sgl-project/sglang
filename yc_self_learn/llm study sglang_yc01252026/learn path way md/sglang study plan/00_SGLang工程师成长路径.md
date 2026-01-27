# SGLang工程师成长路径 - 从使用者到贡献者

## 📚 文档位置

**本文档位于**: `yc_self_learn/llm study sglang_yc01252026/learn path way md/sglang study plan/`

---

## 🎯 概述

本文档帮助你从SGLang的使用者成长为SGLang的贡献者和工程师。包含SGLang RL团队的招募信息、贡献指南、学习路径和实践计划。

---

## 📖 目录

1. [SGLang RL团队招募信息](#1-sglang-rl团队招募信息)
2. [成为SGLang工程师的路径](#2-成为sglang工程师的路径)
3. [贡献指南](#3-贡献指南)
4. [Warm-up Program](#4-warm-up-program)
5. [学习路径](#5-学习路径)
6. [实践计划](#6-实践计划)
7. [资源汇总](#7-资源汇总)

---

## 1. SGLang RL团队招募信息

### 1.1 团队介绍

**SGLang RL团队**正在积极招募新成员，致力于SGLang RL生态系统的工作。

### 1.2 最近工作

1. **Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL**
   - 链接: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/fp8/readme_en.md
   - 内容: 统一FP8格式，超越混合精度，实现稳定和加速的MoE RL

2. **Power Up Speculative Decoding In Reinforcement Learning**
   - 链接: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md
   - 内容: 在强化学习中增强Speculative Decoding

### 1.3 招募类型

#### 社区成员（Community Member）
- **时间要求**: 10+小时/周
- **地点**: 全球（不需要在美国）
- **适合**: 对开源项目有热情，想参与SGLang开发

#### 研究实习生（Research Intern）
- **时间要求**: 20+小时/周
- **地点**: 美国学生
- **要求**: 符合CPT/OPT资格
- **类型**: 兼职/全职研究实习生

### 1.4 核心要求

**自驱力（Self-motivation）**：
- 能够定义问题
- 能够找到合适的解决方案
- 在开源项目中，自驱力与技术能力同样重要

### 1.5 如何申请

**完成Warm-up Program**（见下文），然后：
- 提交你的结果和思考
- 发送到: **rl_team@lmsys.org**

---

## 2. 成为SGLang工程师的路径

### 2.1 路径概览

```
SGLang使用者
    ↓
理解SGLang架构
    ↓
阅读代码和文档
    ↓
完成Warm-up Program
    ↓
贡献小功能/修复Bug
    ↓
参与核心功能开发
    ↓
成为SGLang工程师
```

### 2.2 阶段1：理解SGLang（1-2周）

**目标**: 从使用者到理解者

**任务**:
- [ ] 熟悉SGLang的基本使用
- [ ] 理解SGLang的整体架构
- [ ] 阅读核心代码模块
- [ ] 理解SGLang的设计理念

**资源**:
- [SGLang官方文档](https://docs.sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through)

### 2.3 阶段2：贡献准备（2-4周）

**目标**: 准备开始贡献

**任务**:
- [ ] 设置开发环境
- [ ] 学习贡献流程
- [ ] 理解代码风格
- [ ] 完成Warm-up Program

**资源**:
- [贡献指南](#3-贡献指南)
- [Warm-up Program](#4-warm-up-program)

### 2.4 阶段3：开始贡献（持续）

**目标**: 成为活跃贡献者

**任务**:
- [ ] 从"good first issue"开始
- [ ] 修复小bug
- [ ] 改进文档
- [ ] 参与讨论

**资源**:
- [GitHub Issues](https://github.com/sgl-project/sglang/issues)
- [Slack社区](https://slack.sglang.ai)

### 2.5 阶段4：深度参与（持续）

**目标**: 成为核心贡献者

**任务**:
- [ ] 参与核心功能开发
- [ ] 参与架构设计讨论
- [ ] 帮助新贡献者
- [ ] 成为维护者

---

## 3. 贡献指南

### 3.1 环境设置

#### Fork和Clone仓库

```bash
# Fork仓库到你的GitHub账号
# 然后clone你的fork
git clone https://github.com/<your_user_name>/sglang.git
cd sglang
```

#### 从源码构建

参考: [Install SGLang from Source](https://docs.sglang.ai/get_started/install.html#method-2-from-source)

### 3.2 代码格式

#### 使用pre-commit

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

**重要**:
- 确保代码通过所有检查**之前**创建Pull Request
- 不要直接提交到`main`分支
- 总是创建新分支（如`feature/my-new-feature`）

### 3.3 代码风格

#### 核心原则

1. **避免代码重复**: 如果相同代码片段（超过5行）出现多次，提取为共享函数
2. **最小化设备同步**: 减少昂贵的CPU-GPU同步操作（如`tensor.item()`或`tensor.cpu()`）
3. **优先考虑极致效率**: SGLang是运行时，大部分代码在每个请求的关键路径上运行
4. **保持函数纯净**: 尽可能避免就地修改参数
5. **保持文件简洁**: 如果文件超过2000行，拆分为多个小文件
6. **保持测试快速**: 
   - 单个测试文件运行时间不超过500秒
   - 单个GitHub workflow job不超过30分钟

### 3.4 测试要求

#### 添加单元测试

- 如果添加新功能或修复bug，请添加相应的单元测试
- 使用Python的unittest框架
- 参考: [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)

#### 准确性测试

如果代码改变模型输出，请运行准确性测试：

```bash
# 启动服务器
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# 评估
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

#### 性能测试

参考: [Benchmark and Profiling](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)

### 3.5 文档要求

**推荐新贡献者从写文档开始**，这有助于快速理解SGLang代码库。

参考: [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)

### 3.6 提交Pull Request

#### PR检查清单

- [ ] 代码格式符合pre-commit要求
- [ ] 添加了单元测试
- [ ] 更新了文档（如果需要）
- [ ] 提供了准确性和速度基准测试结果（如果适用）
- [ ] 链接到相关Issue

#### 触发CI

PR必须有"run-ci"标签才能触发CI：
- 如果有write权限，会自动添加标签
- 如果有triage权限，可以手动添加标签
- 否则，请求审查并请维护者添加标签

---

## 4. Warm-up Program

### 4.1 什么是Warm-up Program？

**Warm-up Program**是SGLang RL团队设计的入门任务，用于评估你的：
- 技术能力
- 自驱力
- 问题解决能力

### 4.2 如何完成Warm-up Program？

#### 步骤1：理解任务

阅读SGLang RL团队的工作：
1. [Unified FP8](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/fp8/readme_en.md)
2. [Speculative Decoding in RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md)

#### 步骤2：完成相关任务

根据你的兴趣和能力，选择相关任务：
- 理解FP8量化在MoE RL中的应用
- 理解Speculative Decoding在RL中的增强
- 阅读相关代码
- 提出改进建议或实现

#### 步骤3：提交结果

**提交内容**:
- 你的理解和分析
- 完成的任务（如果有）
- 你的思考和建议
- 你的代码（如果有）

**提交方式**: 发送到 **rl_team@lmsys.org**

### 4.3 Warm-up Program建议

#### 对于初学者

1. **理解概念**:
   - 深入理解FP8量化
   - 理解Speculative Decoding
   - 理解MoE架构

2. **阅读代码**:
   - 阅读SGLang中相关的实现
   - 理解代码结构
   - 提出改进建议

3. **写文档**:
   - 改进相关文档
   - 添加使用示例
   - 翻译文档（如果需要）

#### 对于有经验的开发者

1. **实现功能**:
   - 实现相关优化
   - 修复相关bug
   - 添加新功能

2. **性能优化**:
   - 分析性能瓶颈
   - 提出优化方案
   - 实现优化

3. **架构改进**:
   - 提出架构改进建议
   - 实现架构改进
   - 参与设计讨论

---

## 5. 学习路径

### 5.1 基础学习（必须完成）

#### Week 1-2: SGLang基础

- [ ] 阅读SGLang官方文档
- [ ] 理解SGLang架构
- [ ] 阅读核心代码
- [ ] 完成基础使用示例

**资源**:
- [SGLang官方文档](https://docs.sglang.ai/)
- [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through)

#### Week 3-4: 量化技术

- [ ] 理解FP8量化
- [ ] 理解FP4量化
- [ ] 理解KV Cache量化
- [ ] 阅读量化相关代码

**资源**:
- [01_Z2_FP4量化_详解.md](../01_Z2_FP4量化_详解.md)
- [01_Z3_量化技术对比_详解.md](../01_Z3_量化技术对比_详解.md)
- [Unified FP8文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/fp8/readme_en.md)

#### Week 5-6: RL和Speculative Decoding

- [ ] 理解强化学习基础
- [ ] 理解Speculative Decoding
- [ ] 理解在RL中的应用
- [ ] 阅读相关代码

**资源**:
- [Speculative Decoding in RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md)

### 5.2 进阶学习（推荐）

#### GPU和CUDA

- [ ] GPU架构基础
- [ ] CUDA编程
- [ ] Kernel开发
- [ ] 性能优化

**资源**:
- [01_Z1_GPU内存层次_详解.md](../01_Z1_GPU内存层次_详解.md)
- [CUDA官方文档](https://docs.nvidia.com/cuda/)

#### MoE架构

- [ ] MoE基础
- [ ] MoE在SGLang中的实现
- [ ] MoE优化

**资源**:
- [03_Issue_17526_学习路径.md](../03_Issue_17526_学习路径.md)

### 5.3 实践学习（必须完成）

#### 贡献实践

- [ ] 完成第一个PR（文档或小bug修复）
- [ ] 完成第二个PR（功能改进）
- [ ] 参与讨论和审查

---

## 6. 实践计划

### 6.1 第1个月：基础准备

**Week 1-2: 环境设置和学习**
- [ ] Fork SGLang仓库
- [ ] 设置开发环境
- [ ] 阅读贡献指南
- [ ] 理解代码结构

**Week 3-4: 完成Warm-up Program**
- [ ] 阅读RL团队的工作
- [ ] 理解相关概念
- [ ] 完成Warm-up任务
- [ ] 提交结果

### 6.2 第2个月：开始贡献

**Week 1-2: 第一个贡献**
- [ ] 选择一个"good first issue"
- [ ] 完成修复或改进
- [ ] 提交PR
- [ ] 根据反馈改进

**Week 3-4: 持续贡献**
- [ ] 完成2-3个小贡献
- [ ] 参与社区讨论
- [ ] 帮助其他贡献者

### 6.3 第3个月及以后：深度参与

**持续任务**:
- [ ] 参与核心功能开发
- [ ] 参与架构讨论
- [ ] 成为活跃贡献者
- [ ] 申请成为维护者（如果合适）

---

## 7. 资源汇总

### 7.1 官方资源

#### 文档
- [SGLang官方文档](https://docs.sglang.ai/) ⭐⭐⭐
- [贡献指南](https://docs.sglang.ai/developer_guide/contribution_guide.html) ⭐⭐⭐
- [性能分析指南](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) ⭐⭐

#### 代码
- [SGLang GitHub](https://github.com/sgl-project/sglang) ⭐⭐⭐
- [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) ⭐⭐⭐

#### 社区
- [Slack社区](https://slack.sglang.ai) ⭐⭐⭐
- [GitHub Issues](https://github.com/sgl-project/sglang/issues) ⭐⭐
- [GitHub Discussions](https://github.com/sgl-project/sglang/discussions) ⭐⭐

### 7.2 RL团队资源

- [Unified FP8](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/fp8/readme_en.md) ⭐⭐⭐
- [Speculative Decoding in RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md) ⭐⭐⭐

### 7.3 学习资源

#### 内部文档
- [01_Z1_GPU内存层次_详解.md](../01_Z1_GPU内存层次_详解.md)
- [01_Z2_FP4量化_详解.md](../01_Z2_FP4量化_详解.md)
- [01_Z3_量化技术对比_详解.md](../01_Z3_量化技术对比_详解.md)
- [03_Issue_17526_学习路径.md](../03_Issue_17526_学习路径.md)

#### 外部资源
- [CUDA官方文档](https://docs.nvidia.com/cuda/) ⭐⭐⭐
- [PyTorch官方文档](https://pytorch.org/docs/) ⭐⭐⭐
- [Flash Attention论文](https://arxiv.org/abs/2205.14135) ⭐⭐

### 7.4 贡献资源

#### Issues筛选
- [Good First Issue](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) ⭐⭐⭐
- [Help Wanted](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) ⭐⭐
- [Documentation](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) ⭐⭐

---

## ✅ 检查清单

### 成为贡献者前

- [ ] 理解SGLang的基本使用
- [ ] 理解SGLang的架构
- [ ] 阅读贡献指南
- [ ] 设置开发环境
- [ ] 理解代码风格

### 完成Warm-up Program

- [ ] 阅读RL团队的工作
- [ ] 理解相关概念
- [ ] 完成相关任务
- [ ] 提交结果和思考

### 开始贡献

- [ ] 完成第一个PR
- [ ] 参与社区讨论
- [ ] 帮助其他贡献者
- [ ] 持续贡献

---

## 🎯 下一步行动

1. **立即行动**:
   - Fork SGLang仓库
   - 设置开发环境
   - 阅读贡献指南

2. **本周任务**:
   - 阅读RL团队的工作
   - 开始Warm-up Program
   - 选择一个"good first issue"

3. **本月目标**:
   - 完成Warm-up Program
   - 提交第一个PR
   - 参与社区讨论

---

## 💡 关键建议

1. **从小开始**: 从文档改进或小bug修复开始
2. **持续学习**: 技术栈在快速发展，保持学习
3. **积极参与**: 参与讨论，帮助他人，建立声誉
4. **自驱力**: 能够定义问题并找到解决方案
5. **耐心**: 成为核心贡献者需要时间

---

**开始你的SGLang工程师之旅！** 🚀

**联系方式**: rl_team@lmsys.org
