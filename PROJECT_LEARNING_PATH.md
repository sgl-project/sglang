# SGLang 项目学习路径

这份文档面向第一次阅读 SGLang 源码的开发者，目标是从“能运行”逐步走到“能定位、修改并验证代码”。

## 1. 建立项目认知

先阅读根目录的 [`README.md`](README.md)，理解 SGLang 的核心定位：它是一个面向大语言模型和多模态模型的高性能推理服务框架，重点解决低延迟、高吞吐和大规模部署问题。

建议先记住这些关键词：

- RadixAttention 与前缀缓存
- Continuous batching（连续批处理）
- Prefill / Decode 分离
- Tensor、Pipeline、Expert 和 Data Parallelism
- OpenAI 兼容 API
- 量化、推测解码与结构化输出

阶段成果：能够用自己的话解释 SGLang 解决什么问题，以及它与模型训练框架的区别。

## 2. 完成安装与最小运行

按照 README 中的 Getting Started 顺序学习：

1. 阅读安装文档并准备匹配的硬件、驱动和 Python 环境。
2. 完成 Quick Start，启动一个模型服务。
3. 使用 OpenAI 兼容接口发送一次请求。
4. 记录启动参数、模型路径、端口和返回结果。

阶段成果：能够独立启动服务、发送请求，并根据日志判断服务是否正常。

## 3. 认识代码结构

从入口到核心逻辑逐层阅读，避免一开始遍历整个仓库：

- `python/sglang/`：Python 包和主要运行时逻辑。
- `sgl-kernel/`：高性能算子与底层内核。
- `sgl-model-gateway/`：模型网关及相关能力。
- `test/`：测试用例，可用于理解预期行为。
- `benchmark/`：性能测试和典型负载。
- `examples/`：可直接运行的使用示例。
- `docs/`：用户与开发者文档。

推荐选择一个最小请求，从 API 入口开始，沿调用链追踪到调度、模型执行和响应返回，并画出简单调用流程图。

阶段成果：能够快速找到功能入口、实现位置、对应测试和示例。

## 4. 掌握核心模块

按以下顺序进行专题学习：

1. 请求生命周期：请求解析、入队、调度、推理和流式返回。
2. 批处理与调度：连续批处理如何提高吞吐量。
3. KV Cache 与 RadixAttention：前缀复用如何减少重复计算。
4. 模型执行：模型加载、权重管理和计算后端。
5. 分布式推理：从单卡逐步扩展到多卡和多节点。
6. 高级能力：量化、推测解码、LoRA、结构化输出和多模态模型。

每学习一个模块，都回答三个问题：它解决什么瓶颈、关键数据结构是什么、如何通过测试或指标验证效果。

## 5. 学会测试与性能分析

修改代码前先找到相关测试，修改后至少完成：

1. 运行受影响模块的单元测试。
2. 运行格式化、静态检查或仓库要求的检查命令。
3. 对性能相关修改执行修改前后的同条件基准测试。
4. 记录吞吐量、首 Token 延迟、Token 间延迟和显存占用。

阶段成果：不仅能判断功能是否正确，还能说明性能是否改善、是否发生回退。

## 6. 完成第一次贡献

第一次修改优先选择范围小、容易验证的任务，例如：

- 修正文档或补充示例。
- 增加缺失的测试。
- 改善错误信息或参数校验。
- 修复带有明确复现步骤的小问题。

贡献流程：创建分支、完成修改、自测、检查差异、提交、推送，并创建 Pull Request。提交前阅读仓库的贡献指南及相关目录中的 `AGENTS.md`。

## 7. 本文档的 Git 实践流程

下面的命令对应本文档从创建到合并的完整过程：

```bash
# 查看当前状态
git status --short --branch

# 从 main 创建并切换到学习分支
git switch main
git pull --ff-only origin main
git switch -c codex/learn-basics

# 修改完成后检查差异
git diff -- PROJECT_LEARNING_PATH.md

# 暂存并提交
git add PROJECT_LEARNING_PATH.md
git commit -m "docs: add SGLang project learning path"

# 首次推送分支，并建立上游跟踪关系
git push -u origin codex/learn-basics

# 切回主分支并合并
git switch main
git pull --ff-only origin main
git merge --no-ff codex/learn-basics

# 推送更新后的主分支
git push origin main
```

在 Codex 中执行这些操作时，仍应清楚每一步的含义：`status` 检查现场，`switch` 管理分支，`add` 选择要纳入提交的修改，`commit` 在本地记录快照，`push` 更新远程仓库，`merge` 把分支历史整合进主分支。

## 建议的四周节奏

- 第 1 周：完成安装、启动服务、调用 API，并熟悉目录结构。
- 第 2 周：追踪请求生命周期，重点理解调度和 KV Cache。
- 第 3 周：阅读测试与 benchmark，完成一次性能测量。
- 第 4 周：完成一个小修改，走通提交、推送、评审和合并流程。

完成这条路径后，再根据兴趣深入分布式推理、内核优化、多模态或强化学习 rollout 后端。
