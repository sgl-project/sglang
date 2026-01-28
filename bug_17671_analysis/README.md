# Issue #17671 分析文档

## 问题概述

**Issue**: [Bug] Can't launch diffusion models by following the official doc #17671

**核心问题**: SGLang Docker镜像（`lmsysorg/sglang:dev`）缺少diffusion功能支持

**状态**: Open

---

## 文档结构

### A01_B01_original_issue.md
- Issue原始内容
- 问题描述
- 讨论和回复
- 问题总结

### A01_B02_problem_analysis.md ✅
- 问题详细分析
- Docker镜像构建问题
- Diffusion依赖缺失分析
- **已确认**: 问题本质是缺少`diffusers`模块，可在4090复现

### A01_B03_solution_analysis.md ✅
- 解决方案分析
- 临时解决方案（workaround）
- 永久解决方案（文档修复、Docker修复）
- PR修复方向

### A01_B04_code_analysis.md ✅
- 相关代码分析
- Dockerfile分析
- pyproject.toml分析
- 代码修改方案

### A01_B05_PR_draft.md ✅
- PR修改草案
- 具体代码修改建议
- PR提交步骤
- 验证方法

### A01_B06_reproduction_steps.md ✅
- Bug复现步骤（从轻到重）
- 详细命令和预期结果
- 如何贡献到Issue
- 快速复现脚本

### reproduce_bug.sh ✅
- 自动化复现脚本（Bash版本）
- 可以一键执行所有复现步骤

### reproduce_bug.ps1 ✅
- 自动化复现脚本（PowerShell版本）
- Windows用户可以使用

### A01_B07_issue_contribution_template.md ✅
- Issue贡献模板
- 如何在Issue中回复
- 收集信息清单

### A01_B08_reproduction_results.md ✅
- Bug复现结果记录
- 镜像信息记录
- Step 1和Step 3的结果记录
- Issue回复内容模板

### A01_B09_issue_reply_to_kevin.md ✅
- 给 kevin 的 Issue 回复草稿
- 测试结果总结
- 问题澄清请求

### A01_B10_disk_space_troubleshooting.md ✅
- 磁盘空间问题排查和解决方案
- Docker 系统空间检查
- 清理 Docker 缓存的方法
- Docker Desktop 磁盘配额检查

### A01_B11_diffusion_parameter_issue.md ✅
- Diffusion 模型 Parameter 参数问题记录
- 用户反馈：多个模型无法使用 sglang 的 parameter 参数
- 问题分析和可能的解决方案
- 需要确认的信息清单

### A01_B12_complete_troubleshooting_marathon.md ✅ ⭐ **重要**
- **完整的排障马拉松记录**
- RTX 4090 在 Windows + Docker + WSL2 环境下的完整测试历程
- 四个阶段的详细记录：Diffusion 初试 → 文件系统博弈 → Windows 锁定 → LLM 核心测试
- **核心结论**：
  - SGLang LLM 模块极强（工业级）
  - Diffusion 模块脆弱（实验性，Windows 挂载是"死亡组合"）
  - 4090 性能没问题，问题在 SGLang Diffusion 模块实现
- 避坑指南和最佳实践
- 下一步建议

### A01_B13_missing_accelerate_dependency.md ✅ 🔴 **致命问题**
- **教科书级别的"打脸"现场**
- **核心问题**: `lmsysorg/sglang:dev` 镜像缺少 `accelerate` 库
- **错误**: `NotImplementedError: Using device_map requires the accelerate library`
- **荒谬之处**: 
  - SGLang 作为高性能推理引擎，默认镜像没有预装管理显存最基础的库
  - 官方文档 vs. 现实货不对板
- **连锁反应**: 导致 Rank 0 scheduler is dead，整个推理进程崩毁
- **解决方案**: 临时安装 `pip install accelerate`，或修复 Dockerfile
- **状态**: ✅ **已复现并确认**

### A01_B14_github_issue_template.md ✅
- GitHub Issue 描述模板
- 完整版本和简化版本
- 可直接复制使用提交 Issue
- 包含完整的错误分析、复现步骤和建议修复方案

### A01_B15_function_rename_bug.md ✅ 🟡
- **函数重命名导致的引用错误**
- **问题**: `set_default_dtype` 被重命名为 `set_default_torch_dtype`，但 `comfyui_qwen_image_pipeline.py` 中仍使用旧函数名
- **错误类型**: 代码重构遗漏
- **影响**: ComfyUI Qwen Image Pipeline 无法运行
- **修复难度**: 🟢 简单 - 只需修改函数名引用
- **预防措施**: 使用 IDE 全局重命名、运行完整测试套件

### A01_B16_cache_dit_diffusion_stress_test.md ✅
- **Cache-DiT + 大模型 Diffusion 深度压力测试总结（4090 本地环境）**
- 分阶段测试：依赖补完 → 源码修复 → 硬件极限与 OOM 挑战
- 不同模型对比：Wan 2.1 1.3B、Z-Image-Turbo、SD-Turbo 的资源占用与表现
- 锁定的问题点：
  - 镜像依赖缺失（accelerate、ftfy）
  - 源码命名坏味道（`set_default_dtype` → `set_default_torch_dtype`）
  - Cache-DiT 与 layerwise-offload 的策略冲突与报错不友好
  - 对 DiT/T5 大模型缺乏低内存环境下的自动降级与保护
- 为后续 PR / Issue 提供了面向工程实践的优先级建议

### A01_B17_final_issue_reply_summary.md ✅ ⭐ **最终总结**
- 对 B07–B16 的**去重整合版总结**
- 汇总了我们实际做过的所有关键测试（环境、依赖、SD1.5/SD-Turbo、Wan/Z-Image、FLUX 路径推断等）
- 明确对 Issue #17671 的结论：
  - 最新 dev 镜像已具备基础 Diffusion 能力（至少 SD 类模型路径是通的）
  - 高阶 DiT / FLUX / Wan 路径在依赖和资源策略上仍有明显缺口
- 提炼出一段可以直接贴回 Issue 的“人话版”回复思路，和建议官方修复的优先级事项

---

## 快速开始

1. **阅读原始Issue**: [A01_B01_original_issue.md](./A01_B01_original_issue.md)
2. **了解问题本质**: SGLang镜像缺少diffusion功能
3. **查看解决方案**: （待补充）

---

## 关键信息

### 问题状态更新（2026-01-27）

**重要发现**：经过测试，当前 `lmsysorg/sglang:dev` 镜像（创建时间：2026-01-27）**已经包含了完整的 SGLang diffusion 支持**：
- ✅ `diffusers` 库（v0.36.0）已安装
- ✅ SGLang diffusion 模块存在
- ✅ `sglang generate` 命令可以正常工作
- ✅ Diffusion 工作流功能完整

**测试中遇到的问题**：
- ❌ 磁盘空间不足（环境问题，不是代码问题）
- 详见：[A01_B10_disk_space_troubleshooting.md](./A01_B10_disk_space_troubleshooting.md)

### 排障马拉松重要发现（2025-01）

**完整记录**: [A01_B12_complete_troubleshooting_marathon.md](./A01_B12_complete_troubleshooting_marathon.md)

**核心结论**：

1. **SGLang 是"严重偏科"的学霸**：
   - ✅ **LLM 模块（极强）**：工业级支持，4090 上推理几乎瞬时完成
   - ⚠️ **Diffusion 模块（脆弱）**：实验性，文件校验机制死板，容错率低

2. **Windows/Docker 路径是"万恶之源"**：
   - ❌ **不要使用 Windows 本地挂载**：在 Diffusion 模块中这是"死亡组合"
   - ✅ **推荐方案**：容器内虚拟磁盘（内存中）或纯 Linux 环境
   - 问题根源：`maybe_download_model` 在 Windows + Docker 挂载环境下无法正确识别已下载文件

3. **4090 性能没问题**：
   - ✅ 硬件和环境配置都是顶配
   - ✅ LLM 推理速度极快
   - ❌ 问题在 SGLang Diffusion 模块的实现，不在硬件

**避坑指南**：
- ✅ 先测试 LLM 模块验证环境
- ✅ 使用官方标注为"Stable"的模型
- ❌ 避免 Windows 挂载（Diffusion 模块）
- ✅ 记录详细日志便于问题定位

### 致命依赖缺失（2025-01-28）🔴

**完整记录**: [A01_B13_missing_accelerate_dependency.md](./A01_B13_missing_accelerate_dependency.md)

**核心问题**：
- ❌ **`lmsysorg/sglang:dev` 镜像缺少 `accelerate` 库**
- ❌ **导致 Diffusion 模块完全无法运行**
- ❌ **错误**: `NotImplementedError: Using device_map requires the accelerate library`

**荒谬之处**：
- SGLang 作为高性能推理引擎，默认镜像没有预装管理显存最基础的库
- 代码中明确使用 `device_map=cuda`，但镜像没有必需的依赖
- 官方文档 vs. 现实货不对板

**临时解决方案**：
```bash
# 在容器内安装缺失的依赖
pip install accelerate
```

**验证**：
- ✅ 已复现并确认
- ✅ 环境 100% 正常，问题在软件层面的依赖断裂

### 函数重命名遗漏（2025-01-28）🟡

**完整记录**: [A01_B15_function_rename_bug.md](./A01_B15_function_rename_bug.md)

**核心问题**：
- ❌ **代码重构遗漏**: `set_default_dtype` 被重命名为 `set_default_torch_dtype`
- ❌ **引用未更新**: `comfyui_qwen_image_pipeline.py` 中仍使用旧函数名
- ❌ **导致错误**: `AttributeError` 或 `NameError`

**影响**：
- ComfyUI Qwen Image Pipeline 无法运行
- 相关图像生成功能受影响

**修复方案**：
- 更新 `comfyui_qwen_image_pipeline.py` 中的函数引用
- 或添加向后兼容别名（临时方案）

**经验教训**：
- 重构时使用 IDE 的全局重命名功能
- 运行完整的测试套件
- 代码审查时检查重构是否完整

### 原始问题假设

- **问题**: Docker镜像缺少`diffusers`模块（`ModuleNotFoundError: No module named 'diffusers'`）
- **影响**: 无法按照官方文档启动diffusion模型
- **复现**: 可在4090等普通GPU上复现，不是Blackwell特定问题
- **临时方案**: 在容器内执行 `uv pip install 'sglang[diffusion]' --prerelease=allow`
- **修复方向**: 
  1. **文档修复**（推荐，新人友好）：在文档中明确说明需要额外安装
  2. **Docker修复**（更硬核）：添加build arg支持可选安装diffusion依赖

**注意**：根据最新测试，当前镜像已经包含 diffusion 支持，kevin 的问题可能是：
1. 使用了更早的镜像版本
2. 遇到了其他运行时错误（需要具体错误信息）
3. `FLUX.1-dev` 模型有特殊要求

---

**最后更新**: 2025年1月
