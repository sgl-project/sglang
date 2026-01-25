# Bug #17680 分析文档

## 📋 文档结构

### A01: MoE Tensor Parallelism Bug 详解
- **[A01_moe_tp_bug.md](./A01_moe_tp_bug.md)** ⭐ **主文档**
  - MoE模型在TP=2时的RuntimeError问题
  - 错误原因分析
  - 修复方案

### A01_Bxx: 平行文档（与 A01 同级）

- **[A01_B01_original_issue.md](./A01_B01_original_issue.md)** ⭐ **原始 Issue 内容**
  - Issue 链接和标题
  - 问题描述和错误堆栈
  - 复现步骤和环境信息

- **[A01_B02_fix_analysis.md](./A01_B02_fix_analysis.md)** ⭐ **修复分析**
  - 问题定位
  - 修复前后代码对比
  - 修复理由与逻辑

- **[A01_B03_code_changes.md](./A01_B03_code_changes.md)** ⭐ **代码变更**
  - 修复的具体代码位置
  - 代码变更说明

- **[A01_B04_problem_analysis.md](./A01_B04_problem_analysis.md)** ⭐ **问题深度分析** 🔍 **重点**
  - 错误信息解析
  - Tensor Parallelism 工作原理
  - 维度不匹配的根本原因
  - 关键变量分析
  - 关键疑问和下一步分析方向

- **[A01_B05_theoretical_analysis.md](./A01_B05_theoretical_analysis.md)** ⭐ **原理深度分析** 🎓 **核心**
  - Tensor Parallelism 基本原理
  - RowParallel vs ColumnParallel
  - MoE权重结构和加载流程
  - 量化对权重的影响
  - 数学原理分析
  - 解决方案的原理

- **[A01_B06_documentation_links.md](./A01_B06_documentation_links.md)** 📚 **官方文档链接**
  - SGLang官方文档网站
  - 文档结构导航
  - 针对当前Bug的推荐阅读顺序
  - 关键文档链接

---

## 🎯 快速导航

1. **想了解问题详情** → [A01_B01_original_issue.md](./A01_B01_original_issue.md) ⭐ **从这里开始**
2. **想阅读官方文档** → [A01_B06_documentation_links.md](./A01_B06_documentation_links.md) 📚 **官方文档**
3. **想从原理理解问题** → [A01_B05_theoretical_analysis.md](./A01_B05_theoretical_analysis.md) 🎓 **核心推荐**
4. **想深入理解问题** → [A01_B04_problem_analysis.md](./A01_B04_problem_analysis.md) 🔍 **重点推荐**
5. **想了解修复分析** → [A01_B02_fix_analysis.md](./A01_B02_fix_analysis.md)
6. **想查看代码变更** → [A01_B03_code_changes.md](./A01_B03_code_changes.md)
7. **想了解整体问题** → [A01_moe_tp_bug.md](./A01_moe_tp_bug.md) ⭐ **主文档**

---

## 📁 其他文件

### 代码相关
- `code/` - 修复代码和补丁

### 测试相关
- `test/` - 测试脚本（如需要）

---

## 📝 文档命名规则

- **A01_xxx.md**: 主文档（A01 系列）
- **A01_B01_xxx.md**: A01 的平行文档（A01_B01 系列）

---

## 🔗 Issue 链接

https://github.com/sgl-project/sglang/issues/17680

## 🐛 Bug 摘要

**问题**: MoE模型 `MedAIBase/AntAngelMed-INT4` 在使用 `--tp-size 2` 时，在第二个GPU (TP1) 上加载权重时抛出 `RuntimeError: start (8) + length (8) exceeds dimension size (8)`。

**错误位置**: `sglang/srt/layers/moe/fused_moe_triton/layer.py`, line 501, in `_load_w2`

**修复方案**: 在 `_load_w2` 方法中添加 padding 逻辑，处理权重维度未正确对齐的情况（类似于 `RowParallelLinear` 的处理方式）。
