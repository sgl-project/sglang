# SGLang Bug分析 - 完整学习资料

## 📚 文档结构

### 核心文档
1. **[00_基础概念完整学习指南.md](./00_基础概念完整学习指南.md)** ⭐ **从这里开始**
   - 理解Issue #17680和#17526所需的所有基础概念
   - 7个主要分类，30+个核心概念
   - 每个概念都有官方文档链接

2. **[01_官方文档快速索引.md](./01_官方文档快速索引.md)** 📖 **文档索引**
   - 按优先级分类的官方文档
   - 快速链接汇总
   - 阅读建议和时间分配

3. **[02_学习计划.md](./02_学习计划.md)** 📅 **学习计划**
   - 快速学习路径（2-3周）
   - 深入学习路径（4-6周）
   - 每日学习任务

---

## 🎯 学习目标

通过系统学习这些基础概念，能够：
1. ✅ 理解Issue #17680 (MoE Tensor Parallelism Bug) 的根本原因
2. ✅ 理解Issue #17526 (GLM Blackwell性能优化) 的优化方法
3. ✅ 能够阅读和修改SGLang代码
4. ✅ 能够实现性能优化方案

---

## 🚀 快速开始

### 第一步：阅读基础概念指南
打开 `00_基础概念完整学习指南.md`，了解需要学习的所有概念。

### 第二步：查看官方文档索引
打开 `01_官方文档快速索引.md`，找到相关文档的链接。

### 第三步：制定学习计划
打开 `02_学习计划.md`，选择适合自己的学习路径。

### 第四步：开始学习
按照学习计划，逐一学习每个概念。

---

## 📋 学习检查清单

### 基础概念（必须掌握）
- [ ] Transformer架构
- [ ] LLM推理流程
- [ ] GPU架构基础
- [ ] CUDA基础
- [ ] 量化基础

### Issue #17680相关（MoE TP Bug）
- [ ] Tensor Parallelism
- [ ] MoE架构
- [ ] RowParallel vs ColumnParallel
- [ ] 权重加载和分片
- [ ] Padding和边界检查

### Issue #17526相关（性能优化）
- [ ] Blackwell GPU架构
- [ ] FP8/FP4量化
- [ ] KV Cache量化
- [ ] Kernel融合
- [ ] 性能分析工具
- [ ] Flashinfer和TRT-LLM backend

### SGLang特定
- [ ] SGLang架构
- [ ] SGLang Backend
- [ ] SGLang权重加载
- [ ] SGLang MoE实现

---

## 🔗 相关Issue

- [Issue #17680](https://github.com/sgl-project/sglang/issues/17680) - MoE Tensor Parallelism Bug
- [Issue #17526](https://github.com/sgl-project/sglang/issues/17526) - GLM Blackwell性能优化

---

## 📁 相关分析文档

- `../../bug_17680_analysis/` - Issue #17680的详细分析
- `../../bug_17526_analysis/` - Issue #17526的详细分析

---

**开始你的学习之旅！** 🎓
