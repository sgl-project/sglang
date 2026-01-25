# A01_B06: 学习路径 - GLM Blackwell性能优化

## 相关文档
- [A01_B01: 原始 Issue 内容](./A01_B01_original_issue.md) - Issue详情
- [A01_B02: 优化项详解](./A01_B02_optimization_items.md) - 每个优化项的详细说明
- [A01_B03: 性能分析](./A01_B03_performance_analysis.md) - 性能对比和瓶颈分析

---

## 🎯 学习目标

通过这个issue学习：
1. **GPU性能优化** - 如何在Blackwell GPU上优化LLM推理
2. **量化技术** - FP4/FP8量化的原理和应用
3. **Kernel融合** - 如何通过kernel融合提升性能
4. **性能分析** - 如何使用profiler识别性能瓶颈
5. **MoE优化** - MoE模型的特有优化技术

---

## 📚 第一阶段：基础知识（1-2周）

### 1.1 理解GPU架构
**目标**: 理解Blackwell GPU的架构特性

**学习资源**:
- NVIDIA Blackwell架构文档
- SM100/SM120架构特性
- Tensor Core和FP8支持

**实践**:
- 阅读NVIDIA官方文档
- 理解不同GPU架构的差异（Hopper vs Blackwell）

### 1.2 理解量化基础
**目标**: 理解FP4/FP8量化的原理

**学习资源**:
- [SGLang量化文档](https://docs.sglang.ai/advanced_features/quantization.html)
- FP4/FP8量化原理
- KV Cache量化

**实践**:
- 运行量化模型
- 对比量化前后的性能差异

### 1.3 理解MoE架构
**目标**: 理解MoE模型的结构和特点

**学习资源**:
- MoE模型原理
- Expert Parallelism
- MoE的权重加载

**实践**:
- 运行MoE模型
- 理解expert routing

---

## 📚 第二阶段：性能分析（1周）

### 2.1 学习使用Profiler
**目标**: 掌握性能分析工具

**学习资源**:
- PyTorch Profiler
- CUDA Profiler
- SGLang的bench_one_batch_server

**实践**:
```bash
# 运行profiler
SGLANG_TORCH_PROFILER_DIR="./" \
python -m sglang.bench_one_batch_server \
  --model baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024 \
  --profile \
  --profile-steps 10 \
  --show-report \
  --profile-by-stage
```

**任务**:
1. 运行profiler并分析结果
2. 识别性能瓶颈
3. 理解kernel执行时间

### 2.2 分析性能瓶颈
**目标**: 识别和定位性能问题

**学习内容**:
- Kernel执行时间分析
- 内存访问模式
- 数据依赖关系

**实践**:
- 对比FP8 vs BF16的性能
- 分析额外kernel的开销
- 识别融合机会

---

## 📚 第三阶段：优化技术（2-3周）

### 3.1 Kernel融合
**目标**: 学习kernel融合技术

**学习资源**:
- CUDA kernel融合原理
- Flashinfer的kernel实现
- TRT-LLM的kernel优化

**实践任务**:
1. **FP8 KV buffer kernel融合**
   - 理解FP8 KV buffer的实现
   - 学习Q cast到FP8的操作
   - 尝试融合这两个操作

2. **FlashinferFP4MoE融合**
   - 理解MoE的forward流程
   - 识别可以融合的elementwise操作
   - 实现融合kernel

### 3.2 量化优化
**目标**: 优化量化操作

**学习资源**:
- CUTLASS量化kernel
- Flashinfer的FP4量化实现
- Scale layout优化

**实践任务**:
1. **改进scaled_fp4_quant**
   - 对比sgl-kernel和flashinfer的实现
   - 切换到CUTLASS backend
   - 测量性能提升

2. **优化scale layout**
   - 理解不同scale layout的影响
   - 实现动态scale layout选择
   - 针对低并发场景优化

### 3.3 Backend优化
**目标**: 优化不同的backend

**学习资源**:
- Flashinfer backend
- TRT-LLM backend
- 不同backend的适用场景

**实践任务**:
1. **TRT-LLM MHA优化**
   - 理解auto-enable的逻辑
   - 确保speculative decoding时正确设置
   - 测量性能提升

2. **Flashinfer TRTLLM MoE**
   - 理解MoE backend的选择
   - 确保GLM模型自动使用flashinfer_trtllm
   - 对比性能差异

---

## 📚 第四阶段：高级优化（2-3周）

### 4.1 AllReduce优化
**目标**: 优化通信操作

**学习资源**:
- TRT AllReduce Fusion
- TRTLLM MNNVL AllReduce
- 单节点和多节点的通信优化

**实践任务**:
1. 理解AllReduce的开销
2. 对比不同的AllReduce实现
3. 实现MNNVL AllReduce

### 4.2 FP8 KV Cache优化
**目标**: 解决FP8 KV cache的性能问题

**学习资源**:
- FP8量化的开销分析
- KV cache的实现
- 数据shuffle优化

**实践任务**:
1. 分析FP8 KV cache的瓶颈
2. 优化量化操作
3. 减少数据shuffle的开销
4. 尝试融合相关kernel

---

## 📚 第五阶段：实践项目（持续）

### 5.1 实现一个优化项
**目标**: 选择一个优化项并实现

**推荐项目**:
1. **动态scale layout选择** (5-10%性能提升)
   - 难度: 中等
   - 影响: 高
   - 适合: 有一定CUDA基础

2. **改进scaled_fp4_quant** (1-2%性能提升)
   - 难度: 低
   - 影响: 小
   - 适合: 初学者

3. **FP8 KV buffer kernel融合** (3%性能提升)
   - 难度: 高
   - 影响: 中等
   - 适合: 有kernel开发经验

### 5.2 性能测试和验证
**目标**: 验证优化的效果

**步骤**:
1. 实现优化
2. 运行性能测试
3. 对比优化前后的性能
4. 验证准确性
5. 提交PR

---

## 🔗 相关文档链接

### SGLang官方文档
- [量化文档](https://docs.sglang.ai/advanced_features/quantization.html)
- [服务器参数](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [性能分析](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)

### 相关Issue
- [#17166](https://github.com/sgl-project/sglang/issues/17166) - GLM 4.7 + NVFP4 + MTP
- [#16755](https://github.com/sgl-project/sglang/issues/16755) - Auto-enable TRT-LLM MHA
- [#17237](https://github.com/sgl-project/sglang/issues/17237) - Very slow trtllm_allreduce_fusion
- [#12787](https://github.com/sgl-project/sglang/issues/12787) - Add trtllm mnnvl allreduce

### 外部资源
- NVIDIA Blackwell架构文档
- Flashinfer文档
- TRT-LLM文档
- CUTLASS文档

---

## 📝 学习检查清单

### 基础知识
- [ ] 理解Blackwell GPU架构
- [ ] 理解FP4/FP8量化原理
- [ ] 理解MoE模型结构
- [ ] 理解Tensor Parallelism

### 性能分析
- [ ] 能够使用PyTorch Profiler
- [ ] 能够分析kernel执行时间
- [ ] 能够识别性能瓶颈
- [ ] 能够对比不同配置的性能

### 优化技术
- [ ] 理解kernel融合原理
- [ ] 能够优化量化操作
- [ ] 理解不同backend的特点
- [ ] 能够优化通信操作

### 实践能力
- [ ] 能够实现一个优化项
- [ ] 能够运行性能测试
- [ ] 能够验证优化的效果
- [ ] 能够提交PR

---

## 🎓 学习建议

### 对于初学者
1. **从基础开始**: 先理解GPU架构和量化原理
2. **动手实践**: 运行profiler，观察性能数据
3. **小步快跑**: 先实现简单的优化项（如改进scaled_fp4_quant）
4. **寻求帮助**: 在Slack或GitHub上提问

### 对于有经验的开发者
1. **深入分析**: 分析FP8 KV cache的性能问题
2. **实现复杂优化**: 尝试kernel融合
3. **贡献代码**: 实现动态scale layout选择
4. **分享经验**: 在社区分享优化经验

---

## 🚀 快速开始

### 第一步：环境准备
```bash
# 安装SGLang
pip install sglang

# 准备GLM模型
# 下载或使用 baseten-admin/glm-4.7-fp8-attn-fp4-mlp
```

### 第二步：运行基准测试
```bash
# 运行性能测试
python3 -m sglang.bench_one_batch_server \
  --model baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024
```

### 第三步：运行Profiler
```bash
# 使用profiler分析性能
SGLANG_TORCH_PROFILER_DIR="./" \
python -m sglang.bench_one_batch_server \
  --model baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --profile \
  --profile-steps 10 \
  --show-report
```

### 第四步：分析结果
1. 查看profiler输出
2. 识别性能瓶颈
3. 对比不同配置的性能
4. 制定优化计划

---

## 💡 学习技巧

1. **理论与实践结合**: 不仅要理解原理，还要动手实践
2. **对比分析**: 对比不同配置的性能差异
3. **逐步深入**: 从简单到复杂，逐步学习
4. **记录笔记**: 记录学习过程中的发现和问题
5. **参与讨论**: 在GitHub issue或Slack上参与讨论

---

## 📈 预期学习成果

完成这个学习路径后，你将能够：
1. ✅ 理解GPU性能优化的基本原理
2. ✅ 掌握量化技术的应用
3. ✅ 能够使用profiler分析性能
4. ✅ 能够实现kernel融合优化
5. ✅ 能够优化MoE模型的性能
6. ✅ 能够为SGLang贡献优化代码
