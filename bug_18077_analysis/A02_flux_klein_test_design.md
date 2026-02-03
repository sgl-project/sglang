# A02: FLUX.2-Klein 基准测试设计方案

## 📋 测试模型确定

### 选择的模型
- **模型名称**: FLUX.2-Klein
- **HuggingFace Model ID**: `black-forest-labs/FLUX.2-klein-4B`
- **参数量**: 4B
- **架构**: DiT (Diffusion Transformer)
- **任务类型**: Text-to-Image (T2I)

### 选择理由
1. ✅ **GPU 需求最小**: 在官方支持的 DiT 模型中，预估显存需求最低（16-20GB）
2. ✅ **无 T5 编码器**: 相比 Wan2.1 系列，没有 T5 文本编码器，显存占用更可控
3. ✅ **图像生成任务**: 与 Issue #18077 的目标一致
4. ✅ **官方支持**: SGLang-D 官方支持，文档完善
5. ✅ **任意分辨率**: 支持任意分辨率，便于测试不同配置

---

## 🎯 测试目标（基于 Issue #18077）

根据 Issue 描述，我们需要完成以下目标：

### 1. Benchmarking（基准测试）
建立性能基准，对比 SGLang-D 和 Diffusers 后端的：
- **延迟 (Latency)**: 端到端生成时间
- **吞吐量 (Throughput)**: 请求/秒 (req/s)
- **VRAM 使用**: 峰值显存占用

### 2. Profiling（性能分析）
识别 SGLang-D 路径中的瓶颈：
- **注意力内核**: 注意力计算是否优化
- **内存开销**: 中间激活的内存占用
- **调度器**: 扩散步进过程的效率
- **VAE 编码/解码**: 图像预处理和后处理

### 3. Optimization Analysis（优化分析）
分析优化机会：
- **Sequence Parallelism (SP)**: 是否可以集成，如何集成
- **内存管理**: 是否有优化空间
- **内核优化**: 是否有更优的内核可用

---

## 📊 测试设计

### Phase 1: 基础性能基准测试

#### 1.1 延迟测试 (Latency Benchmark)

**目标**: 测量端到端图像生成延迟

**测试配置**:
```yaml
模型: black-forest-labs/FLUX.2-klein-4B
后端: [sglang, diffusers]
批次大小: [1, 2, 4, 8]
分辨率: 
  - 512x512
  - 768x768
  - 1024x1024
推理步数: [20, 30, 50]
提示词: 固定使用 "A beautiful landscape with mountains and lakes"
运行次数: 每个配置运行 10 次
```

**测量指标**:
- 平均延迟 (Mean Latency)
- 中位数延迟 (Median Latency)
- P50, P90, P99 延迟
- 最小/最大延迟
- 标准差

**输出**:
- 延迟对比表格（SGLang vs Diffusers）
- 延迟随批次大小变化曲线
- 延迟随分辨率变化曲线
- 延迟随推理步数变化曲线

#### 1.2 吞吐量测试 (Throughput Benchmark)

**目标**: 测量系统最大吞吐量

**测试配置**:
```yaml
模型: black-forest-labs/FLUX.2-klein-4B
后端: [sglang, diffusers]
并发级别: [1, 4, 8, 16]
批次大小: 1
分辨率: 512x512
推理步数: 30
请求总数: 100
提示词池: 10 个不同的提示词（循环使用）
```

**测量指标**:
- 请求/秒 (req/s)
- 成功请求数
- 失败请求数
- 平均响应时间
- 系统资源利用率（GPU, CPU, Memory）

**输出**:
- 吞吐量对比表格
- 吞吐量随并发级别变化曲线
- 资源利用率对比

#### 1.3 VRAM 使用测试 (Memory Benchmark)

**目标**: 测量峰值显存占用

**测试配置**:
```yaml
模型: black-forest-labs/FLUX.2-klein-4B
后端: [sglang, diffusers]
批次大小: [1, 2, 4]
分辨率: [512x512, 768x768, 1024x1024]
推理步数: 30
```

**测量指标**:
- 峰值显存占用 (Peak VRAM)
- 平均显存占用 (Mean VRAM)
- 显存占用随时间变化曲线
- 显存分配模式

**输出**:
- 显存使用对比表格
- 显存占用随批次大小变化曲线
- 显存占用随分辨率变化曲线

---

### Phase 2: 性能分析 (Profiling)

#### 2.1 端到端 Pipeline 分析

**目标**: 识别整个 pipeline 的瓶颈

**测试方法**:
- 使用 PyTorch Profiler 进行全 pipeline 分析
- 使用 `--profile-all-stages` 标志

**分析内容**:
```yaml
阶段分解:
  - Text Encoding: 文本编码时间
  - VAE Encoding: 图像编码时间（如果有）
  - Denoising: 去噪阶段时间（主要计算）
  - VAE Decoding: 图像解码时间
  - 其他开销: 数据传输、同步等

每个阶段的指标:
  - 执行时间
  - GPU 利用率
  - 内存访问模式
  - 内核执行时间
```

**输出**:
- Pipeline 阶段时间分解图
- 热点函数列表
- 内核执行时间统计

#### 2.2 Denoising 阶段深度分析

**目标**: 深入分析去噪阶段的性能瓶颈

**测试方法**:
- 使用 PyTorch Profiler 专门分析 denoising 阶段
- 使用 `--profile` 标志（只分析 denoising）

**分析内容**:
```yaml
Denoising 阶段分解:
  - Transformer Forward: Transformer 前向传播
    - Attention 计算
    - Feed-Forward 网络
    - Layer Norm
  - Scheduler Step: 调度器步进
  - 其他操作

每个步骤的指标:
  - 执行时间
  - 内存分配
  - 内核选择（Flash Attention, SDPA 等）
  - 计算强度
```

**输出**:
- Denoising 步骤时间分解
- 注意力计算时间分析
- 内存分配模式分析

#### 2.3 注意力内核分析

**目标**: 分析注意力计算的性能

**测试方法**:
- 对比不同注意力后端
- 分析内核执行时间

**测试配置**:
```yaml
注意力后端:
  - Flash Attention (fa)
  - PyTorch SDPA (torch_sdpa)
  - 其他可用后端

测试场景:
  - 不同序列长度（不同分辨率）
  - 不同批次大小
```

**分析内容**:
- 注意力计算时间
- 内存访问模式
- 内核利用率
- 不同后端的性能对比

**输出**:
- 注意力后端性能对比
- 内核执行时间统计
- 内存带宽利用率

---

### Phase 3: Sequence Parallelism 分析

#### 3.1 SP 支持情况检查

**目标**: 检查当前实现是否支持 Sequence Parallelism

**检查内容**:
```yaml
代码分析:
  - 检查 pipeline 配置中是否有 SP 相关参数
  - 检查 Transformer 层是否支持 SP
  - 检查 VAE 是否支持 SP (vae_sp)

配置检查:
  - GlmImagePipelineConfig 中的 vae_sp 参数
  - 是否有 sp-degree 或类似参数
  - SP 相关的实现代码
```

**输出**:
- SP 支持情况报告
- 代码位置和实现方式
- 集成点分析

#### 3.2 SP 集成方案设计

**目标**: 设计 Sequence Parallelism 集成方案

**分析内容**:
```yaml
集成点识别:
  - VAE 编码/解码: 空间维度分割
  - Transformer 注意力: 序列维度分割
  - 图像 Patch 处理: Patch 级别分割

技术方案:
  - SP 实现方式（Ulysses, Ring Attention 等）
  - 通信模式
  - 同步点
  - 内存管理
```

**输出**:
- SP 集成方案文档
- 代码修改计划
- 预期性能提升

---

### Phase 4: 优化验证测试

#### 4.1 优化前后对比

**目标**: 验证优化效果

**测试方法**:
- 在实施优化后，重复 Phase 1 的测试
- 对比优化前后的性能

**对比指标**:
- 延迟改善
- 吞吐量提升
- 显存使用优化
- 性能瓶颈缓解

**输出**:
- 优化前后性能对比报告
- 性能提升百分比
- 优化效果分析

---

## 📝 测试脚本设计

### 脚本结构

```
bug_18077_analysis/code/
├── bench_flux_klein.py          # 主基准测试脚本
├── profile_flux_klein.py         # 性能分析脚本
├── compare_backends.py           # 后端对比脚本
├── analyze_sp_support.py        # SP 支持分析脚本
└── utils/
    ├── metrics.py                # 指标计算工具
    ├── visualization.py           # 结果可视化
    └── report_generator.py       # 报告生成
```

### 主要脚本功能

#### 1. bench_flux_klein.py
```python
功能:
  - 执行 Phase 1 的所有基准测试
  - 支持 SGLang 和 Diffusers 后端
  - 收集延迟、吞吐量、VRAM 数据
  - 生成 JSON 格式的结果文件

参数:
  --backend: sglang | diffusers | both
  --batch-sizes: 1,2,4,8
  --resolutions: 512x512,768x768,1024x1024
  --num-inference-steps: 20,30,50
  --num-runs: 10
  --output-dir: results/
```

#### 2. profile_flux_klein.py
```python
功能:
  - 执行 Phase 2 的性能分析
  - 使用 PyTorch Profiler
  - 生成 trace 文件
  - 分析 pipeline 阶段时间

参数:
  --backend: sglang | diffusers
  --profile-all-stages: 是否分析所有阶段
  --num-profiled-timesteps: 分析的步数
  --output-dir: profile_results/
```

#### 3. compare_backends.py
```python
功能:
  - 对比 SGLang 和 Diffusers 的结果
  - 生成对比报告（Markdown 表格）
  - 计算性能差距

参数:
  --sglang-results: sglang_results.json
  --diffusers-results: diffusers_results.json
  --output: comparison_report.md
```

#### 4. analyze_sp_support.py
```python
功能:
  - 分析 SP 支持情况
  - 检查代码中的 SP 相关实现
  - 生成 SP 集成方案

参数:
  --model-path: black-forest-labs/FLUX.2-klein-4B
  --output: sp_analysis_report.md
```

---

## 📈 测试结果输出

### 结果文件结构

```
bug_18077_analysis/benchmark/
├── results/
│   ├── sglang_results.json
│   ├── diffusers_results.json
│   └── comparison_report.md
├── profiles/
│   ├── sglang_trace.json.gz
│   ├── diffusers_trace.json.gz
│   └── analysis_report.md
└── reports/
    ├── phase1_benchmark_report.md
    ├── phase2_profiling_report.md
    ├── phase3_sp_analysis.md
    └── final_summary.md
```

### 报告内容

#### Phase 1 报告
- 性能基准测试结果表格
- 延迟、吞吐量、VRAM 对比图表
- 性能差距分析

#### Phase 2 报告
- Pipeline 阶段时间分解
- 性能瓶颈识别
- 优化建议

#### Phase 3 报告
- SP 支持情况
- SP 集成方案
- 预期性能提升

#### 最终总结报告
- 所有测试结果汇总
- 关键发现
- 优化建议
- 下一步行动计划

---

## 🔧 测试环境要求

### 硬件要求
- **GPU**: RTX 4090 (24GB) 或更高
- **系统 RAM**: 至少 32GB（推荐 64GB）
- **存储**: 足够的空间存储模型和结果

### 软件要求
- Python: 3.8+
- PyTorch: 2.0+
- SGLang: 最新版本（支持 diffusion）
- Diffusers: 最新版本
- CUDA: 11.8+

### 环境变量
```bash
# PyTorch Profiler
export SGLANG_TORCH_PROFILER_DIR=/path/to/profile_log

# Cache-DiT (可选)
export SGLANG_CACHE_DIT_ENABLED=false  # 测试时先禁用，确保公平对比
```

---

## 📅 测试计划时间表

### Week 1: 环境准备和 Phase 1
- Day 1-2: 环境搭建，模型下载
- Day 3-5: Phase 1 基准测试（延迟、吞吐量、VRAM）
- Day 6-7: 结果分析和报告生成

### Week 2: Phase 2 和 Phase 3
- Day 1-3: Phase 2 性能分析（Profiling）
- Day 4-5: Phase 3 SP 支持分析
- Day 6-7: 结果整合和优化方案设计

### Week 3: 优化实施和验证（可选）
- Day 1-3: 实施优化（如果时间允许）
- Day 4-5: Phase 4 优化验证测试
- Day 6-7: 最终报告撰写

---

## ✅ 测试检查清单

### Phase 1: 基准测试
- [ ] 延迟测试完成（所有配置）
- [ ] 吞吐量测试完成（所有并发级别）
- [ ] VRAM 使用测试完成（所有配置）
- [ ] 结果数据收集完成
- [ ] 对比报告生成完成

### Phase 2: 性能分析
- [ ] Pipeline 阶段分析完成
- [ ] Denoising 深度分析完成
- [ ] 注意力内核分析完成
- [ ] Trace 文件生成完成
- [ ] 瓶颈识别报告完成

### Phase 3: SP 分析
- [ ] SP 支持情况检查完成
- [ ] 代码分析完成
- [ ] SP 集成方案设计完成
- [ ] 集成点识别完成

### Phase 4: 优化验证（可选）
- [ ] 优化实施完成
- [ ] 优化前后对比测试完成
- [ ] 性能提升验证完成

---

## 🔗 相关文档

- [A01_B01: 原始 Issue](./A01_B01_original_issue.md) - Issue 详细内容
- [A01_B08: 支持的模型和最小 GPU 需求](./A01_B08_supported_models_and_minimum_gpu.md) - 模型选择
- [A01_B05: 基准测试设置](./A01_B05_benchmark_setup.md) - 测试工具使用

---

**状态**: 📝 测试设计完成，等待实施
