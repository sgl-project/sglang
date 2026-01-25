# A01_B07: 完整学习流程 - 从零开始解决GLM Blackwell性能优化

## 🎯 学习目标

**最终目标**: 能够理解和解决GLM 4.7在Blackwell GPU上的性能优化问题

**具体目标**:
1. 理解性能优化的基本原理
2. 能够使用工具分析性能瓶颈
3. 能够实现和验证优化方案
4. 能够为SGLang贡献优化代码

---

## 📅 学习时间规划

**总时间**: 8-12周（根据个人基础调整）

- **第1-2周**: 基础概念和环境搭建
- **第3-4周**: 深入理解GPU和量化
- **第5-6周**: 性能分析和工具使用
- **第7-8周**: 优化技术学习
- **第9-10周**: 实践项目
- **第11-12周**: 深入优化和贡献

---

## 🚀 第一阶段：基础概念和环境搭建（第1-2周）

### 目标
- 理解LLM推理的基本概念
- 搭建开发环境
- 能够运行SGLang

### 第1周：理解LLM推理基础

#### Day 1-2: LLM推理基础概念

**学习内容**:
1. **什么是LLM推理？**
   - Transformer架构基础
   - 前向传播过程
   - Token生成流程

**学习资源**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [How GPT Works](https://www.youtube.com/watch?v=wjZofJX0v4M)

**实践任务**:
- [ ] 阅读Transformer论文（至少理解架构）
- [ ] 理解token、prompt、generation的概念
- [ ] 画一个简单的Transformer前向传播流程图

#### Day 3-4: SGLang基础

**学习内容**:
1. **SGLang是什么？**
   - SGLang的定位和作用
   - 与其他框架的对比（vLLM, TensorRT-LLM）

**学习资源**:
- [SGLang官方文档 - Get Started](https://docs.sglang.ai/get_started/install.html)
- [SGLang README](https://github.com/sgl-project/sglang)

**实践任务**:
- [ ] 阅读SGLang README
- [ ] 理解SGLang的核心特性
- [ ] 了解SGLang的架构（Runtime + Language）

#### Day 5-7: 环境搭建和第一个模型

**学习内容**:
1. **安装SGLang**
2. **运行第一个模型**
3. **理解基本命令**

**实践任务**:
```bash
# 1. 安装SGLang
pip install sglang

# 2. 运行一个简单模型（如果没有GPU，可以用CPU模式）
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000

# 3. 发送一个测试请求
python3 -m sglang.test.send_one \
  --prompt "Hello, how are you?" \
  --max-new-tokens 100
```

**检查点**:
- [ ] 能够成功安装SGLang
- [ ] 能够启动服务器
- [ ] 能够发送请求并收到回复
- [ ] 理解基本的命令行参数

### 第2周：理解性能优化基础

#### Day 1-3: 性能指标和测量

**学习内容**:
1. **性能指标**
   - Throughput (token/s)
   - Latency (ms)
   - Memory usage
   - GPU utilization

2. **如何测量性能**
   - 使用benchmark工具
   - 理解性能报告

**实践任务**:
```bash
# 运行benchmark
python3 -m sglang.bench_one_batch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 512 \
  --output-len 256

# 观察输出：
# - Throughput (token/s)
# - Latency
# - Memory usage
```

**学习资源**:
- [SGLang Benchmark文档](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)

**检查点**:
- [ ] 理解各种性能指标的含义
- [ ] 能够运行benchmark
- [ ] 能够解读benchmark结果

#### Day 4-5: GPU基础概念

**学习内容**:
1. **GPU架构基础**
   - CUDA核心
   - Tensor Core
   - 内存层次结构（Global Memory, Shared Memory, Register）

2. **GPU编程基础**
   - Kernel概念
   - Grid, Block, Thread
   - 内存访问模式

**学习资源**:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

**实践任务**:
- [ ] 阅读CUDA基础文档
- [ ] 理解kernel、grid、block的概念
- [ ] 理解GPU内存层次结构

#### Day 6-7: 量化基础

**学习内容**:
1. **什么是量化？**
   - FP32, FP16, BF16, FP8, FP4, INT8, INT4
   - 量化的目的（减少内存、加速计算）

2. **量化方法**
   - 静态量化 vs 动态量化
   - 权重量化 vs 激活量化

**学习资源**:
- [SGLang量化文档](https://docs.sglang.ai/advanced_features/quantization.html)
- [Quantization Explained](https://pytorch.org/docs/stable/quantization.html)

**实践任务**:
```bash
# 运行量化模型
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --quantization awq

# 对比量化前后的性能
# 1. 无量化
python3 -m sglang.bench_one_batch_server --model-path ... --batch-size 4

# 2. 有量化
python3 -m sglang.bench_one_batch_server --model-path ... --quantization awq --batch-size 4

# 对比结果
```

**检查点**:
- [ ] 理解不同精度格式的特点
- [ ] 理解量化的目的和原理
- [ ] 能够运行量化模型
- [ ] 能够对比量化前后的性能

---

## 🔍 第二阶段：深入理解GPU和量化（第3-4周）

### 目标
- 深入理解Blackwell GPU架构
- 理解FP4/FP8量化原理
- 理解MoE模型

### 第3周：Blackwell GPU架构

#### Day 1-3: Blackwell架构特性

**学习内容**:
1. **Blackwell GPU特性**
   - SM100/SM120架构
   - FP8 Tensor Core支持
   - 内存带宽提升
   - 新的指令集

**学习资源**:
- [NVIDIA Blackwell架构白皮书](https://www.nvidia.com/en-us/data-center/blackwell/)
- [Blackwell GPU技术文档](https://developer.nvidia.com/blackwell)

**实践任务**:
- [ ] 阅读Blackwell架构文档
- [ ] 理解FP8 Tensor Core的优势
- [ ] 对比Blackwell和Hopper的差异
- [ ] 理解为什么Blackwell适合FP4/FP8量化

#### Day 4-5: Tensor Core和矩阵乘法

**学习内容**:
1. **Tensor Core原理**
   - 矩阵乘法的硬件加速
   - FP16, BF16, FP8, FP4支持
   - 混合精度计算

2. **GEMM (General Matrix Multiply)**
   - 矩阵乘法的优化
   - 不同精度的GEMM性能

**学习资源**:
- [Tensor Core文档](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [CUTLASS文档](https://github.com/NVIDIA/cutlass)

**实践任务**:
- [ ] 理解Tensor Core的工作原理
- [ ] 理解为什么FP8/FP4可以加速
- [ ] 了解CUTLASS库的作用

#### Day 6-7: GPU内存和带宽

**学习内容**:
1. **GPU内存层次**
   - Global Memory
   - Shared Memory
   - L2 Cache
   - Register

2. **内存访问优化**
   - 合并访问（Coalesced Access）
   - 内存对齐
   - 减少内存访问

**学习资源**:
- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)

**检查点**:
- [ ] 理解Blackwell GPU的关键特性
- [ ] 理解Tensor Core的作用
- [ ] 理解GPU内存层次结构

### 第4周：FP4/FP8量化深入

#### Day 1-3: FP8量化原理

**学习内容**:
1. **FP8格式**
   - E4M3格式（4位指数，3位尾数）
   - E5M2格式（5位指数，2位尾数）
   - 表示范围和精度

2. **FP8量化过程**
   - Scale计算
   - 量化公式
   - 反量化公式

**学习资源**:
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA FP8文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.fp8)

**实践任务**:
- [ ] 理解FP8的两种格式
- [ ] 理解scale的作用
- [ ] 手动计算一个FP8量化例子

#### Day 4-5: FP4量化原理

**学习内容**:
1. **FP4格式**
   - 4位浮点数表示
   - 量化方法
   - 与FP8的对比

2. **FP4在MoE中的应用**
   - MoE权重量化
   - Expert权重的量化

**学习资源**:
- [SGLang FP4量化实现](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/quantization)

**实践任务**:
```bash
# 运行FP4量化模型
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --quantization modelopt_fp4 \
  --tp 4

# 观察内存使用和性能
```

#### Day 6-7: KV Cache量化

**学习内容**:
1. **KV Cache是什么？**
   - Attention机制中的K和V缓存
   - 为什么需要缓存
   - 缓存的内存占用

2. **KV Cache量化**
   - FP8 KV Cache
   - BF16 KV Cache
   - 量化的权衡（内存 vs 精度）

**实践任务**:
```bash
# 对比不同KV Cache dtype
# 1. BF16 KV Cache
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --kv-cache-dtype bfloat16

# 2. FP8 KV Cache
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --kv-cache-dtype fp8_e4m3

# 对比性能和内存使用
```

**检查点**:
- [ ] 理解FP8和FP4的量化原理
- [ ] 理解KV Cache的作用和量化
- [ ] 能够运行和对比不同量化配置

---

## 🔬 第三阶段：性能分析和工具使用（第5-6周）

### 目标
- 掌握性能分析工具
- 能够识别性能瓶颈
- 理解性能优化的方法

### 第5周：性能分析工具

#### Day 1-3: PyTorch Profiler

**学习内容**:
1. **PyTorch Profiler基础**
   - 如何启动profiler
   - 如何查看结果
   - 理解timeline

**实践任务**:
```bash
# 运行profiler
SGLANG_TORCH_PROFILER_DIR="./profile_results" \
python -m sglang.bench_one_batch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024 \
  --profile \
  --profile-steps 10 \
  --show-report \
  --profile-by-stage

# 查看结果
# 1. 在终端查看报告
# 2. 使用TensorBoard查看详细timeline
tensorboard --logdir=./profile_results
```

**学习资源**:
- [PyTorch Profiler文档](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [SGLang Profiling文档](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)

**检查点**:
- [ ] 能够运行profiler
- [ ] 能够查看profiler报告
- [ ] 理解timeline的含义

#### Day 4-5: 分析性能瓶颈

**学习内容**:
1. **如何识别瓶颈**
   - Kernel执行时间
   - 内存访问时间
   - 通信时间
   - 数据依赖

2. **分析Issue #17526中的瓶颈**
   - FP8 KV cache的额外kernel
   - 量化操作的开销
   - 数据shuffle的开销

**实践任务**:
```bash
# 运行profiler并分析
# 1. FP8 KV cache配置
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --kv-cache-dtype fp8_e4m3 \
  --profile

# 2. BF16 KV cache配置
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --kv-cache-dtype bfloat16 \
  --profile

# 3. 对比两个配置的profiler结果
# 找出FP8配置中的额外kernel：
# - DeviceGemmFp4GemmSm100
# - cvt_fp16_to_fp4
# - float8_copy_kernel_cuda
# - _fused_fp8_set_kv_buffer_kernel
```

**检查点**:
- [ ] 能够识别性能瓶颈
- [ ] 能够对比不同配置的性能
- [ ] 理解Issue中提到的瓶颈kernel

#### Day 6-7: CUDA Profiler

**学习内容**:
1. **NVIDIA Nsight Systems**
   - 系统级性能分析
   - GPU和CPU的timeline
   - 内存使用分析

2. **NVIDIA Nsight Compute**
   - Kernel级性能分析
   - 指令级分析
   - 内存访问分析

**实践任务**:
```bash
# 使用Nsight Systems
nsys profile --trace=cuda,nvtx \
  python3 -m sglang.launch_server \
    --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp

# 查看结果
nsys-ui report.nsys-rep
```

**学习资源**:
- [Nsight Systems文档](https://developer.nvidia.com/nsight-systems)
- [Nsight Compute文档](https://developer.nvidia.com/nsight-compute)

**检查点**:
- [ ] 能够使用Nsight Systems
- [ ] 能够分析kernel性能
- [ ] 理解GPU利用率

### 第6周：性能优化方法

#### Day 1-3: Kernel融合

**学习内容**:
1. **什么是Kernel融合？**
   - 减少kernel启动开销
   - 减少内存访问
   - 提高GPU利用率

2. **融合策略**
   - Elementwise操作融合
   - 数据转换融合
   - 减少中间结果

**实践任务**:
- [ ] 理解Issue中的kernel融合任务
  - FP8 KV buffer kernel融合
  - FlashinferFP4MoE中的elementwise融合
- [ ] 阅读相关代码
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - Flashinfer的FP4 MoE实现

**学习资源**:
- [Kernel Fusion文档](https://developer.nvidia.com/blog/cuda-pro-tip-kepler-shuffle-improves-performance-more/)

#### Day 4-5: 内存优化

**学习内容**:
1. **内存访问优化**
   - 合并访问
   - 减少内存带宽使用
   - 使用Shared Memory

2. **KV Cache优化**
   - Paged Attention
   - KV Cache量化
   - 减少内存占用

**实践任务**:
- [ ] 理解Paged Attention的原理
- [ ] 理解KV Cache量化的权衡
- [ ] 分析FP8 KV cache的内存节省

#### Day 6-7: 通信优化

**学习内容**:
1. **Tensor Parallelism通信**
   - AllReduce操作
   - AllGather操作
   - 通信开销

2. **AllReduce优化**
   - TRT AllReduce Fusion
   - TRTLLM MNNVL AllReduce
   - 单节点优化

**实践任务**:
- [ ] 理解Issue #12787中的AllReduce优化
- [ ] 理解Issue #17237中的AllReduce Fusion问题
- [ ] 阅读相关代码

**检查点**:
- [ ] 理解kernel融合的原理
- [ ] 理解内存优化的方法
- [ ] 理解通信优化的方法

---

## 🛠️ 第四阶段：优化技术学习（第7-8周）

### 目标
- 学习具体的优化技术
- 理解SGLang的优化实现
- 能够阅读和修改优化代码

### 第7周：量化优化

#### Day 1-3: FP4量化优化

**学习内容**:
1. **scaled_fp4_quant优化**
   - 当前的实现
   - CUTLASS backend
   - Flashinfer的FP4量化

**实践任务**:
```python
# 阅读代码
# python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
# 理解scaled_fp4_quant的实现

# 对比不同实现
# 1. sgl-kernel的实现
# 2. flashinfer.fp4_quantization.fp4_quantize
# 3. CUTLASS的实现
```

**学习资源**:
- [Flashinfer FP4量化](https://github.com/flashinfer-ai/flashinfer)
- [CUTLASS FP4 GEMM](https://github.com/NVIDIA/cutlass)

#### Day 4-5: Scale Layout优化

**学习内容**:
1. **Scale Layout是什么？**
   - 量化的scale存储方式
   - 不同layout的影响
   - 低并发场景的优化

**实践任务**:
- [ ] 理解Issue中提到的scale layout优化
- [ ] 理解为什么低并发场景需要特殊优化
- [ ] 阅读相关代码

#### Day 6-7: FP8 KV Cache优化

**学习内容**:
1. **FP8 KV Cache的问题**
   - 额外的量化操作
   - 数据shuffle开销
   - 如何优化

**实践任务**:
- [ ] 分析Issue中提到的瓶颈kernel
- [ ] 理解如何减少这些开销
- [ ] 思考优化方案

**检查点**:
- [ ] 理解FP4量化的优化方法
- [ ] 理解scale layout的影响
- [ ] 理解FP8 KV cache的问题

### 第8周：Backend优化

#### Day 1-3: Flashinfer Backend

**学习内容**:
1. **Flashinfer是什么？**
   - Flash Attention的实现
   - MoE backend
   - TRTLLM backend

2. **Flashinfer TRTLLM MoE**
   - 为什么性能更好
   - 如何自动启用

**实践任务**:
```bash
# 对比不同MoE backend
# 1. 默认backend
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --moe-runner-backend standard

# 2. Flashinfer TRTLLM backend
python3 -m sglang.launch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --moe-runner-backend flashinfer_trtllm

# 对比性能
```

**学习资源**:
- [Flashinfer文档](https://github.com/flashinfer-ai/flashinfer)

#### Day 4-5: TRT-LLM MHA Backend

**学习内容**:
1. **TRT-LLM MHA Backend**
   - 为什么性能更好
   - 自动启用逻辑
   - Speculative decoding的兼容性

**实践任务**:
- [ ] 理解Issue #16755中的auto-enable逻辑
- [ ] 理解为什么speculative topk=1时需要特殊处理
- [ ] 阅读相关代码

#### Day 6-7: Backend选择策略

**学习内容**:
1. **如何选择Backend**
   - 不同backend的适用场景
   - 自动选择逻辑
   - 性能对比

**实践任务**:
- [ ] 理解SGLang的backend选择逻辑
- [ ] 能够根据场景选择合适的backend
- [ ] 能够优化backend选择

**检查点**:
- [ ] 理解不同backend的特点
- [ ] 理解backend选择的逻辑
- [ ] 能够优化backend使用

---

## 🎯 第五阶段：实践项目（第9-10周）

### 目标
- 选择一个优化项实现
- 验证优化效果
- 提交PR

### 第9周：选择和实践优化项

#### Day 1-2: 选择优化项

**推荐项目（按难度排序）**:

1. **改进scaled_fp4_quant** (难度: ⭐⭐)
   - 切换到CUTLASS backend
   - 预期提升: 1-2%
   - 适合: 有一定Python和CUDA基础

2. **动态scale layout选择** (难度: ⭐⭐⭐)
   - 根据并发级别选择scale layout
   - 预期提升: 5-10%
   - 适合: 有量化基础

3. **FP8 KV buffer kernel融合** (难度: ⭐⭐⭐⭐)
   - 融合FP8 KV buffer和Q cast
   - 预期提升: 3%
   - 适合: 有CUDA kernel开发经验

**实践任务**:
- [ ] 选择一个优化项
- [ ] 阅读相关代码
- [ ] 理解当前的实现
- [ ] 制定实现计划

#### Day 3-7: 实现优化

**实践步骤**:
1. **Fork和Clone代码**
```bash
# Fork SGLang仓库
# Clone到本地
git clone https://github.com/YOUR_USERNAME/sglang.git
cd sglang
```

2. **创建分支**
```bash
git checkout -b optimize/your-optimization-name
```

3. **实现优化**
   - 修改代码
   - 添加测试
   - 确保代码质量

4. **测试优化**
```bash
# 运行测试
pytest tests/...

# 运行benchmark
python3 -m sglang.bench_one_batch_server \
  --model-path baseten-admin/glm-4.7-fp4 \
  --your-optimization-flag
```

**检查点**:
- [ ] 代码实现完成
- [ ] 测试通过
- [ ] 性能有提升

### 第10周：验证和提交

#### Day 1-3: 性能验证

**实践任务**:
1. **运行完整的性能测试**
```bash
# 使用Issue中的测试命令
python3 -m sglang.bench_one_batch_server \
  --model-path baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --your-optimization

# 对比优化前后的性能
```

2. **准确性验证**
```bash
# 运行GSM8K测试
python3 benchmark/gsm8k/bench_sglang.py \
  --num-shots 8 \
  --num-questions 1209 \
  --parallel 1209 \
  --platinum
```

3. **Profiler验证**
```bash
# 使用profiler验证优化效果
SGLANG_TORCH_PROFILER_DIR="./" \
python -m sglang.bench_one_batch_server \
  --profile \
  --your-optimization
```

#### Day 4-5: 代码审查和文档

**实践任务**:
1. **代码审查**
   - 检查代码风格
   - 添加注释
   - 确保可读性

2. **编写文档**
   - 说明优化的原理
   - 说明如何使用
   - 说明性能提升

3. **更新测试**
   - 添加单元测试
   - 添加集成测试
   - 确保测试覆盖

#### Day 6-7: 提交PR

**实践任务**:
1. **准备PR**
```bash
# 提交代码
git add .
git commit -m "feat: your optimization description"
git push origin optimize/your-optimization-name
```

2. **创建PR**
   - 在GitHub上创建PR
   - 链接到Issue #17526
   - 说明优化的内容和效果

3. **响应审查**
   - 根据review修改代码
   - 回答问题
   - 更新文档

**检查点**:
- [ ] 性能验证通过
- [ ] 准确性验证通过
- [ ] 代码审查通过
- [ ] PR已提交

---

## 🚀 第六阶段：深入优化和贡献（第11-12周）

### 目标
- 实现更多优化项
- 深入理解性能优化
- 成为活跃贡献者

### 第11周：实现更多优化

**实践任务**:
- [ ] 实现另一个优化项
- [ ] 优化FP8 KV cache的性能问题
- [ ] 实现AllReduce优化

### 第12周：总结和分享

**实践任务**:
- [ ] 总结学习经验
- [ ] 编写技术博客
- [ ] 在社区分享经验
- [ ] 帮助其他贡献者

---

## 📚 学习资源汇总

### 官方文档
- [SGLang官方文档](https://docs.sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)

### GPU和CUDA
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA开发者文档](https://developer.nvidia.com/)

### 量化
- [PyTorch量化文档](https://pytorch.org/docs/stable/quantization.html)
- [FP8论文](https://arxiv.org/abs/2209.05433)

### 性能分析
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)

### 相关Issue
- [#17526](https://github.com/sgl-project/sglang/issues/17526) - 主Issue
- [#17166](https://github.com/sgl-project/sglang/issues/17166) - GLM 4.7 + NVFP4 + MTP
- [#16755](https://github.com/sgl-project/sglang/issues/16755) - Auto-enable TRT-LLM MHA
- [#12787](https://github.com/sgl-project/sglang/issues/12787) - TRTLLM MNNVL AllReduce

---

## ✅ 学习检查清单

### 基础阶段（第1-2周）
- [ ] 理解LLM推理基础
- [ ] 能够运行SGLang
- [ ] 理解性能指标
- [ ] 理解GPU基础概念
- [ ] 理解量化基础

### 深入阶段（第3-4周）
- [ ] 理解Blackwell GPU架构
- [ ] 理解Tensor Core
- [ ] 理解FP8/FP4量化
- [ ] 理解KV Cache量化

### 分析阶段（第5-6周）
- [ ] 能够使用PyTorch Profiler
- [ ] 能够使用Nsight Systems
- [ ] 能够识别性能瓶颈
- [ ] 理解优化方法

### 优化阶段（第7-8周）
- [ ] 理解量化优化
- [ ] 理解Backend优化
- [ ] 能够阅读优化代码
- [ ] 能够修改优化代码

### 实践阶段（第9-10周）
- [ ] 实现了一个优化项
- [ ] 验证了优化效果
- [ ] 提交了PR
- [ ] PR被合并

---

## 💡 学习建议

1. **循序渐进**: 不要跳过基础阶段，扎实的基础很重要
2. **动手实践**: 每学一个概念都要动手实践
3. **记录笔记**: 记录学习过程中的问题和发现
4. **参与讨论**: 在GitHub issue或Slack上提问和讨论
5. **持续学习**: 性能优化是一个持续的过程

---

## 🎯 最终目标

完成这个学习流程后，你将能够：
1. ✅ 理解LLM推理和性能优化的基本原理
2. ✅ 能够使用工具分析性能瓶颈
3. ✅ 能够实现和验证优化方案
4. ✅ 能够为SGLang贡献优化代码
5. ✅ 成为SGLang社区的活跃贡献者

**开始你的学习之旅吧！** 🚀
