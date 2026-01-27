# 00_Z7_GPU基本计算单元_SM_详解

## 📚 问题

**什么叫做基本计算单元？**

在GPU架构中，**SM（Streaming Multiprocessor，流式多处理器）**是GPU的**基本计算单元**，类似于CPU中的核心（Core），但设计理念完全不同。

---

## 🎯 最简单理解：NVIDIA开发SM的目的是什么？

### 一句话总结
**NVIDIA开发SM的目的是：让GPU能够同时做很多很多简单的计算，而不是做几个复杂的计算。**

### 类比理解

#### 类比1：工厂生产
- [ ] **CPU = 高级技工**: 
  - [ ] 只有几个高级技工
  - [ ] 每个技工能做复杂的工作
  - [ ] 但一次只能做几件事
- [ ] **GPU = 工厂**: 
  - [ ] 有很多车间（SM）
  - [ ] 每个车间有很多工人（CUDA Core）
  - [ ] 可以同时做很多简单的工作
- [ ] **SM = 车间**: 
  - [ ] 可以独立生产
  - [ ] 有独立的设备和资源
  - [ ] 多个车间可以同时工作

#### 类比2：军队作战
- [ ] **CPU = 特种部队**: 
  - [ ] 人数少，但每个人都很强
  - [ ] 适合复杂任务
- [ ] **GPU = 大军队**: 
  - [ ] 有很多连队（SM）
  - [ ] 每个连队有很多士兵（CUDA Core）
  - [ ] 适合大规模简单任务
- [ ] **SM = 连队**: 
  - [ ] 可以独立执行任务
  - [ ] 有独立的指挥官和资源

---

### NVIDIA开发SM的核心目的

#### 目的1：并行计算
- [ ] **问题**: 如何同时做很多计算？
- [ ] **解决方案**: 设计很多SM，每个SM可以独立工作
- [ ] **结果**: 100个SM可以同时做100个不同的计算

#### 目的2：简单高效
- [ ] **问题**: 如何让计算更高效？
- [ ] **解决方案**: 每个SM做简单的计算，但做得很快
- [ ] **结果**: 简单计算比复杂计算快很多

#### 目的3：适合AI
- [ ] **问题**: AI计算有什么特点？
- [ ] **特点**: 主要是矩阵运算（简单但重复很多次）
- [ ] **解决方案**: SM内有Tensor Core专门做矩阵运算
- [ ] **结果**: 非常适合AI计算

---

### 实际例子

#### 例子1：图像处理
- [ ] **任务**: 处理一张1000x1000的图片
- [ ] **CPU方式**: 
  - [ ] 4个核心，一个一个像素处理
  - [ ] 很慢
- [ ] **GPU方式**: 
  - [ ] 100个SM，每个SM处理一部分像素
  - [ ] 100倍并行，很快

#### 例子2：矩阵乘法（AI的核心）
- [ ] **任务**: 计算 A @ B（两个大矩阵相乘）
- [ ] **CPU方式**: 
  - [ ] 几个核心，串行计算
  - [ ] 很慢
- [ ] **GPU方式**: 
  - [ ] 100个SM，每个SM用Tensor Core计算一部分
  - [ ] 100倍并行，很快

---

### 为什么SM这么设计？

#### 设计理念
- [ ] **"多"比"强"更重要**: 
  - [ ] 100个简单的SM > 4个强大的核心
  - [ ] 对于并行计算，数量比单个能力更重要
- [ ] **"简单"比"复杂"更快**: 
  - [ ] 简单的计算可以做得很快
  - [ ] 复杂的计算需要更多时间
- [ ] **"专用"比"通用"更快**: 
  - [ ] Tensor Core专门做矩阵运算
  - [ ] 比通用计算单元快很多

---

### Z7.9 调度机制：统一调度 vs SM独立控制

#### Z7.9.1 你的理解是对的！
- [ ] **你的理解**: "以前计算是统一调度的，现在是SM在控制，类似于dynamic balance"
- [ ] **答案**: **基本正确！** 但更准确的说法是：**两层调度机制**

#### Z7.9.2 GPU的两层调度机制

**第一层：全局调度（GPU级别）**
- [ ] **全局调度器**: 
  - [ ] GPU有一个全局调度器
  - [ ] 负责将Thread Block分配给不同的SM
  - [ ] 类似于"任务分配中心"
- [ ] **分配策略**: 
  - [ ] 动态负载均衡（Dynamic Load Balancing）
  - [ ] 哪个SM空闲，就分配给哪个SM
  - [ ] 确保所有SM都有工作做

**第二层：局部调度（SM级别）**
- [ ] **SM的Warp Scheduler**: 
  - [ ] 每个SM有自己的Warp Scheduler
  - [ ] 负责调度SM内的warp
  - [ ] 类似于"车间内的工头"
- [ ] **调度策略**: 
  - [ ] 当一个warp等待内存时，切换到另一个warp
  - [ ] 确保SM的计算单元不闲置
  - [ ] 这也是动态负载均衡

#### Z7.9.3 为什么需要两层调度？

**全局调度的作用**:
- [ ] **任务分配**: 将大的任务分配给不同的SM
- [ ] **负载均衡**: 确保所有SM都有工作
- [ ] **资源管理**: 管理GPU级别的资源

**SM局部调度的作用**:
- [ ] **细粒度调度**: 在SM内部调度warp
- [ ] **隐藏延迟**: 当一个warp等待时，执行另一个warp
- [ ] **提高利用率**: 确保SM的计算单元不闲置

#### Z7.9.4 类比理解

**工厂类比**:
- [ ] **全局调度 = 工厂经理**: 
  - [ ] 决定哪个车间（SM）做什么任务
  - [ ] 动态分配任务，确保所有车间都有工作
- [ ] **SM调度 = 车间主任**: 
  - [ ] 决定车间内的工人（warp）做什么
  - [ ] 当一个工人等待材料时，让另一个工人工作
  - [ ] 确保车间不闲置

**军队类比**:
- [ ] **全局调度 = 总指挥**: 
  - [ ] 决定哪个连队（SM）执行什么任务
  - [ ] 动态分配任务
- [ ] **SM调度 = 连长**: 
  - [ ] 决定连队内的士兵（warp）做什么
  - [ ] 当一个士兵等待时，让另一个士兵工作

#### Z7.9.5 动态负载均衡（Dynamic Load Balancing）

**为什么叫"动态"？**
- [ ] **动态分配**: 
  - [ ] 不是预先分配好所有任务
  - [ ] 而是根据SM的负载情况动态分配
- [ ] **负载均衡**: 
  - [ ] 确保所有SM都有工作
  - [ ] 避免有些SM很忙，有些SM很闲
- [ ] **自适应**: 
  - [ ] 根据任务特点自动调整
  - [ ] 根据SM状态自动调整

**实际例子**:
```
任务: 处理1000个Thread Block

全局调度器:
  - 看到SM0空闲 → 分配Thread Block 1-10给SM0
  - 看到SM1空闲 → 分配Thread Block 11-20给SM1
  - 看到SM2空闲 → 分配Thread Block 21-30给SM2
  - ...（动态分配，直到所有Thread Block分配完）

SM0的Warp Scheduler:
  - Thread Block 1的warp1在执行
  - Thread Block 1的warp2在等待内存
  - → 切换到Thread Block 1的warp3执行（动态切换）
```

**参考文档**: [00_Z07_Y01_Thread_Block_详解.md](./00_Z07_Y01_Thread_Block_详解.md) ⭐ **详细讲解什么是Thread Block**

---

### Z7.9.6 调度器自身的算力开销

#### Z7.9.6.1 你的问题：调度器不吃算力吗？
- [ ] **问题**: 动态分配的时候，调度器自身不吃算力吗？
- [ ] **答案**: **吃，但非常少！** 而且调度带来的性能提升远大于调度开销

#### Z7.9.6.2 调度器的算力开销

**开销很小**:
- [ ] **硬件实现**: 
  - [ ] 调度器是**硬件实现的**，不是软件
  - [ ] 硬件调度器非常高效
  - [ ] 开销通常小于1%的计算资源
- [ ] **为什么开销小**: 
  - [ ] 调度逻辑很简单（检查状态、分配任务）
  - [ ] 硬件并行执行，不需要CPU参与
  - [ ] 调度器是专用的硬件电路

**性能提升远大于开销**:
- [ ] **调度带来的提升**: 
  - [ ] 动态负载均衡 → 所有SM都有工作 → 性能提升10-100倍
  - [ ] Warp切换 → 隐藏内存延迟 → 性能提升2-10倍
- [ ] **开销对比**: 
  - [ ] 调度开销：< 1%
  - [ ] 性能提升：10-100倍
  - [ ] **结论**: 开销可以忽略不计

#### Z7.9.6.3 类比理解

**工厂类比**:
- [ ] **调度器 = 工厂经理和车间主任**: 
  - [ ] 他们需要花时间分配任务（算力开销）
  - [ ] 但他们的时间很少（< 1%）
  - [ ] 而他们带来的效率提升很大（10-100倍）
  - [ ] **结论**: 值得！

**军队类比**:
- [ ] **调度器 = 总指挥和连长**: 
  - [ ] 他们需要花时间制定计划（算力开销）
  - [ ] 但他们的时间很少
  - [ ] 而他们带来的战斗力提升很大
  - [ ] **结论**: 值得！

#### Z7.9.6.4 实际数据

**调度开销**:
- [ ] **全局调度器**: 
  - [ ] 开销：< 0.1%的GPU资源
  - [ ] 频率：只在分配Thread Block时运行
  - [ ] 硬件实现，非常高效
- [ ] **Warp Scheduler**: 
  - [ ] 开销：< 0.5%的SM资源
  - [ ] 频率：每个时钟周期都可能切换
  - [ ] 硬件实现，并行执行

**性能提升**:
- [ ] **没有调度**: 
  - [ ] SM利用率：10-20%（很多SM闲置）
  - [ ] 性能：很低
- [ ] **有调度**: 
  - [ ] SM利用率：80-95%（几乎所有SM都在工作）
  - [ ] 性能：提升10-100倍
- [ ] **结论**: 调度开销可以忽略不计

#### Z7.9.6.5 为什么调度器这么高效？

**硬件实现**:
- [ ] **专用硬件**: 
  - [ ] 调度器是专用的硬件电路
  - [ ] 不是软件，不需要CPU执行
  - [ ] 并行执行，不影响计算单元
- [ ] **简单逻辑**: 
  - [ ] 调度逻辑很简单（检查状态、分配任务）
  - [ ] 硬件可以非常快速地执行
  - [ ] 开销极小

**并行设计**:
- [ ] **独立运行**: 
  - [ ] 调度器和计算单元独立运行
  - [ ] 调度不影响计算
  - [ ] 可以同时进行
- [ ] **流水线**: 
  - [ ] 调度和计算可以流水线执行
  - [ ] 调度下一个任务时，当前任务还在执行
  - [ ] 几乎没有额外开销

---

## 🔍 知识点分解

### Z7.1 什么是"基本计算单元"？

#### Z7.1.1 基本定义
- [ ] **基本计算单元的含义**: 
  - [ ] 能够独立执行计算的最小功能单元
  - [ ] 可以并行工作的计算资源
  - [ ] 是GPU架构的基础组成部分
- [ ] **类比理解**: 
  - [ ] **CPU**: 基本计算单元是"核心"（Core）
  - [ ] **GPU**: 基本计算单元是"SM"（Streaming Multiprocessor）
  - [ ] 但设计理念完全不同

#### Z7.1.2 CPU vs GPU的基本计算单元
- [ ] **CPU的核心（Core）**: 
  - [ ] 数量少（4-64个）
  - [ ] 每个核心功能强大
  - [ ] 适合串行计算和复杂控制流
  - [ ] 有大量缓存（L1, L2, L3）
- [ ] **GPU的SM（Streaming Multiprocessor）**: 
  - [ ] 数量多（几十到上百个）
  - [ ] 每个SM功能相对简单
  - [ ] 适合并行计算和简单控制流
  - [ ] 缓存相对较小，但内存带宽高

---

### Z7.2 SM（Streaming Multiprocessor）详解

#### Z7.2.1 SM是什么？
- [ ] **SM的定义**: 
  - [ ] Streaming Multiprocessor（流式多处理器）
  - [ ] GPU的基本计算单元
  - [ ] 可以独立执行CUDA kernel
- [ ] **SM的作用**: 
  - [ ] 执行CUDA kernel（GPU程序）
  - [ ] 管理线程调度
  - [ ] 管理内存访问
  - [ ] 执行矩阵运算（通过Tensor Core）

#### Z7.2.2 SM的内部结构
- [ ] **Warp Scheduler（Warp调度器）**: 
  - [ ] 调度和管理warp（32个线程的组）
  - [ ] 决定哪个warp执行
  - [ ] 处理warp的切换
- [ ] **CUDA Core（CUDA核心）**: 
  - [ ] 执行通用计算（FP32, FP64）
  - [ ] 执行算术和逻辑运算
  - [ ] 数量：每个SM有几十到上百个
- [ ] **Tensor Core（张量核心）**: 
  - [ ] 执行矩阵运算（专用硬件）
  - [ ] 支持FP16, BF16, FP8, INT8, INT4
  - [ ] 性能比CUDA Core高很多
- [ ] **Shared Memory（共享内存）**: 
  - [ ] 每个SM的快速共享内存
  - [ ] 用于线程块（Thread Block）内的数据共享
  - [ ] 比Global Memory快很多
- [ ] **Register File（寄存器文件）**: 
  - [ ] 存储每个线程的寄存器
  - [ ] 最快的访问速度
  - [ ] 数量有限

#### Z7.2.3 SM的可视化结构
```
SM (Streaming Multiprocessor) - 基本计算单元
├── Warp Scheduler (Warp调度器)
│   ├── 管理多个warp
│   └── 调度warp执行
├── CUDA Core Array (CUDA核心阵列)
│   ├── 64-128个CUDA Core
│   └── 执行通用计算
├── Tensor Core Array (张量核心阵列) ⭐
│   ├── 4-8个Tensor Core
│   └── 执行矩阵运算
├── Shared Memory (共享内存)
│   ├── 48KB-164KB
│   └── 线程块共享
└── Register File (寄存器文件)
    ├── 65536个32位寄存器
    └── 线程私有
```

---

### Z7.3 为什么SM是"基本"计算单元？

#### Z7.3.1 "基本"的含义
- [ ] **不可再分**: 
  - [ ] SM是GPU架构中不可再分的最小功能单元
  - [ ] 不能把SM再拆分成更小的独立计算单元
  - [ ] 但SM内部有多个CUDA Core和Tensor Core
- [ ] **独立执行**: 
  - [ ] 每个SM可以独立执行kernel
  - [ ] 不同SM之间可以并行工作
  - [ ] SM之间通过Global Memory通信
- [ ] **资源管理**: 
  - [ ] 每个SM有独立的资源（内存、寄存器）
  - [ ] 每个SM独立调度线程
  - [ ] 每个SM独立管理内存访问

#### Z7.3.2 为什么不是CUDA Core？
- [ ] **CUDA Core太小**: 
  - [ ] CUDA Core只是执行单元，不能独立工作
  - [ ] 需要SM来管理和调度
  - [ ] 不能独立执行kernel
- [ ] **SM是完整单元**: 
  - [ ] SM包含调度器、内存、执行单元
  - [ ] SM可以独立执行kernel
  - [ ] SM是GPU调度的基本单位

---

### Z7.4 SM如何工作？

#### Z7.4.1 Kernel执行流程
- [ ] **步骤1: Kernel启动**: 
  - [ ] CPU启动CUDA kernel
  - [ ] GPU分配SM执行kernel
  - [ ] 每个SM分配一个或多个Thread Block
- [ ] **步骤2: Thread Block分配**: 
  - [ ] 每个Thread Block分配给一个SM
  - [ ] Thread Block在SM上执行
  - [ ] 多个Thread Block可以分配给同一个SM（时间分片）
- [ ] **步骤3: Warp调度**: 
  - [ ] SM的Warp Scheduler调度warp
  - [ ] 32个线程组成一个warp
  - [ ] 多个warp在SM上并行执行（时间分片）
- [ ] **步骤4: 指令执行**: 
  - [ ] CUDA Core执行算术运算
  - [ ] Tensor Core执行矩阵运算
  - [ ] 访问Shared Memory或Global Memory

#### Z7.4.2 并行执行
- [ ] **SM级并行**: 
  - [ ] 多个SM同时执行不同的Thread Block
  - [ ] 这是GPU并行计算的主要来源
  - [ ] 例如：100个SM可以同时执行100个Thread Block
- [ ] **Warp级并行**: 
  - [ ] 同一个SM内的多个warp并行执行（时间分片）
  - [ ] Warp Scheduler在warp之间切换
  - [ ] 当一个warp等待内存时，执行另一个warp

---

### Z7.5 SM的数量和性能

#### Z7.5.1 不同GPU的SM数量
- [ ] **Blackwell B100**: 
  - [ ] 约100+个SM（具体数量取决于型号）
  - [ ] 每个SM有更多Tensor Core
- [ ] **Hopper H100**: 
  - [ ] 132个SM
  - [ ] 每个SM有4个Tensor Core
- [ ] **Ampere A100**: 
  - [ ] 108个SM
  - [ ] 每个SM有4个Tensor Core
- [ ] **规律**: 
  - [ ] 新一代GPU通常有更多SM
  - [ ] 每个SM有更强的计算能力

#### Z7.5.2 SM数量与性能的关系
- [ ] **更多SM = 更高并行度**: 
  - [ ] 更多SM可以同时执行更多Thread Block
  - [ ] 更高的并行计算能力
  - [ ] 但需要足够的并行任务
- [ ] **性能瓶颈**: 
  - [ ] 如果任务不够并行，SM会闲置
  - [ ] 需要足够的Thread Block来利用所有SM
  - [ ] 内存带宽也可能成为瓶颈

---

### Z7.6 SM在LLM推理中的作用

#### Z7.6.1 Attention计算
- [ ] **矩阵运算**: 
  - [ ] Q @ K^T使用Tensor Core
  - [ ] Attention @ V使用Tensor Core
  - [ ] 多个SM并行计算不同的head
- [ ] **并行策略**: 
  - [ ] 不同SM处理不同的batch或head
  - [ ] 利用SM的并行能力
  - [ ] 提升整体性能

#### Z7.6.2 MoE计算
- [ ] **Expert并行**: 
  - [ ] 不同SM处理不同的expert
  - [ ] 利用SM的并行能力
  - [ ] 提升MoE模型性能
- [ ] **路由计算**: 
  - [ ] Router计算可以在多个SM上并行
  - [ ] 利用SM的并行能力

---

### Z7.7 类比理解

#### Z7.7.1 工厂类比
- [ ] **GPU = 工厂**: 
  - [ ] 整个GPU是一个工厂
- [ ] **SM = 车间**: 
  - [ ] 每个SM是一个车间
  - [ ] 可以独立生产（执行计算）
  - [ ] 有独立的设备和资源
- [ ] **CUDA Core = 工人**: 
  - [ ] 每个车间有多个工人
  - [ ] 工人在车间内工作
  - [ ] 不能独立工作，需要车间管理
- [ ] **Tensor Core = 专业设备**: 
  - [ ] 每个车间有专业设备（如3D打印机）
  - [ ] 专门做特定工作（矩阵运算）
  - [ ] 效率比工人高很多

#### Z7.7.2 军队类比
- [ ] **GPU = 军队**: 
  - [ ] 整个GPU是一个军队
- [ ] **SM = 连队**: 
  - [ ] 每个SM是一个连队
  - [ ] 可以独立执行任务
  - [ ] 有独立的指挥官和资源
- [ ] **CUDA Core = 士兵**: 
  - [ ] 每个连队有多个士兵
  - [ ] 士兵在连队内工作
  - [ ] 不能独立执行任务，需要连队组织
- [ ] **Tensor Core = 特种兵**: 
  - [ ] 每个连队有特种兵
  - [ ] 专门执行特殊任务（矩阵运算）
  - [ ] 能力比普通士兵强很多

---

### Z7.8 实际例子

#### Z7.8.1 简单的Kernel执行
```cuda
// 一个简单的CUDA kernel
__global__ void add(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

// 启动kernel
add<<<100, 256>>>(a, b, c);
// 100个Thread Block，每个256个线程
```

**执行过程**:
1. ⚡ **Kernel启动（混合过程）**:
   - 📝 **软件层**: 程序员调用`add<<<100, 256>>>(a, b, c)`
   - 📝 **软件层**: GPU驱动准备kernel参数和启动配置
   - ⚡ **硬件层**: GPU硬件接收启动命令，自动创建Grid和Thread Block
2. ⚡ **Thread Block分配（硬件自动）**:
   - GPU硬件自动分配100个Thread Block到不同的SM
   - 全局调度器（硬件电路）自动完成分配
3. ⚡ **SM执行（硬件自动）**:
   - 每个SM硬件自动执行分配给它的Thread Block
   - 每个SM内的warp并行执行
4. ⚡ **并行计算（硬件自动）**:
   - 所有SM并行工作，完成计算

---

#### Z7.8.1.1 Kernel启动是硬件还是软件的？

**答案：是混合过程！** Kernel启动涉及**软件层和硬件层**的协作：

- 📝 **软件层**: 负责准备代码、参数、命令（编写代码、编译、准备参数、发送命令）
- ⚡ **硬件层**: 负责创建Grid、分配Thread Block、执行计算（接收命令、创建Grid、分配Thread Block、执行计算）

**详细讲解**: [00_Z07_Y02_Kernel启动机制_详解.md](./00_Z07_Y02_Kernel启动机制_详解.md) ⭐ **详细讲解Kernel启动的软件层和硬件层协作机制**

#### Z7.8.2 Attention计算
```python
# Attention计算在GPU上的执行
# Q, K, V: [batch_size, num_heads, seq_len, head_dim]

# 1. 分配到多个SM
# 每个SM处理一部分head或batch

# 2. 每个SM使用Tensor Core计算
# Q @ K^T -> scores
# softmax(scores) -> attention_weights
# attention_weights @ V -> output

# 3. 多个SM并行工作
# 提升整体性能
```

---

## 📊 可视化：SM的层次结构

### GPU整体结构
```
GPU (整个芯片)
├── SM 0 (基本计算单元)
│   ├── Warp Scheduler
│   ├── CUDA Core Array
│   ├── Tensor Core Array
│   ├── Shared Memory
│   └── Register File
├── SM 1 (基本计算单元)
│   └── ...
├── SM 2 (基本计算单元)
│   └── ...
└── ... (更多SM)

Global Memory (所有SM共享)
L2 Cache (所有SM共享)
```

### SM内部结构
```
SM (基本计算单元)
├── 调度层
│   └── Warp Scheduler (管理warp)
├── 执行层
│   ├── CUDA Core Array (通用计算)
│   └── Tensor Core Array (矩阵运算) ⭐
├── 内存层
│   ├── Shared Memory (快速共享)
│   └── Register File (最快)
└── 通信层
    └── 访问Global Memory
```

---

## ✅ 总结

### 核心要点

1. **基本计算单元的定义**: 
   - SM是GPU中不可再分的最小功能单元
   - 可以独立执行kernel
   - 是GPU调度的基本单位

2. **SM的结构**: 
   - Warp Scheduler（调度器）
   - CUDA Core Array（通用计算）
   - Tensor Core Array（矩阵运算）⭐
   - Shared Memory（共享内存）
   - Register File（寄存器）

3. **为什么是"基本"**: 
   - 不可再分的最小功能单元
   - 可以独立执行kernel
   - 有独立的资源管理

4. **并行执行**: 
   - 多个SM并行工作
   - 每个SM内多个warp并行
   - 这是GPU高性能的来源

### 关键理解

- ✅ **SM = 基本计算单元**: GPU的基本计算单元是SM，不是CUDA Core
- ✅ **独立执行**: 每个SM可以独立执行kernel
- ✅ **并行工作**: 多个SM并行工作，提升性能
- ✅ **资源管理**: 每个SM有独立的资源（内存、寄存器）

---

## 🔗 相关文档

### 内部文档
- [00_Z07_Y02_Kernel启动机制_详解.md](./00_Z07_Y02_Kernel启动机制_详解.md) - Kernel启动机制详解
- [00_Z07_Y01_Thread_Block_详解.md](./00_Z07_Y01_Thread_Block_详解.md) - Thread Block详解
- [00_Z6_Blackwell_GPU_架构_详解.md](./00_Z6_Blackwell_GPU_架构_详解.md) - Blackwell GPU架构详解
- [00_基础概念完整学习指南.md](./00_基础概念完整学习指南.md) - GPU架构基础

---

## 🔗 外部资源

### 官方文档
- [NVIDIA CUDA C++ Programming Guide - GPU Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability) ⭐⭐⭐ - NVIDIA官方GPU架构文档，详细讲解SM
- [NVIDIA CUDA C++ Programming Guide - Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration) ⭐⭐⭐ - Kernel启动配置官方文档
- [NVIDIA CUDA C++ Programming Guide - Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) ⭐⭐⭐ - GPU硬件实现详解，包括调度机制

### 技术博客
- [NVIDIA Developer Blog - CUDA Kernel Launch](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/) ⭐⭐ - CUDA Kernel启动和Occupancy优化
- [AnandTech - GPU Architecture Deep Dive](https://www.anandtech.com) ⭐⭐ - 深度技术分析，包含SM架构详解
- [CUDA by Example - GPU Architecture](https://developer.nvidia.com/cuda-example) ⭐⭐ - 通过例子讲解GPU架构

### GitHub资源
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) ⭐⭐ - NVIDIA官方CUDA示例代码，包含SM和Kernel启动示例

---

## 💡 记忆技巧

1. **SM = 车间**: 把SM想象成工厂的车间，可以独立生产
2. **基本 = 不可再分**: SM是GPU中不可再分的最小功能单元
3. **并行 = 多个SM**: GPU的高性能来自多个SM并行工作
4. **Tensor Core = 专业设备**: SM内的Tensor Core是专门做矩阵运算的设备
