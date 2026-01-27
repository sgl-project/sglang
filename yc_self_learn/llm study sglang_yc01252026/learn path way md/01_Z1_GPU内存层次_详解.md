# GPU内存层次详解

## 📚 文档位置

**本文档位于**: `yc_self_learn/llm study sglang_yc01252026/learn path way md/`

---

## 🎯 概述

GPU内存层次结构是理解CUDA编程和GPU性能优化的核心概念。本文档详细解释GPU中的各种内存类型、它们的特性、区别和使用场景。

---

## 📖 目录

1. [GPU内存层次概览](#1-gpu内存层次概览)
2. [Global Memory详解](#2-global-memory详解)
3. [Shared Memory详解](#3-shared-memory详解)
4. [Register详解](#4-register详解)
5. [L1/L2 Cache详解](#5-l1l2-cache详解)
6. [内存层次对比](#6-内存层次对比)
7. [实际应用场景](#7-实际应用场景)

---

## 1. GPU内存层次概览

### 1.1 内存层次结构图

```
GPU内存层次（从快到慢，从小到大）

┌─────────────────────────────────────────┐
│  Register（寄存器）                      │ 最快，最小，线程私有
│  - 每个线程私有                          │
│  - 访问延迟：~1 cycle                    │
│  - 容量：每个线程几十到几百个寄存器      │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Shared Memory（共享内存）               │ 很快，较小，线程块共享
│  - 每个SM共享                            │
│  - 访问延迟：~20-30 cycles               │
│  - 容量：每个SM 48KB-164KB               │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  L1 Cache（一级缓存）                    │ 较快，自动缓存
│  - 每个SM的L1缓存                        │
│  - 访问延迟：~100 cycles                 │
│  - 容量：每个SM 128KB                    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  L2 Cache（二级缓存）                    │ 较慢，所有SM共享
│  - 所有SM共享                            │
│  - 访问延迟：~300 cycles                 │
│  - 容量：几十MB                          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Global Memory（全局内存）               │ 最慢，最大，所有线程共享
│  - HBM（High Bandwidth Memory）          │
│  - 访问延迟：~400-800 cycles             │
│  - 容量：几十GB                          │
└─────────────────────────────────────────┘
```

### 1.2 关键特性对比

| 内存类型 | 位置 | 访问速度 | 容量 | 作用域 | 程序员控制 |
|---------|------|---------|------|--------|-----------|
| **Register** | SM内部 | 最快 | 最小 | 线程私有 | 自动/显式 |
| **Shared Memory** | SM内部 | 很快 | 较小 | 线程块共享 | 显式控制 |
| **L1 Cache** | SM内部 | 较快 | 中等 | SM内共享 | 自动 |
| **L2 Cache** | GPU芯片 | 较慢 | 较大 | 所有SM共享 | 自动 |
| **Global Memory** | GPU板卡 | 最慢 | 最大 | 所有线程共享 | 显式控制 |

---

## 2. Global Memory详解

### 2.1 什么是Global Memory？

**Global Memory（全局内存）**是GPU上最大、最慢的内存类型，通常指的是GPU的HBM（High Bandwidth Memory）。

#### 2.1.1 基本特性

- **位置**: GPU板卡上的独立内存芯片（HBM）
- **容量**: 几十GB（例如：H100有80GB，B200有192GB）
- **访问速度**: 最慢，延迟约400-800个时钟周期
- **带宽**: 虽然延迟高，但带宽很高（TB/s级别）
- **作用域**: 所有SM、所有线程都可以访问
- **持久性**: 在kernel执行期间和kernel之间都存在

#### 2.1.2 物理实现

```
GPU芯片
├── SM 0 ──┐
├── SM 1 ──┤
├── SM 2 ──┤
└── ...    │
           │
           ↓ 通过内存控制器访问
           │
┌──────────────────────┐
│  HBM (Global Memory) │  ← 独立的内存芯片
│  - 容量：几十GB       │
│  - 带宽：TB/s级别     │
└──────────────────────┘
```

### 2.2 Global Memory的用途

#### 2.2.1 存储大型数据

- **模型权重**: LLM的权重参数（几十GB）
- **KV Cache**: 存储Key-Value缓存（可能几十GB）
- **输入数据**: 批量输入数据
- **中间结果**: 大型中间计算结果

#### 2.2.2 示例代码

```cuda
// 在CUDA kernel中访问Global Memory
__global__ void example_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 从Global Memory读取
        float value = input[idx];
        // 计算
        value = value * 2.0f;
        // 写回Global Memory
        output[idx] = value;
    }
}
```

### 2.3 Global Memory的访问模式

#### 2.3.1 Coalesced Access（合并访问）

**关键概念**: 当warp中的32个线程访问连续的内存地址时，GPU可以将这些访问合并成一个或几个内存事务，大幅提升效率。

**好的访问模式（Coalesced）**:
```cuda
// 32个线程访问连续地址
thread 0: input[0]
thread 1: input[1]
thread 2: input[2]
...
thread 31: input[31]
// → 可以合并成1个128字节的内存事务
```

**差的访问模式（Non-Coalesced）**:
```cuda
// 32个线程访问随机地址
thread 0: input[0]
thread 1: input[100]
thread 2: input[200]
...
thread 31: input[3100]
// → 需要32个独立的内存事务，效率很低
```

#### 2.3.2 内存对齐

- **对齐要求**: 访问地址应该对齐到32字节（或更大）
- **对齐的好处**: 减少内存事务数量，提升带宽利用率

### 2.4 Global Memory的性能特点

#### 2.4.1 优点

- ✅ **容量大**: 可以存储整个模型和大量数据
- ✅ **带宽高**: 虽然延迟高，但总带宽很高（TB/s）
- ✅ **持久性**: 数据在kernel之间保持

#### 2.4.2 缺点

- ❌ **延迟高**: 访问延迟约400-800个时钟周期
- ❌ **需要优化**: 必须使用coalesced access才能获得高带宽
- ❌ **竞争**: 所有SM都访问同一块内存，可能产生竞争

---

## 3. Shared Memory详解

### 3.1 什么是Shared Memory？

**Shared Memory（共享内存）**是每个SM内部的快速内存，用于线程块（Thread Block）内的线程之间共享数据。

#### 3.1.1 基本特性

- **位置**: 每个SM内部
- **容量**: 每个SM 48KB-164KB（取决于GPU架构和配置）
- **访问速度**: 很快，延迟约20-30个时钟周期
- **带宽**: 非常高（TB/s级别，比Global Memory更快）
- **作用域**: 同一个Thread Block内的所有线程共享
- **生命周期**: 只在Thread Block执行期间存在

#### 3.1.2 物理实现

```
SM (Streaming Multiprocessor)
├── CUDA Core Array
├── Tensor Core Array
├── Shared Memory (48KB-164KB)  ← 快速SRAM
│   └── 所有Thread Block内的线程共享
└── Register File
```

### 3.2 Shared Memory的用途

#### 3.2.1 线程间数据共享

- **协作计算**: 多个线程协作处理同一块数据
- **数据重用**: 多次访问相同数据时，先加载到Shared Memory
- **减少Global Memory访问**: 减少对慢速Global Memory的访问

#### 3.2.2 典型应用场景

1. **矩阵乘法**: 将矩阵块加载到Shared Memory，然后计算
2. **归约操作**: 多个线程协作进行sum、max等归约
3. **卷积**: 共享卷积核和输入数据块
4. **排序**: 线程块内的小规模排序

#### 3.2.3 示例代码

```cuda
__global__ void matrix_multiply(float* A, float* B, float* C, int N) {
    // 声明Shared Memory
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块加载到Shared Memory
    for (int tile = 0; tile < N / 16; tile++) {
        // 协作加载数据块到Shared Memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        
        // 同步，确保所有线程都加载完成
        __syncthreads();
        
        // 从Shared Memory读取并计算
        for (int k = 0; k < 16; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        // 同步，确保计算完成
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3.3 Shared Memory的访问模式

#### 3.3.1 Bank Conflict（存储体冲突）

**关键概念**: Shared Memory被分成多个bank（通常是32个），如果多个线程同时访问同一个bank的不同地址，就会产生bank conflict，降低性能。

**避免Bank Conflict的方法**:

1. **使用不同的bank**: 确保线程访问不同的bank
2. **填充（Padding）**: 在数组末尾添加padding，改变内存布局
3. **访问模式优化**: 使用stride=1的访问模式

**示例**:
```cuda
// 有Bank Conflict的代码
__shared__ float data[32];
// thread 0访问data[0], thread 1访问data[1], ...
// 如果它们映射到同一个bank，就会冲突

// 避免Bank Conflict的方法：使用padding
__shared__ float data[33];  // 添加1个元素作为padding
```

### 3.4 Shared Memory的性能特点

#### 3.4.1 优点

- ✅ **速度快**: 访问延迟只有20-30个时钟周期
- ✅ **带宽高**: 比Global Memory更快
- ✅ **可编程**: 程序员可以显式控制
- ✅ **线程协作**: 支持线程块内的数据共享

#### 3.4.2 缺点

- ❌ **容量小**: 每个SM只有48KB-164KB
- ❌ **作用域有限**: 只能在同一个Thread Block内共享
- ❌ **需要同步**: 需要`__syncthreads()`来同步线程
- ❌ **Bank Conflict**: 需要注意访问模式，避免冲突

---

## 4. Register详解

### 4.1 什么是Register？

**Register（寄存器）**是每个线程私有的最快内存，用于存储局部变量和中间计算结果。

#### 4.1.1 基本特性

- **位置**: SM内部的Register File（硬件组件）
- **容量**: 每个线程几十到几百个32位寄存器
- **访问速度**: 最快，延迟约1个时钟周期
- **作用域**: 线程私有
- **生命周期**: 线程执行期间存在
- **硬件实现**: 片上SRAM，与计算单元紧密集成

**寄存器是硬件吗？**

**是的，寄存器是硬件！** ⚙️ 寄存器是GPU芯片内部的**物理硬件组件**，由片上SRAM（Static Random Access Memory）组成，集成在SM内部的Register File中，与CUDA Core、Tensor Core一起制造在同一块GPU芯片上。

**查看寄存器硬件架构和资源**:

- [NVIDIA CUDA Programming Guide - Register](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#registers) ⭐⭐⭐ - CUDA官方文档，包含寄存器硬件说明
- [NVIDIA GPU Architecture Documentation](https://www.nvidia.com/en-us/data-center/technologies/) ⭐⭐⭐ - GPU架构白皮书，包含SM架构图（展示寄存器文件位置）
- 搜索 "NVIDIA GPU SM architecture diagram" 查看详细的SM架构图，可以看到Register File在SM中的物理位置
- 搜索 "NVIDIA GPU die shot" 查看GPU芯片的物理照片（虽然看不到单个寄存器，但可以看到整体结构）

#### 4.1.2 物理实现

```
SM (Streaming Multiprocessor) - GPU芯片上的硬件单元
├── CUDA Core Array (硬件计算单元)
├── Tensor Core Array (硬件矩阵运算单元)
├── Shared Memory (硬件SRAM，48KB-164KB)
└── Register File (硬件寄存器文件) ⚙️
    ├── Thread 0的寄存器组 (物理SRAM单元)
    ├── Thread 1的寄存器组 (物理SRAM单元)
    ├── Thread 2的寄存器组 (物理SRAM单元)
    └── ... (每个线程都有独立的物理寄存器)
```

### 4.2 Register的用途

#### 4.2.1 存储局部变量

- **循环变量**: `for (int i = 0; i < n; i++)`
- **临时计算结果**: `float sum = 0.0f;`
- **函数参数**: 传递给函数的参数
- **中间值**: 计算过程中的中间结果

#### 4.2.2 示例代码

```cuda
__global__ void example_kernel(float* input, float* output, int n) {
    // 这些变量通常存储在Register中
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Register
    float sum = 0.0f;  // Register
    float temp;  // Register
    
    if (idx < n) {
        for (int i = 0; i < 10; i++) {  // i存储在Register中
            temp = input[idx] * i;  // 中间结果在Register中
            sum += temp;
        }
        output[idx] = sum;
    }
}
```

### 4.3 Register溢出（Register Spilling）

#### 4.3.1 什么是Register溢出？

当线程使用的寄存器数量超过SM可用的寄存器数量时，编译器会将多余的变量"溢出"到Local Memory（实际上是Global Memory的一部分），这会严重影响性能。

#### 4.3.2 如何避免Register溢出？

1. **减少局部变量**: 重用变量，减少变量数量
2. **限制循环展开**: 减少循环展开的深度
3. **使用Shared Memory**: 将一些数据移到Shared Memory
4. **编译器选项**: 使用`-maxrregcount`限制寄存器使用

---

## 5. L1/L2 Cache详解

### 5.1 L1 Cache（一级缓存）

#### 5.1.1 基本特性

- **位置**: 每个SM内部
- **容量**: 每个SM约128KB（可配置）
- **访问速度**: 较快，延迟约100个时钟周期
- **作用域**: SM内共享
- **管理方式**: 硬件自动管理

#### 5.1.2 工作原理

- **自动缓存**: 当访问Global Memory时，硬件自动将数据缓存到L1 Cache
- **缓存行**: 以缓存行（cache line）为单位缓存数据
- **替换策略**: 使用LRU（最近最少使用）等策略

### 5.2 L2 Cache（二级缓存）

#### 5.2.1 基本特性

- **位置**: GPU芯片上，所有SM共享
- **容量**: 几十MB（例如：H100有50MB）
- **访问速度**: 较慢，延迟约300个时钟周期
- **作用域**: 所有SM共享
- **管理方式**: 硬件自动管理

#### 5.2.2 工作原理

- **统一缓存**: 缓存Global Memory和Shared Memory的数据
- **缓存一致性**: 保证所有SM看到一致的数据
- **大容量**: 可以缓存更多数据，减少Global Memory访问

### 5.3 Cache vs Shared Memory

#### 5.3.1 关键区别

| 特性 | L1/L2 Cache | Shared Memory |
|------|------------|---------------|
| **控制方式** | 硬件自动管理 | 程序员显式控制 |
| **可预测性** | 不可预测（取决于访问模式） | 可预测（程序员控制） |
| **容量** | 较大（L2有几十MB） | 较小（每个SM 48KB-164KB） |
| **用途** | 自动缓存Global Memory访问 | 显式数据共享和协作 |

#### 5.3.2 使用建议

- **Cache**: 依赖硬件自动优化，适合不可预测的访问模式
- **Shared Memory**: 程序员显式控制，适合可预测的、需要协作的访问模式

---

## 6. 内存层次对比

### 6.1 速度对比

```
访问延迟（时钟周期）:
Register:        ~1 cycle      ⚡⚡⚡⚡⚡
Shared Memory:   ~20-30 cycles  ⚡⚡⚡⚡
L1 Cache:        ~100 cycles    ⚡⚡⚡
L2 Cache:        ~300 cycles    ⚡⚡
Global Memory:   ~400-800 cycles ⚡
```

### 6.2 容量对比

```
容量（典型值）:
Register:        ~100个/线程 × 2048线程 = ~800KB/SM
Shared Memory:   48KB-164KB/SM
L1 Cache:        ~128KB/SM
L2 Cache:        ~50MB (所有SM共享)
Global Memory:   ~80GB (整个GPU)
```

### 6.3 带宽对比

```
带宽（典型值）:
Register:        >10 TB/s
Shared Memory:   >10 TB/s
L1 Cache:        ~5 TB/s
L2 Cache:        ~3 TB/s
Global Memory:   ~3 TB/s
```

### 6.4 使用场景对比

| 内存类型 | 适合的场景 | 不适合的场景 |
|---------|-----------|-------------|
| **Register** | 局部变量、循环变量、临时计算 | 大型数组、需要共享的数据 |
| **Shared Memory** | 线程块内数据共享、数据重用、协作计算 | 跨线程块共享、大型数据 |
| **L1/L2 Cache** | 不可预测的访问模式、自动优化 | 需要精确控制、可预测的访问 |
| **Global Memory** | 大型数据、模型权重、持久化数据 | 频繁访问的小数据、临时数据 |

---

## 7. 实际应用场景

### 7.1 LLM推理中的内存使用

#### 7.1.1 模型权重

- **存储位置**: Global Memory（HBM）
- **大小**: 几十GB（例如：70B模型约140GB FP16）
- **访问模式**: 顺序访问，可以很好地利用L2 Cache

#### 7.1.2 KV Cache

- **存储位置**: Global Memory（HBM）
- **大小**: 可能几十GB（取决于序列长度和batch size）
- **访问模式**: 顺序访问，L2 Cache可以很好地缓存

#### 7.1.3 Attention计算

- **Shared Memory**: 存储Q、K、V的tile，用于矩阵乘法
- **Register**: 存储中间计算结果
- **Global Memory**: 读取输入，写入输出

#### 7.1.4 量化操作

- **Shared Memory**: 存储量化/反量化的临时数据
- **Register**: 存储scale、offset等参数
- **Global Memory**: 读取原始数据，写入量化数据

### 7.2 性能优化策略

#### 7.2.1 减少Global Memory访问

1. **使用Shared Memory**: 将频繁访问的数据加载到Shared Memory
2. **利用Cache**: 优化访问模式，提高Cache命中率
3. **Coalesced Access**: 使用合并访问模式

#### 7.2.2 优化Shared Memory使用

1. **避免Bank Conflict**: 优化访问模式
2. **合理分配**: 平衡Shared Memory和Register的使用
3. **数据重用**: 最大化数据重用，减少重复加载

#### 7.2.3 优化Register使用

1. **减少变量**: 重用变量，减少寄存器使用
2. **避免溢出**: 控制寄存器使用，避免溢出到Local Memory
3. **循环优化**: 合理使用循环展开

---

## ✅ 总结

### 核心要点

1. **内存层次**: GPU有5层内存，从快到慢、从小到大
2. **Global Memory**: 最大最慢，存储模型权重、KV Cache等大型数据
3. **Shared Memory**: 快速共享内存，用于线程块内数据共享和协作
4. **Register**: 最快最小，存储局部变量和临时结果
5. **Cache**: 硬件自动管理，减少Global Memory访问延迟

### 关键理解

- ✅ **Global Memory = 慢但大**: 适合存储大型数据，需要优化访问模式
- ✅ **Shared Memory = 快但小**: 适合线程协作和数据重用
- ✅ **Register = 最快最小**: 适合局部变量和临时计算
- ✅ **Cache = 自动优化**: 硬件自动管理，减少内存访问延迟

### 性能优化原则

1. **最小化Global Memory访问**: 使用Shared Memory和Cache
2. **优化访问模式**: Coalesced access，避免Bank Conflict
3. **合理分配资源**: 平衡Register、Shared Memory的使用
4. **数据重用**: 最大化数据重用，减少重复访问

---

## 🔗 相关文档

### 内部文档
- [00_Z6_Blackwell_GPU_架构_详解.md](./00_Z6_Blackwell_GPU_架构_详解.md) - Blackwell GPU架构详解
- [00_Z7_GPU基本计算单元_SM_详解.md](./00_Z7_GPU基本计算单元_SM_详解.md) - SM详解
- [03_Issue_17526_学习路径.md](./03_Issue_17526_学习路径.md) - Issue 17526学习路径

### 外部资源
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Memory Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)

---

**开始你的GPU内存学习之旅！** 🎓
