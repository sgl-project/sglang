# A01_B05: 原理深度分析

## 相关文档
- [A01_B01: 原始 Issue 内容](./A01_B01_original_issue.md) - Issue详情
- [A01_B04: 问题深度分析](./A01_B04_problem_analysis.md) - 问题分析
- [A01: MoE Tensor Parallelism Bug 详解](./A01_moe_tp_bug.md) - 整体问题

---

## 📚 第一部分：Tensor Parallelism 基本原理

### 1.1 什么是 Tensor Parallelism？

Tensor Parallelism (TP) 是一种模型并行策略，将模型的**权重矩阵**分割到多个GPU上，每个GPU只存储和计算部分权重。

### 1.2 线性层的矩阵乘法

一个标准的线性层可以表示为：
```
Y = XW + b
```
其中：
- `X`: 输入矩阵 `[batch_size, input_size]`
- `W`: 权重矩阵 `[input_size, output_size]`
- `b`: 偏置向量 `[output_size]`
- `Y`: 输出矩阵 `[batch_size, output_size]`

### 1.3 TP的两种分片方式

#### ColumnParallel (列并行)
**分片方式**: 在**输出维度**上分片

```
原始权重 W: [input_size, output_size]
TP=2分片:
  W_0: [input_size, output_size/2]  # 前一半列
  W_1: [input_size, output_size/2]  # 后一半列

计算:
  Y_0 = X @ W_0  # GPU0计算
  Y_1 = X @ W_1  # GPU1计算
  Y = concat([Y_0, Y_1], dim=1)  # 拼接结果
```

**特点**:
- 输入 `X` 需要**广播**到所有GPU（每个GPU都有完整的输入）
- 输出需要**拼接**或**all-gather**
- 权重在**最后一个维度**（输出维度）上分片

#### RowParallel (行并行)
**分片方式**: 在**输入维度**上分片

```
原始权重 W: [input_size, output_size]
TP=2分片:
  W_0: [input_size/2, output_size]  # 前一半行
  W_1: [input_size/2, output_size]  # 后一半行

计算:
  X_0 = X[:, 0:input_size/2]  # 输入的前一半
  X_1 = X[:, input_size/2:]   # 输入的后一半
  Y_0 = X_0 @ W_0  # GPU0计算
  Y_1 = X_1 @ W_1  # GPU1计算
  Y = Y_0 + Y_1  # 通过all-reduce求和
```

**特点**:
- 输入 `X` 需要**分片**（每个GPU只有部分输入）
- 输出需要**all-reduce**求和
- 权重在**第一个维度**（输入维度）上分片

---

## 📚 第二部分：MoE模型中的权重结构

### 2.1 MoE的MLP结构

MoE模型的MLP层通常包含：
- **w1 (gate_proj)**: ColumnParallel
- **w3 (up_proj)**: ColumnParallel  
- **w2 (down_proj)**: RowParallel

### 2.2 w2 (down_proj) 的RowParallel分片

从代码注释可以看到：
```python
# Line 407: down_proj: "RowParallel" so tp sharding on input_dim
```

**权重矩阵形状**:
```
w2: [intermediate_size, d_model]
```

**TP=2时的分片**:
```
TP0: w2_0: [intermediate_size/2, d_model]  # 前一半行
TP1: w2_1: [intermediate_size/2, d_model]  # 后一半行
```

**期望的权重文件**:
- 应该包含完整的 `[intermediate_size, d_model]` 权重
- TP0加载 `[0:intermediate_size/2, :]`
- TP1加载 `[intermediate_size/2:, :]`

---

## 📚 第三部分：权重加载的完整流程

### 3.1 `_load_w2` 方法的执行流程

```python
def _load_w2(self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank, is_bias=False):
    # 1. 计算每个rank的shard大小
    shard_size = expert_data.shape[shard_dim]  # 例如: 8
    
    # 2. 计算要加载的权重切片
    start = shard_size * tp_rank  # TP0: 0, TP1: 8
    length = shard_size          # 8
    
    # 3. 从loaded_weight中提取切片
    loaded_weight.narrow(shard_dim, start, length)  # TP1: [8:16]
```

### 3.2 关键变量的含义

#### `expert_data` (目标张量)
- **作用**: 存储加载后的权重
- **形状**: `[num_experts, intermediate_size/tp_size, d_model]`
- **`shard_size`**: `expert_data.shape[shard_dim]` = `intermediate_size/tp_size`

#### `loaded_weight` (源张量)
- **作用**: 从checkpoint文件加载的原始权重
- **期望形状**: `[num_experts, intermediate_size, d_model]`
- **实际形状**: `[num_experts, 8, d_model]` (只有期望的一半)

### 3.3 问题的核心

**代码假设**:
```python
loaded_weight.shape[shard_dim] >= shard_size * (tp_rank + 1)
```

**实际情况**:
```python
loaded_weight.shape[shard_dim] = 8  # 只有8
shard_size * (tp_rank + 1) = 8 * 2 = 16  # 需要16
8 < 16  # ❌ 不满足假设
```

---

## 📚 第四部分：量化对权重的影响

### 4.1 INT4量化的特点

**INT4量化**:
- 将FP16/BF16的权重压缩到4位
- 通常使用**分组量化**或**块量化**
- 可能改变权重的存储格式

### 4.2 CompressedTensorsWNA16MarlinMoEMethod

从错误日志可以看到：
```
[2026-01-25 01:19:07 TP1] Using CompressedTensorsWNA16MarlinMoEMethod
```

**可能的影响**:
1. **权重压缩**: 量化可能改变了权重的实际维度
2. **对齐要求**: 量化通常要求维度是8或16的倍数
3. **存储格式**: 压缩后的权重可能不是标准的 `[intermediate_size, d_model]` 格式

### 4.3 量化后的维度问题

**可能的情况**:
- 原始 `intermediate_size = 16`
- 量化后可能变成 `8`（压缩了一半）
- 但代码仍然假设是 `16`

**或者**:
- 权重文件已经是**分片格式**
- 每个文件只包含一个TP rank的数据
- 但代码假设文件包含完整数据

---

## 📚 第五部分：RowParallel vs ColumnParallel 的权重加载对比

### 5.1 ColumnParallel 的权重加载

**`_load_w13` 方法** (w1和w3):
```python
# ColumnParallel: 在输出维度上分片
shard_size = expert_data.shape[shard_dim]  # 每个rank的大小
start = shard_size * tp_rank  # TP0: 0, TP1: shard_size
loaded_weight.narrow(shard_dim, start, shard_size)
```

**为什么ColumnParallel不会出错？**
- 因为 `loaded_weight` 的维度通常是 `output_size`，足够大
- 或者ColumnParallel的权重加载逻辑已经处理了边界情况

### 5.2 RowParallel 的权重加载

**`_load_w2` 方法** (w2):
```python
# RowParallel: 在输入维度上分片
shard_size = expert_data.shape[shard_dim]  # 每个rank的大小
start = shard_size * tp_rank  # TP0: 0, TP1: shard_size
loaded_weight.narrow(shard_dim, start, shard_size)  # ❌ 可能失败
```

**为什么RowParallel会出错？**
- `loaded_weight` 的维度可能只有 `shard_size`，而不是 `tp_size * shard_size`
- 没有边界检查，直接使用 `narrow` 会失败

### 5.3 对比标准实现

**`RowvLLMParameter.load_row_parallel_weight`** (标准实现):
```python
start_idx = tp_rank * shard_size
end_idx = start_idx + shard_size
if end_idx > loaded_weight.shape[self.input_dim]:
    loaded_weight = pad_or_narrow_weight(...)  # ✅ 有边界检查
else:
    loaded_weight = loaded_weight.narrow(...)
```

**`_load_w2` 方法** (MoE实现):
```python
loaded_weight = loaded_weight.narrow(...)  # ❌ 没有边界检查
```

---

## 📚 第六部分：问题的根本原因

### 6.1 维度不匹配的根本原因

**问题1: 量化/压缩改变了权重维度**
- INT4量化可能改变了权重的实际维度
- 压缩算法可能将 `intermediate_size` 从16压缩到8
- 但代码仍然假设是16

**问题2: 权重文件格式问题**
- 权重文件可能已经是分片格式
- 每个文件只包含一个TP rank的数据
- 但代码假设文件包含完整数据

**问题3: 缺少边界检查**
- `_load_w2` 方法直接使用 `narrow`，没有检查边界
- 标准实现 `RowvLLMParameter` 有边界检查和padding逻辑
- MoE实现缺少这个保护机制

### 6.2 为什么只有TP1失败？

**TP0 (rank=0)**:
```python
start = shard_size * 0 = 0
length = shard_size = 8
loaded_weight.narrow(shard_dim, 0, 8)  # ✅ 成功，访问 [0:8]
```

**TP1 (rank=1)**:
```python
start = shard_size * 1 = 8
length = shard_size = 8
loaded_weight.narrow(shard_dim, 8, 8)  # ❌ 失败，尝试访问 [8:16]，但维度只有8
```

**结论**: 
- TP0可以成功加载，因为 `[0:8]` 在有效范围内
- TP1失败，因为 `[8:16]` 超出了维度大小

---

## 📚 第七部分：数学原理分析

### 7.1 矩阵乘法的分片

**RowParallel的数学原理**:
```
Y = XW + b

将W按行分片:
  W = [W_0; W_1]  # 垂直拼接

将X按列分片:
  X = [X_0, X_1]  # 水平拼接

计算:
  Y = X_0 @ W_0 + X_1 @ W_1
    = [X_0, X_1] @ [W_0; W_1]
    = X @ W  # 等价
```

**关键**: 输入X也需要分片，每个GPU只有部分输入特征。

### 7.2 权重加载的数学要求

**假设**:
- 完整权重: `W: [M, N]`
- TP大小: `tp_size = 2`
- 每个rank的权重: `W_i: [M/tp_size, N]`

**要求**:
- 权重文件应该包含完整的 `W: [M, N]`
- TP0加载 `W[0:M/2, :]`
- TP1加载 `W[M/2:, :]`

**问题**:
- 如果权重文件只有 `W: [M/2, N]`
- TP0可以加载 `W[0:M/2, :]` ✅
- TP1尝试加载 `W[M/2:, :]`，但 `M/2` 已经是边界 ❌

---

## 📚 第八部分：解决方案的原理

### 8.1 Padding的原理

**为什么需要padding？**
- 当权重维度不足时，用零填充
- 零填充不会影响计算结果（因为对应的输入也是零）

**数学原理**:
```
如果 W 只有 [M/2, N]，但需要 [M, N]:
  W_padded = [W; 0]  # 下半部分用零填充

计算时:
  Y = X @ W_padded
    = X @ [W; 0]
    = X_0 @ W + X_1 @ 0
    = X_0 @ W  # 如果X_1是零（因为输入也被分片了）
```

### 8.2 `pad_or_narrow_weight` 的工作原理

```python
def pad_or_narrow_weight(loaded_weight, input_dim, start_idx, shard_size):
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)
    
    if valid_size > 0:
        # 提取有效部分
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
        # 计算需要填充的大小
        pad_size = shard_size - valid_size
        # 创建零填充
        pad = torch.zeros(..., pad_size, ...)
        # 拼接
        return torch.cat([loaded_slice, pad], dim=input_dim)
    else:
        # 全部用零填充
        return torch.zeros(..., shard_size, ...)
```

**关键点**:
1. 计算有效大小: `valid_size = dim_size - start_idx`
2. 提取有效部分: `narrow(input_dim, start_idx, valid_size)`
3. 零填充剩余部分: `zeros(..., shard_size - valid_size, ...)`
4. 拼接: `cat([slice, pad], dim=input_dim)`

---

## 📚 第九部分：总结

### 9.1 问题的本质

1. **Tensor Parallelism原理**: RowParallel在输入维度上分片权重
2. **MoE权重结构**: w2是RowParallel，需要在输入维度上分片
3. **量化影响**: INT4量化可能改变了权重维度
4. **代码缺陷**: `_load_w2` 缺少边界检查和padding逻辑

### 9.2 为什么会出现这个问题？

**根本原因**: 
- 量化/压缩后的权重维度可能不是 `tp_size * shard_size` 的倍数
- 代码假设权重维度足够大，没有处理边界情况
- MoE的 `_load_w2` 实现缺少标准 `RowParallelLinear` 中的padding逻辑

### 9.3 解决方案的原理

**Padding策略**:
- 当权重维度不足时，用零填充
- 零填充不会影响计算结果（因为对应的输入特征也被分片了）
- 这是处理维度未对齐的标准方法

---

## 🔗 相关代码位置

- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` - `_load_w2` 方法
- `python/sglang/srt/layers/parameter.py` - `RowvLLMParameter.load_row_parallel_weight` (标准实现)
- `python/sglang/srt/layers/utils.py` - `pad_or_narrow_weight` 函数
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` - 量化方法
