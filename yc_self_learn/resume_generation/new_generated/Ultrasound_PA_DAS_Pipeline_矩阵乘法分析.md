# Ultrasound 和 Photoacoustic DAS Pipeline 逻辑与矩阵乘法分析

## 📋 目录

1. [Ultrasound DAS Pipeline](#1-ultrasound-das-pipeline)
2. [Photoacoustic DAS Pipeline](#2-photoacoustic-das-pipeline)
3. [矩阵乘法部分详细分析](#3-矩阵乘法部分详细分析)
4. [可缓存机会识别](#4-可缓存机会识别)

---

## 1. Ultrasound DAS Pipeline

### 1.1 基本流程

```
输入：传感器阵列接收信号
  ↓
步骤1：信号预处理（滤波、去噪）
  ↓
步骤2：计算延迟（Delay Calculation）
  ↓
步骤3：信号对齐（Signal Alignment）
  ↓
步骤4：加权求和（Weighted Summation）
  ↓
步骤5：后处理（插值、平滑）
  ↓
输出：重建图像
```

### 1.2 详细步骤与矩阵表示

#### 步骤1：信号预处理

**输入**：
- 原始信号矩阵：`S_raw ∈ ℝ^(N×T)`
  - N = 传感器通道数
  - T = 时间采样点数

**操作**：
- 带通滤波：`S_filtered = H_filter · S_raw`
  - `H_filter ∈ ℝ^(T×T)`：滤波矩阵（通常是 Toeplitz 矩阵）

**矩阵乘法**：
```python
# 伪代码
S_filtered = H_filter @ S_raw  # 矩阵乘法：T×T × N×T → N×T
```

**可缓存性**：⭐⭐⭐⭐⭐
- `H_filter` 可以缓存（如果采样率、频率范围固定）

---

#### 步骤2：计算延迟（Delay Calculation）

**输入**：
- 传感器位置：`P_sensors ∈ ℝ^(N×3)` (x, y, z 坐标)
- 重建像素位置：`P_pixels ∈ ℝ^(M×3)` (M = 像素数)
- 声速：`c` (标量)

**计算**：
```python
# 对于每个像素 m 和每个传感器 n
distance[n, m] = ||P_sensors[n] - P_pixels[m]||
delay[n, m] = distance[n, m] / c
```

**矩阵表示**：
- 距离矩阵：`D ∈ ℝ^(N×M)`
- 延迟矩阵：`τ ∈ ℝ^(N×M)` = `D / c`

**矩阵乘法**：❌ 无直接矩阵乘法（主要是距离计算）

**可缓存性**：⭐⭐⭐⭐⭐
- `τ` 矩阵可以缓存（如果几何固定）
- 这是**最关键的缓存机会**！

---

#### 步骤3：信号对齐（Signal Alignment）

**输入**：
- 滤波后信号：`S_filtered ∈ ℝ^(N×T)`
- 延迟矩阵：`τ ∈ ℝ^(N×M)`

**操作**：
- 对于每个像素 m，对齐所有通道的信号：
```python
# 对于每个像素 m
for m in range(M):
    aligned_signals = []
    for n in range(N):
        # 延迟对齐：s_aligned[n] = s[n, t - τ[n, m]]
        aligned_signals.append(interpolate(S_filtered[n], τ[n, m]))
    S_aligned[:, m] = aligned_signals
```

**矩阵表示**：
- 对齐后的信号：`S_aligned ∈ ℝ^(N×M×T_samples)`
  - 每个像素 m 对应一个对齐后的信号矩阵

**矩阵乘法**：⚠️ 主要是插值操作，不是标准矩阵乘法
- 但可以用矩阵形式表示插值：
  ```python
  # 插值可以表示为矩阵乘法
  S_aligned[n, m] = I_interp @ S_filtered[n]
  # I_interp ∈ ℝ^(T_samples×T)：插值矩阵
  ```

**可缓存性**：⭐⭐⭐
- 插值矩阵 `I_interp` 可以缓存（如果采样模式固定）

---

#### 步骤4：加权求和（Weighted Summation）

**输入**：
- 对齐后的信号：`S_aligned ∈ ℝ^(N×M×T_samples)`
- 权重矩阵：`W ∈ ℝ^(N×M)`

**操作**：
```python
# 对于每个像素 m
for m in range(M):
    # 加权求和
    output[m] = sum(W[:, m] * S_aligned[:, m, :])
```

**矩阵表示**：
```python
# 向量化表示
# 对于每个像素 m
output[m] = W[:, m]^T @ S_aligned[:, m, :]
# 或者整体矩阵形式
Output = (W^T @ S_aligned)  # 需要 reshape
```

**矩阵乘法**：✅ **核心矩阵乘法**
```python
# 伪代码
# 方式1：逐像素计算
for m in range(M):
    output[m] = W[:, m].T @ S_aligned[:, m, :]  # N×1 × N×T_samples → 1×T_samples

# 方式2：整体矩阵形式（如果 reshape 合适）
Output = W.T @ S_aligned_reshaped  # M×N × N×(M×T_samples) → M×(M×T_samples)
```

**可缓存性**：⭐⭐⭐⭐⭐
- `W` 权重矩阵可以缓存（如果权重策略固定，如 apodization）

---

#### 步骤5：后处理

**操作**：
- 插值到最终图像网格
- 平滑、去噪

**矩阵乘法**：
```python
# 插值到图像网格
Image = I_grid @ Output  # 插值矩阵 × 输出
```

**可缓存性**：⭐⭐⭐⭐
- 插值矩阵 `I_grid` 可以缓存

---

## 2. Photoacoustic DAS Pipeline

### 2.1 基本流程

```
输入：传感器阵列接收的 PA 信号
  ↓
步骤1：信号预处理（滤波、去噪）
  ↓
步骤2：计算传播时间（Time-of-Flight）
  ↓
步骤3：信号对齐（Time Alignment）
  ↓
步骤4：加权求和（Weighted Summation）
  ↓
步骤5：后处理（插值、平滑）
  ↓
输出：重建图像
```

### 2.2 详细步骤与矩阵表示

#### 步骤1：信号预处理

**与 Ultrasound 相同**：
- 原始信号：`S_raw ∈ ℝ^(N×T)`
- 滤波：`S_filtered = H_filter · S_raw`

**可缓存性**：⭐⭐⭐⭐⭐
- `H_filter` 可以缓存

---

#### 步骤2：计算传播时间（Time-of-Flight）

**输入**：
- 传感器位置：`P_sensors ∈ ℝ^(N×3)`
- 重建像素位置：`P_pixels ∈ ℝ^(M×3)`
- 声速：`c` (标量)

**计算**：
```python
# 对于每个像素 m 和每个传感器 n
distance[n, m] = ||P_sensors[n] - P_pixels[m]||
tof[n, m] = distance[n, m] / c  # Time-of-Flight
```

**矩阵表示**：
- 距离矩阵：`D ∈ ℝ^(N×M)`
- 传播时间矩阵：`TOF ∈ ℝ^(N×M)` = `D / c`

**矩阵乘法**：❌ 无直接矩阵乘法

**可缓存性**：⭐⭐⭐⭐⭐
- `TOF` 矩阵可以缓存（如果几何固定）
- **这是最关键的缓存机会！**

---

#### 步骤3：信号对齐（Time Alignment）

**输入**：
- 滤波后信号：`S_filtered ∈ ℝ^(N×T)`
- 传播时间矩阵：`TOF ∈ ℝ^(N×M)`

**操作**：
```python
# 对于每个像素 m
for m in range(M):
    aligned_signals = []
    for n in range(N):
        # 时间对齐：s_aligned[n] = s[n, t - TOF[n, m]]
        aligned_signals.append(interpolate(S_filtered[n], TOF[n, m]))
    S_aligned[:, m] = aligned_signals
```

**矩阵表示**：
- 对齐后的信号：`S_aligned ∈ ℝ^(N×M×T_samples)`

**矩阵乘法**：⚠️ 主要是插值操作
```python
# 插值矩阵形式
S_aligned[n, m] = I_interp @ S_filtered[n]
```

**可缓存性**：⭐⭐⭐
- 插值矩阵可以缓存

---

#### 步骤4：加权求和（Weighted Summation）

**输入**：
- 对齐后的信号：`S_aligned ∈ ℝ^(N×M×T_samples)`
- 权重矩阵：`W ∈ ℝ^(N×M)`

**操作**：
```python
# 对于每个像素 m
for m in range(M):
    output[m] = W[:, m].T @ S_aligned[:, m, :]
```

**矩阵乘法**：✅ **核心矩阵乘法**
```python
# 与 Ultrasound 相同
output[m] = W[:, m].T @ S_aligned[:, m, :]  # N×1 × N×T_samples → 1×T_samples
```

**可缓存性**：⭐⭐⭐⭐⭐
- `W` 权重矩阵可以缓存

---

## 3. 矩阵乘法部分详细分析

### 3.1 核心矩阵乘法总结

| 步骤 | 矩阵乘法 | 维度 | 可缓存性 |
|------|---------|------|---------|
| **滤波** | `H_filter @ S_raw` | T×T × N×T → N×T | ⭐⭐⭐⭐⭐ |
| **延迟/TOF计算** | 无（距离计算） | - | ⭐⭐⭐⭐⭐ (结果矩阵) |
| **信号对齐** | `I_interp @ S_filtered` | T_samples×T × N×T → N×T_samples | ⭐⭐⭐ |
| **加权求和** | `W.T @ S_aligned` | M×N × N×T_samples → M×T_samples | ⭐⭐⭐⭐⭐ |
| **后处理插值** | `I_grid @ Output` | Grid×M × M×T_samples → Grid×T_samples | ⭐⭐⭐⭐ |

### 3.2 矩阵乘法的计算复杂度

#### Ultrasound DAS

```python
# 总计算量
1. 滤波：O(N × T²)  # H_filter @ S_raw
2. 延迟计算：O(N × M)  # 距离计算（无矩阵乘法）
3. 信号对齐：O(N × M × T × log(T))  # 插值操作
4. 加权求和：O(N × M × T_samples)  # W.T @ S_aligned
5. 后处理：O(M × Grid × T_samples)  # I_grid @ Output

总复杂度：O(N × T² + N × M × T_samples)
```

#### Photoacoustic DAS

```python
# 总计算量（与 Ultrasound 类似）
1. 滤波：O(N × T²)
2. TOF计算：O(N × M)
3. 信号对齐：O(N × M × T × log(T))
4. 加权求和：O(N × M × T_samples)
5. 后处理：O(M × Grid × T_samples)

总复杂度：O(N × T² + N × M × T_samples)
```

---

## 4. 可缓存机会识别

### 4.1 高优先级缓存（⭐⭐⭐⭐⭐）

#### 1. 延迟/TOF 矩阵 `τ` 或 `TOF`

**为什么可以缓存**：
- 如果传感器几何固定（如 ring array），`τ` 或 `TOF` 矩阵只依赖于几何配置
- 计算复杂度：O(N × M)，但只需要计算一次

**缓存策略**：
```python
class DASCache:
    def __init__(self):
        self.tau_cache = {}  # 缓存延迟矩阵
        self.tof_cache = {}   # 缓存 TOF 矩阵
    
    def get_tau(self, geometry_key):
        if geometry_key in self.tau_cache:
            return self.tau_cache[geometry_key]
        else:
            tau = compute_delay_matrix(geometry_key)  # 计算
            self.tau_cache[geometry_key] = tau  # 缓存
            return tau
```

**加速效果**：
- 延迟计算：O(N × M) → O(1)（缓存命中）
- **加速比**：N × M 倍（对于固定几何）

---

#### 2. 权重矩阵 `W`

**为什么可以缓存**：
- 如果权重策略固定（如 apodization、coherence factor），权重矩阵可以预先计算

**缓存策略**：
```python
def get_weight_matrix(self, weight_type, geometry_key):
    cache_key = f"{weight_type}_{geometry_key}"
    if cache_key in self.weight_cache:
        return self.weight_cache[cache_key]
    else:
        W = compute_weight_matrix(weight_type, geometry_key)
        self.weight_cache[cache_key] = W
        return W
```

**加速效果**：
- 权重计算：O(N × M) → O(1)（缓存命中）

---

#### 3. 滤波矩阵 `H_filter`

**为什么可以缓存**：
- 如果采样率、频率范围固定，滤波矩阵可以预先计算

**缓存策略**：
```python
def get_filter_matrix(self, fs, f_low, f_high):
    cache_key = f"{fs}_{f_low}_{f_high}"
    if cache_key in self.filter_cache:
        return self.filter_cache[cache_key]
    else:
        H = design_filter(fs, f_low, f_high)
        self.filter_cache[cache_key] = H
        return H
```

**加速效果**：
- 滤波矩阵设计：O(T²) → O(1)（缓存命中）

---

### 4.2 中等优先级缓存（⭐⭐⭐）

#### 1. 插值矩阵 `I_interp`

**为什么可以缓存**：
- 如果插值方法、采样模式固定，插值矩阵可以缓存

**限制**：
- 插值矩阵可能依赖于具体的延迟值（非整数采样）
- 可能需要多个插值矩阵模板

**缓存策略**：
```python
def get_interp_matrix(self, interp_type, delay_pattern):
    # delay_pattern 可能是延迟值的模式（如整数延迟、小数延迟）
    cache_key = f"{interp_type}_{delay_pattern}"
    if cache_key in self.interp_cache:
        return self.interp_cache[cache_key]
    else:
        I = compute_interp_matrix(interp_type, delay_pattern)
        self.interp_cache[cache_key] = I
        return I
```

---

### 4.3 低优先级缓存（⭐⭐）

#### 1. 后处理插值矩阵 `I_grid`

**为什么可以缓存**：
- 如果图像网格固定，插值矩阵可以缓存

**限制**：
- 图像网格可能变化（不同 ROI、不同分辨率）

---

## 5. 缓存策略总结

### 5.1 缓存优先级

| 矩阵 | 缓存优先级 | 加速效果 | 适用场景 |
|------|-----------|---------|---------|
| **延迟/TOF 矩阵** | ⭐⭐⭐⭐⭐ | 极高 | 固定几何（ring array） |
| **权重矩阵** | ⭐⭐⭐⭐⭐ | 高 | 固定权重策略 |
| **滤波矩阵** | ⭐⭐⭐⭐⭐ | 高 | 固定采样参数 |
| **插值矩阵** | ⭐⭐⭐ | 中等 | 固定插值模式 |
| **后处理矩阵** | ⭐⭐ | 低 | 固定图像网格 |

### 5.2 关键洞察

1. **延迟/TOF 矩阵是最关键的缓存机会**
   - 计算复杂度：O(N × M)
   - 如果几何固定，可以完全缓存
   - **这是最类似 KV Cache 的场景**：一次计算，多次使用

2. **权重矩阵是第二优先级**
   - 如果权重策略固定，可以缓存
   - 计算复杂度：O(N × M)

3. **滤波矩阵可以缓存**
   - 如果采样参数固定，可以缓存
   - 计算复杂度：O(T²)

4. **信号对齐和加权求和的矩阵乘法**
   - 这些是**每次重建都需要计算的**
   - 但可以优化矩阵乘法的实现（如使用 GPU、优化库）

---

## 6. 与 KV Cache 的类比

### 6.1 相似性

| KV Cache | DAS 缓存 |
|---------|---------|
| **缓存 K/V 向量** | **缓存延迟/TOF 矩阵** |
| **一次计算，多次使用** | **一次计算，多次使用** |
| **前缀匹配** | **几何匹配** |
| **Radix Tree** | **Geometry Hash Table** |

### 6.2 关键区别

| KV Cache | DAS 缓存 |
|---------|---------|
| **序列生成** | **图像重建** |
| **Token-by-token** | **Pixel-by-pixel** |
| **自回归依赖** | **几何依赖** |
| **前缀共享** | **几何共享** |

---

## 7. 实现建议

### 7.1 缓存管理器设计

```python
class DASReconstructionCache:
    def __init__(self):
        # 高优先级缓存
        self.tau_cache = {}      # 延迟矩阵缓存
        self.tof_cache = {}      # TOF 矩阵缓存
        self.weight_cache = {}   # 权重矩阵缓存
        self.filter_cache = {}   # 滤波矩阵缓存
        
        # 中等优先级缓存
        self.interp_cache = {}   # 插值矩阵缓存
        
        # 低优先级缓存
        self.grid_cache = {}     # 图像网格插值矩阵缓存
    
    def get_tau(self, geometry_key):
        """获取延迟矩阵（最高优先级缓存）"""
        if geometry_key in self.tau_cache:
            return self.tau_cache[geometry_key]
        else:
            tau = compute_delay_matrix(geometry_key)
            self.tau_cache[geometry_key] = tau
            return tau
    
    def get_weight(self, weight_type, geometry_key):
        """获取权重矩阵"""
        cache_key = f"{weight_type}_{geometry_key}"
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        else:
            W = compute_weight_matrix(weight_type, geometry_key)
            self.weight_cache[cache_key] = W
            return W
```

### 7.2 使用示例

```python
# 初始化缓存
cache = DASReconstructionCache()

# 第一次重建（计算并缓存）
tau = cache.get_tau(geometry_key="ring_array_128")
W = cache.get_weight("apodization", geometry_key="ring_array_128")
output1 = das_reconstruction(signals1, tau, W)

# 第二次重建（使用缓存）
tau = cache.get_tau(geometry_key="ring_array_128")  # 从缓存读取
W = cache.get_weight("apodization", geometry_key="ring_array_128")  # 从缓存读取
output2 = das_reconstruction(signals2, tau, W)  # 加速！
```

---

## 8. 总结

### 8.1 核心发现

1. **延迟/TOF 矩阵是最关键的缓存机会**
   - 如果几何固定（如 ring array），可以完全缓存
   - 加速效果：O(N × M) → O(1)

2. **权重矩阵可以缓存**
   - 如果权重策略固定，可以缓存
   - 加速效果：O(N × M) → O(1)

3. **滤波矩阵可以缓存**
   - 如果采样参数固定，可以缓存
   - 加速效果：O(T²) → O(1)

4. **信号对齐和加权求和的矩阵乘法**
   - 这些是每次重建都需要计算的
   - 但可以优化实现（GPU、优化库）

### 8.2 关键洞察

- **不是缓存完整的系统矩阵 A**（几何参数多变性）
- **而是缓存算子分解的中间结果**（延迟矩阵、权重矩阵等）
- **最类似 KV Cache 的场景**：延迟/TOF 矩阵的缓存（一次计算，多次使用）

---

*文档创建时间：2025-01-26*
*基于 DAS 算法的矩阵乘法分析与 KV Cache 思想迁移*
