# A01: MoE Tensor Parallelism Bug 详解

## 📋 相关文档
- [A01_B01: 原始 Issue 内容](./A01_B01_original_issue.md) - Issue详情
- [A01_B02: 修复分析](./A01_B02_fix_analysis.md) - 修复前后代码对比
- [A01_B03: 代码变更](./A01_B03_code_changes.md) - 修复的具体代码位置

---

## 🎯 问题概述

### Bug描述
MoE模型 `MedAIBase/AntAngelMed-INT4` 在使用 tensor parallelism (`--tp-size 2`) 时，在第二个GPU (TP1) 上加载权重时抛出 `RuntimeError`。

### 错误信息
```
RuntimeError: start (8) + length (8) exceeds dimension size (8).
```

### 错误位置
- **文件**: `sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **方法**: `_load_w2`
- **行号**: line 501

---

## 🔍 问题分析

### 问题场景
1. **模型**: MoE模型 `MedAIBase/AntAngelMed-INT4` (INT4量化)
2. **配置**: 使用 `--tp-size 2` (tensor parallelism)
3. **问题**: 单GPU OOM，必须使用TP=2
4. **错误**: TP1 rank加载权重时失败

### 根本原因

#### 1. Tensor Parallelism 权重分片
在TP模式下，每个rank需要加载不同的权重切片：
- **TP0**: 加载权重的前半部分 `[0:shard_size]`
- **TP1**: 加载权重的后半部分 `[shard_size:2*shard_size]`

#### 2. 维度未对齐问题
- **期望**: 权重维度应该是 `2 * shard_size` (例如16)
- **实际**: 权重维度只有 `shard_size` (例如8)
- **原因**: 量化/压缩可能导致维度不是8的倍数

#### 3. 代码缺陷
原代码直接使用 `narrow` 操作，没有检查边界：
```python
loaded_weight = loaded_weight.narrow(
    shard_dim, shard_size * tp_rank, shard_size
)
```

当 `tp_rank=1` 时：
- `start = 8 * 1 = 8`
- `length = 8`
- 尝试访问 `[8:16]`，但维度只有8，导致错误

---

## 💡 解决方案

### 修复思路
参考 `RowParallelLinear.load_row_parallel_weight` 的处理方式，添加边界检查和padding逻辑。

### 修复步骤

#### 1. 添加导入
```python
from sglang.srt.layers.utils import pad_or_narrow_weight
```

#### 2. 修改 `_load_w2` 方法
在调用 `narrow` 之前，先检查边界：
```python
start_idx = shard_size * tp_rank
end_idx = start_idx + shard_size
if end_idx > loaded_weight.shape[shard_dim]:
    loaded_weight = pad_or_narrow_weight(
        loaded_weight, shard_dim, start_idx, shard_size
    )
else:
    loaded_weight = loaded_weight.narrow(
        shard_dim, start_idx, shard_size
    )
```

### `pad_or_narrow_weight` 工作原理

1. **计算有效大小**: `valid_size = max(dim_size - start_idx, 0)`
2. **提取有效部分**: 如果 `valid_size > 0`，提取 `[start_idx:start_idx+valid_size]`
3. **零填充**: 如果 `valid_size < shard_size`，用零填充剩余部分
4. **拼接**: 将有效部分和填充部分拼接

---

## ✅ 修复效果

### 修复前
- ❌ TP=2时，TP1 rank加载权重失败
- ❌ 抛出 `RuntimeError`
- ❌ 服务器无法启动

### 修复后
- ✅ TP=2时，两个rank都能正确加载权重
- ✅ 自动处理维度未对齐的情况
- ✅ 服务器可以正常启动
- ✅ 向后兼容，正常对齐的权重不受影响

---

## 🔗 相关代码

### 参考实现
- `sglang/srt/layers/parameter.py` - `RowParallelLinear.load_row_parallel_weight`
- `sglang/srt/layers/utils.py` - `pad_or_narrow_weight`

### 类似问题
这个问题与 `qwen2_5_VL` 的MLP层未8对齐的问题类似，都使用了相同的padding处理方式。

---

## 📝 测试建议

### 基本测试
```bash
python3 -m sglang.launch_server  \
    --model-path MedAIBase/AntAngelMed-INT4 \
    --host 0.0.0.0 --port 30012  \
    --trust-remote-code  \
    --attention-backend fa3  \
    --mem-fraction-static 0.9 \
    --tp-size 2
```

### 验证点
1. ✅ 服务器能正常启动
2. ✅ 两个GPU都能正确加载权重
3. ✅ 推理功能正常
4. ✅ 性能不受影响

---

## 🎓 学习要点

### 1. Tensor Parallelism 权重分片
- 每个rank加载不同的权重切片
- 需要正确处理边界情况

### 2. 量化/压缩模型
- 维度可能不是8的倍数
- 需要padding处理

### 3. 代码复用
- 参考已有的成熟实现
- 保持代码风格一致性

### 4. 边界检查
- 在访问数组/张量之前检查边界
- 使用padding而不是直接失败

---

## 📚 参考资料

- [Issue #17680](https://github.com/sgl-project/sglang/issues/17680)
- `sglang/srt/layers/parameter.py` - RowParallelLinear实现
- `sglang/srt/layers/utils.py` - padding工具函数
