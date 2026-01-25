# A01_B03: 代码变更

## 相关文档
- [A01: MoE Tensor Parallelism Bug 详解](./A01_moe_tp_bug.md) - 了解整体问题
- [A01_B01: 原始 Issue 内容](./A01_B01_original_issue.md) - 原始Issue详情
- [A01_B02: 修复分析](./A01_B02_fix_analysis.md) - 修复前后代码对比与详细解释

---

## 修改文件

### 文件路径
`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

---

## 变更1: 添加导入

### 位置
文件顶部，导入部分

### 变更前
```python
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_flashinfer_available,
    is_hip,
    next_power_of_2,
    round_up,
)
```

### 变更后
```python
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_flashinfer_available,
    is_hip,
    next_power_of_2,
    round_up,
)
```

### 说明
添加了 `pad_or_narrow_weight` 的导入，这个函数用于处理权重维度未对齐的情况。

---

## 变更2: 修复 `_load_w2` 方法

### 位置
`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, line 428-434

### 变更前
```python
else:
    if not is_bias and not self.use_presharded_weights:
        if self.use_triton_kernels:
            loaded_weight = loaded_weight.transpose(-2, -1)
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
```

### 变更后
```python
else:
    if not is_bias and not self.use_presharded_weights:
        if self.use_triton_kernels:
            loaded_weight = loaded_weight.transpose(-2, -1)
        # Padding for special case where weight dimension is not properly aligned
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

### 说明
1. **添加边界检查**: 计算 `start_idx` 和 `end_idx`
2. **条件分支**: 
   - 如果 `end_idx` 超出维度大小，使用 `pad_or_narrow_weight` 进行padding
   - 否则，使用 `narrow` 正常切片
3. **保持兼容**: 对于正常对齐的权重，行为不变

---

## `pad_or_narrow_weight` 函数说明

### 函数位置
`sglang/srt/layers/utils.py`, line 18-38

### 函数签名
```python
def pad_or_narrow_weight(
    loaded_weight: torch.Tensor, 
    input_dim: int, 
    start_idx: int, 
    shard_size: int
) -> torch.Tensor:
```

### 功能
- 如果 `start_idx` 在有效范围内，从 `start_idx` 开始取 `valid_size` 个元素
- 如果 `valid_size < shard_size`，用零填充剩余部分
- 如果 `start_idx` 超出范围，返回全零张量

### 实现逻辑
```python
valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

if valid_size > 0:
    loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
    pad_shape = list(loaded_weight.shape)
    pad_shape[input_dim] = shard_size - valid_size
    pad = torch.zeros(
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )
    return torch.cat([loaded_slice, pad], dim=input_dim)

# All padding
pad_shape = list(loaded_weight.shape)
pad_shape[input_dim] = shard_size
return torch.zeros(
    pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
)
```

---

## 变更总结

### 修改的文件数
1个文件：`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

### 修改的行数
- 添加导入：1行
- 修改方法：约10行（添加边界检查和padding逻辑）

### 影响范围
- **影响**: 仅影响MoE模型的权重加载，特别是使用tensor parallelism时
- **兼容性**: 向后兼容，正常对齐的权重不受影响
- **性能**: 仅在需要padding时才有额外开销，且开销很小

---

## 测试代码位置

如果需要添加测试，建议在以下位置：
- `test/srt/test_moe_tp_loading.py` (新建)
- 或者添加到现有的MoE测试文件中

### 测试用例建议
```python
def test_moe_tp_weight_loading_with_padding():
    """Test MoE weight loading with TP when weight dimension is not aligned."""
    # Test case: weight dimension = 8, tp_size = 2, tp_rank = 1
    # Expected: should use padding instead of failing
    pass
```
