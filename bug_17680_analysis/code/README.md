# Fix for Issue #17680: MoE Tensor Parallelism Bug

## 问题描述
MoE模型在使用 `--tp-size 2` 时，在第二个GPU (TP1) 上加载权重时抛出 `RuntimeError: start (8) + length (8) exceeds dimension size (8)`。

## 修复位置
`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

## 修复内容

### 1. 添加导入 (line ~36)
在导入部分添加：
```python
from sglang.srt.layers.utils import pad_or_narrow_weight
```

### 2. 修复 `_load_w2` 方法 (line ~428-434)
将原来的直接 `narrow` 调用改为带边界检查的版本。

## 使用方法

### 应用修复
```bash
# 切换到修复分支
git checkout fix/issue-17680-moe-tp-bug

# 应用修复
patch -p1 < code/fix_moe_tp_layer.py.patch
```

### 或者手动修改
1. 在文件顶部添加导入
2. 修改 `_load_w2` 方法中的权重加载逻辑

## 测试
```bash
python3 -m sglang.launch_server  \
    --model-path MedAIBase/AntAngelMed-INT4 \
    --host 0.0.0.0 --port 30012  \
    --trust-remote-code  \
    --attention-backend fa3  \
    --mem-fraction-static 0.9 \
    --tp-size 2
```
