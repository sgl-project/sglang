# Lightning Attention Backend 重构说明

## 概述

本次重构将 MiniMaxText01 模型的 AttentionMetadata 相关代码重构为了一个标准的 attention backend 实现。重构参考了 `flashinfer_mla_backend.py` 的架构设计。

## 主要变更

### 1. 新增 LightningAttnBackend 类

- **文件位置**: `sglang/srt/layers/attention/lightning_attn.py`
- **继承关系**: `LightningAttnBackend` <- `AttentionBackend`
- **核心功能**:
  - 实现标准的 attention backend 接口
  - 支持 prefill 和 decode 模式
  - 集成 lightning attention 算法
  - 支持 CUDA 图优化

### 2. 新增 LightningAttentionMetadata 类

- **文件位置**: `sglang/srt/layers/attention/lightning_attn.py`
- **功能**:
  - 存储 lightning attention 的元数据
  - 包含 slope rates、KV cache 等特定信息
  - 兼容原有的 metadata 接口

### 3. 重构 MiniMaxText01 模型

- **删除**: 原有的 `AttentionMetadata`、`AttentionMetadataManager`、`ForwardContext` 等类
- **更新**: `MiniMaxText01LinearAttention` 现在使用 `LightningAttnBackend`
- **简化**: 移除了全局状态管理，改为显式传递 backend

## 主要接口

### LightningAttnBackend 初始化

```python
from sglang.srt.layers.attention.lightning_attn import LightningAttnBackend

backend = LightningAttnBackend(
    model_runner=model_runner,
    num_heads=32,           # 注意力头数
    head_dim=64,            # 每个头的维度
    max_context_len=4096,   # 最大上下文长度
    block_size=256,         # 计算块大小
)
```

### 使用方式

```python
# 1. 初始化 metadata
backend.init_forward_metadata(forward_batch)

# 2. 前向推理
# Prefill 模式
output = backend.forward_extend(q, k, v, layer, forward_batch)

# Decode 模式
output = backend.forward_decode(q, k, v, layer, forward_batch)
```

## 技术特点

### 1. 模块化设计

- 将 lightning attention 封装为独立的 backend
- 支持与其他 attention backend 并存
- 清晰的接口定义

### 2. 高性能

- 保留原有的 Triton kernel 实现
- 支持 CUDA 图优化
- 内存高效的设计

### 3. 兼容性

- 兼容现有的 ForwardBatch 接口
- 支持混合 prefill/decode 模式
- 向后兼容原有代码

## 文件结构

```
sglang/srt/layers/attention/
├── lightning_attn.py              # 新增：Lightning Attention Backend
├── flashinfer_mla_backend.py      # 参考实现
└── base_attn_backend.py           # 基础接口

sglang/srt/models/
└── minimax_text_01.py             # 更新：使用新 backend
```

## 使用示例

详见 `lightning_attn.py` 文件末尾的 `create_lightning_attention_backend_example()` 函数。

## 优势

1. **标准化**: 符合 SGLang 的 attention backend 架构
2. **可维护性**: 清晰的模块边界，便于维护和扩展
3. **性能**: 保留了原有的高性能实现
4. **灵活性**: 支持不同的配置和优化选项

## 测试建议

1. 单元测试 `LightningAttnBackend` 的各个方法
2. 端到端测试 `MiniMaxText01` 模型的推理
3. 性能基准测试，确保重构后性能无损失
4. CUDA 图兼容性测试

## 后续工作

1. 添加更多的配置选项
2. 优化内存使用
3. 支持更多的 speculative decoding 特性
4. 集成到 SGLang 的自动 backend 选择机制中
