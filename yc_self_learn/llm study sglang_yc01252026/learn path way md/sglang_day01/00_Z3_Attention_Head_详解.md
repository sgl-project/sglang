# 00_Z3_Attention_Head_详解

## 📚 问题

**什么是Attention Head？**

在Multi-Head Attention中，我们经常听到"head"这个概念，但什么是head？为什么需要多个head？

---

## 🔍 知识点分解

### Z3.1 Attention Head基础概念

#### Z3.1.1 什么是Attention Head
- [ ] **定义**: Attention Head是Multi-Head Attention中的一个独立的attention计算单元
- [ ] **单个Head的作用**: 
  - [ ] 计算一组Q, K, V矩阵
  - [ ] 执行一次完整的attention计算
  - [ ] 产生一个attention输出
- [ ] **Head的数量**: 通常8, 16, 32, 64等（num_heads）
- [ ] **Head的独立性**: 每个head有自己独立的Q, K, V权重矩阵

**官方文档**:
- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762)

#### Z3.1.2 为什么叫"Head"
- [ ] **Head的含义**: 
  - [ ] 可以理解为"注意力头"或"关注点"
  - [ ] 每个head关注不同类型的特征或关系
- [ ] **类比**: 
  - [ ] 就像人有多个感官（视觉、听觉等）
  - [ ] 每个head关注不同的信息维度
- [ ] **Head的多样性**: 
  - [ ] 不同的head可能关注不同的位置
  - [ ] 不同的head可能关注不同的语义关系

---

### Z3.2 Single-Head vs Multi-Head

#### Z3.2.1 Single-Head Attention
- [ ] **结构**: 只有一个attention head
- [ ] **Q, K, V的维度**: 
  - [ ] Q: `[batch_size, seq_len, hidden_size]`
  - [ ] K: `[batch_size, seq_len, hidden_size]`
  - [ ] V: `[batch_size, seq_len, hidden_size]`
- [ ] **计算过程**: 
  - [ ] 一次attention计算
  - [ ] 输出: `[batch_size, seq_len, hidden_size]`
- [ ] **限制**: 
  - [ ] 只能学习一种类型的依赖关系
  - [ ] 表达能力有限

#### Z3.2.2 Multi-Head Attention
- [ ] **结构**: 有多个attention head（例如8个）
- [ ] **Q, K, V的维度**: 
  - [ ] 每个head: `[batch_size, seq_len, head_dim]`
  - [ ] head_dim = hidden_size / num_heads
  - [ ] 例如: hidden_size=512, num_heads=8 → head_dim=64
- [ ] **计算过程**: 
  - [ ] 每个head独立计算attention
  - [ ] 每个head产生: `[batch_size, seq_len, head_dim]`
  - [ ] 所有head拼接: `[batch_size, seq_len, hidden_size]`
- [ ] **优势**: 
  - [ ] 可以学习多种类型的依赖关系
  - [ ] 表达能力更强

**官方文档**:
- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762)

---

### Z3.3 Multi-Head Attention的详细结构

#### Z3.3.1 权重矩阵的分割
- [ ] **原始权重矩阵**: 
  - [ ] W_q: `[hidden_size, hidden_size]`
  - [ ] W_k: `[hidden_size, hidden_size]`
  - [ ] W_v: `[hidden_size, hidden_size]`
- [ ] **分割成多个head**: 
  - [ ] 每个head的权重: `[hidden_size, head_dim]`
  - [ ] 例如: hidden_size=512, num_heads=8, head_dim=64
  - [ ] W_q被分割成8个: 每个`[512, 64]`
- [ ] **计算方式**: 
  - [ ] `Q_i = X @ W_q_i`（第i个head的Q）
  - [ ] `K_i = X @ W_k_i`（第i个head的K）
  - [ ] `V_i = X @ W_v_i`（第i个head的V）

#### Z3.3.2 每个Head的独立计算
- [ ] **Head 1的计算**:
  - [ ] `Q1 = X @ W_q1`
  - [ ] `K1 = X @ W_k1`
  - [ ] `V1 = X @ W_v1`
  - [ ] `Attention1 = softmax(Q1 @ K1^T / √d_k) @ V1`
  - [ ] 输出: `[batch_size, seq_len, head_dim]`
- [ ] **Head 2的计算**:
  - [ ] 独立计算，使用不同的权重矩阵
  - [ ] 输出: `[batch_size, seq_len, head_dim]`
- [ ] **所有Head并行计算**: 
  - [ ] 8个head同时计算（可以并行）
  - [ ] 每个head产生独立的输出

#### Z3.3.3 Head输出的拼接
- [ ] **拼接操作**: 
  - [ ] 将所有head的输出拼接（concatenate）
  - [ ] 形状: `[batch_size, seq_len, num_heads * head_dim]`
  - [ ] 即: `[batch_size, seq_len, hidden_size]`
- [ ] **拼接公式**: 
  - [ ] `MultiHead(Q,K,V) = Concat(head1, head2, ..., head_h) @ W_o`
  - [ ] 其中W_o是输出投影矩阵: `[hidden_size, hidden_size]`
- [ ] **最终输出**: 
  - [ ] `[batch_size, seq_len, hidden_size]`
  - [ ] 与Single-Head的输出维度相同

**官方文档**:
- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762)

---

### Z3.4 为什么需要多个Head

#### Z3.4.1 不同类型的依赖关系
- [ ] **Head 1可能关注**: 语法关系（主谓宾）
- [ ] **Head 2可能关注**: 语义关系（同义词、反义词）
- [ ] **Head 3可能关注**: 长距离依赖（句子开头和结尾）
- [ ] **Head 4可能关注**: 局部依赖（相邻词）
- [ ] **多个Head**: 可以同时捕获多种关系

**重要说明**: 这些关注模式是**训练学习出来的**，不是预先设定的。参考 [00_Z4_Multi_Head_如何学习不同关注模式_详解.md](./00_Z4_Multi_Head_如何学习不同关注模式_详解.md) ⭐ **深入理解**

**如何验证**: 训练完成后，可以通过可视化attention权重来观察不同head的关注模式。参考 [00_Z5_如何验证Head的关注模式_详解.md](./00_Z5_如何验证Head的关注模式_详解.md) ⭐ **验证方法**

#### Z3.4.2 表达能力的提升
- [ ] **Single-Head的限制**: 
  - [ ] 只能学习一种attention模式
  - [ ] 表达能力有限
- [ ] **Multi-Head的优势**: 
  - [ ] 可以学习多种attention模式
  - [ ] 每个head关注不同的特征
  - [ ] 表达能力更强

#### Z3.4.3 实际例子
- [ ] **翻译任务**: 
  - [ ] Head 1: 关注源语言和目标语言的对应关系
  - [ ] Head 2: 关注语法结构
  - [ ] Head 3: 关注语义一致性
- [ ] **文本理解**: 
  - [ ] Head 1: 关注词与词的关系
  - [ ] Head 2: 关注短语级别的关系
  - [ ] Head 3: 关注句子级别的关系

**官方文档**:
- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762)

---

### Z3.5 Head的维度计算

#### Z3.5.1 维度关系
- [ ] **基本关系**: `hidden_size = num_heads * head_dim`
- [ ] **head_dim的计算**: `head_dim = hidden_size / num_heads`
- [ ] **常见配置**: 
  - [ ] hidden_size=512, num_heads=8 → head_dim=64
  - [ ] hidden_size=768, num_heads=12 → head_dim=64
  - [ ] hidden_size=1024, num_heads=16 → head_dim=64
  - [ ] hidden_size=2048, num_heads=32 → head_dim=64（大模型）

#### Z3.5.1.1 Head数量的常见选择
- [ ] **最常见的配置**: **8个head**（原始Transformer论文的标准配置）
- [ ] **中等模型**: 12个head（如BERT-base）
- [ ] **大型模型**: 16个head（如GPT-3的部分层）
- [ ] **超大模型**: 32个head（如GPT-3的某些层，或hidden_size=2048的模型）
- [ ] **关键原则**: 
  - [ ] `hidden_size` 必须能被 `num_heads` 整除
  - [ ] `head_dim` 通常保持在64左右（经验值）
  - [ ] 更多head = 更多不同的关注模式，但也增加计算成本

#### Z3.5.1.2 实际模型示例（公开架构信息）

**Meta LLaMA系列**:
- [ ] **LLaMA-7B**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **LLaMA-13B**: hidden_size=5120, num_heads=40 → head_dim=128
- [ ] **LLaMA-30B**: hidden_size=6656, num_heads=52 → head_dim=128
- [ ] **LLaMA-65B**: hidden_size=8192, num_heads=64 → head_dim=128
- [ ] **LLaMA-2-7B**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **LLaMA-2-13B**: hidden_size=5120, num_heads=40 → head_dim=128
- [ ] **LLaMA-2-70B**: hidden_size=8192, num_heads=64 → head_dim=128
- [ ] **LLaMA-3-8B**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **LLaMA-3-70B**: hidden_size=8192, num_heads=64 → head_dim=128

**Mistral AI系列**:
- [ ] **Mistral-7B**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **Mistral-7B-Instruct**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **Mixtral-8x7B**: hidden_size=4096, num_heads=32 → head_dim=128 (MoE模型)
- [ ] **Mixtral-8x22B**: hidden_size=6144, num_heads=48 → head_dim=128 (MoE模型)

**Qwen系列（阿里）**:
- [ ] **Qwen-7B**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **Qwen-14B**: hidden_size=5120, num_heads=40 → head_dim=128
- [ ] **Qwen-72B**: hidden_size=8192, num_heads=64 → head_dim=128
- [ ] **Qwen2-7B**: hidden_size=4096, num_heads=32 → head_dim=128

**Google系列**:
- [ ] **Gemma-2B**: hidden_size=2048, num_heads=8 → head_dim=256
- [ ] **Gemma-7B**: hidden_size=3072, num_heads=16 → head_dim=192
- [ ] **Gemma-2-9B**: hidden_size=3584, num_heads=14 → head_dim=256
- [ ] **Gemma-2-27B**: hidden_size=6144, num_heads=24 → head_dim=256

**其他公开模型**:
- [ ] **Phi-3**: hidden_size=3072, num_heads=32 → head_dim=96
- [ ] **PhiMoE**: hidden_size=4096, num_heads=32 → head_dim=128
- [ ] **FalconH1-9.8B**: hidden_size=4096, num_heads=32, num_kv_heads=8 (GQA)
- [ ] **Step3**: hidden_size=7168, num_heads=64 → head_dim=256
- [ ] **DeepSeek-V2**: hidden_size=3584, num_heads=28 → head_dim=128 (MoE)

**传统模型（参考）**:
- [ ] **GPT-2 Small**: hidden_size=768, num_heads=12 → head_dim=64
- [ ] **GPT-2 Medium**: hidden_size=1024, num_heads=16 → head_dim=64
- [ ] **BERT-base**: hidden_size=768, num_heads=12 → head_dim=64
- [ ] **BERT-large**: hidden_size=1024, num_heads=16 → head_dim=64
- [ ] **GPT-4**: 未公开详细配置（OpenAI未披露架构细节）
- [ ] **GPT-4o**: 未公开详细配置
- [ ] **o1 (o1-preview/o1-mini)**: 未公开详细配置（可能使用不同的架构）
- [ ] **GPT-5**: **尚未正式发布**，无公开架构信息

#### Z3.5.1.3 关于GPT-5和最新模型
- [ ] **GPT-5状态**: 
  - [ ] 截至2025年1月，GPT-5尚未正式发布
  - [ ] OpenAI未公开任何GPT-5的架构细节
  - [ ] 无法确定head数量等具体配置
- [ ] **GPT-4系列**: 
  - [ ] GPT-4、GPT-4 Turbo、GPT-4o的详细架构均未公开
  - [ ] OpenAI出于竞争考虑，通常不公开大模型的详细架构
  - [ ] 推测：可能使用64+个head（基于模型规模）
- [ ] **o1系列**: 
  - [ ] o1-preview和o1-mini可能使用不同的架构
  - [ ] 可能不是传统的Transformer架构
  - [ ] 架构细节未公开
- [ ] **趋势推测**: 
  - [ ] 如果GPT-5继续增大模型规模，head数量可能会增加
  - [ ] 但head数量不是唯一指标，head_dim也会相应调整
  - [ ] 关键公式：`hidden_size = num_heads × head_dim` 仍然适用
- [ ] **获取最新信息**: 
  - [ ] 关注OpenAI官方技术博客
  - [ ] 关注相关技术论文（如果发布）
  - [ ] 关注开源社区的分析（如通过模型权重逆向工程）

#### Z3.5.2 为什么head_dim通常是64
- [ ] **经验值**: 64是一个常用的head_dim值
- [ ] **原因**: 
  - [ ] 64足够表达复杂的attention模式
  - [ ] 不会太小（信息丢失）也不会太大（计算开销）
- [ ] **其他选择**: 
  - [ ] 32: 更小的模型（较少见）
  - [ ] 128: 更大的模型（如LLaMA系列）
- [ ] **重要**: head_dim的选择会影响head数量的选择
  - [ ] 如果head_dim=64，那么num_heads = hidden_size / 64
  - [ ] 如果hidden_size=2048，那么num_heads = 2048 / 64 = 32
  - [ ] 所以32个head通常出现在hidden_size=2048或更大的模型中

#### Z3.5.3 维度示例
- [ ] **输入**: `X: [batch_size=2, seq_len=10, hidden_size=512]`
- [ ] **num_heads=8**: 
  - [ ] head_dim = 512 / 8 = 64
  - [ ] 每个head的Q: `[2, 10, 64]`
  - [ ] 每个head的K: `[2, 10, 64]`
  - [ ] 每个head的V: `[2, 10, 64]`
  - [ ] 每个head的输出: `[2, 10, 64]`
  - [ ] 拼接后: `[2, 10, 512]`

---

### Z3.6 Multi-Head Attention的完整流程

#### Z3.6.1 步骤1: 线性投影并分割
- [ ] **输入**: `X: [batch_size, seq_len, hidden_size]`
- [ ] **计算Q, K, V**: 
  - [ ] `Q = X @ W_q` → `[batch_size, seq_len, hidden_size]`
  - [ ] `K = X @ W_k` → `[batch_size, seq_len, hidden_size]`
  - [ ] `V = X @ W_v` → `[batch_size, seq_len, hidden_size]`
- [ ] **分割成多个head**: 
  - [ ] `Q` → `[batch_size, seq_len, num_heads, head_dim]`
  - [ ] `K` → `[batch_size, seq_len, num_heads, head_dim]`
  - [ ] `V` → `[batch_size, seq_len, num_heads, head_dim]`
- [ ] **转置**: 
  - [ ] `Q` → `[batch_size, num_heads, seq_len, head_dim]`
  - [ ] `K` → `[batch_size, num_heads, seq_len, head_dim]`
  - [ ] `V` → `[batch_size, num_heads, seq_len, head_dim]`

#### Z3.6.2 步骤2: 每个Head独立计算Attention
- [ ] **对每个head i**:
  - [ ] `Q_i = Q[:, i, :, :]` → `[batch_size, seq_len, head_dim]`
  - [ ] `K_i = K[:, i, :, :]` → `[batch_size, seq_len, head_dim]`
  - [ ] `V_i = V[:, i, :, :]` → `[batch_size, seq_len, head_dim]`
  - [ ] `scores_i = Q_i @ K_i^T / √head_dim`
  - [ ] `attention_i = softmax(scores_i) @ V_i`
  - [ ] 输出: `[batch_size, seq_len, head_dim]`

#### Z3.6.3 步骤3: 拼接所有Head
- [ ] **拼接**: 
  - [ ] 将所有head的输出拼接
  - [ ] `Concat(head1, head2, ..., head_h)` → `[batch_size, seq_len, hidden_size]`
- [ ] **输出投影**: 
  - [ ] `Output = Concat(...) @ W_o`
  - [ ] `W_o: [hidden_size, hidden_size]`
  - [ ] 最终输出: `[batch_size, seq_len, hidden_size]`

**官方文档**:
- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762)

---

### Z3.7 Head的并行计算

#### Z3.7.1 为什么可以并行
- [ ] **独立性**: 每个head的计算是独立的
- [ ] **无依赖**: Head之间没有数据依赖
- [ ] **并行执行**: 可以在GPU上并行计算所有head

#### Z3.7.2 并行计算的实现
- [ ] **批量矩阵乘法**: 
  - [ ] 使用`torch.bmm`或`torch.einsum`
  - [ ] 一次性计算所有head
- [ ] **GPU加速**: 
  - [ ] 所有head的计算可以同时进行
  - [ ] 充分利用GPU的并行能力

#### Z3.7.3 计算复杂度
- [ ] **Single-Head**: O(seq_len² × hidden_size)
- [ ] **Multi-Head**: O(seq_len² × hidden_size)（相同！）
- [ ] **原因**: 
  - [ ] 虽然head数量增加了，但每个head的维度减小了
  - [ ] 总计算量相同
  - [ ] 但可以并行计算，速度更快

---

### Z3.8 Head数量的选择

#### Z3.8.1 常见配置
- [ ] **小模型**: 4-8 heads
- [ ] **中等模型**: 8-16 heads
- [ ] **大模型**: 16-32 heads
- [ ] **超大模型**: 32-64 heads

#### Z3.8.2 选择原则
- [ ] **hidden_size的限制**: 
  - [ ] head_dim = hidden_size / num_heads
  - [ ] head_dim不能太小（通常≥32）
- [ ] **计算资源**: 
  - [ ] 更多head需要更多内存
  - [ ] 但可以并行计算
- [ ] **任务需求**: 
  - [ ] 复杂任务可能需要更多head
  - [ ] 简单任务可能不需要太多head

#### Z3.8.3 实际例子
- [ ] **BERT-base**: hidden_size=768, num_heads=12, head_dim=64
- [ ] **BERT-large**: hidden_size=1024, num_heads=16, head_dim=64
- [ ] **GPT-3**: hidden_size=12288, num_heads=96, head_dim=128

---

## 📊 可视化理解 - 流程图

### Z3.9 Single-Head Attention流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入: X                                   │
│            [batch_size, seq_len, hidden_size]               │
│                    例如: [2, 10, 512]                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   线性投影 (Linear)          │
        │   Q = X @ W_q                │
        │   K = X @ W_k                │
        │   V = X @ W_v                │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼                              ▼
    Q: [2,10,512]              K: [2,10,512]              V: [2,10,512]
        │                              │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Attention计算              │
        │   scores = Q @ K^T / √d_k    │
        │   weights = softmax(scores)  │
        │   output = weights @ V       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   输出                        │
        │   [batch_size, seq_len,      │
        │    hidden_size]              │
        │   例如: [2, 10, 512]         │
        └──────────────────────────────┘
```

---

### Z3.10 Multi-Head Attention流程图（详细版）

```
┌─────────────────────────────────────────────────────────────┐
│                    输入: X                                   │
│            [batch_size, seq_len, hidden_size]               │
│                    例如: [2, 10, 512]                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   线性投影 (Linear)          │
        │   Q = X @ W_q  [2,10,512]    │
        │   K = X @ W_k  [2,10,512]    │
        │   V = X @ W_v  [2,10,512]    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   分割成多个Head              │
        │   num_heads = 8              │
        │   head_dim = 512/8 = 64      │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼                              ▼
    Q: [2,10,8,64]            K: [2,10,8,64]            V: [2,10,8,64]
        │                              │                              │
        │  转置和重塑                   │                              │
        ▼                              ▼                              ▼
    Q: [2,8,10,64]            K: [2,8,10,64]            V: [2,8,10,64]
        │                              │                              │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │    8个Head并行计算            │
        └──────────────┬───────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Head 1  │      │ Head 2  │ ...  │ Head 8  │
│         │      │         │      │         │
│ Q1[2,10,│      │ Q2[2,10,│      │ Q8[2,10,│
│   64]   │      │   64]   │      │   64]   │
│         │      │         │      │         │
│ K1[2,10,│      │ K2[2,10,│      │ K8[2,10,│
│   64]   │      │   64]   │      │   64]   │
│         │      │         │      │         │
│ V1[2,10,│      │ V2[2,10,│      │ V8[2,10,│
│   64]   │      │   64]   │      │   64]   │
│         │      │         │      │         │
│ Attention1     │ Attention2     │ Attention8
│ [2,10,64]      │ [2,10,64]      │ [2,10,64]
└─────────┘      └─────────┘      └─────────┘
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   拼接 (Concat)              │
        │   所有head的输出拼接         │
        │   [2, 10, 512]               │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   输出投影 (Output Projection)│
        │   Output = Concat(...) @ W_o │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   最终输出                    │
        │   [batch_size, seq_len,      │
        │    hidden_size]              │
        │   例如: [2, 10, 512]         │
        └──────────────────────────────┘
```

---

### Z3.11 单个Head的详细计算流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Head i 的计算流程                          │
└─────────────────────────────────────────────────────────────┘

输入: 
  Q_i: [batch_size, seq_len, head_dim] 例如: [2, 10, 64]
  K_i: [batch_size, seq_len, head_dim] 例如: [2, 10, 64]
  V_i: [batch_size, seq_len, head_dim] 例如: [2, 10, 64]

步骤1: 计算相似度矩阵
  ┌─────────────┐      ┌─────────────┐
  │ Q_i         │      │ K_i^T       │
  │ [2,10,64]   │  @   │ [2,64,10]   │
  └──────┬──────┘      └──────┬──────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ scores = Q_i @ K_i^T │
         │ [2, 10, 10]          │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ 缩放: scores / √64    │
         │ [2, 10, 10]          │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Softmax              │
         │ weights = softmax(   │
         │   scores)            │
         │ [2, 10, 10]          │
         │ 每行和为1            │
         └──────────┬───────────┘
                    │
                    ▼
步骤2: 加权求和
  ┌─────────────┐      ┌─────────────┐
  │ weights     │      │ V_i         │
  │ [2,10,10]   │  @   │ [2,10,64]   │
  └──────┬──────┘      └──────┬──────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Attention输出        │
         │ [2, 10, 64]         │
         └──────────────────────┘
```

---

### Z3.12 Multi-Head的并行计算示意图

```
时间轴: ────────────────────────────────────────────►

Single-Head (串行):
  Head1计算 ──────────┐
                      │
                      ▼ 等待Head1完成
  Head2计算 ──────────┐
                      │
                      ▼ 等待Head2完成
  Head3计算 ──────────┐
                      │
                      ▼ 总时间 = 3 × 单个head时间

Multi-Head (并行):
  Head1计算 ──────────┐
  Head2计算 ──────────┤
  Head3计算 ──────────┤ 同时进行
  ...                 │
  Head8计算 ──────────┘
                      │
                      ▼ 总时间 = 单个head时间
                      │
  拼接所有head输出 ────┘
```

---

### Z3.13 维度变化可视化

```
输入 X
┌─────────────────────┐
│ [2, 10, 512]        │  batch_size=2, seq_len=10, hidden_size=512
└──────────┬───────────┘
           │
           │ 线性投影
           ▼
┌─────────────────────┐
│ Q: [2, 10, 512]     │
│ K: [2, 10, 512]     │
│ V: [2, 10, 512]     │
└──────────┬───────────┘
           │
           │ 分割成8个head (num_heads=8, head_dim=64)
           ▼
┌─────────────────────┐
│ Q: [2, 10, 8, 64]   │  重塑: [batch, seq, heads, head_dim]
│ K: [2, 10, 8, 64]   │
│ V: [2, 10, 8, 64]   │
└──────────┬───────────┘
           │
           │ 转置: 将head维度提前
           ▼
┌─────────────────────┐
│ Q: [2, 8, 10, 64]   │  转置: [batch, heads, seq, head_dim]
│ K: [2, 8, 10, 64]   │
│ V: [2, 8, 10, 64]   │
└──────────┬───────────┘
           │
           │ 每个head独立计算
           ▼
┌─────────────────────┐
│ Head1: [2, 10, 64]  │
│ Head2: [2, 10, 64]  │
│ Head3: [2, 10, 64]  │
│ ...                 │
│ Head8: [2, 10, 64]  │
└──────────┬───────────┘
           │
           │ 拼接 (Concat)
           ▼
┌─────────────────────┐
│ [2, 10, 512]        │  8个head × 64 = 512
└──────────┬───────────┘
           │
           │ 输出投影
           ▼
┌─────────────────────┐
│ [2, 10, 512]        │  最终输出
└─────────────────────┘
```

---

### Z3.14 Attention权重矩阵可视化（单个Head）

```
假设 seq_len = 5, 计算"Hello world how are you"的attention

Head 1的Attention权重矩阵 (weights):
        Hello  world  how   are   you
Hello  [ 0.3   0.2   0.1   0.2   0.2 ]  ← Hello关注所有词
world  [ 0.1   0.4   0.2   0.1   0.2 ]  ← world主要关注自己
how    [ 0.1   0.1   0.5   0.2   0.1 ]  ← how关注自己
are    [ 0.1   0.1   0.1   0.5   0.2 ]  ← are关注自己
you    [ 0.1   0.1   0.1   0.1   0.6 ]  ← you关注自己

Head 2的Attention权重矩阵 (可能关注语法关系):
        Hello  world  how   are   you
Hello  [ 0.6   0.3   0.05  0.03  0.02 ]  ← Hello主要关注world
world  [ 0.3   0.5   0.1   0.05  0.05 ]  ← world关注Hello和how
how    [ 0.05  0.1   0.6   0.2   0.05 ]  ← how关注are
are    [ 0.02  0.05  0.2   0.6   0.13 ]  ← are关注you
you    [ 0.01  0.02  0.05  0.12  0.8  ]  ← you主要关注自己

不同的Head关注不同的关系！
```

---

### Z3.15 完整Multi-Head Attention数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入序列                                   │
│              "The cat sat on the mat"                          │
│              Token IDs: [1, 2, 3, 4, 5, 6]                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Embedding     │
                    │  [6, 512]      │  6个token，每个512维
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Q, K, V投影   │
                    │  Q: [6, 512]   │
                    │  K: [6, 512]   │
                    │  V: [6, 512]   │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  分割成8个Head │
                    │  Q: [6, 8, 64] │
                    │  K: [6, 8, 64] │
                    │  V: [6, 8, 64] │
                    └────────┬───────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Head 1  │         │ Head 2  │   ...   │ Head 8  │
   │         │         │         │         │         │
   │ 关注:   │         │ 关注:   │         │ 关注:   │
   │ 语法    │         │ 语义    │         │ 位置    │
   │         │         │         │         │         │
   │ Output: │         │ Output: │         │ Output: │
   │ [6,64]  │         │ [6,64]  │         │ [6,64]  │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   拼接         │
                    │   [6, 512]     │  8×64=512
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   输出投影     │
                    │   [6, 512]     │
                    └────────────────┘
```

---

### Z3.16 矩阵运算可视化

#### Head 1的Attention计算（详细）

```
步骤1: Q @ K^T
  Q1: [2, 10, 64]          K1^T: [2, 64, 10]
  ┌─────────────┐          ┌─────────────┐
  │ token1: 64维│          │ token1: 64维│
  │ token2: 64维│    @     │ token2: 64维│
  │ ...         │          │ ...         │
  │ token10:64维│          │ token10:64维│
  └─────────────┘          └─────────────┘
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
            scores: [2, 10, 10]
            ┌─────────────────────┐
            │ token1对token1的相似度│
            │ token1对token2的相似度│
            │ ...                 │
            │ token10对token10的   │
            └─────────────────────┘

步骤2: Softmax
  scores: [2, 10, 10]
           │
           ▼
  weights: [2, 10, 10]
  ┌─────────────────────┐
  │ 每行和为1            │
  │ token1的attention权重│
  │ token2的attention权重│
  │ ...                 │
  └─────────────────────┘

步骤3: weights @ V
  weights: [2,10,10]    V1: [2, 10, 64]
           │                    │
           └──────────┬─────────┘
                      │
                      ▼
           output: [2, 10, 64]
           ┌─────────────────┐
           │ token1的输出:64维│
           │ token2的输出:64维│
           │ ...             │
           └─────────────────┘
```

---

## 📊 可视化理解 - 流程图

---

## ✅ 总结

### 核心概念
1. **Attention Head**: Multi-Head Attention中的一个独立attention计算单元
2. **每个Head**: 有自己独立的Q, K, V权重矩阵
3. **Head的维度**: head_dim = hidden_size / num_heads
4. **多个Head**: 可以学习不同类型的依赖关系
5. **并行计算**: 所有head可以并行计算

### 关键公式
```
head_dim = hidden_size / num_heads
MultiHead(Q,K,V) = Concat(head1, head2, ..., head_h) @ W_o
```

### 为什么需要多个Head
- ✅ 学习多种类型的依赖关系
- ✅ 提升表达能力
- ✅ 可以并行计算，不增加计算复杂度

---

## 🔗 相关文档

- [Attention Is All You Need - Section 3.2.2](https://arxiv.org/abs/1706.03762) ⭐ **原始论文**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化理解
- [00_Z1_Scaled_Dot_Product_Attention_详解.md](./00_Z1_Scaled_Dot_Product_Attention_详解.md) - Attention机制详解

---

## 💡 记忆技巧

1. **Head = 注意力头**: 每个head关注不同的特征
2. **多个Head = 多个视角**: 就像从多个角度观察同一个问题
3. **Head维度**: hidden_size = num_heads × head_dim
4. **并行计算**: 所有head可以同时计算，速度快
