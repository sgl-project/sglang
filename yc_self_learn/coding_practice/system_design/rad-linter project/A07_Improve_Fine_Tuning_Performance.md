# A07: Improve Fine-Tuning Performance Without Changing Dataset
# A07: 数据集不变的情况下如何提升 Fine-Tuning 性能

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：How to improve model performance through better fine-tuning strategies when dataset remains unchanged  
**Key Question**: How to improve performance without changing the dataset?

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Hyperparameter Tuning](#hyperparameter-tuning)
3. [Training Strategy Improvements](#training-strategy-improvements)
4. [Model Architecture Adjustments](#model-architecture-adjustments)
5. [Loss Function Optimization](#loss-function-optimization)
6. [Data Utilization Improvements](#data-utilization-improvements)
7. [Upstream Component Optimization](#upstream-component-optimization)
8. [Complete Optimization Workflow](#complete-optimization-workflow)

---

## Overview

### The Question（问题）

**If dataset doesn't change, how can we improve model performance through fine-tuning?**

**中文解释**：如果数据集不变，我们如何通过改进 fine-tuning 策略来提升模型性能？

### Key Insight（核心洞察）

**Even with the same dataset, you can improve performance by:**
1. **Hyperparameter tuning**（超参数调优）
2. **Better training strategies**（更好的训练策略）
3. **Model architecture adjustments**（模型架构调整）
4. **Loss function optimization**（损失函数优化）
5. **Better data utilization**（更好的数据利用）
6. **Upstream component optimization**（上游组件优化）

### Optimization Dimensions（优化维度）

```
Dataset (Fixed) 数据集固定
    ↓
┌─────────────────────────────────────────┐
│ Optimization Opportunities 优化机会     │
├─────────────────────────────────────────┤
│ 1. Hyperparameters 超参数               │
│    - Learning rate, batch size, epochs  │
│    - LoRA rank, LoRA alpha              │
│                                         │
│ 2. Training Strategy 训练策略            │
│    - Learning rate scheduling           │
│    - Warmup, gradient accumulation      │
│    - Mixed precision training           │
│                                         │
│ 3. Model Architecture 模型架构          │
│    - LoRA target modules                │
│    - LoRA rank adjustment               │
│                                         │
│ 4. Loss Function 损失函数               │
│    - Weighted loss                      │
│    - Focal loss for imbalanced data    │
│                                         │
│ 5. Data Utilization 数据利用            │
│    - Data augmentation                 │
│    - Better data sampling              │
│                                         │
│ 6. Upstream Components 上游组件          │
│    - Prompt optimization (Step 3.5)    │
│    - Rule optimization (Step 3)         │
└─────────────────────────────────────────┘
    ↓
Improved Model Performance 改进的模型性能
```

---

## Hyperparameter Tuning（超参数调优）

### 1. Learning Rate Tuning（学习率调优）

#### Current Configuration（当前配置）

```yaml
Current:
  Learning Rate: 2e-4  # 固定学习率
```

#### Optimization Strategies（优化策略）

**Strategy A: Learning Rate Scheduling（学习率调度）**

```python
# Instead of fixed learning rate
# 不使用固定学习率，而是使用学习率调度

from transformers import get_linear_schedule_with_warmup

training_args = TrainingArguments(
    learning_rate=2e-4,  # Initial learning rate（初始学习率）
    lr_scheduler_type="cosine",  # Cosine annealing（余弦退火）
    warmup_steps=100,  # Warmup steps（预热步数）
    num_train_epochs=5
)

# Or use custom scheduler（或使用自定义调度器）
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,  # Warmup: gradually increase LR（预热：逐渐增加学习率）
    num_training_steps=total_steps  # Then decrease linearly（然后线性下降）
)
```

**Benefits（好处）**：
- **Warmup**：避免训练初期的不稳定
- **Scheduling**：后期降低学习率，更精细的优化
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2% 准确率）

**Strategy B: Learning Rate Search（学习率搜索）**

```python
# Learning rate range test（学习率范围测试）
learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

for lr in learning_rates:
    # Train for 1 epoch with this learning rate
    # 用这个学习率训练 1 个 epoch
    trainer = Trainer(
        learning_rate=lr,
        num_train_epochs=1  # Quick test（快速测试）
    )
    trainer.train()
    # Evaluate and record performance（评估并记录性能）
    # Choose best learning rate（选择最佳学习率）
```

**Recommended Learning Rates（推荐学习率）**：
- **Conservative**: 1e-4（保守：1e-4）
- **Default**: 2e-4（默认：2e-4）
- **Aggressive**: 5e-4（激进：5e-4，可能不稳定）

### 2. Batch Size Optimization（批次大小优化）

#### Current Configuration（当前配置）

```yaml
Current:
  Batch Size: 4 (per GPU)
```

#### Optimization Strategies（优化策略）

**Strategy A: Increase Batch Size（增加批次大小）**

```python
# Larger batch size = more stable gradients（更大的批次大小 = 更稳定的梯度）
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Increase from 4 to 8（从 4 增加到 8）
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16（有效批次大小）
    # 如果 GPU 显存不够，可以用 gradient accumulation
)
```

**Benefits（好处）**：
- **More stable training**：更大的批次大小带来更稳定的梯度
- **Better convergence**：可能收敛到更好的局部最优
- **Expected improvement**: +0.5-1% accuracy（预期提升：+0.5-1%）

**Trade-off（权衡）**：
- **GPU Memory**：需要更多显存
- **Training Speed**：可能稍慢（但可以用 gradient accumulation 补偿）

**Strategy B: Gradient Accumulation（梯度累积）**

```python
# Simulate larger batch size without more GPU memory
# 模拟更大的批次大小，不需要更多 GPU 显存

training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Small batch（小批次）
    gradient_accumulation_steps=4,  # Accumulate gradients（累积梯度）
    # Effective batch size = 4 * 4 = 16（有效批次大小 = 16）
)
```

**Benefits（好处）**：
- **Same effect as larger batch**：和更大批次效果相同
- **No extra GPU memory**：不需要额外显存
- **Flexible**：可以根据显存情况调整

### 3. Epochs and Early Stopping（训练轮数和早停）

#### Current Configuration（当前配置）

```yaml
Current:
  Epochs: 5 (fixed)
```

#### Optimization Strategies（优化策略）

**Strategy A: Early Stopping（早停）**

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    num_train_epochs=10,  # Max epochs（最大轮数）
    load_best_model_at_end=True,  # Load best model（加载最佳模型）
    metric_for_best_model="eval_silver_agreement",  # Metric to monitor（监控指标）
    greater_is_better=True,
    evaluation_strategy="epoch",  # Evaluate every epoch（每个 epoch 评估）
    save_strategy="epoch"
)

trainer = Trainer(
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop if no improvement for 3 epochs（3 个 epoch 无改进就停止）
            early_stopping_threshold=0.001  # Minimum improvement threshold（最小改进阈值）
        )
    ]
)
```

**Benefits（好处）**：
- **Prevent overfitting**：防止过拟合
- **Save time**：如果提前收敛，节省训练时间
- **Better generalization**：可能获得更好的泛化性能

**Strategy B: More Epochs with Monitoring（更多轮数 + 监控）**

```python
# Train longer but monitor closely（训练更长时间但密切监控）
training_args = TrainingArguments(
    num_train_epochs=10,  # Increase from 5 to 10（从 5 增加到 10）
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps（每 500 步评估）
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True  # Always use best checkpoint（总是使用最佳 checkpoint）
)
```

**Benefits（好处）**：
- **More training**：更多训练可能带来更好性能
- **Best model selection**：自动选择最佳 checkpoint
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

---

## Training Strategy Improvements（训练策略改进）

### 1. Learning Rate Scheduling（学习率调度）

#### Cosine Annealing（余弦退火）

```python
training_args = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # Cosine annealing（余弦退火）
    warmup_steps=100,  # Warmup steps（预热步数）
    num_train_epochs=5
)
```

**How it works（工作原理）**：
```
Learning Rate
    ↑
2e-4 |     ╱╲
     |    ╱  ╲
1e-4 |   ╱    ╲
     |  ╱      ╲
   0 |╱────────╲────→ Steps
     0  warmup  end
```

**Benefits（好处）**：
- **Warmup**：训练初期逐渐增加学习率，避免不稳定
- **Gradual decay**：后期逐渐降低学习率，精细优化
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

#### Linear Schedule with Warmup（线性调度 + 预热）

```python
from transformers import get_linear_schedule_with_warmup

# Create optimizer（创建优化器）
optimizer = AdamW(model.parameters(), lr=2e-4)

# Create scheduler（创建调度器）
total_steps = len(train_dataset) // batch_size * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup（10% 预热）
    num_training_steps=total_steps
)
```

### 2. Mixed Precision Training（混合精度训练）

#### Enable FP16/BF16（启用 FP16/BF16）

```python
training_args = TrainingArguments(
    fp16=True,  # Use FP16（使用 FP16）
    # or
    bf16=True,  # Use BF16 (better for some GPUs)（使用 BF16，某些 GPU 更好）
    
    # Benefits（好处）:
    # - Faster training（训练更快）
    # - Less GPU memory（显存更少）
    # - Can use larger batch size（可以使用更大批次）
)
```

**Benefits（好处）**：
- **2x faster training**：训练速度提升约 2 倍
- **Less GPU memory**：显存占用减少，可以用更大 batch size
- **Same accuracy**：通常精度相同或略有提升

### 3. Gradient Checkpointing（梯度检查点）

#### Enable Gradient Checkpointing（启用梯度检查点）

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Enable（启用）
    
    # Benefits（好处）:
    # - Save GPU memory（节省 GPU 显存）
    # - Can use larger batch size（可以使用更大批次）
    # - Trade-off: slightly slower（权衡：稍慢一些）
)
```

**Benefits（好处）**：
- **Save 40-50% GPU memory**：节省 40-50% 显存
- **Larger batch size**：可以用更大批次，可能提升性能
- **Trade-off**：训练速度稍慢（约 20%），但通常值得

### 4. Label Smoothing（标签平滑）

#### For Imbalanced Data（用于不平衡数据）

```python
from torch.nn import CrossEntropyLoss

# Standard loss（标准损失）
loss_fn = CrossEntropyLoss()

# Label smoothing loss（标签平滑损失）
loss_fn = CrossEntropyLoss(
    label_smoothing=0.1  # Smooth labels（平滑标签）
)

# How it works（工作原理）:
# Instead of hard label [0, 1], use soft label [0.05, 0.95]
# 不使用硬标签 [0, 1]，而是使用软标签 [0.05, 0.95]
```

**Benefits（好处）**：
- **Prevent overconfidence**：防止模型过度自信
- **Better generalization**：更好的泛化性能
- **Useful for imbalanced data**：对不平衡数据特别有用
- **Expected improvement**: +0.5-1% accuracy（预期提升：+0.5-1%）

---

## Model Architecture Adjustments（模型架构调整）

### 1. LoRA Rank Tuning（LoRA 秩调优）

#### Current Configuration（当前配置）

```yaml
Current:
  LoRA Rank: 16
```

#### Optimization Strategies（优化策略）

**Strategy A: Increase LoRA Rank（增加 LoRA 秩）**

```python
# Higher rank = more capacity（更高的秩 = 更大的容量）
lora_config = LoraConfig(
    r=32,  # Increase from 16 to 32（从 16 增加到 32）
    lora_alpha=64,  # Keep alpha = 2 * rank（保持 alpha = 2 * rank）
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
```

**Benefits（好处）**：
- **More model capacity**：更大的模型容量
- **Can learn more complex patterns**：可以学习更复杂的模式
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

**Trade-off（权衡）**：
- **More parameters**：更多参数（但仍然是 LoRA，参数很少）
- **Slightly slower**：稍慢一些（通常可忽略）

**Strategy B: LoRA Rank Search（LoRA 秩搜索）**

```python
# Test different ranks（测试不同的秩）
ranks = [8, 16, 32, 64]

for rank in ranks:
    lora_config = LoraConfig(r=rank, lora_alpha=2*rank)
    model = get_peft_model(base_model, lora_config)
    # Train and evaluate（训练并评估）
    # Choose best rank（选择最佳秩）
```

**Recommended Ranks（推荐秩）**：
- **Small dataset (< 5K)**: r=8-16（小数据集：r=8-16）
- **Medium dataset (5K-10K)**: r=16-32（中等数据集：r=16-32）
- **Large dataset (> 10K)**: r=32-64（大数据集：r=32-64）

### 2. LoRA Target Modules（LoRA 目标模块）

#### Current Configuration（当前配置）

```python
Current:
  target_modules=["q_proj", "v_proj"]  # Only Q and V projections（只有 Q 和 V 投影）
```

#### Optimization Strategies（优化策略）

**Strategy A: Add More Modules（添加更多模块）**

```python
# Add more attention modules（添加更多注意力模块）
lora_config = LoraConfig(
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # All attention projections（所有注意力投影）
        "gate_proj", "up_proj", "down_proj"  # MLP projections（MLP 投影）
    ]
)
```

**Benefits（好处）**：
- **More trainable parameters**：更多可训练参数
- **Better adaptation**：更好的适配能力
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

**Trade-off（权衡）**：
- **More parameters**：更多参数（但仍然很少）
- **Slightly slower training**：训练稍慢（通常可接受）

**Strategy B: Module-Specific Ranks（模块特定秩）**

```python
# Different ranks for different modules（不同模块使用不同秩）
# This requires custom implementation（这需要自定义实现）

# Example: Higher rank for attention, lower for MLP
# 示例：注意力模块用更高秩，MLP 用较低秩
attention_rank = 32
mlp_rank = 16
```

### 3. LoRA Alpha Tuning（LoRA Alpha 调优）

#### Current Configuration（当前配置）

```python
Current:
  LoRA Alpha: 32 (usually 2 * rank)
```

#### Optimization Strategies（优化策略）

**Strategy A: Tune Alpha/Rank Ratio（调优 Alpha/Rank 比例）**

```python
# Alpha controls the scaling of LoRA weights（Alpha 控制 LoRA 权重的缩放）
# Common ratios: 1x, 2x, 4x rank（常见比例：1x、2x、4x 秩）

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 2x rank（2 倍秩）
    # Try: 16 (1x), 32 (2x), 64 (4x)（尝试：16、32、64）
)
```

**Benefits（好处）**：
- **Control LoRA strength**：控制 LoRA 的强度
- **Fine-tune adaptation**：精细调整适配程度
- **Expected improvement**: +0.5-1% accuracy（预期提升：+0.5-1%）

---

## Loss Function Optimization（损失函数优化）

### 1. Weighted Loss（加权损失）

#### For Imbalanced Classes（用于不平衡类别）

```python
from torch.nn import CrossEntropyLoss
import torch

# Calculate class weights（计算类别权重）
# If some issue types are rare（如果某些问题类型很少见）
class_weights = torch.tensor([
    1.0,  # Common issue type（常见问题类型）
    2.0,  # Rare issue type（罕见问题类型）
    1.5,  # Medium frequency（中等频率）
])

loss_fn = CrossEntropyLoss(weight=class_weights)
```

**Benefits（好处）**：
- **Handle imbalanced data**：处理不平衡数据
- **Focus on rare cases**：关注罕见案例
- **Expected improvement**: +1-2% on rare classes（预期提升：罕见类别 +1-2%）

### 2. Focal Loss（Focal Loss）

#### For Hard Examples（用于困难样本）

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Use focal loss（使用 Focal Loss）
loss_fn = FocalLoss(alpha=1, gamma=2)
```

**Benefits（好处）**：
- **Focus on hard examples**：关注困难样本
- **Better for imbalanced data**：对不平衡数据更好
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

### 3. Multi-Task Loss（多任务损失）

#### Combine Multiple Objectives（组合多个目标）

```python
def multi_task_loss(predictions, targets, rule_labels, judge_labels):
    """
    Combine multiple loss functions（组合多个损失函数）
    """
    # Task 1: Issue type prediction（任务 1：问题类型预测）
    loss1 = CrossEntropyLoss()(predictions['issue_type'], targets['issue_type'])
    
    # Task 2: Severity prediction（任务 2：严重程度预测）
    loss2 = CrossEntropyLoss()(predictions['severity'], targets['severity'])
    
    # Task 3: Rule adherence（任务 3：规则遵循）
    loss3 = MSELoss()(predictions['rule_score'], rule_labels['score'])
    
    # Weighted combination（加权组合）
    total_loss = 0.4 * loss1 + 0.4 * loss2 + 0.2 * loss3
    return total_loss
```

**Benefits（好处）**：
- **Multi-objective optimization**：多目标优化
- **Better alignment with evaluation**：更好地对齐评估指标
- **Expected improvement**: +1-2% on combined metrics（预期提升：组合指标 +1-2%）

---

## Data Utilization Improvements（数据利用改进）

### 1. Data Augmentation（数据增强）

#### Even with Fixed Dataset（即使数据集固定）

**Strategy A: Text Augmentation（文本增强）**

```python
# Augment report text（增强报告文本）
def augment_report_text(report_text):
    """
    Augment without changing meaning（增强但不改变含义）
    """
    # Synonym replacement（同义词替换）
    # "effusion" → "pleural effusion" (context-aware)（上下文感知）
    
    # Paraphrasing（改写）
    # "No effusion" → "Effusion is absent"
    
    # Add/remove punctuation（添加/删除标点）
    # "Left lung clear" → "Left lung, clear"
    
    return augmented_text
```

**Strategy B: Fact Augmentation（事实增强）**

```python
# Augment visual_facts and report_facts（增强视觉事实和报告事实）
def augment_facts(visual_facts, report_facts):
    """
    Create variations（创建变体）
    """
    # Add noise to measurements（给测量值添加噪声）
    # "size: 5cm" → "size: 4.8-5.2cm" (range)（范围）
    
    # Vary confidence scores（变化置信度）
    # "confidence: 0.95" → "confidence: 0.92" (slight variation)（轻微变化）
    
    return augmented_visual_facts, augmented_report_facts
```

**Benefits（好处）**：
- **More training data**：更多训练数据（通过增强）
- **Better generalization**：更好的泛化性能
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

### 2. Better Data Sampling（更好的数据采样）

#### Stratified Sampling（分层采样）

```python
from sklearn.model_selection import train_test_split

# Stratified split by issue type（按问题类型分层划分）
train_data, val_data = train_test_split(
    dataset,
    test_size=0.2,
    stratify=dataset['issue_type'],  # Maintain distribution（保持分布）
    random_state=42
)
```

**Benefits（好处）**：
- **Balanced training**：平衡的训练
- **Better validation**：更好的验证
- **Expected improvement**: +0.5-1% accuracy（预期提升：+0.5-1%）

#### Hard Example Mining（困难样本挖掘）

```python
# Focus on hard examples（关注困难样本）
def mine_hard_examples(model, dataset, top_k=1000):
    """
    Find examples with high loss（找到高损失的样本）
    """
    losses = []
    for example in dataset:
        loss = compute_loss(model, example)
        losses.append((loss, example))
    
    # Sort by loss, take top K（按损失排序，取前 K 个）
    hard_examples = sorted(losses, reverse=True)[:top_k]
    return hard_examples

# Use hard examples for training（使用困难样本训练）
hard_examples = mine_hard_examples(model, dataset)
trainer.train_dataset = hard_examples + regular_examples
```

**Benefits（好处）**：
- **Focus on difficult cases**：关注困难案例
- **Better learning**：更好的学习效果
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

### 3. Curriculum Learning（课程学习）

#### Train from Easy to Hard（从易到难训练）

```python
# Phase 1: Train on easy examples（阶段 1：训练简单样本）
easy_examples = filter_easy_examples(dataset)
trainer.train_dataset = easy_examples
trainer.train()  # Train for 2 epochs（训练 2 个 epoch）

# Phase 2: Add medium examples（阶段 2：添加中等样本）
medium_examples = filter_medium_examples(dataset)
trainer.train_dataset = easy_examples + medium_examples
trainer.train()  # Train for 2 epochs（训练 2 个 epoch）

# Phase 3: Add hard examples（阶段 3：添加困难样本）
hard_examples = filter_hard_examples(dataset)
trainer.train_dataset = easy_examples + medium_examples + hard_examples
trainer.train()  # Train for 1 epoch（训练 1 个 epoch）
```

**Benefits（好处）**：
- **Gradual learning**：渐进式学习
- **Better convergence**：更好的收敛
- **Expected improvement**: +1-2% accuracy（预期提升：+1-2%）

---

## Upstream Component Optimization（上游组件优化）

### 1. Prompt Optimization (Step 3.5)（Prompt 优化）

#### Even Dataset Fixed, Prompt Can Improve（即使数据集固定，Prompt 可以改进）

**Current Prompt（当前 Prompt）**：
```jinja
# prompt_v1.2.jinja
You are a radiology report quality checker.
Analyze the visual facts and report facts.
Identify issues: {{ issue_types }}
```

**Optimized Prompt（优化后的 Prompt）**：
```jinja
# prompt_v1.3.jinja
You are an expert radiology report quality checker.

Task: Compare visual facts with report facts and identify discrepancies.

Visual Facts:
{% for fact in visual_facts %}
- {{ fact.type }}: {{ fact.description }} (confidence: {{ fact.confidence }})
{% endfor %}

Report Facts:
{% for fact in report_facts %}
- {{ fact.entity }}: "{{ fact.text }}" (span: {{ fact.span_ref }})
{% endfor %}

Instructions:
1. Check for laterality mismatches（检查左右侧不匹配）
2. Check for omissions（检查遗漏）
3. Check for contradictions（检查矛盾）
4. Provide evidence references（提供证据引用）

Output format: JSON with issue_type, severity, supporting_facts
```

**Benefits（好处）**：
- **Better judge_labels quality**：更好的 judge_labels 质量
- **More consistent labels**：更一致的标签
- **Expected improvement**: +1-2% on Silver Agreement（预期提升：Silver Agreement +1-2%）

**How to Optimize（如何优化）**：
1. **Analyze judge_labels quality**：分析 judge_labels 质量
2. **Identify common errors**：识别常见错误
3. **Refine prompt instructions**：细化 prompt 指令
4. **Re-run Step 3.5**：重新运行 Step 3.5
5. **Re-train with new judge_labels**：用新的 judge_labels 重新训练

### 2. Rule Optimization (Step 3)（规则优化）

#### Improve Rule Labels Quality（改进规则标签质量）

**Current Rules（当前规则）**：
```python
# rules_v1.0.py
def check_laterality(visual_facts, report_facts):
    visual_laterality = extract_laterality(visual_facts)
    report_laterality = extract_laterality(report_facts)
    
    if visual_laterality != report_laterality:
        return "hard_fail", "Laterality mismatch"
    return "pass"
```

**Optimized Rules（优化后的规则）**：
```python
# rules_v1.1.py
def check_laterality(visual_facts, report_facts):
    visual_laterality = extract_laterality(visual_facts)
    report_laterality = extract_laterality(report_facts)
    
    # Add confidence threshold（添加置信度阈值）
    visual_confidence = get_laterality_confidence(visual_facts)
    
    if visual_confidence < 0.9:  # Low confidence, skip（低置信度，跳过）
        return "soft_flag", "Low confidence laterality"
    
    if visual_laterality != report_laterality:
        return "hard_fail", "Laterality mismatch"
    return "pass"
```

**Benefits（好处）**：
- **Better rule_labels quality**：更好的 rule_labels 质量
- **More accurate labels**：更准确的标签
- **Expected improvement**: +0.5-1% on Rule Adherence（预期提升：Rule Adherence +0.5-1%）

**How to Optimize（如何优化）**：
1. **Analyze rule_labels errors**：分析 rule_labels 错误
2. **Add confidence thresholds**：添加置信度阈值
3. **Refine rule logic**：细化规则逻辑
4. **Re-run Step 3**：重新运行 Step 3
5. **Re-train with new rule_labels**：用新的 rule_labels 重新训练

---

## Complete Optimization Workflow（完整优化工作流）

### Optimization Pipeline（优化 Pipeline）

```
Fixed Dataset（固定数据集）
    ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Upstream Optimization（阶段 1：上游优化）       │
│                                                          │
│ • Optimize Prompt (Step 3.5)                            │
│   - Analyze judge_labels quality                        │
│   - Refine prompt instructions                          │
│   - Re-run Step 3.5 → Better judge_labels              │
│                                                          │
│ • Optimize Rules (Step 3)                               │
│   - Analyze rule_labels errors                          │
│   - Add confidence thresholds                           │
│   - Re-run Step 3 → Better rule_labels                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Better Labels（更好的标签）
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Hyperparameter Tuning（阶段 2：超参数调优）    │
│                                                          │
│ • Learning Rate: 2e-4 → Try [1e-4, 2e-4, 5e-4]         │
│ • Batch Size: 4 → Try [4, 8, 16]                        │
│ • LoRA Rank: 16 → Try [16, 32, 64]                      │
│ • Epochs: 5 → Try [5, 10] with early stopping          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Better Config（更好的配置）
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Training Strategy（阶段 3：训练策略）          │
│                                                          │
│ • Learning Rate Scheduling: cosine annealing            │
│ • Warmup: 100 steps                                     │
│ • Mixed Precision: FP16/BF16                           │
│ • Gradient Checkpointing: Enable                       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Better Training（更好的训练）
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Loss Function（阶段 4：损失函数）              │
│                                                          │
│ • Weighted Loss: Handle imbalanced data                 │
│ • Focal Loss: Focus on hard examples                    │
│ • Multi-Task Loss: Combine objectives                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Better Loss（更好的损失）
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Data Utilization（阶段 5：数据利用）          │
│                                                          │
│ • Data Augmentation: Text + Facts                       │
│ • Stratified Sampling: Balanced splits                  │
│ • Hard Example Mining: Focus on difficult cases         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Improved Model（改进的模型）
```

### Step-by-Step Optimization Guide（逐步优化指南）

#### Step 1: Baseline（基线）

```python
# Baseline configuration（基线配置）
baseline_config = {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "lora_rank": 16,
    "epochs": 5,
    "lr_scheduler": "constant",
    "warmup_steps": 0
}

# Train and evaluate（训练并评估）
baseline_performance = train_and_evaluate(baseline_config)
# Result: Silver Agreement = 88.74%（结果：Silver Agreement = 88.74%）
```

#### Step 2: Hyperparameter Tuning（超参数调优）

```python
# Try different learning rates（尝试不同学习率）
for lr in [1e-4, 2e-4, 5e-4]:
    config = baseline_config.copy()
    config["learning_rate"] = lr
    performance = train_and_evaluate(config)
    # Record best（记录最佳）

# Try different batch sizes（尝试不同批次大小）
for batch_size in [4, 8, 16]:
    config = baseline_config.copy()
    config["batch_size"] = batch_size
    performance = train_and_evaluate(config)
    # Record best（记录最佳）

# Best config so far（目前最佳配置）
best_config = {
    "learning_rate": 2e-4,  # Best LR（最佳学习率）
    "batch_size": 8,  # Best batch size（最佳批次大小）
    "lora_rank": 16,
    "epochs": 5
}
# Expected: +0.5-1% improvement（预期：+0.5-1% 提升）
```

#### Step 3: Training Strategy（训练策略）

```python
# Add learning rate scheduling（添加学习率调度）
best_config["lr_scheduler"] = "cosine"
best_config["warmup_steps"] = 100

# Enable mixed precision（启用混合精度）
best_config["fp16"] = True

# Enable gradient checkpointing（启用梯度检查点）
best_config["gradient_checkpointing"] = True

performance = train_and_evaluate(best_config)
# Expected: +1-2% improvement（预期：+1-2% 提升）
```

#### Step 4: Loss Function（损失函数）

```python
# Use weighted loss（使用加权损失）
best_config["loss_function"] = "weighted_cross_entropy"
best_config["class_weights"] = calculate_class_weights(dataset)

performance = train_and_evaluate(best_config)
# Expected: +0.5-1% improvement（预期：+0.5-1% 提升）
```

#### Step 5: Upstream Optimization（上游优化）

```python
# Optimize prompt（优化 prompt）
optimize_prompt()  # prompt_v1.2 → prompt_v1.3
re_run_step_3_5()  # Generate better judge_labels（生成更好的 judge_labels）

# Optimize rules（优化规则）
optimize_rules()  # rules_v1.0 → rules_v1.1
re_run_step_3()  # Generate better rule_labels（生成更好的 rule_labels）

# Re-train with better labels（用更好的标签重新训练）
performance = train_and_evaluate(best_config)
# Expected: +1-2% improvement（预期：+1-2% 提升）
```

### Expected Improvements（预期改进）

| Optimization | Expected Improvement | Effort |
|-------------|---------------------|--------|
| **Hyperparameter Tuning** | +0.5-1% | Low（低） |
| **Training Strategy** | +1-2% | Medium（中） |
| **Loss Function** | +0.5-1% | Medium（中） |
| **LoRA Rank Increase** | +1-2% | Low（低） |
| **Prompt Optimization** | +1-2% | Medium（中） |
| **Rule Optimization** | +0.5-1% | Low（低） |
| **Data Augmentation** | +1-2% | High（高） |
| **Combined** | **+5-10%** | **High（高）** |

---

## Best Practices（最佳实践）

### 1. Optimization Order（优化顺序）

**Recommended Order（推荐顺序）**：

1. **Quick Wins（快速见效）**：
   - Learning rate scheduling（学习率调度）
   - Increase batch size（增加批次大小）
   - Enable mixed precision（启用混合精度）
   - **Expected**: +1-2% improvement（预期：+1-2%）

2. **Medium Effort（中等努力）**：
   - LoRA rank tuning（LoRA 秩调优）
   - Prompt optimization（Prompt 优化）
   - Loss function tuning（损失函数调优）
   - **Expected**: +2-3% improvement（预期：+2-3%）

3. **High Effort（高努力）**：
   - Data augmentation（数据增强）
   - Hard example mining（困难样本挖掘）
   - Curriculum learning（课程学习）
   - **Expected**: +2-3% improvement（预期：+2-3%）

### 2. Evaluation Strategy（评估策略）

```python
# Always evaluate on golden set（总是在 Golden Set 上评估）
def evaluate_improvements(new_config, baseline_config):
    """
    Compare new config with baseline（比较新配置和基线）
    """
    baseline_perf = evaluate(baseline_config, golden_set)
    new_perf = evaluate(new_config, golden_set)
    
    improvements = {
        "rule_adherence": new_perf["rule_adherence"] - baseline_perf["rule_adherence"],
        "silver_agreement": new_perf["silver_agreement"] - baseline_perf["silver_agreement"],
        "judge_rule_gap": new_perf["judge_rule_gap"] - baseline_perf["judge_rule_gap"]
    }
    
    # All metrics must improve or maintain（所有指标必须改进或保持）
    if improvements["rule_adherence"] < -0.01:  # Regression（回归）
        return False, "Rule Adherence regressed"
    
    if improvements["silver_agreement"] < 0:  # No improvement（无改进）
        return False, "Silver Agreement didn't improve"
    
    return True, improvements
```

### 3. Iterative Optimization（迭代优化）

```
Iteration 1: Quick Wins（迭代 1：快速见效）
  - Learning rate scheduling
  - Increase batch size
  - Result: 88.74% → 90.0% (+1.26%)

Iteration 2: Medium Effort（迭代 2：中等努力）
  - LoRA rank: 16 → 32
  - Prompt optimization
  - Result: 90.0% → 91.5% (+1.5%)

Iteration 3: Fine-tuning（迭代 3：精细调优）
  - Loss function tuning
  - Data augmentation
  - Result: 91.5% → 92.5% (+1.0%)

Total Improvement: 88.74% → 92.5% (+3.76%)
```

---

## Summary（总结）

### Key Takeaways（关键要点）

1. ✅ **Dataset Fixed ≠ No Optimization**（数据集固定 ≠ 无法优化）
   - Many optimization opportunities without changing data（不改变数据也有很多优化机会）

2. ✅ **Optimization Dimensions（优化维度）**：
   - Hyperparameters（超参数）
   - Training strategies（训练策略）
   - Model architecture（模型架构）
   - Loss functions（损失函数）
   - Data utilization（数据利用）
   - Upstream components（上游组件）

3. ✅ **Expected Total Improvement（预期总改进）**：
   - **+5-10% accuracy** with comprehensive optimization（全面优化可提升 +5-10%）

4. ✅ **Optimization Order（优化顺序）**：
   - Start with quick wins（从快速见效开始）
   - Then medium effort（然后中等努力）
   - Finally high effort（最后高努力）

### Quick Reference（快速参考）

| Optimization | Expected Gain | Effort | Priority |
|-------------|--------------|--------|----------|
| Learning Rate Scheduling | +1-2% | Low | High |
| Increase Batch Size | +0.5-1% | Low | High |
| LoRA Rank Increase | +1-2% | Low | Medium |
| Prompt Optimization | +1-2% | Medium | High |
| Loss Function Tuning | +0.5-1% | Medium | Medium |
| Data Augmentation | +1-2% | High | Low |

---

**Remember**: Even with fixed dataset, there are many ways to improve fine-tuning performance. Start with quick wins, then iterate.
