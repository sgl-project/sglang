# A06_B01: LoRA Fine-Tuning Deep Dive
# LoRA Fine-Tuning 深度教学：如何把控性能提升

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Comprehensive guide to LoRA Fine-Tuning details and performance control  
**Target Audience**：Developers who want to master LoRA fine-tuning for rad-linter

---

## 📋 Table of Contents

1. [What is LoRA? (LoRA 基础原理)](#what-is-lora)
2. [LoRA Configuration Deep Dive (配置详解)](#lora-configuration-deep-dive)
3. [Training Process Step-by-Step (训练流程详解)](#training-process-step-by-step)
4. [Performance Monitoring (性能监控)](#performance-monitoring)
5. [How to Improve Performance (如何提升性能)](#how-to-improve-performance)
6. [Complete Training Code (完整训练代码)](#complete-training-code)
7. [Best Practices & Troubleshooting (最佳实践与问题排查)](#best-practices--troubleshooting)

---

## What is LoRA? (LoRA 基础原理)

### Core Concept (核心概念)

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that:
- **Freezes** the base model weights (doesn't update them)
- **Adds** small trainable adapters (LoRA matrices) to specific layers
- **Only trains** ~0.1-1% of total parameters (vs 100% in full fine-tuning)

**中文解释**：LoRA 是一种参数高效微调方法，冻结基础模型权重，只在特定层添加小的可训练适配器，只训练总参数的 0.1-1%。

### Why LoRA? (为什么使用 LoRA？)

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Fine-Tuning                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Base Model (7B parameters)                              │ │
│ │ • Train ALL 7B parameters                                │ │
│ │ • Memory: ~28GB GPU (FP32)                               │ │
│ │ • Time: ~24 hours                                        │ │
│ │ • Storage: 14GB per checkpoint                           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│                    LoRA Fine-Tuning                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Base Model (7B parameters) - FROZEN                    │ │
│ │ + LoRA Adapters (~10M parameters) - TRAINABLE          │ │
│ │ • Train ONLY ~10M parameters (0.14% of total)          │ │
│ │ • Memory: ~12GB GPU (4-bit quantized)                   │ │
│ │ • Time: ~4 hours                                        │ │
│ │ • Storage: 500MB per checkpoint                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ Benefits:                                                    │
│ ✅ 6x faster training                                       │
│ ✅ 2.3x less GPU memory                                    │
│ ✅ 28x smaller checkpoints                                 │
│ ✅ Can train on consumer GPUs (RTX 4090)                    │
└─────────────────────────────────────────────────────────────┘
```

### LoRA Mathematical Foundation (LoRA 数学原理)

**Original Layer**:
```
W₀ (pretrained weights, frozen)
```

**LoRA Adaptation**:
```
W = W₀ + ΔW
  = W₀ + BA

Where:
- B: rank r × d matrix (trainable)
- A: d × rank r matrix (trainable)
- r << d (rank << dimension)
- Only BA is trained, W₀ is frozen
```

**Key Insight**:
- **Rank r** controls adapter capacity (r=16 means 16-dimensional adaptation space)
- **Lower r** = fewer parameters, faster training, but less capacity
- **Higher r** = more parameters, slower training, but more capacity

---

## LoRA Configuration Deep Dive (配置详解)

### Complete Configuration (完整配置)

```python
from peft import LoraConfig

lora_config = LoraConfig(
    # ========== Rank & Capacity ==========
    r=16,                    # LoRA rank (秩)
    # What it does: Controls adapter capacity
    # Lower r (8):  Fewer params, faster, less capacity
    # Higher r (32): More params, slower, more capacity
    # Sweet spot: 16 for rad-linter task
    
    lora_alpha=32,           # LoRA alpha (缩放因子)
    # What it does: Scales LoRA weights
    # Formula: ΔW = (alpha/r) * BA
    # Common ratio: alpha = 2*r (alpha/r = 2)
    # Higher alpha/r = stronger adaptation
    
    # ========== Target Modules ==========
    target_modules=[
        "q_proj",            # Query projection (注意力查询)
        "v_proj",            # Value projection (注意力值)
        # Optional additions:
        # "k_proj",          # Key projection (注意力键)
        # "o_proj",          # Output projection (输出投影)
        # "gate_proj",        # Gate projection (MoE gates)
        # "up_proj",          # Up projection (MLP)
        # "down_proj",        # Down projection (MLP)
    ],
    # Strategy:
    # - Start with q_proj + v_proj (minimal, fast)
    # - Add k_proj + o_proj if need more capacity
    # - Add MLP layers only if task is very different
    
    # ========== Regularization ==========
    lora_dropout=0.1,        # Dropout rate
    # What it does: Prevents overfitting
    # Range: 0.0 - 0.5
    # Lower (0.05): Less regularization, risk overfitting
    # Higher (0.2): More regularization, risk underfitting
    # Sweet spot: 0.1 for rad-linter
    
    # ========== Task Type ==========
    task_type="CAUSAL_LM",   # Task type
    # Options: CAUSAL_LM, SEQ_2_SEQ_LM, TOKEN_CLS, etc.
    # Rad-linter uses CAUSAL_LM (generative task)
    
    # ========== Advanced Options ==========
    bias="none",             # Bias training
    # Options: "none", "all", "lora_only"
    # "none": Don't train bias (recommended)
    # "all": Train all biases (rarely needed)
    # "lora_only": Train only LoRA biases
    
    inference_mode=False,    # Inference mode
    # False: Training mode (enable dropout, etc.)
    # True: Inference mode (disable dropout, etc.)
)
```

### Configuration Impact Analysis (配置影响分析)

#### 1. Rank (r) Impact

```
┌─────────────────────────────────────────────────────────────┐
│ Rank (r) Comparison                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ r=8:                                                         │
│ • Parameters: ~5M                                           │
│ • Training Speed: Fastest                                    │
│ • Capacity: Low (may underfit complex tasks)                │
│ • Use Case: Simple tasks, limited GPU                       │
│                                                              │
│ r=16: ⭐ RECOMMENDED FOR RAD-LINTER                         │
│ • Parameters: ~10M                                           │
│ • Training Speed: Fast                                      │
│ • Capacity: Balanced (good for most tasks)                   │
│ • Use Case: Rad-linter (balanced complexity)                │
│                                                              │
│ r=32:                                                        │
│ • Parameters: ~20M                                          │
│ • Training Speed: Moderate                                  │
│ • Capacity: High (may overfit simple tasks)                │
│ • Use Case: Complex tasks, abundant GPU                     │
│                                                              │
│ r=64:                                                        │
│ • Parameters: ~40M                                           │
│ • Training Speed: Slow                                       │
│ • Capacity: Very High (often unnecessary)                    │
│ • Use Case: Very complex tasks, research                    │
└─────────────────────────────────────────────────────────────┘
```

**How to Choose Rank**:
1. **Start with r=16** (sweet spot for most tasks)
2. **If underfitting** (low training accuracy): Increase to r=32
3. **If overfitting** (high train, low val): Decrease to r=8
4. **Monitor validation loss** to find optimal r

#### 2. Alpha (lora_alpha) Impact

```
┌─────────────────────────────────────────────────────────────┐
│ Alpha/Ratio Impact                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ alpha/r = 1:                                                 │
│ • Weak adaptation                                            │
│ • May not learn task-specific patterns                      │
│                                                              │
│ alpha/r = 2: ⭐ RECOMMENDED                                 │
│ • Balanced adaptation                                        │
│ • Good for most tasks                                        │
│                                                              │
│ alpha/r = 4:                                                 │
│ • Strong adaptation                                          │
│ • May cause instability                                      │
│                                                              │
│ Rule of Thumb: alpha = 2*r                                   │
│ - r=16 → alpha=32                                            │
│ - r=32 → alpha=64                                            │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Target Modules Impact

```
┌─────────────────────────────────────────────────────────────┐
│ Target Modules Strategy                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Minimal (q_proj + v_proj):                                  │
│ • Fastest training                                           │
│ • Lowest memory                                              │
│ • Good for: Similar tasks to base model                     │
│                                                              │
│ Standard (q_proj + v_proj + k_proj + o_proj):              │
│ • Balanced                                                   │
│ • Good for: Moderate task differences                        │
│                                                              │
│ Full Attention (all attention layers):                       │
│ • More capacity                                              │
│ • Good for: Significant task differences                     │
│                                                              │
│ Full Model (attention + MLP):                               │
│ • Maximum capacity                                           │
│ • Good for: Very different tasks                            │
│ • Warning: May overfit, slower training                    │
└─────────────────────────────────────────────────────────────┘
```

**Rad-Linter Strategy**:
- **Start**: q_proj + v_proj (fast, efficient)
- **If underfitting**: Add k_proj + o_proj
- **Only if needed**: Add MLP layers (rarely necessary)

---

## Training Process Step-by-Step (训练流程详解)

### Step 1: Load Base Model (加载基础模型)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization (saves memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit quantization
    bnb_4bit_quant_type="nf4",             # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype
    bnb_4bit_use_double_quant=True,        # Double quantization (saves more memory)
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",         # Base model
    quantization_config=bnb_config,       # 4-bit quantization
    device_map="auto",                     # Auto device mapping
    trust_remote_code=True,                # Trust remote code
    torch_dtype=torch.float16,            # Model dtype
)

# Freeze base model (important!)
for param in base_model.parameters():
    param.requires_grad = False

print(f"Base model loaded: {base_model.num_parameters():,} parameters (frozen)")
```

**Key Points**:
- **4-bit quantization**: Reduces memory from 28GB → 12GB
- **Freeze base model**: Only LoRA adapters will be trained
- **Device map auto**: Automatically handles multi-GPU

### Step 2: Configure LoRA (配置 LoRA)

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for LoRA training
base_model = prepare_model_for_kbit_training(base_model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"Total params: {total_params:,}")

# Expected output:
# Trainable params: 10,485,760 (0.14%)
# Total params: 7,000,000,000
```

**Key Points**:
- **prepare_model_for_kbit_training**: Prepares quantized model for training
- **get_peft_model**: Adds LoRA adapters to model
- **Only 0.14% trainable**: Massive memory savings!

### Step 3: Prepare Training Data (准备训练数据)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    trust_remote_code=True,
    padding_side="right",
)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load training data
train_data = load_dataset("json", data_files="s3://rad-linter-data/labels/judge_labels_v1.0.jsonl")

# Format: Each example has:
# {
#   "input": "visual_facts: {...}, report_facts: {...}",
#   "output": "lint_result: {...}"
# }

def format_prompt(example):
    """Format input-output pair as prompt"""
    prompt = f"""You are a radiology report quality checker. Analyze the following case and identify any issues.

Visual Facts:
{example['visual_facts']}

Report Facts:
{example['report_facts']}

Identify issues and provide recommendations:"""
    
    return {
        "text": prompt + "\n" + example['lint_result']
    }

# Format dataset
train_dataset = train_data["train"].map(format_prompt)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,          # Max sequence length
        padding="max_length",      # Pad to max_length
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

# Split into train/val (80/20)
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
```

**Key Points**:
- **Prompt formatting**: Structure input-output pairs
- **Tokenization**: Convert text to token IDs
- **Train/val split**: 80/20 split for validation

### Step 4: Configure Training Arguments (配置训练参数)

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # ========== Output & Logging ==========
    output_dir="./models/lora_model_v1.0",
    logging_dir="./logs",
    logging_steps=50,                    # Log every 50 steps
    save_strategy="steps",                # Save by steps
    save_steps=500,                       # Save every 500 steps
    evaluation_strategy="steps",          # Evaluate by steps
    eval_steps=500,                       # Evaluate every 500 steps
    
    # ========== Training Hyperparameters ==========
    learning_rate=2e-4,                   # Learning rate
    per_device_train_batch_size=4,        # Batch size per GPU
    per_device_eval_batch_size=4,         # Eval batch size
    gradient_accumulation_steps=4,         # Gradient accumulation (effective batch = 4 * 4 = 16)
    num_train_epochs=5,                    # Number of epochs
    max_steps=-1,                         # Max steps (-1 = use epochs)
    
    # ========== Learning Rate Schedule ==========
    lr_scheduler_type="cosine",           # Cosine annealing
    warmup_ratio=0.1,                     # 10% warmup steps
    warmup_steps=100,                     # Warmup steps
    
    # ========== Optimization ==========
    optim="paged_adamw_32bit",            # Optimizer (memory efficient)
    weight_decay=0.01,                    # Weight decay (L2 regularization)
    max_grad_norm=1.0,                     # Gradient clipping
    
    # ========== Mixed Precision ==========
    fp16=True,                             # Use FP16 (faster, less memory)
    # bf16=True,                            # Or use BF16 (better for some models)
    
    # ========== Reproducibility ==========
    seed=42,                               # Random seed
    data_seed=42,                          # Data seed
    
    # ========== Other ==========
    load_best_model_at_end=True,          # Load best model at end
    metric_for_best_model="eval_loss",     # Metric for best model
    greater_is_better=False,               # Lower is better for loss
    report_to="tensorboard",               # Logging backend
    remove_unused_columns=False,           # Keep all columns
)
```

**Key Hyperparameters Explained**:

1. **Learning Rate (2e-4)**:
   - **Too high (>5e-4)**: Training unstable, loss spikes
   - **Too low (<1e-5)**: Slow convergence, may not learn
   - **Sweet spot**: 1e-4 to 3e-4 for LoRA

2. **Batch Size (4 per GPU)**:
   - **Effective batch size** = per_device_batch_size × gradient_accumulation_steps × num_gpus
   - **Example**: 4 × 4 × 1 = 16 effective batch size
   - **Larger batch**: More stable gradients, but needs more memory
   - **Smaller batch**: Less memory, but noisier gradients

3. **Gradient Accumulation (4)**:
   - **Purpose**: Simulate larger batch size without more memory
   - **How it works**: Accumulate gradients over 4 steps before updating
   - **Effective batch**: 4 × 4 = 16

4. **Learning Rate Schedule (cosine)**:
   - **Cosine**: Smooth decay from lr to 0
   - **Linear**: Linear decay
   - **Constant**: No decay (not recommended)

### Step 5: Train Model (训练模型)

```python
from transformers import Trainer, DataCollatorForLanguageModeling

# Data collator (handles padding)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not masked language modeling (causal LM)
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("./models/lora_model_v1.0/final")
tokenizer.save_pretrained("./models/lora_model_v1.0/final")

print("Training complete!")
```

---

## Performance Monitoring (性能监控)

### Key Metrics to Monitor (关键监控指标)

#### 1. Training Loss (训练损失)

```python
# Expected training loss curve:
# Epoch 1: Loss starts high (~2.5), decreases rapidly
# Epoch 2-3: Loss decreases steadily (~1.5 → 1.0)
# Epoch 4-5: Loss plateaus (~0.8 → 0.7)

# Good signs:
# ✅ Loss decreases smoothly
# ✅ No sudden spikes
# ✅ Validation loss follows training loss

# Bad signs:
# ❌ Loss spikes (learning rate too high)
# ❌ Loss doesn't decrease (learning rate too low)
# ❌ Validation loss increases while training decreases (overfitting)
```

#### 2. Validation Loss (验证损失)

```python
# Validation loss should:
# - Follow training loss (slightly higher)
# - Decrease over epochs
# - Not diverge from training loss

# Overfitting detection:
# Training loss: 0.5 (decreasing)
# Validation loss: 1.2 (increasing)  # ❌ Overfitting!

# Good training:
# Training loss: 0.7 (decreasing)
# Validation loss: 0.9 (decreasing)  # ✅ Good!
```

#### 3. Learning Rate Schedule (学习率调度)

```python
# Monitor learning rate over training:
# Step 0-100:    2e-4 → 2e-4 (warmup)
# Step 100-2500:  2e-4 → 1e-4 (cosine decay)
# Step 2500-5000: 1e-4 → 0 (cosine decay)

# Expected: Smooth decay, no sudden changes
```

#### 4. Gradient Norm (梯度范数)

```python
# Monitor gradient norm:
# Good: 0.1 - 1.0 (stable training)
# Warning: > 5.0 (gradient explosion, reduce learning rate)
# Warning: < 0.01 (vanishing gradients, increase learning rate)

# Gradient clipping (max_grad_norm=1.0) prevents explosion
```

### Monitoring Dashboard (监控面板)

```python
# TensorBoard logs are saved to ./logs
# View with: tensorboard --logdir ./logs

# Key metrics to watch:
# 1. train/loss: Training loss
# 2. eval/loss: Validation loss
# 3. train/learning_rate: Learning rate
# 4. train/grad_norm: Gradient norm
# 5. train/epoch: Current epoch
```

### Performance Metrics (性能指标)

After training, evaluate on test set:

```python
# Evaluation metrics for rad-linter:
metrics = {
    "rule_adherence": 1.0,        # Model vs Rule labels (target: 100%)
    "silver_agreement": 0.8874,   # Model vs Judge labels (target: >85%)
    "judge_rule_gap": 0.8874,     # Judge vs Rule agreement (target: >85%)
    "precision": 0.85,            # Precision (target: >80%)
    "recall": 0.82,               # Recall (target: >80%)
    "f1_score": 0.80,             # F1 score (target: >80%)
}

# Rule Adherence: 100% means model learned rule patterns perfectly
# Silver Agreement: Measures model quality vs high-quality Judge labels
# Judge-Rule Gap: Reveals rule limitations
```

---

## How to Improve Performance (如何提升性能)

### Strategy 1: Hyperparameter Tuning (超参数调优)

#### Learning Rate Tuning

```python
# Strategy: Learning rate finder
# 1. Start with wide range: [1e-5, 1e-3]
# 2. Train for 100 steps with each LR
# 3. Plot loss vs learning rate
# 4. Choose LR at steepest descent point

learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

# Expected curve:
# LR too low (<1e-5): Loss decreases slowly
# LR optimal (1e-4 to 3e-4): Loss decreases fastest
# LR too high (>5e-4): Loss spikes, unstable

# For rad-linter: Start with 2e-4, adjust based on loss curve
```

#### Batch Size Tuning

```python
# Strategy: Increase batch size if:
# 1. Training is unstable (loss spikes)
# 2. Validation loss is noisy
# 3. GPU memory allows

# Current: batch_size=4, gradient_accumulation=4 (effective=16)
# Try: batch_size=8, gradient_accumulation=2 (effective=16, same but faster)

# Larger batch benefits:
# - More stable gradients
# - Better convergence
# - Faster training (fewer gradient accumulation steps)
```

#### Rank Tuning

```python
# Strategy: Increase rank if underfitting, decrease if overfitting

# Underfitting signs:
# - Training loss plateaus high (>1.0)
# - Validation loss high
# - Low accuracy on training set
# Solution: Increase r from 16 → 32

# Overfitting signs:
# - Training loss low (<0.5) but validation loss high (>1.0)
# - High accuracy on train, low on val
# Solution: Decrease r from 16 → 8, or increase dropout
```

### Strategy 2: Training Strategy Improvements (训练策略改进)

#### Learning Rate Scheduling

```python
# Current: Cosine annealing
# Alternative: Linear warmup + cosine decay

training_args = TrainingArguments(
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,        # 10% warmup
    # warmup_steps=100,     # Or specify steps directly
)

# Warmup benefits:
# - Prevents early training instability
# - Allows model to adapt gradually
# - Common: 5-10% of total steps
```

#### Gradient Accumulation

```python
# Increase gradient accumulation for larger effective batch:
training_args = TrainingArguments(
    per_device_train_batch_size=2,      # Smaller per-device batch
    gradient_accumulation_steps=8,       # More accumulation
    # Effective batch: 2 * 8 = 16 (same as before, but uses less memory)
)

# Benefits:
# - Larger effective batch size
# - More stable training
# - Less GPU memory usage
```

#### Mixed Precision Training

```python
# Use FP16 for faster training:
training_args = TrainingArguments(
    fp16=True,              # FP16 (faster, less memory)
    # bf16=True,            # Or BF16 (better for some models)
)

# Benefits:
# - 2x faster training
# - 50% less memory
# - Minimal accuracy loss
```

### Strategy 3: Data Quality Improvements (数据质量改进)

#### Data Filtering

```python
# Filter low-quality examples:
def filter_high_quality(examples):
    """Keep only high-confidence labels"""
    return [
        ex for ex in examples
        if ex['judge_confidence'] > 0.8  # High confidence threshold
    ]

# Benefits:
# - Better training signal
# - Faster convergence
# - Higher final accuracy
```

#### Data Augmentation

```python
# Augment training data:
# 1. Paraphrase report facts (keep meaning, change wording)
# 2. Add noise to visual facts (small confidence variations)
# 3. Mix rule_labels and judge_labels

# Benefits:
# - More training examples
# - Better generalization
# - Reduced overfitting
```

### Strategy 4: Model Architecture Adjustments (模型架构调整)

#### Add More Target Modules

```python
# Start minimal:
target_modules = ["q_proj", "v_proj"]

# If underfitting, add more:
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Benefits:
# - More capacity
# - Better task adaptation
# - But: Slower training, more memory
```

#### Increase Rank

```python
# Start with r=16:
lora_config = LoraConfig(r=16, ...)

# If underfitting, increase:
lora_config = LoraConfig(r=32, ...)

# Benefits:
# - More parameters
# - More capacity
# - But: Slower training
```

### Strategy 5: Loss Function Optimization (损失函数优化)

#### Weighted Loss

```python
# Weight loss by severity:
def compute_weighted_loss(predictions, labels, severity_weights):
    """
    severity_weights: {
        "high": 3.0,    # 3x weight for high severity
        "med": 1.5,     # 1.5x weight for medium severity
        "low": 1.0,     # 1x weight for low severity
    }
    """
    base_loss = cross_entropy_loss(predictions, labels)
    weights = [severity_weights[label['severity']] for label in labels]
    weighted_loss = base_loss * weights
    return weighted_loss.mean()

# Benefits:
# - Focus on high-severity errors
# - Better safety (critical for rad-linter)
```

#### Focal Loss (for imbalanced data)

```python
# Use focal loss if classes are imbalanced:
def focal_loss(predictions, labels, alpha=0.25, gamma=2.0):
    """
    Focal loss focuses on hard examples
    """
    ce_loss = cross_entropy_loss(predictions, labels)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Benefits:
# - Better handling of imbalanced classes
# - Focus on hard examples
```

---

## Complete Training Code (完整训练代码)

### Full Training Script

```python
#!/usr/bin/env python3
"""
Rad-Linter LoRA Fine-Tuning Script
Complete training pipeline with monitoring and best practices
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb  # Optional: for experiment tracking

# ========== Configuration ==========
CONFIG = {
    "base_model": "Qwen/Qwen2-VL-7B-Instruct",
    "train_data_path": "s3://rad-linter-data/labels/judge_labels_v1.0.jsonl",
    "output_dir": "./models/lora_model_v1.0",
    "logging_dir": "./logs",
    
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],
    
    # Training config
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "max_length": 2048,
    "seed": 42,
}

# ========== Step 1: Load Base Model ==========
print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Freeze base model
for param in base_model.parameters():
    param.requires_grad = False

# ========== Step 2: Configure LoRA ==========
print("Configuring LoRA...")
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"Total params: {total_params:,}")

# ========== Step 3: Load Tokenizer ==========
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["base_model"],
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== Step 4: Prepare Data ==========
print("Loading training data...")
train_data = load_dataset("json", data_files=CONFIG["train_data_path"])

def format_prompt(example):
    prompt = f"""You are a radiology report quality checker. Analyze the following case and identify any issues.

Visual Facts:
{example['visual_facts']}

Report Facts:
{example['report_facts']}

Identify issues and provide recommendations:"""
    return {"text": prompt + "\n" + example['lint_result']}

train_dataset = train_data["train"].map(format_prompt)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

# Split train/val
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=CONFIG["seed"])
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# ========== Step 5: Configure Training ==========
print("Configuring training...")
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    logging_dir=CONFIG["logging_dir"],
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    num_train_epochs=CONFIG["num_epochs"],
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG["warmup_ratio"],
    optim="paged_adamw_32bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
)

# ========== Step 6: Train ==========
print("Starting training...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save final model
print("Saving final model...")
trainer.save_model(os.path.join(CONFIG["output_dir"], "final"))
tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "final"))

print("Training complete!")
print(f"Model saved to: {CONFIG['output_dir']}/final")
```

---

## Best Practices & Troubleshooting (最佳实践与问题排查)

### Best Practices (最佳实践)

1. **Start Simple**:
   - Begin with r=16, q_proj+v_proj only
   - Use default learning rate (2e-4)
   - Monitor training, adjust as needed

2. **Monitor Closely**:
   - Watch training/validation loss curves
   - Check gradient norms
   - Monitor learning rate schedule

3. **Save Checkpoints**:
   - Save every 500 steps
   - Keep best model based on validation loss
   - Version all checkpoints

4. **Reproducibility**:
   - Fix random seeds (seed=42)
   - Version data, config, code
   - Document all hyperparameters

5. **Iterate Gradually**:
   - Change one hyperparameter at a time
   - Compare results systematically
   - Keep experiment logs

### Common Issues & Solutions (常见问题与解决方案)

#### Issue 1: Loss Not Decreasing

**Symptoms**:
- Training loss stays high (>2.0)
- No improvement over epochs

**Solutions**:
1. **Increase learning rate**: Try 3e-4 or 5e-4
2. **Increase rank**: Try r=32
3. **Add more target modules**: Add k_proj, o_proj
4. **Check data quality**: Ensure labels are correct

#### Issue 2: Overfitting

**Symptoms**:
- Training loss low (<0.5)
- Validation loss high (>1.0)
- Large gap between train/val

**Solutions**:
1. **Increase dropout**: Try lora_dropout=0.2
2. **Decrease rank**: Try r=8
3. **Add regularization**: Increase weight_decay
4. **Early stopping**: Stop when val loss stops improving

#### Issue 3: Training Unstable (Loss Spikes)

**Symptoms**:
- Loss spikes randomly
- Gradient norm > 5.0

**Solutions**:
1. **Decrease learning rate**: Try 1e-4
2. **Increase gradient clipping**: max_grad_norm=0.5
3. **Increase batch size**: More stable gradients
4. **Add warmup**: More warmup steps

#### Issue 4: Out of Memory (OOM)

**Symptoms**:
- CUDA out of memory error

**Solutions**:
1. **Decrease batch size**: Try batch_size=2
2. **Increase gradient accumulation**: Compensate with more steps
3. **Use 4-bit quantization**: Already using, check config
4. **Reduce max_length**: Try 1024 instead of 2048

#### Issue 5: Slow Training

**Symptoms**:
- Training takes too long

**Solutions**:
1. **Use FP16**: Already using, check if enabled
2. **Increase batch size**: Fewer gradient accumulation steps
3. **Reduce max_length**: Shorter sequences = faster
4. **Use more GPUs**: Multi-GPU training

### Performance Checklist (性能检查清单)

Before training:
- [ ] Base model loaded correctly (frozen)
- [ ] LoRA config appropriate (r=16, alpha=32)
- [ ] Data formatted correctly
- [ ] Train/val split done (80/20)
- [ ] Hyperparameters set (lr=2e-4, batch=4, epochs=5)

During training:
- [ ] Loss decreases smoothly (no spikes)
- [ ] Validation loss follows training loss
- [ ] Gradient norm stable (0.1-1.0)
- [ ] Learning rate decays correctly
- [ ] Checkpoints saved regularly

After training:
- [ ] Best model loaded (lowest val loss)
- [ ] Model saved to output directory
- [ ] Evaluation metrics computed
- [ ] Performance meets targets:
  - [ ] Rule Adherence > 95%
  - [ ] Silver Agreement > 85%
  - [ ] F1 Score > 80%

---

## Summary (总结)

### Key Takeaways (关键要点)

1. **LoRA is Parameter-Efficient**:
   - Only trains 0.1-1% of parameters
   - 6x faster, 2.3x less memory
   - 28x smaller checkpoints

2. **Configuration Matters**:
   - Start with r=16, alpha=32 (sweet spot)
   - Use q_proj+v_proj (minimal, fast)
   - Adjust based on underfitting/overfitting

3. **Monitor Closely**:
   - Watch training/validation loss
   - Check gradient norms
   - Monitor learning rate schedule

4. **Iterate Systematically**:
   - Change one thing at a time
   - Compare results
   - Document everything

5. **Performance Targets**:
   - Rule Adherence: >95%
   - Silver Agreement: >85%
   - F1 Score: >80%

### Next Steps (下一步)

1. **Run initial training** with default config
2. **Monitor metrics** closely
3. **Adjust hyperparameters** based on results
4. **Evaluate on test set**
5. **Iterate** to improve performance

---

**Happy Training! 🚀**
