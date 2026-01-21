# A06: Fine-Tuning vs Post-Training
# A06: 模型微调 vs 训练后持续改进

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Clarify the Difference Between Fine-Tuning and Post-Training  
**Key Question**: Are Fine-Tuning and Post-Training the Same?

---

## 📋 Table of Contents

1. [Quick Answer](#quick-answer)
2. [Fine-Tuning (Initial Training)](#fine-tuning-initial-training)
3. [Post-Training (Continuous Improvement)](#post-training-continuous-improvement)
4. [Key Differences Comparison](#key-differences-comparison)
5. [When to Use Each](#when-to-use-each)
6. [Complete Lifecycle](#complete-lifecycle)
7. [Best Practices](#best-practices)

---

## Quick Answer

### Are They the Same?（它们一样吗？）

**No, they are different stages in the model lifecycle（不，它们是模型生命周期中的不同阶段）:**

| Aspect | Fine-Tuning | Post-Training |
|--------|------------|---------------|
| **Stage** | Initial training (Step 4) | Production improvement |
| **Timing** | Before deployment | After deployment |
| **Data Source** | Labeled training data | Production feedback |
| **Purpose** | Learn task-specific patterns | Improve based on real usage |
| **Frequency** | One-time (per version) | Continuous (iterative) |
| **Base Model** | Pre-trained base model | Previously fine-tuned model |

### Simple Analogy（简单类比）

- **Fine-Tuning**: 学习新技能（从通用模型到任务专用模型）
  - 就像从"会说话"到"会写医学报告"的学习过程
  - 使用高质量的标注数据（rule_labels + judge_labels）
  
- **Post-Training**: 在工作中持续改进（基于实际使用反馈优化）
  - 就像医生在实际工作中不断改进诊断技能
  - 使用生产环境的真实反馈数据（Accept/Ignore/Review）

---

## Fine-Tuning (Initial Training)（初始训练）

### Definition（定义）

**Fine-tuning（微调）** is the initial training phase where a pre-trained base model is adapted to a specific task using labeled training data.

**中文解释**：Fine-tuning 是初始训练阶段，使用标注好的训练数据，将预训练的基础模型适配到特定任务（rad-linter 医学报告质量检查）。

### In Rad-Linter Pipeline（在 Rad-Linter Pipeline 中的位置）

**Location**: Step 4 of Training Pipeline（位置：训练 Pipeline 的第 4 步）

```
Step 0-1: Data Preprocessing
    ↓
Step 2: Visual Feature Extraction
    ↓
Step 3: Rule-Based Label Generation
    ↓
Step 3.5: LLM Judge Label Generation
    ↓
Step 4: LoRA Fine-Tuning ⭐ (Initial Training)
    ↓
Step 5: Evaluation
```

### Fine-Tuning Process（Fine-Tuning 流程）

#### 1. Input Data（输入数据）

```yaml
Fine-Tuning Data:
  Source: Step 3 + Step 3.5 outputs  # 数据来源：Step 3 和 Step 3.5 的输出
  - rule_labels_v1.0.jsonl (Rule-based labels)  # 规则基础标签
  - judge_labels_v1.0.jsonl (LLM Judge labels)  # LLM Judge 标签
  
  Format:
    Input: visual_facts + report_facts  # 输入：视觉事实 + 报告事实
    Output: lint_result (issues, severity, recommended_action)  # 输出：检查结果
  
  Size: 10,000 cases (training set)  # 数据量：10,000 个案例（训练集）
  Quality: High-quality labeled data  # 数据质量：高质量的标注数据
```

**关键点**：
- 数据来源是 Pipeline 前几步生成的标签（rule_labels + judge_labels）
- 数据质量高，因为经过了规则引擎和 LLM Judge 的双重验证
- 数据量大（10,000 cases），覆盖了各种场景

#### 2. Training Configuration（训练配置）

```yaml
Fine-Tuning Config (Step 4):
  Base Model: qwen2.5-vl-7b (4-bit quantized, frozen)  # 基础模型：qwen2.5-vl-7b（4-bit 量化，冻结）
  Method: LoRA (Low-Rank Adaptation)  # 方法：LoRA（低秩适配，参数高效微调）
  LoRA Rank: 16  # LoRA 秩：16（控制适配器容量）
  Learning Rate: 2e-4 (initial training rate)  # 学习率：2e-4（初始训练的学习率）
  Batch Size: 4 (per GPU)  # 批次大小：4（每个 GPU）
  Epochs: 5  # 训练轮数：5
  Random Seed: 42 (fixed)  # 随机种子：42（固定，保证可复现）
  
  Training Data:
    - rule_labels (deterministic)  # 规则标签（确定性）
    - judge_labels (high-quality)  # Judge 标签（高质量）
    - Total: 10,000 cases  # 总计：10,000 个案例
  
  Output: lora_model_v1.0.pt  # 输出：lora_model_v1.0.pt
```

**配置说明**：
- **LoRA**：只训练少量参数（LoRA adapters），不训练整个模型，节省计算资源
- **Learning Rate 2e-4**：初始训练的学习率，相对较高，因为模型需要学习任务特定的模式
- **Random Seed 42**：固定随机种子，确保每次训练结果可复现

#### 3. Training Code Example（训练代码示例）

```python
# Fine-tuning (Step 4) - 初始训练阶段
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load base model（加载基础模型）
# 使用 4-bit 量化加载，节省显存
base_model = AutoModelForCausalLM.from_pretrained(
    "qwen2.5-vl-7b",
    load_in_4bit=True,  # 4-bit 量化
    device_map="auto"  # 自动设备映射
)

# Configure LoRA（配置 LoRA）
# LoRA 是一种参数高效微调方法，只训练少量参数
lora_config = LoraConfig(
    r=16,  # LoRA rank（秩）：控制适配器容量，16 是平衡点
    lora_alpha=32,  # LoRA alpha：缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标模块：只在这些模块上添加 LoRA
    lora_dropout=0.1  # Dropout：防止过拟合
)

# Apply LoRA（应用 LoRA）
# 在基础模型上添加 LoRA adapters
model = get_peft_model(base_model, lora_config)

# Training arguments（训练参数）
training_args = TrainingArguments(
    output_dir="./models/lora_model_v1.0",  # 输出目录
    learning_rate=2e-4,  # 学习率：2e-4（初始训练的学习率）
    per_device_train_batch_size=4,  # 每个设备的批次大小
    num_train_epochs=5,  # 训练轮数：5
    logging_steps=100,  # 每 100 步记录一次日志
    save_steps=500,  # 每 500 步保存一次 checkpoint
    seed=42  # 随机种子：固定为 42，保证可复现
)

# Train（开始训练）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 训练数据：rule_labels + judge_labels
    data_collator=data_collator  # 数据整理器
)

trainer.train()  # 执行训练
```

### Characteristics of Fine-Tuning（Fine-Tuning 的特点）

#### ✅ Advantages（优势）

1. **High-Quality Data（高质量数据）**: Uses carefully labeled training data
   - 使用经过规则引擎和 LLM Judge 双重验证的标签
   - 数据质量可控，标注一致性好

2. **Controlled Environment（可控环境）**: Training in controlled AWS environment
   - 在 AWS 上训练，环境可控
   - 可以固定所有版本（Docker、数据、配置）

3. **Reproducible（可复现）**: Fixed random seed, versioned data
   - 固定随机种子（seed=42）
   - 所有数据、配置都版本化
   - 可以完全重现训练过程

4. **Comprehensive（全面）**: Covers all expected scenarios
   - 数据量大（10,000 cases）
   - 覆盖了所有预期的场景和问题类型

#### ⚠️ Limitations（局限性）

1. **Static Data（静态数据）**: Based on initial dataset, may not cover all edge cases
   - 基于初始数据集，可能无法覆盖所有边界情况
   - 无法预知生产环境中的真实场景

2. **No Real-World Feedback（没有真实反馈）**: Doesn't know how model performs in production
   - 不知道模型在生产环境中的实际表现
   - 无法知道医生的接受度和满意度

3. **One-Time（一次性）**: Done once per model version
   - 每个模型版本只做一次
   - 无法基于实际使用反馈进行改进

### Fine-Tuning Output

```yaml
Output:
  Model: lora_model_v1.0.pt
  Location: s3://rad-linter-data/models/lora_model_v1.0/
  Size: ~500MB (LoRA adapters only)
  Performance:
    - Rule Adherence: 100%
    - Silver Agreement: 88.74%
    - Judge-Rule Gap: 88.74%
```

---

## Post-Training (Continuous Improvement)（训练后持续改进）

### Definition（定义）

**Post-training（训练后改进）** is the continuous improvement phase after model deployment, using production feedback to refine the model, rules, and prompts.

**中文解释**：Post-training 是模型部署到生产环境后的持续改进阶段，使用生产环境的真实反馈数据来优化模型、规则和 prompt。

### In Rad-Linter Lifecycle（在 Rad-Linter 生命周期中的位置）

**Location**: After production deployment（位置：生产部署之后）

```
Fine-Tuning (Step 4)
    ↓
Evaluation (Step 5)
    ↓
Release Gate
    ↓
Deploy to Production (v1.0)
    ↓
Collect Feedback (Weeks 1-4)
    ↓
Post-Training ⭐ (Continuous Improvement)
    ↓
New Model Version (v1.1)
    ↓
(Repeat Cycle)
```

### Post-Training Process

#### 1. Input Data（输入数据）

```yaml
Post-Training Data:
  Source: Production feedback  # 数据来源：生产环境反馈
  - Accept cases (doctor agrees with prediction)  # Accept：医生同意系统预测
  - Ignore cases (doctor disagrees - false positives)  # Ignore：医生不同意（误报）
  - Review cases (expert corrections)  # Review：专家修正
  
  Format:
    Input: visual_facts + report_facts + original_prediction  # 输入：视觉事实 + 报告事实 + 原始预测
    Output: doctor_feedback (action, reason, corrected_content)  # 输出：医生反馈（操作、原因、修正内容）
  
  Size: 1000+ cases (accumulated over weeks)  # 数据量：1000+ 个案例（几周内积累）
  Quality: Real-world production data  # 数据质量：真实世界的生产数据
```

**关键点**：
- 数据来源是生产环境的真实反馈，不是人工标注
- 数据量相对较小（1000+ cases），但质量高（真实场景）
- 包含三种类型的反馈：Accept（正样本）、Ignore（误报）、Review（专家标注）

#### 2. Training Methods（训练方法）

```yaml
Post-Training Methods:

  Method 1: Supervised Fine-Tuning (SFT)  # 方法 1：监督微调
    Data: Accept cases (1000+ examples)  # 数据：Accept 案例（1000+ 个样本）
    Purpose: Reinforce correct predictions  # 目的：强化正确预测
    Learning Rate: 1e-4 (lower than initial)  # 学习率：1e-4（比初始训练低）
    Epochs: 3-5  # 训练轮数：3-5
    
  Method 2: Direct Preference Optimization (DPO)  # 方法 2：直接偏好优化
    Data: Review cases (500+ preference pairs)  # 数据：Review 案例（500+ 个偏好对）
    Purpose: Learn from expert corrections  # 目的：从专家修正中学习
    Learning Rate: 5e-6 (much lower)  # 学习率：5e-6（比 SFT 更低）
    Epochs: 2-3  # 训练轮数：2-3
    
  Method 3: Rule Optimization  # 方法 3：规则优化
    Data: Ignore cases (500+ examples)  # 数据：Ignore 案例（500+ 个样本）
    Purpose: Reduce false positives  # 目的：降低误报率
    Method: Non-model training (threshold adjustment)  # 方法：非模型训练（调整阈值）
```

**方法选择**：
- **SFT**：用于 Accept cases，强化模型已经做对的地方
- **DPO**：用于 Review cases，学习专家的偏好和修正
- **Rule Optimization**：用于 Ignore cases，调整规则阈值，不涉及模型训练

#### 3. Training Code Example（训练代码示例）

```python
# Post-Training: SFT（训练后改进：监督微调）
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load previously fine-tuned model（加载之前微调好的模型）
# 注意：这里加载的是已经 fine-tuned 的模型，不是原始 base model
base_model = AutoModelForCausalLM.from_pretrained(
    "qwen2.5-vl-7b",
    load_in_4bit=True
)

# Load existing LoRA adapters（加载现有的 LoRA 适配器）
# 这是关键区别：post-training 是在 fine-tuned 模型基础上继续训练
model = PeftModel.from_pretrained(
    base_model,
    "lora_model_v1.0.pt"  # Previous fine-tuned model（之前微调好的模型）
)

# Post-training: SFT with feedback data（使用反馈数据进行 SFT）
training_args = TrainingArguments(
    output_dir="./models/lora_model_v1.1",  # 输出目录：新版本模型
    learning_rate=1e-4,  # Lower learning rate（更低的学习率：1e-4，比初始训练的 2e-4 低）
    per_device_train_batch_size=8,  # 批次大小可以稍大（因为数据量小）
    num_train_epochs=5,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=feedback_dataset,  # Accept cases from production（生产环境的 Accept 案例）
    data_collator=data_collator
)

trainer.train()

# Post-Training: DPO (optional)（训练后改进：DPO，可选）
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,  # Review cases（Review 案例，包含偏好对）
    beta=0.1  # DPO temperature（DPO 温度参数）
)

dpo_trainer.train()
```

**关键区别**：
- **Fine-Tuning**：从 base model 开始训练
- **Post-Training**：从 fine-tuned model（lora_model_v1.0.pt）开始继续训练
- **Learning Rate**：Post-training 使用更低的学习率（1e-4），因为模型已经训练过，需要更温和的更新

### Characteristics of Post-Training（Post-Training 的特点）

#### ✅ Advantages（优势）

1. **Real-World Data（真实世界数据）**: Based on actual production usage
   - 基于生产环境的真实使用数据
   - 反映了模型在实际场景中的表现

2. **Targeted Improvement（针对性改进）**: Addresses specific issues found in production
   - 针对生产环境中发现的具体问题
   - 可以解决 fine-tuning 时无法预见的边界情况

3. **Continuous（持续）**: Iterative improvement cycle
   - 迭代改进循环
   - 每 2-3 个月进行一次，持续优化

4. **Feedback-Driven（反馈驱动）**: Uses doctor feedback to guide improvements
   - 使用医生的真实反馈指导改进
   - 确保改进方向符合实际需求

#### ⚠️ Limitations（局限性）

1. **Data Quality（数据质量）**: May have noise, requires careful filtering
   - 可能有噪声，需要仔细过滤
   - 需要数据清洗和质量控制

2. **Smaller Dataset（数据集较小）**: Typically smaller than initial training data
   - 通常比初始训练数据小（1000+ vs 10,000）
   - 需要积累一段时间才能有足够的数据

3. **Requires Deployment（需要先部署）**: Must deploy model first to collect feedback
   - 必须先部署模型才能收集反馈
   - 需要等待反馈数据积累（通常 4 周）

### Post-Training Output

```yaml
Output:
  Model: lora_model_v1.1.pt (improved version)
  Location: s3://rad-linter-data/models/lora_model_v1.1/
  Improvements:
    - False Positive Rate: 10% → 5%
    - Doctor Satisfaction: 80% → 90%
    - Automation Rate: 82% → 87%
```

---

## Key Differences Comparison（关键区别对比）

### Comprehensive Comparison Table（全面对比表）

| Dimension | Fine-Tuning (Step 4) | Post-Training |
|-----------|---------------------|---------------|
| **Stage（阶段）** | Initial training（初始训练） | Production improvement（生产改进） |
| **Timing（时机）** | Before deployment（部署前） | After deployment（部署后） |
| **Location in Pipeline（Pipeline 位置）** | Step 4（第 4 步） | Separate cycle（独立循环） |
| **Data Source（数据来源）** | Step 3 + Step 3.5 labels（Step 3 和 3.5 的标签） | Production feedback（生产反馈） |
| **Data Type（数据类型）** | Labeled training data（标注训练数据） | Accept/Ignore/Review（接受/忽略/复核） |
| **Data Size（数据量）** | 10,000 cases（10,000 个案例） | 1000+ cases (accumulated)（1000+ 个案例，积累） |
| **Data Quality（数据质量）** | High (carefully labeled)（高，精心标注） | Variable (real-world)（可变，真实世界） |
| **Base Model（基础模型）** | Pre-trained base model（预训练基础模型） | Previously fine-tuned model（之前微调的模型） |
| **Learning Rate（学习率）** | 2e-4 (initial)（2e-4，初始） | 1e-4 (SFT) or 5e-6 (DPO)（1e-4 SFT 或 5e-6 DPO） |
| **Epochs（训练轮数）** | 5 | 3-5 (SFT) or 2-3 (DPO)（SFT 3-5 轮，DPO 2-3 轮） |
| **Purpose（目的）** | Learn task-specific patterns（学习任务特定模式） | Improve based on real usage（基于实际使用改进） |
| **Methods（方法）** | LoRA Fine-Tuning（LoRA 微调） | SFT, DPO, Rule Optimization（SFT、DPO、规则优化） |
| **Frequency（频率）** | One-time per version（每个版本一次） | Continuous (every 2-3 months)（持续，每 2-3 个月） |
| **Evaluation（评估）** | Golden Set（Golden Set） | Golden Set + Feedback validation（Golden Set + 反馈验证集） |
| **Output（输出）** | lora_model_v1.0.pt | lora_model_v1.1.pt (improved)（改进版本） |
| **Risk Level（风险级别）** | Medium (initial training)（中等，初始训练） | Low (incremental improvement)（低，增量改进） |

### Visual Comparison

```
Fine-Tuning (Initial Training):
┌─────────────────────────────────────────┐
│ Pre-trained Base Model                  │
│ (qwen2.5-vl-7b)                         │
└──────────────┬──────────────────────────┘
               │
               │ LoRA Fine-Tuning
               │ (Learning Rate: 2e-4)
               │ (Epochs: 5)
               │ (Data: 10K labeled cases)
               ▼
┌─────────────────────────────────────────┐
│ Fine-Tuned Model                        │
│ (lora_model_v1.0.pt)                    │
│ - Rule Adherence: 100%                  │
│ - Silver Agreement: 88.74%              │
└─────────────────────────────────────────┘

Post-Training (Continuous Improvement):
┌─────────────────────────────────────────┐
│ Fine-Tuned Model                        │
│ (lora_model_v1.0.pt)                    │
└──────────────┬──────────────────────────┘
               │
               │ Production Deployment
               │ ↓
               │ Collect Feedback
               │ (Accept/Ignore/Review)
               │
               │ Post-Training
               │ - SFT (Accept cases)
               │ - DPO (Review cases)
               │ (Learning Rate: 1e-4 / 5e-6)
               │ (Epochs: 3-5 / 2-3)
               │ (Data: 1K+ feedback cases)
               ▼
┌─────────────────────────────────────────┐
│ Improved Model                          │
│ (lora_model_v1.1.pt)                    │
│ - False Positive Rate: 5% (was 10%)    │
│ - Doctor Satisfaction: 90% (was 80%)    │
└─────────────────────────────────────────┘
```

### Data Flow Comparison（数据流对比）

#### Fine-Tuning Data Flow（Fine-Tuning 数据流）

```
Step 3: Rule Labels（规则标签）
    ↓
Step 3.5: Judge Labels（Judge 标签）
    ↓
Step 4: Fine-Tuning（Fine-Tuning）
    ├─ Input: rule_labels + judge_labels（输入：规则标签 + Judge 标签）
    ├─ Method: LoRA Fine-Tuning（方法：LoRA 微调）
    ├─ Learning Rate: 2e-4（学习率：2e-4）
    └─ Output: lora_model_v1.0.pt（输出：lora_model_v1.0.pt）
```

**数据流说明**：
- 数据来源：Pipeline 前几步生成的标签（确定性 + 高质量）
- 数据流向：Step 3/3.5 → Step 4 Fine-Tuning → 输出模型

#### Post-Training Data Flow（Post-Training 数据流）

```
Production Environment（生产环境）
    ↓
Doctor Actions (Accept/Ignore/Review)（医生操作）
    ↓
Feedback Data Collection（反馈数据收集）
    ↓
Post-Training（Post-Training）
    ├─ Accept Cases → SFT（Accept 案例 → SFT）
    ├─ Review Cases → DPO（Review 案例 → DPO）
    └─ Ignore Cases → Rule Optimization（Ignore 案例 → 规则优化）
    ↓
Output: lora_model_v1.1.pt（输出：lora_model_v1.1.pt）
```

**数据流说明**：
- 数据来源：生产环境的真实反馈（医生操作）
- 数据分类：三种反馈类型对应三种改进方法
- 数据流向：生产反馈 → 分类处理 → Post-Training → 输出改进模型

---

## When to Use Each

### Use Fine-Tuning When（何时使用 Fine-Tuning）:

✅ **Initial Model Training（初始模型训练）**
- Building the first version of the model（构建第一个版本的模型）
- Adapting pre-trained model to rad-linter task（将预训练模型适配到 rad-linter 任务）
- Have high-quality labeled data (rule_labels + judge_labels)（有高质量的标注数据）

✅ **Major Model Updates（重大模型更新）**
- Changing base model（更换基础模型）
- Significant architecture changes（重大架构变更）
- New task requirements（新任务需求）

✅ **Version Creation（版本创建）**
- Creating new model version (v1.0, v2.0)（创建新模型版本）
- Starting from scratch（从零开始）

**总结**：Fine-Tuning 用于**创建新模型版本**，是**一次性**的训练过程。

### Use Post-Training When（何时使用 Post-Training）:

✅ **Production Improvement（生产改进）**
- Model already deployed and running（模型已经部署并运行）
- Have production feedback data（有生产反馈数据）
- Need incremental improvements（需要增量改进）

✅ **Addressing Specific Issues（解决特定问题）**
- High false positive rate（高误报率）
- Low doctor satisfaction（低医生满意度）
- Specific edge cases not covered（特定边界情况未覆盖）

✅ **Continuous Optimization（持续优化）**
- Regular model updates (every 2-3 months)（定期模型更新，每 2-3 个月）
- Iterative improvement cycle（迭代改进循环）
- Feedback-driven optimization（反馈驱动的优化）

**总结**：Post-Training 用于**改进已部署的模型**，是**持续迭代**的改进过程。

### Decision Tree（决策树）

```
Do you have a deployed model in production?（你有部署在生产环境的模型吗？）
    │
    ├─ No（没有） → Use Fine-Tuning (Step 4)（使用 Fine-Tuning）
    │        - Initial training（初始训练）
    │        - Use labeled training data（使用标注训练数据）
    │        - Create lora_model_v1.0.pt（创建 lora_model_v1.0.pt）
    │
    └─ Yes（有） → Do you have production feedback?（你有生产反馈数据吗？）
             │
             ├─ No（没有） → Wait for feedback collection（等待反馈收集）
             │        - Deploy model first（先部署模型）
             │        - Collect 1000+ feedback cases（收集 1000+ 反馈案例）
             │
             └─ Yes（有） → Use Post-Training（使用 Post-Training）
                      - SFT: Accept cases（SFT：Accept 案例）
                      - DPO: Review cases（DPO：Review 案例）
                      - Rule Optimization: Ignore cases（规则优化：Ignore 案例）
                      - Create lora_model_v1.1.pt（创建 lora_model_v1.1.pt）
```

**决策逻辑**：
1. **没有部署模型** → 必须先用 Fine-Tuning 创建第一个版本
2. **有部署模型但没有反馈** → 先收集反馈（通常需要 4 周）
3. **有部署模型且有反馈** → 使用 Post-Training 进行改进

---

## Complete Lifecycle（完整生命周期）

### Full Model Lifecycle（完整模型生命周期）

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Initial Training (Fine-Tuning)                 │
│         阶段 1：初始训练（Fine-Tuning）                  │
│                                                          │
│ Step 0-1: Data Preprocessing                            │
│ Step 2: Visual Feature Extraction                       │
│ Step 3: Rule-Based Label Generation                     │
│ Step 3.5: LLM Judge Label Generation                    │
│ Step 4: LoRA Fine-Tuning ⭐                             │
│   - Base Model: qwen2.5-vl-7b                           │
│   - Data: rule_labels + judge_labels                    │
│   - Learning Rate: 2e-4                                 │
│   - Output: lora_model_v1.0.pt                          │
│ Step 5: Evaluation                                       │
│ Release Gate → Deploy v1.0                              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Production Deployment                         │
│         阶段 2：生产部署                                 │
│                                                          │
│ • Deploy lora_model_v1.0.pt to production              │
│ • Shadow Mode (提示但不拦截)                            │
│ • Collect feedback (Weeks 1-4)                          │
│ • Target: 1000+ feedback cases                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Post-Training (Continuous Improvement)        │
│         阶段 3：训练后改进（持续改进）                   │
│                                                          │
│ • Analyze feedback (Accept/Ignore/Review)              │
│ • Prepare training data                                 │
│ • Post-Training Methods:                                │
│   - SFT: Accept cases (1000+ examples)                  │
│   - DPO: Review cases (500+ pairs)                      │
│   - Rule Optimization: Ignore cases                     │
│ • Learning Rate: 1e-4 (SFT) or 5e-6 (DPO)              │
│ • Output: lora_model_v1.1.pt                            │
│ • Evaluation → Release Gate → Deploy v1.1               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Iterative Improvement (Repeat Post-Training)  │
│         阶段 4：迭代改进（重复 Post-Training）          │
│                                                          │
│ • Collect new feedback                                  │
│ • Post-Training v1.1 → v1.2                             │
│ • Continuous improvement cycle                          │
│ • Every 2-3 months                                      │
└─────────────────────────────────────────────────────────┘
```

**生命周期说明**：
- **Phase 1**：Fine-Tuning 创建第一个模型版本（v1.0）
- **Phase 2**：部署到生产环境，收集反馈（4 周）
- **Phase 3**：基于反馈进行 Post-Training，创建改进版本（v1.1）
- **Phase 4**：持续迭代改进（v1.1 → v1.2 → v1.3 ...）

### Timeline Example（时间线示例）

```
Month 1-2: Initial Training (Fine-Tuning)（第 1-2 月：初始训练）
  Week 1-2: Data preprocessing (Step 0-2)（数据预处理）
  Week 3: Label generation (Step 3-3.5)（标签生成）
  Week 4: Fine-tuning (Step 4) ⭐（Fine-Tuning，关键步骤）
  Week 5: Evaluation (Step 5)（评估）
  Week 6: Release Gate → Deploy v1.0（发布门禁 → 部署 v1.0）

Month 3-4: Production Feedback Collection（第 3-4 月：生产反馈收集）
  Week 1-4: Deploy v1.0, collect feedback（部署 v1.0，收集反馈）
  Target: 1000+ feedback cases（目标：1000+ 反馈案例）

Month 5-6: First Post-Training（第 5-6 月：第一次 Post-Training）
  Week 1: Analyze feedback（分析反馈）
  Week 2-3: Post-Training (SFT + DPO)（Post-Training：SFT + DPO）
  Week 4: Evaluation → Deploy v1.1（评估 → 部署 v1.1）

Month 7+: Continuous Improvement（第 7 月+：持续改进）
  Every 2-3 months: Post-Training cycle（每 2-3 个月：Post-Training 循环）
  v1.1 → v1.2 → v1.3 → ...（持续迭代改进）
```

**时间线说明**：
- **前 2 个月**：完成初始训练（Fine-Tuning），创建 v1.0
- **第 3-4 个月**：部署并收集反馈，积累数据
- **第 5-6 个月**：第一次 Post-Training，创建 v1.1
- **第 7 个月后**：每 2-3 个月进行一次 Post-Training，持续改进

---

## Best Practices

### Fine-Tuning Best Practices（Fine-Tuning 最佳实践）

#### 1. Data Quality（数据质量）

```yaml
Fine-Tuning Data Requirements:
  - High-quality labels (rule_labels + judge_labels)  # 高质量标签
  - Balanced dataset (all issue types covered)  # 平衡数据集（覆盖所有问题类型）
  - Sufficient size (10,000+ cases)  # 足够的数据量（10,000+ 案例）
  - Data isolation (training set ≠ golden set)  # 数据隔离（训练集 ≠ Golden Set）
```

**关键点**：
- 确保数据质量：rule_labels（确定性）+ judge_labels（高质量）
- 数据平衡：覆盖所有问题类型（laterality、omission、contradiction 等）
- 数据隔离：训练集和评估集严格分离，避免数据泄漏

#### 2. Training Configuration（训练配置）

```yaml
Fine-Tuning Config:
  Learning Rate: 2e-4 (initial, can adjust)  # 学习率：2e-4（初始值，可调整）
  Batch Size: 4-8 (depends on GPU memory)  # 批次大小：4-8（取决于 GPU 显存）
  Epochs: 5 (monitor for overfitting)  # 训练轮数：5（监控过拟合）
  LoRA Rank: 16 (balance capacity vs efficiency)  # LoRA 秩：16（平衡容量和效率）
  Random Seed: 42 (for reproducibility)  # 随机种子：42（保证可复现）
```

**配置建议**：
- **Learning Rate 2e-4**：初始训练可以稍高，如果 loss 不下降可以降低
- **Batch Size**：根据 GPU 显存调整，LoRA 训练显存占用较小
- **Epochs 5**：监控 validation loss，如果开始上升就停止（early stopping）

#### 3. Monitoring（监控）

```yaml
Fine-Tuning Monitoring:
  - Training loss (should decrease)  # 训练损失（应该下降）
  - Validation loss (watch for overfitting)  # 验证损失（注意过拟合）
  - Rule Adherence (target: 100%)  # 规则遵循率（目标：100%）
  - Silver Agreement (target: > 85%)  # Silver Agreement（目标：> 85%）
  - Checkpoint saving (every epoch)  # Checkpoint 保存（每个 epoch）
```

**监控要点**：
- **Training loss**：应该持续下降，如果震荡可能是 learning rate 太高
- **Validation loss**：如果开始上升，说明过拟合，需要 early stopping
- **Rule Adherence**：必须达到 100%，这是硬性要求

### Post-Training Best Practices（Post-Training 最佳实践）

#### 1. Feedback Collection（反馈收集）

```yaml
Feedback Collection Requirements:
  - Minimum: 1000+ feedback cases  # 最少：1000+ 反馈案例
  - Diversity: All issue types, departments  # 多样性：所有问题类型、科室
  - Quality: Filter noisy feedback  # 质量：过滤噪声反馈
  - Time Period: 4+ weeks of collection  # 时间周期：4+ 周收集
```

**收集建议**：
- **最少 1000+ cases**：数据量太少无法进行有意义的改进
- **多样性**：确保覆盖所有问题类型和科室，避免偏向某些场景
- **质量控制**：过滤掉明显错误的反馈（例如误操作、测试数据等）
- **时间周期**：至少收集 4 周，确保有足够的数据积累

#### 2. Training Strategy（训练策略）

```yaml
Post-Training Strategy:
  Priority 1: Rule Optimization (fastest, lowest risk)  # 优先级 1：规则优化（最快，风险最低）
    - Analyze ignore cases  # 分析 Ignore 案例
    - Update rules  # 更新规则
    - Deploy immediately  # 立即部署
  
  Priority 2: Prompt Optimization (medium speed)  # 优先级 2：Prompt 优化（中等速度）
    - Analyze review cases  # 分析 Review 案例
    - Update prompt  # 更新 prompt
    - Re-run Step 3.5  # 重新运行 Step 3.5
  
  Priority 3: Model Re-training (slowest, highest risk)  # 优先级 3：模型重训练（最慢，风险最高）
    - SFT: Accept cases  # SFT：Accept 案例
    - DPO: Review cases  # DPO：Review 案例
    - Full pipeline re-run  # 完整 Pipeline 重新运行
```

**策略优先级**：
1. **规则优化**：最快、风险最低，可以先做，立即见效
2. **Prompt 优化**：中等速度，只需要重新运行 Step 3.5
3. **模型重训练**：最慢、风险最高，需要完整 Pipeline，但改进效果最明显

#### 3. Learning Rate Selection（学习率选择）

```yaml
Post-Training Learning Rates:
  SFT (Supervised Fine-Tuning):
    Learning Rate: 1e-4 (lower than initial 2e-4)  # 学习率：1e-4（比初始的 2e-4 低）
    Reason: Model already trained, need gentle updates  # 原因：模型已经训练过，需要温和更新
  
  DPO (Direct Preference Optimization):
    Learning Rate: 5e-6 (much lower)  # 学习率：5e-6（更低）
    Reason: Preference learning needs careful tuning  # 原因：偏好学习需要仔细调优
    Beta: 0.1 (DPO temperature parameter)  # Beta：0.1（DPO 温度参数）
```

**学习率说明**：
- **SFT 1e-4**：比初始训练的 2e-4 低一半，因为模型已经训练过，需要更温和的更新
- **DPO 5e-6**：比 SFT 更低，因为偏好学习（preference learning）需要更仔细的调优
- **Beta 0.1**：DPO 的温度参数，控制模型对偏好的敏感度

#### 4. Evaluation

```yaml
Post-Training Evaluation:
  - Golden Set (baseline, must maintain)
  - Feedback Validation Set (new, must improve)
  - Release Gate Checks:
    - Rule Adherence > 99%
    - Silver Agreement > 85%
    - No regression on golden set
    - Improvement on feedback validation set
```

### Common Mistakes to Avoid（常见错误避免）

#### ❌ Fine-Tuning Mistakes（Fine-Tuning 常见错误）

1. **Too High Learning Rate（学习率太高）**
   - Problem: Model diverges or overfits（模型发散或过拟合）
   - Solution: Start with 2e-4, adjust based on loss（从 2e-4 开始，根据 loss 调整）
   - **中文说明**：学习率太高会导致训练不稳定，loss 震荡或发散

2. **Insufficient Data（数据不足）**
   - Problem: Model doesn't learn patterns（模型学不到模式）
   - Solution: Ensure 10,000+ labeled cases（确保 10,000+ 标注案例）
   - **中文说明**：数据量太少，模型无法学习到足够的模式，性能会差

3. **Data Leakage（数据泄漏）**
   - Problem: Training data overlaps with evaluation（训练数据和评估数据重叠）
   - Solution: Strict data isolation (training ≠ golden set)（严格数据隔离）
   - **中文说明**：如果训练集和评估集有重叠，评估结果会虚高，无法反映真实性能

#### ❌ Post-Training Mistakes（Post-Training 常见错误）

1. **Too Aggressive Updates（更新太激进）**
   - Problem: Model forgets previous knowledge（模型忘记之前的知识）
   - Solution: Use lower learning rates (1e-4 for SFT, 5e-6 for DPO)（使用更低的学习率）
   - **中文说明**：学习率太高会导致 catastrophic forgetting（灾难性遗忘），模型会忘记之前学到的知识

2. **Ignoring Rule Optimization（忽略规则优化）**
   - Problem: Only focusing on model training（只关注模型训练）
   - Solution: Rule optimization is faster and lower risk（规则优化更快、风险更低）
   - **中文说明**：规则优化不需要重新训练模型，可以立即部署，应该优先考虑

3. **Insufficient Feedback（反馈不足）**
   - Problem: Not enough data for meaningful improvement（数据不足以进行有意义的改进）
   - Solution: Collect 1000+ feedback cases before post-training（Post-Training 前收集 1000+ 反馈案例）
   - **中文说明**：反馈数据太少，改进效果不明显，甚至可能变差

4. **No Validation Set（没有验证集）**
   - Problem: Can't measure improvement（无法衡量改进效果）
   - Solution: Hold out 20% of feedback data for validation（保留 20% 反馈数据作为验证集）
   - **中文说明**：没有验证集就无法知道改进是否有效，可能只是过拟合到训练数据

---

## Summary（总结）

### Key Takeaways（关键要点）

1. ✅ **Fine-Tuning**: Initial training (Step 4), one-time per version（初始训练，每个版本一次）
2. ✅ **Post-Training**: Continuous improvement, iterative cycle（持续改进，迭代循环）
3. ✅ **Different Stages**: Fine-tuning before deployment, post-training after（不同阶段：Fine-tuning 在部署前，Post-Training 在部署后）
4. ✅ **Different Data**: Fine-tuning uses labeled data, post-training uses feedback（不同数据：Fine-tuning 用标注数据，Post-Training 用反馈数据）
5. ✅ **Different Learning Rates**: Fine-tuning 2e-4, Post-training 1e-4 (SFT) or 5e-6 (DPO)（不同学习率）
6. ✅ **Complementary**: Both are needed for a complete model lifecycle（互补：两者都是完整模型生命周期所必需的）

### Quick Reference（快速参考）

| Question | Fine-Tuning | Post-Training |
|----------|------------|---------------|
| **When?（何时）** | Before deployment（部署前） | After deployment（部署后） |
| **What data?（什么数据）** | Labeled training data（标注训练数据） | Production feedback（生产反馈） |
| **Learning rate?（学习率）** | 2e-4 | 1e-4 (SFT) or 5e-6 (DPO) |
| **Epochs?（训练轮数）** | 5 | 3-5 (SFT) or 2-3 (DPO) |
| **Frequency?（频率）** | One-time（一次性） | Continuous (every 2-3 months)（持续，每 2-3 个月） |
| **Purpose?（目的）** | Learn task patterns（学习任务模式） | Improve based on usage（基于使用改进） |

### Final Answer（最终答案）

**Fine-Tuning and Post-Training are NOT the same（Fine-Tuning 和 Post-Training 不一样）:**

- **Fine-Tuning** = Initial training (Step 4) → Creates first model version（初始训练 → 创建第一个模型版本）
- **Post-Training** = Continuous improvement → Improves deployed model（持续改进 → 改进已部署的模型）

**Both are essential（两者都必不可少）** for a complete model lifecycle:
1. Fine-Tuning creates the initial model（Fine-Tuning 创建初始模型）
2. Post-Training continuously improves it based on real-world usage（Post-Training 基于真实使用持续改进它）

**核心区别总结**：
- **Fine-Tuning**：从 base model 开始，创建第一个版本（v1.0）
- **Post-Training**：从 fine-tuned model 开始，持续改进（v1.0 → v1.1 → v1.2 ...）
- **两者互补**：Fine-Tuning 是基础，Post-Training 是持续改进的引擎

---

## Related Documents

- **A02**: Production Training Pipeline (Fine-Tuning details)
- **A04**: Production Training Pipeline Revised (Fine-Tuning configuration)
- **A05**: Post-Training Mechanism (Post-Training details)

---

**Remember**: Fine-Tuning is the foundation, Post-Training is the engine for continuous improvement.
