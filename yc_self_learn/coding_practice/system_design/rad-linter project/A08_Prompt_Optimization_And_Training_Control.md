# A08: Prompt Optimization & Training Control
# A08: Prompt 优化与训练控制（工业级持续提升方案）

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Industrial-grade approach to ensure monotonic improvement through prompt optimization  
**Key Insight**: Prompt optimization → Better judge_labels → Better training data → Better model performance  
**Core Principle**: **"Controlled Iteration: No Regression, No Release"（受控迭代：不准退步就不发布）**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prompt Optimization Strategy](#prompt-optimization-strategy)
3. [Training Control Mechanisms](#training-control-mechanisms)
4. [Ensuring Continuous Improvement](#ensuring-continuous-improvement)
5. [Complete Workflow](#complete-workflow)
6. [Best Practices](#best-practices)

---

## Overview

### Key Insight（核心洞察）

**"Prompt/Rubric/Contracts are the Highest Leverage Optimization"**

**中文解释**：Prompt/Rubric/Contracts 是"同数据下提升最大的杠杆"。

**Why（为什么）**：
- **Dataset fixed**：数据集本身不变
- **But judge_labels improve**：但通过优化 prompt，可以生成更好的 judge_labels
- **Better training data**：更好的训练数据 → 更好的模型性能
- **Low cost, high impact**：成本低（只需要重新运行 Step 3.5），影响大（提升训练数据质量）

### The Challenge（挑战）

**How to ensure monotonic improvement?（如何确保持续提升？）**

**工业级定义**：
- **不是"绝对单调"**：现实里做不到绝对单调
- **而是"受控迭代"**：不准退步就不发布（No Regression, No Release）
- **核心**：把 prompt 迭代变成可版本化、可评测、可回滚的工程流程

### Core Principle（核心原则）

**"Controlled Iteration: No Regression, No Release"**

**中文解释**：受控迭代：不准退步就不发布。

**实现方式**：
1. **Release Gate**：不是控制 training，而是控制 Release Gate
2. **版本化**：Prompt 当成代码 + 模型权重同等级的 artifact
3. **分层指标**：三道门禁，任何一道不过都不发
4. **Regression Set**：专门的回归集，任何版本不能变差

---

## Prompt Versioning & Artifact Management（Prompt 版本化与 Artifact 管理）

### 1. Prompt as First-Class Artifact（Prompt 作为一等 Artifact）

**Key Principle（核心原则）**：
- **Prompt 必须当成"代码 + 模型权重"同等级的 artifact**
- **每次改 prompt，都要完整记录版本信息**

```python
class PromptArtifact:
    """
    Prompt artifact with full versioning（带完整版本信息的 Prompt artifact）
    """
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        self.prompt_version = self.extract_version(prompt_path)  # v1.2, v1.3
        self.prompt_hash = self.calculate_hash(prompt_path)  # SHA256
        self.judge_model_version = self.get_judge_model_version()  # base model + lora + docker
        self.timestamp = datetime.now().isoformat()
    
    def calculate_hash(self, prompt_path):
        """
        Calculate SHA256 hash of prompt file（计算 Prompt 文件的 SHA256 hash）
        """
        import hashlib
        with open(prompt_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    
    def get_judge_model_version(self):
        """
        Get judge model version（获取 Judge 模型版本）
        """
        return {
            "base_model": "qwen2.5-vl-7b",
            "lora_version": "lora_model_v1.0.pt",
            "docker_digest": "sha256:abc123..."  # Docker image digest
        }
    
    def to_dict(self):
        """
        Serialize to dict for logging（序列化为字典用于日志）
        """
        return {
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "judge_model_version": self.judge_model_version,
            "timestamp": self.timestamp
        }
```

### 2. Version Tracking in Eval Report（在评估报告中记录版本）

```python
def generate_eval_report(model, prompt_artifact, eval_results):
    """
    Generate eval report with full version tracking（生成带完整版本跟踪的评估报告）
    """
    report = {
        "model_version": model.version,
        "prompt_artifact": prompt_artifact.to_dict(),
        "eval_results": eval_results,
        "golden_set_version": "golden_set_v1.0.jsonl",
        "regression_set_version": "regression_set_v1.0.jsonl",
        "release_gate_results": release_gate_results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to S3 with version（保存到 S3，带版本）
    report_path = f"s3://rad-linter-data/eval_reports/report_{prompt_artifact.prompt_version}_{model.version}.json"
    save_to_s3(report, report_path)
    
    return report
```

## Prompt Optimization Strategy（Prompt 优化策略）

### 1. Why Prompt Optimization Works（为什么 Prompt 优化有效）

#### Current Flow（当前流程）

```
Fixed Dataset（固定数据集）
    ↓
Step 3.5: LLM Judge Label Generation
    ├─ Prompt: prompt_v1.2.jinja（当前 prompt）
    ├─ Input: visual_facts + report_facts
    └─ Output: judge_labels_v1.2.jsonl（当前 judge_labels）
    ↓
Step 4: Fine-Tuning
    ├─ Training Data: rule_labels + judge_labels_v1.2
    └─ Output: lora_model_v1.0.pt（基于当前 judge_labels）
```

#### Optimized Flow（优化后的流程）

```
Same Dataset（相同数据集）
    ↓
Step 3.5: LLM Judge Label Generation（优化后）
    ├─ Prompt: prompt_v1.3.jinja（优化后的 prompt）⭐
    ├─ Input: visual_facts + report_facts（相同输入）
    └─ Output: judge_labels_v1.3.jsonl（更好的 judge_labels）⭐
    ↓
Step 4: Fine-Training（重新训练）
    ├─ Training Data: rule_labels + judge_labels_v1.3（更好的数据）
    └─ Output: lora_model_v1.1.pt（更好的模型）⭐
```

**关键点**：
- **Dataset 不变**：visual_facts 和 report_facts 不变
- **Prompt 改变**：prompt_v1.2 → prompt_v1.3
- **judge_labels 质量提升**：更准确、更一致的标签
- **模型性能提升**：基于更好的训练数据，模型性能提升

### 2. Prompt Optimization Process（Prompt 优化流程）

#### Step 1: Analyze Current judge_labels Quality（分析当前 judge_labels 质量）

```python
def analyze_judge_labels_quality(judge_labels, rule_labels, golden_set):
    """
    Analyze judge_labels quality（分析 judge_labels 质量）
    """
    analysis = {
        "consistency": analyze_consistency(judge_labels),
        "agreement_with_rules": calculate_agreement(judge_labels, rule_labels),
        "coverage": analyze_coverage(judge_labels, golden_set),
        "error_patterns": identify_error_patterns(judge_labels)
    }
    
    return analysis

# Identify issues（识别问题）
issues = {
    "low_consistency": "Same case gets different labels（相同案例得到不同标签）",
    "missing_evidence": "Labels lack evidence references（标签缺少证据引用）",
    "severity_mismatch": "Severity doesn't match issue type（严重程度不匹配问题类型）",
    "incomplete_coverage": "Some issue types not covered（某些问题类型未覆盖）"
}
```

#### Step 2: Optimize Prompt（优化 Prompt）

**Current Prompt（当前 Prompt）**：
```jinja
# prompt_v1.2.jinja
You are a radiology report quality checker.
Analyze visual facts and report facts.
Identify issues.
```

**Optimized Prompt - Two-Stage Generation（优化后的 Prompt - 两段式生成）**：

**Key Improvement（关键改进）**：两段式生成，稳定性大幅提高

**Pass 1: Decision JSON（第一阶段：决策 JSON）**
```jinja
# prompt_v1.3_pass1_decision.jinja
You are an expert radiology report quality checker.

## Task（任务）
Compare visual facts with report facts and identify discrepancies（比较视觉事实和报告事实，识别差异）.

## Input（输入）
Visual Facts:
{% for fact in visual_facts %}
- vf_{{ loop.index0 }}: {{ fact.type }} | {{ fact.laterality }} | confidence: {{ fact.confidence }}
{% endfor %}

Report Facts:
{% for fact in report_facts %}
- rf_{{ loop.index0 }}: {{ fact.entity }} | "{{ fact.text }}" | span: {{ fact.span_ref.start }}-{{ fact.span_ref.end }}
{% endfor %}

## Instructions（指令）
1. **Output ONLY structured JSON（只输出结构化 JSON）**
2. **No explanations（不要解释）**
3. **Must reference fact IDs（必须引用 fact IDs）**

## Output Format（输出格式）
```json
{
  "issues": [
    {
      "issue_type": "laterality_mismatch",
      "severity": "high",
      "supporting_facts": ["vf_001", "rf_002"],
      "report_spans": [{"start": 120, "end": 135}]
    }
  ]
}
```
```

**Pass 2: Explanation (Optional)（第二阶段：解释（可选））**
```jinja
# prompt_v1.3_pass2_explanation.jinja
You are an expert radiology report quality checker.

## Task（任务）
Generate explanation for the decision（为决策生成解释）.

## Decision（决策）
{{ decision_json }}

## Instructions（指令）
1. **Explanation must reference supporting_facts（解释必须引用 supporting_facts）**
2. **Explanation must reference report_spans（解释必须引用 report_spans）**
3. **Do not change the decision（不要改变决策）**

## Output Format（输出格式）
```json
{
  "issues": [
    {
      "issue_type": "laterality_mismatch",
      "severity": "high",
      "supporting_facts": ["vf_001", "rf_002"],
      "report_spans": [{"start": 120, "end": 135}],
      "explanation": "Visual fact vf_001 shows left effusion, but report states right effusion (span 120-135)"
    }
  ]
}
```
```

**Benefits of Two-Stage Generation（两段式生成的好处）**：
- **Schema success rate higher**：Schema 成功率更高
- **Hallucination less**：幻觉更少
- **Evaluation more stable**：评测更稳定（波动更小）
- **Easier to ensure continuous improvement**：更容易确保持续提升

**Key Improvements（关键改进）**：
1. **More detailed instructions**：更详细的指令
2. **Structured input format**：结构化的输入格式
3. **Clear output format**：清晰的输出格式
4. **Evidence requirements**：证据要求
5. **Examples**：示例

#### Step 3: Re-run Step 3.5（重新运行 Step 3.5）

```bash
# Re-run Step 3.5 with new prompt（用新 prompt 重新运行 Step 3.5）
python step3_5_judge.py \
  --prompt prompt_v1.3.jinja \
  --input visual_facts_v1.0.jsonl \
  --input rule_labels_v1.0.jsonl \
  --output judge_labels_v1.3.jsonl \
  --model qwen2.5-vl-7b \
  --batch_size 8
```

#### Step 4: Validate judge_labels Quality（验证 judge_labels 质量）

```python
def validate_judge_labels_quality(new_judge_labels, old_judge_labels, rule_labels):
    """
    Validate that new judge_labels are better（验证新 judge_labels 更好）
    """
    metrics = {
        "consistency": calculate_consistency(new_judge_labels),
        "evidence_coverage": calculate_evidence_coverage(new_judge_labels),
        "agreement_with_rules": calculate_agreement(new_judge_labels, rule_labels),
        "schema_compliance": validate_schema(new_judge_labels)
    }
    
    # Compare with old（与旧的对比）
    old_metrics = calculate_metrics(old_judge_labels)
    
    improvements = {
        "consistency": metrics["consistency"] - old_metrics["consistency"],
        "evidence_coverage": metrics["evidence_coverage"] - old_metrics["evidence_coverage"],
        "agreement": metrics["agreement_with_rules"] - old_metrics["agreement_with_rules"]
    }
    
    # All metrics should improve（所有指标都应该改进）
    if all(imp > 0 for imp in improvements.values()):
        return True, improvements
    else:
        return False, improvements
```

---

## Training Control Mechanisms（训练控制机制）

### 1. Validation Set Strategy（验证集策略）

#### Create Validation Set（创建验证集）

```python
def create_validation_set(dataset, validation_ratio=0.2):
    """
    Create validation set from training data（从训练数据创建验证集）
    """
    # Stratified split（分层划分）
    train_data, val_data = train_test_split(
        dataset,
        test_size=validation_ratio,
        stratify=dataset['issue_type'],  # Maintain distribution（保持分布）
        random_state=42
    )
    
    return train_data, val_data

# Usage（使用）
train_data, val_data = create_validation_set(
    rule_labels + judge_labels,
    validation_ratio=0.2  # 20% for validation（20% 用于验证）
)
```

**关键点**：
- **从训练数据中划分**：不是用 golden_set（golden_set 只用于最终评估）
- **Stratified split**：保持问题类型分布
- **固定 random seed**：确保可复现

#### Use Validation Set During Training（训练时使用验证集）

```python
training_args = TrainingArguments(
    # Training data（训练数据）
    train_dataset=train_data,  # 80% of data（80% 数据）
    
    # Validation data（验证数据）
    eval_dataset=val_data,  # 20% of data（20% 数据）
    evaluation_strategy="steps",  # Evaluate every N steps（每 N 步评估）
    eval_steps=500,  # Evaluate every 500 steps（每 500 步评估）
    
    # Save best model（保存最佳模型）
    load_best_model_at_end=True,  # Load best checkpoint（加载最佳 checkpoint）
    metric_for_best_model="eval_silver_agreement",  # Metric to monitor（监控指标）
    greater_is_better=True,  # Higher is better（越高越好）
    
    # Save strategy（保存策略）
    save_strategy="steps",
    save_steps=500,  # Save checkpoint every 500 steps（每 500 步保存 checkpoint）
    save_total_limit=3,  # Keep only 3 best checkpoints（只保留 3 个最佳 checkpoint）
)
```

### 2. Early Stopping（早停）

#### Implement Early Stopping（实现早停）

```python
from transformers import EarlyStoppingCallback

# Early stopping configuration（早停配置）
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop if no improvement for 3 evaluations（3 次评估无改进就停止）
    early_stopping_threshold=0.001,  # Minimum improvement threshold（最小改进阈值）
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[early_stopping]  # Add early stopping（添加早停）
)

trainer.train()
```

**How it works（工作原理）**：
```
Training Progress（训练进度）
    ↓
Eval every 500 steps（每 500 步评估）
    ↓
Monitor eval_silver_agreement（监控 eval_silver_agreement）
    ↓
If no improvement for 3 evaluations（如果 3 次评估无改进）
    ↓
Stop training（停止训练）
    ↓
Load best checkpoint（加载最佳 checkpoint）
```

**Benefits（好处）**：
- **Prevent overfitting**：防止过拟合
- **Save time**：如果提前收敛，节省时间
- **Best model**：自动选择最佳模型

### 3. Training Monitoring（训练监控）

#### Real-Time Monitoring（实时监控）

```python
import wandb  # or tensorboard

# Initialize logging（初始化日志）
wandb.init(
    project="rad-linter-training",
    config={
        "learning_rate": 2e-4,
        "batch_size": 8,
        "lora_rank": 16,
        "prompt_version": "v1.3"
    }
)

# Monitor during training（训练时监控）
def log_metrics(metrics, step):
    """
    Log training metrics（记录训练指标）
    """
    wandb.log({
        "train_loss": metrics["train_loss"],
        "eval_loss": metrics["eval_loss"],
        "eval_silver_agreement": metrics["eval_silver_agreement"],
        "eval_rule_adherence": metrics["eval_rule_adherence"],
        "learning_rate": metrics["learning_rate"],
        "step": step
    })

# In training loop（在训练循环中）
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    
    if step % 100 == 0:
        # Evaluate on validation set（在验证集上评估）
        eval_metrics = evaluate(val_data)
        
        # Log metrics（记录指标）
        log_metrics(eval_metrics, step)
        
        # Check for improvement（检查改进）
        if eval_metrics["eval_silver_agreement"] > best_agreement:
            best_agreement = eval_metrics["eval_silver_agreement"]
            save_checkpoint(model, step)
```

#### Key Metrics to Monitor（关键监控指标）

```yaml
Training Metrics（训练指标）:
  - train_loss: Should decrease（应该下降）
  - eval_loss: Should decrease, watch for increase（应该下降，注意上升）
  - learning_rate: Track scheduler（跟踪调度器）

Performance Metrics（性能指标）:
  - eval_silver_agreement: Target > 85%（目标 > 85%）
  - eval_rule_adherence: Target = 100%（目标 = 100%）
  - eval_judge_rule_gap: Monitor（监控）

Warning Signs（警告信号）:
  - eval_loss increasing: Overfitting（eval_loss 上升：过拟合）
  - train_loss not decreasing: Learning rate too low（train_loss 不下降：学习率太低）
  - eval_loss oscillating: Learning rate too high（eval_loss 震荡：学习率太高）
```

### 4. Checkpoint Selection（Checkpoint 选择）

#### Save Multiple Checkpoints（保存多个 Checkpoint）

```python
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,  # Save every 500 steps（每 500 步保存）
    save_total_limit=5,  # Keep 5 best checkpoints（保留 5 个最佳 checkpoint）
    
    # Evaluation（评估）
    evaluation_strategy="steps",
    eval_steps=500,
    
    # Best model selection（最佳模型选择）
    load_best_model_at_end=True,
    metric_for_best_model="eval_silver_agreement",
    greater_is_better=True
)
```

#### Evaluate All Checkpoints（评估所有 Checkpoint）

```python
def evaluate_all_checkpoints(checkpoint_dir, val_data, golden_set):
    """
    Evaluate all checkpoints and select best（评估所有 checkpoint 并选择最佳）
    """
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint-*")
    
    best_checkpoint = None
    best_score = 0
    
    for checkpoint in checkpoints:
        # Load checkpoint（加载 checkpoint）
        model = load_checkpoint(checkpoint)
        
        # Evaluate on validation set（在验证集上评估）
        val_metrics = evaluate(model, val_data)
        
        # Evaluate on golden set（在 Golden Set 上评估）
        golden_metrics = evaluate(model, golden_set)
        
        # Combined score（组合分数）
        score = (
            0.4 * val_metrics["silver_agreement"] +
            0.3 * golden_metrics["silver_agreement"] +
            0.3 * golden_metrics["rule_adherence"]
        )
        
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
    
    return best_checkpoint, best_score
```

---

## Ensuring Continuous Improvement（确保持续改进）

### 1. Iterative Prompt Optimization（迭代 Prompt 优化）

#### Optimization Cycle（优化循环）

```
Iteration 1: Baseline（迭代 1：基线）
  Prompt: prompt_v1.2.jinja
  judge_labels: judge_labels_v1.2.jsonl
  Model: lora_model_v1.0.pt
  Performance: Silver Agreement = 88.74%
    ↓
Analyze judge_labels quality（分析 judge_labels 质量）
  - Identify issues（识别问题）
  - Find error patterns（找到错误模式）
    ↓
Iteration 2: Optimize Prompt（迭代 2：优化 Prompt）
  Prompt: prompt_v1.3.jinja（改进的 prompt）
  judge_labels: judge_labels_v1.3.jsonl（重新生成）
  Model: lora_model_v1.1.pt（重新训练）
  Performance: Silver Agreement = 90.5% (+1.76%) ⭐
    ↓
Validate improvement（验证改进）
  - Compare judge_labels quality（比较 judge_labels 质量）
  - Compare model performance（比较模型性能）
  - Ensure all metrics improve（确保所有指标改进）
    ↓
Iteration 3: Further Optimization（迭代 3：进一步优化）
  Prompt: prompt_v1.4.jinja
  judge_labels: judge_labels_v1.4.jsonl
  Model: lora_model_v1.2.pt
  Performance: Silver Agreement = 91.8% (+1.3%) ⭐
```

### 2. Quality Gates（质量门禁）

#### judge_labels Quality Gates（judge_labels 质量门禁）

```python
def check_judge_labels_quality_gates(new_judge_labels, old_judge_labels, rule_labels):
    """
    Check if new judge_labels pass quality gates（检查新 judge_labels 是否通过质量门禁）
    """
    gates = {
        "consistency": {
            "threshold": 0.95,  # Consistency score（一致性分数）
            "current": calculate_consistency(new_judge_labels),
            "improvement": calculate_consistency(new_judge_labels) - calculate_consistency(old_judge_labels),
            "status": "PASS" if calculate_consistency(new_judge_labels) >= 0.95 else "FAIL"
        },
        "evidence_coverage": {
            "threshold": 0.90,  # Evidence coverage（证据覆盖率）
            "current": calculate_evidence_coverage(new_judge_labels),
            "improvement": calculate_evidence_coverage(new_judge_labels) - calculate_evidence_coverage(old_judge_labels),
            "status": "PASS" if calculate_evidence_coverage(new_judge_labels) >= 0.90 else "FAIL"
        },
        "agreement_with_rules": {
            "threshold": 0.85,  # Agreement with rules（与规则的一致性）
            "current": calculate_agreement(new_judge_labels, rule_labels),
            "improvement": calculate_agreement(new_judge_labels, rule_labels) - calculate_agreement(old_judge_labels, rule_labels),
            "status": "PASS" if calculate_agreement(new_judge_labels, rule_labels) >= 0.85 else "FAIL"
        },
        "schema_compliance": {
            "threshold": 0.98,  # Schema compliance（Schema 合规性）
            "current": validate_schema_compliance(new_judge_labels),
            "status": "PASS" if validate_schema_compliance(new_judge_labels) >= 0.98 else "FAIL"
        }
    }
    
    # All gates must pass（所有门禁必须通过）
    all_pass = all(gate["status"] == "PASS" for gate in gates.values())
    
    # At least one metric must improve（至少一个指标必须改进）
    has_improvement = any(gate["improvement"] > 0 for gate in gates.values() if "improvement" in gate)
    
    return all_pass and has_improvement, gates
```

#### Model Training Quality Gates（模型训练质量门禁）

```python
def check_training_quality_gates(new_model, baseline_model, val_data, golden_set):
    """
    Check if new model passes quality gates（检查新模型是否通过质量门禁）
    """
    # Evaluate new model（评估新模型）
    new_val_metrics = evaluate(new_model, val_data)
    new_golden_metrics = evaluate(new_model, golden_set)
    
    # Evaluate baseline model（评估基线模型）
    baseline_val_metrics = evaluate(baseline_model, val_data)
    baseline_golden_metrics = evaluate(baseline_model, golden_set)
    
    gates = {
        "val_silver_agreement": {
            "threshold": baseline_val_metrics["silver_agreement"] * 0.98,  # No regression（无回归）
            "current": new_val_metrics["silver_agreement"],
            "improvement": new_val_metrics["silver_agreement"] - baseline_val_metrics["silver_agreement"],
            "status": "PASS" if new_val_metrics["silver_agreement"] >= baseline_val_metrics["silver_agreement"] * 0.98 else "FAIL"
        },
        "golden_silver_agreement": {
            "threshold": baseline_golden_metrics["silver_agreement"] * 0.98,
            "current": new_golden_metrics["silver_agreement"],
            "improvement": new_golden_metrics["silver_agreement"] - baseline_golden_metrics["silver_agreement"],
            "status": "PASS" if new_golden_metrics["silver_agreement"] >= baseline_golden_metrics["silver_agreement"] * 0.98 else "FAIL"
        },
        "rule_adherence": {
            "threshold": 0.99,  # Must maintain（必须保持）
            "current": new_golden_metrics["rule_adherence"],
            "status": "PASS" if new_golden_metrics["rule_adherence"] >= 0.99 else "FAIL"
        }
    }
    
    # All gates must pass（所有门禁必须通过）
    all_pass = all(gate["status"] == "PASS" for gate in gates.values())
    
    # At least one metric must improve（至少一个指标必须改进）
    has_improvement = any(gate["improvement"] > 0.005 for gate in gates.values() if "improvement" in gate)
    
    return all_pass and has_improvement, gates
```

### 3. Training Control Workflow（训练控制工作流）

#### Complete Control Flow（完整控制流程）

```python
def controlled_training_workflow(
    prompt_version,
    judge_labels_version,
    model_version,
    train_data,
    val_data,
    golden_set
):
    """
    Controlled training workflow with quality gates（带质量门禁的受控训练工作流）
    """
    # Step 1: Generate judge_labels with new prompt（步骤 1：用新 prompt 生成 judge_labels）
    print(f"Step 1: Generating judge_labels with prompt {prompt_version}")
    judge_labels = generate_judge_labels(
        prompt=f"prompt_{prompt_version}.jinja",
        visual_facts=train_data["visual_facts"],
        rule_labels=train_data["rule_labels"]
    )
    
    # Step 2: Validate judge_labels quality（步骤 2：验证 judge_labels 质量）
    print("Step 2: Validating judge_labels quality")
    old_judge_labels = load_judge_labels(f"judge_labels_{judge_labels_version - 1}.jsonl")
    judge_labels_pass, judge_gates = check_judge_labels_quality_gates(
        judge_labels, old_judge_labels, train_data["rule_labels"]
    )
    
    if not judge_labels_pass:
        print("❌ judge_labels quality gates failed")
        print(f"Gates: {judge_gates}")
        return None, "judge_labels_quality_failed"
    
    print("✅ judge_labels quality gates passed")
    
    # Step 3: Prepare training data（步骤 3：准备训练数据）
    print("Step 3: Preparing training data")
    training_dataset = prepare_training_data(
        rule_labels=train_data["rule_labels"],
        judge_labels=judge_labels
    )
    
    # Step 4: Train with monitoring（步骤 4：带监控的训练）
    print("Step 4: Training model with monitoring")
    model = train_with_monitoring(
        training_dataset=training_dataset,
        val_dataset=val_data,
        model_version=model_version,
        callbacks=[
            EarlyStoppingCallback(patience=3),
            WandbCallback()  # Logging（日志）
        ]
    )
    
    # Step 5: Evaluate on validation set（步骤 5：在验证集上评估）
    print("Step 5: Evaluating on validation set")
    val_metrics = evaluate(model, val_data)
    
    # Step 6: Evaluate on golden set（步骤 6：在 Golden Set 上评估）
    print("Step 6: Evaluating on golden set")
    golden_metrics = evaluate(model, golden_set)
    
    # Step 7: Check training quality gates（步骤 7：检查训练质量门禁）
    print("Step 7: Checking training quality gates")
    baseline_model = load_model(f"lora_model_v{model_version - 1}.pt")
    training_pass, training_gates = check_training_quality_gates(
        model, baseline_model, val_data, golden_set
    )
    
    if not training_pass:
        print("❌ Training quality gates failed")
        print(f"Gates: {training_gates}")
        return None, "training_quality_failed"
    
    print("✅ Training quality gates passed")
    print(f"Improvements: {training_gates}")
    
    # Step 8: Save model（步骤 8：保存模型）
    save_model(model, f"lora_model_v{model_version}.pt")
    
    return model, "success"
```

### 4. Preventing Regression（防止回归）

#### Regression Detection（回归检测）

```python
def detect_regression(new_metrics, baseline_metrics, threshold=0.02):
    """
    Detect if new model regressed（检测新模型是否回归）
    """
    regressions = []
    
    # Check each metric（检查每个指标）
    for metric_name in baseline_metrics:
        if metric_name in new_metrics:
            baseline_value = baseline_metrics[metric_name]
            new_value = new_metrics[metric_name]
            
            # Calculate relative change（计算相对变化）
            relative_change = (new_value - baseline_value) / baseline_value
            
            # If regression > threshold（如果回归 > 阈值）
            if relative_change < -threshold:
                regressions.append({
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "new": new_value,
                    "regression": relative_change
                })
    
    return regressions

# Usage（使用）
regressions = detect_regression(new_golden_metrics, baseline_golden_metrics)

if regressions:
    print("⚠️ Regression detected:")
    for reg in regressions:
        print(f"  - {reg['metric']}: {reg['baseline']} → {reg['new']} ({reg['regression']*100:.2f}% regression)")
    # Block release（阻止发布）
    return False
else:
    print("✅ No regression detected")
    return True
```

#### Automatic Rollback（自动回滚）

```python
def automatic_rollback_if_regression(new_model, baseline_model, val_data, golden_set):
    """
    Automatically rollback if regression detected（如果检测到回归，自动回滚）
    """
    new_metrics = evaluate(new_model, golden_set)
    baseline_metrics = evaluate(baseline_model, golden_set)
    
    regressions = detect_regression(new_metrics, baseline_metrics)
    
    if regressions:
        print("⚠️ Regression detected, rolling back to baseline model")
        return baseline_model, "rolled_back"
    else:
        print("✅ No regression, keeping new model")
        return new_model, "kept"
```

---

## Complete Workflow（完整工作流）

### Prompt Optimization → Training → Validation Cycle（Prompt 优化 → 训练 → 验证循环）

```
┌─────────────────────────────────────────────────────────┐
│ Iteration N: Prompt Optimization（迭代 N：Prompt 优化） │
│                                                          │
│ 1. Analyze current judge_labels quality                 │
│    - Identify error patterns（识别错误模式）            │
│    - Find improvement opportunities（找到改进机会）    │
│                                                          │
│ 2. Optimize prompt                                      │
│    - Refine instructions（细化指令）                    │
│    - Add examples（添加示例）                            │
│    - Improve output format（改进输出格式）              │
│                                                          │
│ 3. Re-run Step 3.5                                      │
│    - Generate new judge_labels（生成新 judge_labels）    │
│    - Validate quality gates（验证质量门禁）             │
│                                                          │
│ 4. If quality gates pass → Continue（如果质量门禁通过） │
│    If fail → Return to step 2（如果失败 → 返回步骤 2） │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Training with Control（带控制的训练）                    │
│                                                          │
│ 1. Split data: 80% train, 20% validation               │
│    - Stratified split（分层划分）                       │
│    - Fixed random seed（固定随机种子）                   │
│                                                          │
│ 2. Train with monitoring                                │
│    - Early stopping（早停）                             │
│    - Checkpoint saving（Checkpoint 保存）                │
│    - Real-time metrics（实时指标）                       │
│                                                          │
│ 3. Evaluate all checkpoints                             │
│    - Validation set（验证集）                            │
│    - Golden set（Golden Set）                            │
│    - Select best（选择最佳）                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Quality Gates Check（质量门禁检查）                     │
│                                                          │
│ 1. Check training quality gates                         │
│    - No regression on validation（验证集无回归）        │
│    - No regression on golden set（Golden Set 无回归）    │
│    - Rule adherence maintained（规则遵循率保持）         │
│                                                          │
│ 2. Check improvement                                   │
│    - At least one metric improves（至少一个指标改进）   │
│    - Minimum improvement threshold（最小改进阈值）       │
│                                                          │
│ 3. If gates pass → Release（如果门禁通过 → 发布）       │
│    If fail → Analyze and iterate（如果失败 → 分析并迭代）│
└─────────────────────────────────────────────────────────┘
```

### Example Iteration（迭代示例）

```python
# Iteration 1: Baseline（迭代 1：基线）
baseline = {
    "prompt": "prompt_v1.2.jinja",
    "judge_labels": "judge_labels_v1.2.jsonl",
    "model": "lora_model_v1.0.pt",
    "performance": {
        "val_silver_agreement": 0.8874,
        "golden_silver_agreement": 0.8874,
        "rule_adherence": 1.0
    }
}

# Iteration 2: Optimize Prompt（迭代 2：优化 Prompt）
# Step 1: Analyze（步骤 1：分析）
analysis = analyze_judge_labels_quality(baseline["judge_labels"])
# Found: Low evidence coverage, inconsistent severity（发现：证据覆盖率低，严重程度不一致）

# Step 2: Optimize prompt（步骤 2：优化 prompt）
optimized_prompt = optimize_prompt(
    baseline["prompt"],
    improvements={
        "add_evidence_requirements": True,
        "add_severity_guidelines": True,
        "add_examples": True
    }
)
# Output: prompt_v1.3.jinja

# Step 3: Re-run Step 3.5（步骤 3：重新运行 Step 3.5）
new_judge_labels = generate_judge_labels(
    prompt="prompt_v1.3.jinja",
    visual_facts=visual_facts,
    rule_labels=rule_labels
)
# Output: judge_labels_v1.3.jsonl

# Step 4: Validate judge_labels quality（步骤 4：验证 judge_labels 质量）
judge_pass, judge_gates = check_judge_labels_quality_gates(
    new_judge_labels, baseline["judge_labels"], rule_labels
)
# Result: ✅ PASS
# Improvements:
#   - consistency: +0.03
#   - evidence_coverage: +0.05
#   - agreement_with_rules: +0.02

# Step 5: Train with new judge_labels（步骤 5：用新 judge_labels 训练）
new_model = train_with_monitoring(
    training_data=rule_labels + new_judge_labels,
    val_data=val_data,
    callbacks=[EarlyStoppingCallback(patience=3)]
)
# Output: lora_model_v1.1.pt

# Step 6: Evaluate（步骤 6：评估）
new_performance = {
    "val_silver_agreement": 0.905,  # +1.76%
    "golden_silver_agreement": 0.902,  # +1.48%
    "rule_adherence": 1.0  # Maintained（保持）
}

# Step 7: Check training quality gates（步骤 7：检查训练质量门禁）
training_pass, training_gates = check_training_quality_gates(
    new_model, baseline["model"], val_data, golden_set
)
# Result: ✅ PASS
# Improvements:
#   - val_silver_agreement: +1.76%
#   - golden_silver_agreement: +1.48%

# Step 8: Release（步骤 8：发布）
if training_pass:
    release_model(new_model, "lora_model_v1.1.pt")
    print("✅ Model v1.1 released successfully")
```

---

## Regression Set Management（回归集管理）

### 1. Build Regression Set（构建回归集）

```python
def build_regression_set():
    """
    Build regression set from historical bad cases（从历史坏案例构建回归集）
    """
    regression_cases = []
    
    # 1. All historical bad cases（所有历史坏案例）
    historical_bad_cases = load_historical_bad_cases()
    regression_cases.extend(historical_bad_cases)
    
    # 2. All judge≠rule controversial cases（所有 judge≠rule 争议案例）
    controversial_cases = find_judge_rule_disagreements(judge_labels, rule_labels)
    regression_cases.extend(controversial_cases)
    
    # 3. All doctor-marked wrong cases (if have UI data)（所有医生标记错误的案例（如果有 UI 数据））
    if has_ui_feedback_data():
        doctor_wrong_cases = load_doctor_marked_wrong_cases()
        regression_cases.extend(doctor_wrong_cases)
    
    # Deduplicate（去重）
    regression_set = deduplicate_cases(regression_cases)
    
    # Save as fixed artifact（保存为固定 artifact）
    save_regression_set(regression_set, "regression_set_v1.0.jsonl")
    
    return regression_set
```

### 2. Regression Set Evolution（回归集进化）

```python
def evolve_regression_set(new_prompt_version, old_prompt_version):
    """
    Evolve regression set based on prompt changes（基于 Prompt 变化进化回归集）
    """
    # Compare judge_labels from two prompt versions（比较两个 Prompt 版本的 judge_labels）
    old_judge_labels = load_judge_labels(f"judge_labels_{old_prompt_version}.jsonl")
    new_judge_labels = load_judge_labels(f"judge_labels_{new_prompt_version}.jsonl")
    
    # Find changed cases（找到变化的案例）
    changed_cases = find_changed_cases(old_judge_labels, new_judge_labels)
    
    # Categorize changes（分类变化）
    changes = {
        "became_more_conservative": [],  # 变得更保守
        "became_more_aggressive": [],  # 变得更激进
        "issue_type_changed": [],  # 问题类型改变
        "severity_changed": []  # 严重程度改变
    }
    
    for case in changed_cases:
        old_label = old_judge_labels[case.case_id]
        new_label = new_judge_labels[case.case_id]
        
        if old_label.severity == "low" and new_label.severity == "high":
            changes["became_more_aggressive"].append(case)
        elif old_label.severity == "high" and new_label.severity == "low":
            changes["became_more_conservative"].append(case)
        # ... more categorization
    
    # Add to regression set（添加到回归集）
    regression_set = load_regression_set("regression_set_v1.0.jsonl")
    
    # Add changed cases as hard examples（添加变化案例作为困难样本）
    regression_set.extend(changes["became_more_conservative"])
    regression_set.extend(changes["became_more_aggressive"])
    
    # Save updated regression set（保存更新的回归集）
    save_regression_set(regression_set, "regression_set_v1.1.jsonl")
    
    return regression_set
```

## Prompt Unit Tests（Prompt 单元测试）

### 1. Prompt Unit Test Framework（Prompt 单元测试框架）

```python
class PromptUnitTests:
    """
    Unit tests for prompt（Prompt 单元测试）
    """
    def __init__(self):
        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self):
        """
        Load minimal test cases（加载最小测试用例）
        """
        return [
            {
                "name": "laterality_conflict_must_red",
                "input": {
                    "visual_facts": [{"type": "effusion", "laterality": "left", "confidence": 0.95}],
                    "report_facts": [{"entity": "effusion", "text": "right effusion", "span_ref": {"start": 10, "end": 25}}]
                },
                "expected": {
                    "issue_type": "laterality_mismatch",
                    "severity": "high"  # Must be high（必须是 high）
                }
            },
            {
                "name": "measurement_unit_error_must_yellow_or_red",
                "input": {
                    "visual_facts": [{"type": "measurement", "value": 5.0, "unit": "cm"}],
                    "report_facts": [{"entity": "measurement", "text": "5 inches", "span_ref": {"start": 30, "end": 40}}]
                },
                "expected": {
                    "issue_type": "measurement_error",
                    "severity": "med"  # Must be med or high（必须是 med 或 high）
                }
            },
            {
                "name": "empty_facts_must_review",
                "input": {
                    "visual_facts": [],
                    "report_facts": []
                },
                "expected": {
                    "action": "review"  # Must be review（必须是 review）
                }
            }
        ]
    
    def run_tests(self, prompt_path, judge_model):
        """
        Run all unit tests（运行所有单元测试）
        """
        results = []
        
        for test_case in self.test_cases:
            # Generate prediction（生成预测）
            prediction = generate_prediction(
                prompt_path=prompt_path,
                judge_model=judge_model,
                input_data=test_case["input"]
            )
            
            # Check if prediction matches expected（检查预测是否匹配期望）
            passed = self.check_prediction(prediction, test_case["expected"])
            
            results.append({
                "test_name": test_case["name"],
                "passed": passed,
                "expected": test_case["expected"],
                "actual": prediction
            })
        
        # All tests must pass（所有测试必须通过）
        all_pass = all(result["passed"] for result in results)
        
        return {
            "all_pass": all_pass,
            "results": results,
            "message": "All unit tests passed" if all_pass else "Some unit tests failed"
        }
```

### 2. One-Change Rule（单次改动规则）

```python
def apply_one_change_rule(prompt_changes):
    """
    Enforce one-change rule（强制执行单次改动规则）
    """
    change_categories = [
        "taxonomy_definition",  # Taxonomy 定义
        "severity_rubric",  # Severity rubric
        "abstain_review_threshold",  # Abstain/review 阈值
        "evidence_reference_format"  # Evidence 引用格式
    ]
    
    changes_made = []
    for category in change_categories:
        if category in prompt_changes:
            changes_made.append(category)
    
    if len(changes_made) > 1:
        raise ValueError(
            f"One-change rule violated: {len(changes_made)} changes detected. "
            f"Only one change allowed per iteration. Changes: {changes_made}"
        )
    
    return changes_made[0] if changes_made else None
```

## Complete Pipeline Integration（完整 Pipeline 集成）

### Step 5: Evaluation with Release Gates（Step 5：带 Release Gate 的评估）

```python
def step5_evaluation_with_release_gates(
    model_path,
    prompt_version,
    baseline_model_path,
    baseline_prompt_version
):
    """
    Step 5: Evaluation with industrial-grade release gates
    Step 5：带工业级 Release Gate 的评估
    """
    print(f"Step 5: Evaluation with Release Gates")
    print(f"  New: prompt {prompt_version}, model {model_path}")
    print(f"  Baseline: prompt {baseline_prompt_version}, model {baseline_model_path}")
    
    # Load models（加载模型）
    new_model = load_model(model_path)
    baseline_model = load_model(baseline_model_path)
    
    # Load prompt artifacts（加载 Prompt artifacts）
    new_prompt_artifact = PromptArtifact(f"prompt_{prompt_version}.jinja")
    baseline_prompt_artifact = PromptArtifact(f"prompt_{baseline_prompt_version}.jinja")
    
    # Step 1: Run unit tests（步骤 1：运行单元测试）
    print("\nStep 1: Running Prompt Unit Tests...")
    unit_test_results = PromptUnitTests().run_tests(
        prompt_path=f"prompt_{prompt_version}.jinja",
        judge_model=judge_model
    )
    
    if not unit_test_results["all_pass"]:
        print("❌ Unit tests failed. Blocking release.")
        return {
            "status": "BLOCKED",
            "reason": "unit_tests_failed",
            "details": unit_test_results
        }
    
    print("✅ Unit tests passed")
    
    # Step 2: Evaluate on Golden Set（步骤 2：在 Golden Set 上评估）
    print("\nStep 2: Evaluating on Golden Set...")
    golden_metrics = evaluate_model(new_model, golden_set)
    baseline_golden_metrics = evaluate_model(baseline_model, golden_set)
    
    print(f"  Silver Agreement: {baseline_golden_metrics['silver_agreement']:.4f} → {golden_metrics['silver_agreement']:.4f}")
    
    # Step 3: Check Release Gates（步骤 3：检查 Release Gate）
    print("\nStep 3: Checking Release Gates...")
    release_gate = IndustrialReleaseGate()
    gate_results = release_gate.check_release_gates(
        new_model=new_model,
        baseline_model=baseline_model,
        prompt_version=prompt_version,
        model_version=new_model.version
    )
    
    # Step 4: Check Regression Set（步骤 4：检查回归集）
    print("\nStep 4: Checking Regression Set...")
    regression_results = release_gate.check_regression_set(new_model, baseline_model)
    
    # Step 5: Final decision（步骤 5：最终决定）
    print("\nStep 5: Final Decision...")
    
    all_checks_pass = (
        unit_test_results["all_pass"] and
        gate_results["all_pass"] and
        regression_results["pass"]
    )
    
    if all_checks_pass:
        print("✅ All checks passed. RELEASE approved.")
        
        # Generate eval report（生成评估报告）
        eval_report = generate_eval_report(
            model=new_model,
            prompt_artifact=new_prompt_artifact,
            eval_results={
                "golden_metrics": golden_metrics,
                "baseline_golden_metrics": baseline_golden_metrics,
                "gate_results": gate_results,
                "regression_results": regression_results
            }
        )
        
        return {
            "status": "APPROVED",
            "recommendation": "RELEASE",
            "eval_report": eval_report
        }
    else:
        print("❌ Some checks failed. BLOCKING release.")
        
        return {
            "status": "BLOCKED",
            "reason": "release_gates_failed",
            "details": {
                "unit_tests": unit_test_results,
                "release_gates": gate_results,
                "regression_set": regression_results
            }
        }
```

## Best Practices（最佳实践）

### 1. Prompt Optimization Best Practices（Prompt 优化最佳实践）

#### Analyze Before Optimizing（优化前分析）

```python
# Always analyze current judge_labels before optimizing prompt
# 优化 prompt 前总是先分析当前 judge_labels

def analyze_before_optimization(judge_labels, rule_labels):
    """
    Comprehensive analysis before prompt optimization（Prompt 优化前的全面分析）
    """
    analysis = {
        "error_patterns": identify_error_patterns(judge_labels),
        "missing_evidence": find_missing_evidence(judge_labels),
        "inconsistent_severity": find_inconsistent_severity(judge_labels),
        "low_agreement_cases": find_low_agreement(judge_labels, rule_labels)
    }
    
    # Generate optimization suggestions（生成优化建议）
    suggestions = generate_optimization_suggestions(analysis)
    
    return analysis, suggestions
```

#### Incremental Optimization（增量优化）

```python
# Don't change everything at once（不要一次性改变所有内容）
# Make small, focused improvements（做小的、聚焦的改进）

# Bad（不好）:
# Change entire prompt structure（改变整个 prompt 结构）

# Good（好）:
# Iteration 1: Add evidence requirements（迭代 1：添加证据要求）
# Iteration 2: Add severity guidelines（迭代 2：添加严重程度指南）
# Iteration 3: Add examples（迭代 3：添加示例）
```

### 2. Training Control Best Practices（训练控制最佳实践）

#### Always Use Validation Set（总是使用验证集）

```python
# Never train without validation set（永远不要在没有验证集的情况下训练）
train_data, val_data = create_validation_set(dataset, validation_ratio=0.2)

# Always evaluate on validation set during training（训练时总是在验证集上评估）
training_args = TrainingArguments(
    eval_dataset=val_data,
    evaluation_strategy="steps",
    eval_steps=500
)
```

#### Monitor Multiple Metrics（监控多个指标）

```python
# Don't just monitor loss（不要只监控 loss）
# Monitor performance metrics（监控性能指标）

metrics_to_monitor = [
    "eval_loss",  # Training quality（训练质量）
    "eval_silver_agreement",  # Main performance metric（主要性能指标）
    "eval_rule_adherence",  # Must maintain（必须保持）
    "eval_judge_rule_gap"  # Consistency（一致性）
]
```

#### Set Clear Improvement Thresholds（设置清晰的改进阈值）

```python
improvement_thresholds = {
    "minimum_improvement": 0.005,  # At least 0.5% improvement（至少 0.5% 改进）
    "regression_tolerance": 0.02,  # Allow 2% regression before blocking（允许 2% 回归才阻止）
    "rule_adherence_tolerance": 0.01  # Rule adherence can drop max 1%（规则遵循率最多下降 1%）
}
```

### 3. Ensuring Continuous Improvement（确保持续改进）

#### Iteration Control（迭代控制）

```python
def controlled_iteration(
    iteration_number,
    baseline_model,
    baseline_metrics,
    prompt_template,
    train_data,
    val_data,
    golden_set
):
    """
    Controlled iteration with quality gates（带质量门禁的受控迭代）
    """
    print(f"Starting iteration {iteration_number}")
    
    # Step 1: Generate judge_labels（步骤 1：生成 judge_labels）
    judge_labels = generate_judge_labels(prompt_template, train_data)
    
    # Step 2: Validate judge_labels（步骤 2：验证 judge_labels）
    judge_pass, judge_metrics = validate_judge_labels(judge_labels)
    if not judge_pass:
        return None, "judge_labels_validation_failed"
    
    # Step 3: Train（步骤 3：训练）
    model = train_with_control(
        rule_labels + judge_labels,
        val_data,
        baseline_model  # Start from baseline（从基线开始）
    )
    
    # Step 4: Evaluate（步骤 4：评估）
    new_metrics = evaluate_model(model, val_data, golden_set)
    
    # Step 5: Check improvement（步骤 5：检查改进）
    improvement = check_improvement(new_metrics, baseline_metrics)
    
    if improvement["has_improvement"] and not improvement["has_regression"]:
        print(f"✅ Iteration {iteration_number} successful")
        print(f"   Improvements: {improvement['details']}")
        return model, "success"
    else:
        print(f"❌ Iteration {iteration_number} failed")
        print(f"   Reasons: {improvement['failure_reasons']}")
        return baseline_model, "no_improvement"  # Keep baseline（保持基线）
```

#### Automatic Rollback（自动回滚）

```python
# If new model doesn't improve, automatically use baseline
# 如果新模型没有改进，自动使用基线

def safe_training_with_rollback(new_config, baseline_model):
    """
    Train with automatic rollback if no improvement（如果没有改进，自动回滚的训练）
    """
    new_model = train(new_config)
    new_metrics = evaluate(new_model)
    baseline_metrics = evaluate(baseline_model)
    
    if is_improvement(new_metrics, baseline_metrics):
        return new_model, "improved"
    else:
        print("⚠️ No improvement, rolling back to baseline")
        return baseline_model, "rolled_back"
```

---

## Summary（总结）

### Key Takeaways（关键要点）

1. ✅ **Prompt/Rubric/Contracts are Highest Leverage（Prompt/Rubric/Contracts 是最高杠杆）**
   - Dataset fixed, but judge_labels improve（数据集固定，但 judge_labels 改进）
   - Better training data → Better model（更好的训练数据 → 更好的模型）
   - **ROI 最高**：成本低，影响大

2. ✅ **Control Release Gate, Not Training（控制 Release Gate，不是训练）**
   - **Fixed Golden Set**：固定 Golden Set（不可变）
   - **Fixed Metric Definitions**：固定指标口径（不可变）
   - **Fixed Release Gates**：固定 Release Gate（不可变）
   - **No Regression, No Release**：不准退步就不发布

3. ✅ **Prompt as First-Class Artifact（Prompt 作为一等 Artifact）**
   - **Version tracking**：prompt_version, prompt_hash, judge_model_version
   - **Full logging**：记录在 eval report 中
   - **Reproducible**：可以完全重现

4. ✅ **Three-Layer Release Gates（三层 Release Gate）**
   - **Gate A: Safety Slice**：高风险类别不能退步
   - **Gate B: Evidence Grounding**：防止幻觉，确保引用真实
   - **Gate C: User Experience**：误报率不能上升
   - **Any gate fails → BLOCK**：任何一道门禁失败就阻止发布

5. ✅ **Two-Stage Generation（两段式生成）**
   - **Pass 1: Decision JSON**：只输出结构化判决
   - **Pass 2: Explanation**：基于 Pass 1 生成解释
   - **Benefits**：Schema 成功率更高，幻觉更少，评测更稳定

6. ✅ **One-Change Rule（单次改动规则）**
   - **Only one change per iteration**：每次迭代只改一件事
   - **Know what improved**：知道改进来自哪里
   - **Easier to debug**：更容易调试

7. ✅ **Regression Set（回归集）**
   - **Historical bad cases**：所有历史坏案例
   - **Controversial cases**：所有争议案例
   - **Doctor feedback**：医生反馈的错误案例
   - **Rule: No regression on regression set**：规则：不能在回归集上退步

8. ✅ **Prompt Unit Tests（Prompt 单元测试）**
   - **Minimal test cases**：最小测试用例（20 个）
   - **Fast feedback**：快速反馈（比全量快很多）
   - **Fail fast**：快速失败

### Industrial-Grade Implementation Checklist（工业级实现检查清单）

#### Minimum Viable Implementation（最小可行实现）

- [ ] **Fixed Golden Set**：固定 Golden Set（golden_set_v1.0.jsonl）
- [ ] **Regression Set**：回归集（regression_set_v1.0.jsonl，至少包含 43 个争议样本）
- [ ] **Three Release Gates**：三道 Release Gate（Safety / Evidence / UX）
- [ ] **One-Change Rule**：单次改动规则（每次只改一件事）
- [ ] **Prompt Versioning**：Prompt 版本化（prompt_version + prompt_hash）
- [ ] **Two-Stage Generation**：两段式生成（Decision JSON → Explanation）
- [ ] **Unit Tests**：单元测试（20 个最小用例）
- [ ] **Automatic Rollback**：自动回滚（如果失败，自动回滚到基线）

#### Complete Implementation（完整实现）

- [ ] **Release Gate Configuration**：Release Gate 配置（YAML 文件）
- [ ] **Prompt Artifact Management**：Prompt Artifact 管理（完整版本跟踪）
- [ ] **Eval Report Generation**：评估报告生成（带完整版本信息）
- [ ] **Regression Set Evolution**：回归集进化（基于 Prompt 变化）
- [ ] **Pipeline Integration**：Pipeline 集成（Step 5 集成 Release Gate）

### Expected Results（预期结果）

**With this implementation（通过这个实现）**：

✅ **Monotonic Improvement**：持续提升（不会倒退）
✅ **Controlled Iteration**：受控迭代（不准退步就不发布）
✅ **Reproducible**：可复现（所有版本信息完整记录）
✅ **Confident Release**：自信发布（通过所有门禁才发布）

**Quote（引用）**：
> "我们通过 prompt-driven post-train + strict release gates，保证系统迭代不会回归，只会上线更安全的版本。"

### Quick Reference（快速参考）

| Step | Action | Control Mechanism |
|------|--------|------------------|
| **1. Prompt Optimization** | Optimize prompt | judge_labels quality gates |
| **2. Generate judge_labels** | Re-run Step 3.5 | Validate consistency, evidence coverage |
| **3. Training** | Train model | Validation set, early stopping |
| **4. Evaluation** | Evaluate model | Quality gates, improvement check |
| **5. Release** | Release if pass | Automatic rollback if regression |

---

**Remember**: Prompt optimization + Training control = Continuous improvement without dataset changes.
