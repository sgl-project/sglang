# A05: Post-Training Mechanism
# A05: 训练后持续改进机制

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Post-Training Continuous Improvement Mechanism  
**Key Principle**: Feedback Loop is the Engine for Scale-Up

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Feedback Collection Mechanism](#feedback-collection-mechanism)
3. [Feedback Data Classification](#feedback-data-classification)
4. [Model Improvement Strategies](#model-improvement-strategies)
5. [Post-Training Pipeline](#post-training-pipeline)
6. [Evaluation & Release](#evaluation--release)
7. [Best Practices](#best-practices)

---

## Overview

### Core Principle

**"Feedback Loop is the Engine for Scale-Up"**

Post-training is not a one-time process, but a continuous improvement cycle that uses production feedback to refine the model, rules, and prompts.

### Post-Training Cycle

```
Production Environment (On-Prem)
    ↓
Doctor Actions (Accept/Ignore/Review)
    ↓
Feedback Data Collection
    ↓
Feedback Analysis & Classification
    ↓
Model Improvement (SFT/DPO/Rule Optimization)
    ↓
Re-training Pipeline (AWS)
    ↓
New Model Version Release
    ↓
Gradual Rollout to Production
    ↓
(Repeat Cycle)
```

### Key Components

1. **Feedback Collection**: Collect doctor actions from production
2. **Data Classification**: Categorize feedback (Accept/Ignore/Review)
3. **Model Improvement**: Apply appropriate improvement method
4. **Re-training**: Run training pipeline with feedback data
5. **Evaluation**: Validate improvements on golden set + feedback validation set
6. **Release**: Gradual rollout with monitoring

---

## Feedback Collection Mechanism

### Production Feedback Sources

#### 1. Doctor Actions in Review UI

**Three Action Types**:

```json
{
  "case_id": "case_001",
  "issue_id": "issue_001",
  "action": "adopt" | "ignore" | "edit",
  "reason": "string (required for ignore)",
  "corrected_content": "string (if edit)",
  "reviewer_id": "doctor_001",
  "timestamp": "2025-01-01T10:00:00Z",
  "model_version": "lora_model_v1.0",
  "original_prediction": {
    "issue_type": "laterality_mismatch",
    "severity": "high",
    "confidence": 0.95,
    "supporting_facts": ["vf_001", "rf_002"]
  },
  "doctor_feedback": {
    "action": "ignore",
    "reason": "False positive - visual fact confidence too low",
    "corrected_severity": null
  }
}
```

#### 2. Feedback Data Schema

```python
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class FeedbackRecord(BaseModel):
    """Feedback record from production"""
    
    # Case identification
    case_id: str
    issue_id: str
    
    # Doctor action
    action: str  # "adopt" | "ignore" | "edit"
    reason: Optional[str]  # Required for "ignore"
    corrected_content: Optional[str]  # If "edit"
    
    # Reviewer info
    reviewer_id: str
    reviewer_role: str  # "doctor" | "expert" | "supervisor"
    timestamp: datetime
    
    # Model version tracking
    model_version: str  # "lora_model_v1.0"
    prompt_version: str  # "prompt_v1.2"
    rule_version: str  # "rules_v1.0"
    
    # Original prediction
    original_prediction: dict
    
    # Context (for analysis)
    visual_facts: List[dict]
    report_facts: List[dict]
    report_text: str
    
    # Metadata
    department: str
    exam_type: str
    metadata: dict
```

### Feedback Storage

#### On-Prem Storage (Production)

```yaml
PostgreSQL Database:
  Table: feedback_records
  Columns:
    - case_id (PK)
    - issue_id (PK)
    - action
    - reason
    - reviewer_id
    - timestamp
    - model_version
    - original_prediction (JSONB)
    - doctor_feedback (JSONB)
  
  Indexes:
    - (case_id, issue_id)
    - (action, timestamp)
    - (model_version, timestamp)
    - (reviewer_id, timestamp)
```

#### S3 Storage (Training)

```bash
s3://rad-linter-data/feedback/
├── accept_cases/
│   └── v1.0/
│       ├── accept_cases_20250101.jsonl
│       ├── accept_cases_20250102.jsonl
│       └── metadata.json
│
├── ignore_cases/
│   └── v1.0/
│       ├── ignore_cases_20250101.jsonl
│       ├── ignore_reasons_analysis.json
│       └── metadata.json
│
└── review_cases/
    └── v1.0/
        ├── review_cases_20250101.jsonl
        ├── preference_pairs.jsonl
        └── metadata.json
```

### Feedback Collection Process

```python
# Production Environment (On-Prem)
def collect_feedback(case_id: str, issue_id: str, action: str, 
                     reason: Optional[str] = None):
    """
    Collect feedback from production review UI
    """
    feedback = FeedbackRecord(
        case_id=case_id,
        issue_id=issue_id,
        action=action,
        reason=reason,
        reviewer_id=get_current_doctor_id(),
        timestamp=datetime.now(),
        model_version=get_current_model_version(),
        original_prediction=get_original_prediction(case_id, issue_id),
        visual_facts=get_visual_facts(case_id),
        report_facts=get_report_facts(case_id)
    )
    
    # Store in PostgreSQL
    save_to_database(feedback)
    
    # Queue for S3 sync (async)
    queue_for_s3_sync(feedback)
    
    return feedback

# Periodic S3 Sync (Daily)
def sync_feedback_to_s3():
    """
    Sync feedback data from PostgreSQL to S3 (daily batch)
    """
    # Query feedback from last 24 hours
    feedback_records = query_feedback_since(yesterday)
    
    # Classify by action type
    accept_cases = [f for f in feedback_records if f.action == "adopt"]
    ignore_cases = [f for f in feedback_records if f.action == "ignore"]
    review_cases = [f for f in feedback_records if f.action == "edit"]
    
    # Upload to S3
    upload_to_s3("accept_cases", accept_cases)
    upload_to_s3("ignore_cases", ignore_cases)
    upload_to_s3("review_cases", review_cases)
```

---

## Feedback Data Classification

### Classification Matrix

| Action | Data Type | Training Method | Purpose |
|--------|-----------|----------------|---------|
| **Accept** | Positive Examples | SFT (Supervised Fine-Tuning) | Reinforce correct predictions |
| **Ignore** | False Positives | Rule Optimization | Reduce false positive rate |
| **Review** | Expert Annotations | DPO/GRPO (Preference Learning) | Learn from expert corrections |

### 1. Accept Cases (Positive Examples)

**Characteristics**:
- Doctor agrees with system prediction
- System prediction is correct
- High confidence cases

**Data Format**:
```json
{
  "input": {
    "visual_facts": [...],
    "report_facts": [...],
    "report_text": "..."
  },
  "output": {
    "issues": [
      {
        "issue_type": "laterality_mismatch",
        "severity": "high",
        "supporting_facts": ["vf_001", "rf_002"],
        "recommended_action": "block"
      }
    ],
    "confidence": 0.95
  },
  "label": "positive",
  "doctor_action": "adopt",
  "case_id": "case_001"
}
```

**Usage**: Supervised Fine-Tuning (SFT)

### 2. Ignore Cases (False Positives)

**Characteristics**:
- Doctor disagrees with system prediction
- System prediction is incorrect (false positive)
- Low confidence or edge cases

**Data Format**:
```json
{
  "case_id": "case_002",
  "issue_id": "issue_002",
  "original_prediction": {
    "issue_type": "missing_measurement",
    "severity": "med",
    "confidence": 0.75
  },
  "doctor_action": "ignore",
  "reason": "Not relevant for this exam type",
  "analysis": {
    "false_positive_type": "rule_too_strict",
    "suggested_action": "adjust_threshold"
  }
}
```

**Usage**: Rule Optimization (threshold adjustment, rule filtering)

### 3. Review Cases (Expert Annotations)

**Characteristics**:
- Doctor provides expert correction
- High-quality preference data
- Used for preference learning

**Data Format**:
```json
{
  "input": {
    "visual_facts": [...],
    "report_facts": [...]
  },
  "preferred": {
    "issues": [
      {
        "issue_type": "contradiction",
        "severity": "high",
        "explanation": "Expert-corrected explanation"
      }
    ]
  },
  "rejected": {
    "issues": [
      {
        "issue_type": "contradiction",
        "severity": "med",
        "explanation": "Original model explanation"
      }
    ]
  },
  "case_id": "case_003"
}
```

**Usage**: DPO (Direct Preference Optimization) / GRPO

---

## Model Improvement Strategies

### Strategy 1: Supervised Fine-Tuning (SFT)

**Use Case**: Accept cases (positive examples)

**Training Data Preparation**:
```python
def prepare_sft_data(accept_cases: List[FeedbackRecord]):
    """
    Prepare SFT training data from accept cases
    """
    sft_data = []
    
    for case in accept_cases:
        # Input: visual_facts + report_facts
        input_data = {
            "visual_facts": case.visual_facts,
            "report_facts": case.report_facts,
            "report_text": case.report_text
        }
        
        # Output: doctor-accepted prediction
        output_data = case.original_prediction
        
        sft_data.append({
            "input": input_data,
            "output": output_data,
            "case_id": case.case_id
        })
    
    return sft_data
```

**Training Configuration**:
```yaml
SFT Training Config:
  Method: LoRA Fine-Tuning
  Base Model: qwen2.5-vl-7b (4-bit quantized)
  LoRA Rank: 16
  Learning Rate: 1e-4  # Lower than initial training
  Epochs: 3-5
  Batch Size: 8
  Data: Accept cases (1000+ examples)
  Loss Function: Cross-Entropy
  Evaluation: 
    - Golden Set
    - Feedback validation set (held-out accept cases)
```

**Training Command**:
```bash
python train_sft.py \
  --base_model qwen2.5-vl-7b \
  --lora_model_path s3://rad-linter-data/models/lora_model_v1.0/lora_model_v1.0.pt \
  --train_data s3://rad-linter-data/feedback/accept_cases/v1.0/ \
  --output_dir s3://rad-linter-data/models/lora_model_v1.1/ \
  --learning_rate 1e-4 \
  --epochs 5 \
  --batch_size 8
```

### Strategy 2: Direct Preference Optimization (DPO)

**Use Case**: Review cases (expert annotations)

**Training Data Preparation**:
```python
def prepare_dpo_data(review_cases: List[FeedbackRecord]):
    """
    Prepare DPO training data from review cases
    """
    dpo_data = []
    
    for case in review_cases:
        # Input: visual_facts + report_facts
        input_data = {
            "visual_facts": case.visual_facts,
            "report_facts": case.report_facts
        }
        
        # Preferred: expert-corrected output
        preferred = case.corrected_content or case.doctor_feedback.get("preferred")
        
        # Rejected: original model output
        rejected = case.original_prediction
        
        dpo_data.append({
            "input": input_data,
            "preferred": preferred,
            "rejected": rejected,
            "case_id": case.case_id
        })
    
    return dpo_data
```

**Training Configuration**:
```yaml
DPO Training Config:
  Method: Direct Preference Optimization
  Base Model: qwen2.5-vl-7b (4-bit quantized)
  LoRA Rank: 16
  Learning Rate: 5e-6  # Much lower for DPO
  Epochs: 2-3
  Batch Size: 4
  Data: Review cases (500+ preference pairs)
  Beta: 0.1  # DPO temperature parameter
  Evaluation:
    - Preference accuracy (preferred > rejected)
    - Golden Set performance
```

**Training Command**:
```bash
python train_dpo.py \
  --base_model qwen2.5-vl-7b \
  --lora_model_path s3://rad-linter-data/models/lora_model_v1.0/lora_model_v1.0.pt \
  --train_data s3://rad-linter-data/feedback/review_cases/v1.0/preference_pairs.jsonl \
  --output_dir s3://rad-linter-data/models/lora_model_v1.1/ \
  --learning_rate 5e-6 \
  --epochs 3 \
  --batch_size 4 \
  --beta 0.1
```

### Strategy 3: Rule Optimization (Non-Model Training)

**Use Case**: Ignore cases (false positives)

**Analysis Process**:
```python
def analyze_ignore_cases(ignore_cases: List[FeedbackRecord]):
    """
    Analyze ignore cases to identify false positive patterns
    """
    # Group by reason
    reason_distribution = {}
    for case in ignore_cases:
        reason = case.reason
        reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
    
    # Analyze patterns
    patterns = {
        "false_positive": [],
        "rule_too_strict": [],
        "not_relevant": [],
        "edge_case": []
    }
    
    for case in ignore_cases:
        if "false positive" in case.reason.lower():
            patterns["false_positive"].append(case)
        elif "not relevant" in case.reason.lower():
            patterns["not_relevant"].append(case)
        # ... more pattern matching
    
    return patterns
```

**Rule Optimization Actions**:

```yaml
Rule Optimization Strategies:

  1. Threshold Adjustment:
     Before:
       laterality_check_threshold: 0.9
     
     After (based on ignore analysis):
       laterality_check_threshold: 0.95  # Increase threshold
    
  2. Rule Filtering:
     Before:
       check_missing_measurement:
         enabled: true
         applies_to: ["all"]
     
     After:
       check_missing_measurement:
         enabled: true
         applies_to: ["chest_xray", "ct_chest"]  # Filter by exam type
         exclude_departments: ["emergency"]  # Exclude edge cases
  
  3. Rule Disabling:
     Before:
       check_style_suggestion:
         enabled: true
     
     After (if high ignore rate):
       check_style_suggestion:
         enabled: false  # Temporarily disable
         reason: "High ignore rate (15%)"
```

**Rule Update Process**:
```bash
# 1. Analyze ignore cases
python analyze_ignore_cases.py \
  --input s3://rad-linter-data/feedback/ignore_cases/v1.0/ \
  --output ignore_analysis.json

# 2. Generate rule updates
python generate_rule_updates.py \
  --analysis ignore_analysis.json \
  --current_rules rules_v1.0.py \
  --output rules_v1.1.py

# 3. Test rule updates
python test_rules.py \
  --rules rules_v1.1.py \
  --test_data golden_set_v1.0.jsonl

# 4. Deploy rule updates
# Update rules_v1.1.py in production
```

---

## Post-Training Pipeline

### Complete Post-Training Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Feedback Collection (On-Prem, Weeks 1-4)      │
│                                                          │
│ • Deploy model v1.0 to production                       │
│ • Shadow Mode (提示但不拦截)                            │
│ • Collect Accept/Ignore/Review feedback                 │
│ • Target: 1000+ feedback cases                          │
│ • Daily sync to S3                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Feedback Analysis (AWS, Week 5)                │
│                                                          │
│ • Download feedback data from S3                        │
│ • Data cleaning and validation                          │
│ • Classify feedback (Accept/Ignore/Review)              │
│ • Analyze false positive patterns                       │
│ • Identify improvement opportunities                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Improvement Strategy (AWS, Week 6-7)          │
│                                                          │
│ Strategy A: Rule Optimization                           │
│   - Analyze ignore cases                                │
│   - Update rules_v1.0.py → rules_v1.1.py              │
│                                                          │
│ Strategy B: Prompt Optimization                         │
│   - Analyze review cases                                │
│   - Update prompt_v1.2.jinja → prompt_v1.3.jinja       │
│                                                          │
│ Strategy C: Model Re-training                           │
│   - Prepare SFT data (Accept cases)                     │
│   - Prepare DPO data (Review cases)                      │
│   - Update train_config_v1.0.yaml → train_config_v1.1.yaml│
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Re-training Pipeline (AWS, Week 8-9)          │
│                                                          │
│ Step 0-1: Data Preprocessing                            │
│   - Include feedback data                               │
│   - Merge with original training data                   │
│                                                          │
│ Step 2: Visual Feature Extraction                       │
│   - Cache hit (skip if no new images)                  │
│                                                          │
│ Step 3: Rule-Based Label Generation                     │
│   - Use new rules_v1.1.py                              │
│                                                          │
│ Step 3.5: LLM Judge Label Generation                    │
│   - Use new prompt_v1.3.jinja                          │
│   - Cache hit for existing cases                        │
│                                                          │
│ Step 4: LoRA Training                                   │
│   - SFT: Accept cases (1000+ examples)                  │
│   - DPO: Review cases (500+ preference pairs)           │
│   - Base: lora_model_v1.0.pt                           │
│   - Output: lora_model_v1.1.pt                         │
│                                                          │
│ Step 5: Evaluation                                      │
│   - Golden Set (baseline)                               │
│   - Feedback validation set (held-out feedback)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Evaluation & Release (AWS, Week 10)            │
│                                                          │
│ • Evaluate new model performance                        │
│ • Release Gate Checks:                                  │
│   - Rule Adherence > 99%                                │
│   - Silver Agreement > 85%                               │
│   - Feedback validation set improvement                  │
│   - No regression on golden set                         │
│                                                          │
│ • If PASS:                                              │
│   - Release to S3 Model Registry (v1.1)                │
│   - Generate release notes                              │
│                                                          │
│ • If FAIL:                                              │
│   - Analyze failure reasons                             │
│   - Return to Phase 3 (improvement strategy)            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 6: Gradual Rollout (On-Prem, Week 11+)           │
│                                                          │
│ • Download new model from S3                            │
│ • Shadow Mode (1 week)                                  │
│ • Small-scale pilot (1% → 5% → 25%)                    │
│ • Monitor metrics:                                      │
│   - Automation Rate                                     │
│   - False Positive Rate                                 │
│   - Doctor Satisfaction                                 │
│ • Full rollout (100%)                                  │
│ • (Repeat cycle)                                        │
└─────────────────────────────────────────────────────────┘
```

### Post-Training Pipeline Script

```bash
#!/bin/bash
# post_train_pipeline.sh

# Configuration
FEEDBACK_VERSION="v1.0"
MODEL_VERSION="v1.1"
BASE_MODEL_VERSION="v1.0"
S3_BUCKET="rad-linter-data"
S3_REGION="us-east-1"

# Phase 1: Download feedback data
echo "Phase 1: Downloading feedback data..."
aws s3 sync s3://${S3_BUCKET}/feedback/${FEEDBACK_VERSION}/ \
  ./data/feedback/${FEEDBACK_VERSION}/ \
  --region ${S3_REGION}

# Phase 2: Analyze feedback
echo "Phase 2: Analyzing feedback..."
python analyze_feedback.py \
  --input ./data/feedback/${FEEDBACK_VERSION}/ \
  --output ./analysis/feedback_analysis.json

# Phase 3: Prepare training data
echo "Phase 3: Preparing training data..."
python prepare_sft_data.py \
  --input ./data/feedback/${FEEDBACK_VERSION}/accept_cases/ \
  --output ./data/training/sft_data.jsonl

python prepare_dpo_data.py \
  --input ./data/feedback/${FEEDBACK_VERSION}/review_cases/ \
  --output ./data/training/dpo_data.jsonl

# Phase 4: Re-training
echo "Phase 4: Re-training model..."

# SFT Training
python train_sft.py \
  --base_model qwen2.5-vl-7b \
  --lora_model_path s3://${S3_BUCKET}/models/lora_model_${BASE_MODEL_VERSION}/lora_model_${BASE_MODEL_VERSION}.pt \
  --train_data ./data/training/sft_data.jsonl \
  --output_dir ./models/lora_model_${MODEL_VERSION}/ \
  --learning_rate 1e-4 \
  --epochs 5 \
  --batch_size 8

# DPO Training (optional, if review cases available)
if [ -f "./data/training/dpo_data.jsonl" ]; then
  python train_dpo.py \
    --base_model qwen2.5-vl-7b \
    --lora_model_path ./models/lora_model_${MODEL_VERSION}/lora_model_${MODEL_VERSION}.pt \
    --train_data ./data/training/dpo_data.jsonl \
    --output_dir ./models/lora_model_${MODEL_VERSION}/ \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 4
fi

# Phase 5: Evaluation
echo "Phase 5: Evaluating model..."
python evaluate_model.py \
  --model_path ./models/lora_model_${MODEL_VERSION}/lora_model_${MODEL_VERSION}.pt \
  --golden_set s3://${S3_BUCKET}/eval/golden_set_v1.0/golden_set_v1.0.jsonl \
  --feedback_validation_set ./data/feedback/${FEEDBACK_VERSION}/feedback_validation_set.jsonl \
  --output ./eval/results_${MODEL_VERSION}.json

# Phase 6: Release Gate Check
echo "Phase 6: Release gate check..."
python check_release_gates.py \
  --eval_results ./eval/results_${MODEL_VERSION}.json \
  --baseline_version ${BASE_MODEL_VERSION}

# If pass, upload to S3 Model Registry
if [ $? -eq 0 ]; then
  echo "Release gates passed. Uploading to S3 Model Registry..."
  aws s3 cp ./models/lora_model_${MODEL_VERSION}/ \
    s3://${S3_BUCKET}/models/registry/${MODEL_VERSION}/ \
    --recursive \
    --region ${S3_REGION}
  
  echo "Model ${MODEL_VERSION} released successfully!"
else
  echo "Release gates failed. Please review evaluation results."
  exit 1
fi
```

---

## Evaluation & Release

### Evaluation Metrics

#### 1. Golden Set Performance (Baseline)

```yaml
Golden Set Metrics:
  Rule Adherence: > 99% (must maintain)
  Silver Agreement: > 85% (must maintain or improve)
  Judge-Rule Gap: Not significantly worse
  Performance Regression: < 20%
```

#### 2. Feedback Validation Set Performance

```yaml
Feedback Validation Set Metrics:
  Accept Case Accuracy: > 90% (SFT improvement)
  Preference Accuracy: > 80% (DPO improvement)
  False Positive Rate: < 5% (rule optimization)
  Ignore Rate Reduction: > 20% (compared to baseline)
```

#### 3. Production Metrics (Post-Deployment)

```yaml
Production Metrics (After Rollout):
  Automation Rate: > 85%
  False Positive Rate: < 5%
  Doctor Satisfaction: > 85%
  Model Performance: No regression
```

### Release Gate Checks

```python
def check_release_gates(eval_results: dict, baseline_version: str):
    """
    Check if new model passes all release gates
    """
    gates = {
        "rule_adherence": {
            "threshold": 0.99,
            "current": eval_results["rule_adherence"],
            "status": "PASS" if eval_results["rule_adherence"] >= 0.99 else "FAIL"
        },
        "silver_agreement": {
            "threshold": 0.85,
            "current": eval_results["silver_agreement"],
            "baseline": get_baseline_metric(baseline_version, "silver_agreement"),
            "status": "PASS" if eval_results["silver_agreement"] >= 0.85 and \
                              eval_results["silver_agreement"] >= get_baseline_metric(baseline_version, "silver_agreement") * 0.95 \
                         else "FAIL"
        },
        "feedback_validation": {
            "threshold": 0.80,
            "current": eval_results["feedback_validation_accuracy"],
            "status": "PASS" if eval_results["feedback_validation_accuracy"] >= 0.80 else "FAIL"
        },
        "performance_regression": {
            "threshold": 0.20,
            "current": calculate_regression(eval_results, baseline_version),
            "status": "PASS" if calculate_regression(eval_results, baseline_version) < 0.20 else "FAIL"
        }
    }
    
    # All gates must pass
    all_pass = all(gate["status"] == "PASS" for gate in gates.values())
    
    return {
        "all_pass": all_pass,
        "gates": gates,
        "recommendation": "RELEASE" if all_pass else "BLOCK"
    }
```

### Gradual Rollout Strategy

```yaml
Rollout Phases:

  Phase 1: Shadow Mode (Week 1)
    - Deploy new model alongside old model
    - Run predictions but don't show to doctors
    - Compare predictions between v1.0 and v1.1
    - Monitor for anomalies
    
  Phase 2: Small-Scale Pilot (Week 2-3)
    - 1% traffic → 5% traffic → 25% traffic
    - Show predictions to doctors
    - Monitor metrics:
      - Automation Rate
      - False Positive Rate
      - Doctor Satisfaction
    - Collect feedback
    
  Phase 3: Full Rollout (Week 4+)
    - 100% traffic
    - Monitor continuously
    - Collect feedback for next iteration
```

---

## Best Practices

### 1. Feedback Data Quality

**Minimum Requirements**:
- **Accept cases**: 1000+ examples for SFT
- **Review cases**: 500+ preference pairs for DPO
- **Ignore cases**: 500+ examples for rule analysis
- **Data diversity**: Cover all issue types, departments, exam types

**Data Quality Checks**:
```python
def validate_feedback_data(feedback_data: List[FeedbackRecord]):
    """
    Validate feedback data quality
    """
    checks = {
        "minimum_samples": len(feedback_data) >= 1000,
        "diversity": check_diversity(feedback_data),
        "completeness": all(has_required_fields(f) for f in feedback_data),
        "consistency": check_consistency(feedback_data)
    }
    
    return all(checks.values())
```

### 2. Model Improvement Prioritization

**Priority Order**:
1. **Rule Optimization** (fastest, lowest risk)
   - Analyze ignore cases
   - Update rules
   - Deploy immediately (no retraining needed)
   
2. **Prompt Optimization** (medium speed, medium risk)
   - Analyze review cases
   - Update prompt
   - Re-run Step 3.5 only
   
3. **Model Re-training** (slowest, highest risk)
   - SFT: Accept cases
   - DPO: Review cases
   - Full pipeline re-run

### 3. Feedback Collection Timeline

**Recommended Timeline**:
- **Week 1-4**: Collect feedback (target: 1000+ cases)
- **Week 5**: Analyze feedback
- **Week 6-7**: Prepare improvements
- **Week 8-9**: Re-training
- **Week 10**: Evaluation & release
- **Week 11+**: Gradual rollout

**Minimum Feedback Threshold**:
- Before starting post-training: **1000+ feedback cases**
- Before model re-training: **500+ accept cases, 200+ review cases**
- Before rule optimization: **200+ ignore cases**

### 4. Version Management

**Versioning Strategy**:
```yaml
Model Versions:
  v1.0: Initial training
  v1.1: First post-training (SFT + DPO)
  v1.2: Second post-training (rule optimization + prompt update)
  v2.0: Major architecture change

Rule Versions:
  v1.0: Initial rules
  v1.1: First optimization (threshold adjustment)
  v1.2: Second optimization (rule filtering)

Prompt Versions:
  v1.2: Initial prompt
  v1.3: First optimization
  v1.4: Second optimization
```

### 5. Monitoring & Alerting

**Key Metrics to Monitor**:
```yaml
Post-Training Metrics:

  Feedback Collection:
    - Daily feedback count
    - Action distribution (Accept/Ignore/Review)
    - Feedback quality score
    
  Model Improvement:
    - Training loss (SFT/DPO)
    - Validation accuracy
    - Golden set performance
    
  Production Performance:
    - Automation Rate
    - False Positive Rate
    - Doctor Satisfaction
    - Model latency (P95/P99)
```

**Alerting Rules**:
```yaml
Alerts:
  - Feedback collection rate drops < 50 cases/day
  - Model performance regression > 5%
  - False positive rate increases > 2%
  - Doctor satisfaction drops < 80%
```

---

## Summary

### Key Takeaways

1. ✅ **Feedback Loop is Engine**: Doctor feedback is the most valuable post-train data
2. ✅ **Three Feedback Types**: Accept (SFT), Ignore (Rule Optimization), Review (DPO)
3. ✅ **Iterative Improvement**: Continuous cycle of collect → analyze → improve → release
4. ✅ **Gradual Rollout**: Shadow Mode → Small-scale → Full rollout
5. ✅ **Version Management**: Track all versions (model, rules, prompt)

### Post-Training Checklist

- [ ] Collect 1000+ feedback cases from production
- [ ] Classify feedback (Accept/Ignore/Review)
- [ ] Analyze false positive patterns
- [ ] Prepare training data (SFT/DPO)
- [ ] Update rules/prompt if needed
- [ ] Run re-training pipeline
- [ ] Evaluate on golden set + feedback validation set
- [ ] Pass release gates
- [ ] Gradual rollout with monitoring
- [ ] Collect new feedback for next iteration

---

## Quick Reference

### Post-Training Commands

```bash
# 1. Download feedback data
aws s3 sync s3://rad-linter-data/feedback/v1.0/ ./data/feedback/

# 2. Analyze feedback
python analyze_feedback.py --input ./data/feedback/

# 3. Prepare training data
python prepare_sft_data.py --input ./data/feedback/accept_cases/
python prepare_dpo_data.py --input ./data/feedback/review_cases/

# 4. Re-train model
python train_sft.py --train_data ./data/training/sft_data.jsonl
python train_dpo.py --train_data ./data/training/dpo_data.jsonl

# 5. Evaluate
python evaluate_model.py --model_path ./models/lora_model_v1.1/

# 6. Release
python check_release_gates.py --eval_results ./eval/results_v1.1.json
```

### Expected Timeline

| Phase | Duration | Key Activities |
|-------|----------|---------------|
| Feedback Collection | 4 weeks | Collect 1000+ cases |
| Analysis | 1 week | Analyze patterns |
| Improvement | 2 weeks | Prepare data, update configs |
| Re-training | 2 weeks | SFT + DPO training |
| Evaluation | 1 week | Release gate checks |
| Rollout | 4+ weeks | Gradual deployment |
| **Total** | **14+ weeks** | **One complete cycle** |

---

**Remember**: Post-training is not a one-time event, but a continuous improvement cycle. The feedback loop is the engine that drives model improvement and scale-up.
