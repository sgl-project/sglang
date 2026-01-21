# A02: Production Training Pipeline
# A02: 工业级训练流程

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Production-Grade Training Pipeline for Rad-Linter

---

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Production Pipeline Architecture](#production-pipeline-architecture)
3. [Version Management & Reproducibility](#version-management--reproducibility)
4. [Key Production Practices](#key-production-practices)
5. [Iteration Workflow](#iteration-workflow)

---

## Pipeline Overview

### Core Principles

**Reproducibility + Version Control + Evaluation Gates**

The production training pipeline ensures:
- ✅ **Reproducibility**: Every run produces identical results
- ✅ **Version Control**: All components are versioned
- ✅ **Evaluation Gates**: Models only released if passing all gates

### Training Steps

1. **Training Launch**: Log versions, generate experiment_id
2. **Step 0-1**: Data Preprocessing (Cached)
3. **Step 2**: Visual Feature Extraction (GPU-Intensive, Cached)
4. **Step 3**: Rule-Based Label Generation (Fast, Deterministic)
5. **Step 3.5**: LLM Judge Label Generation (GPU, High Cost)
6. **Step 4**: LoRA Training (GPU, Longest Duration, Checkpoint)
7. **Step 5**: Evaluation (Golden Set + Release Gate)
8. **Release Gate Decision**: Release only if all gates pass

---

## Production Pipeline Architecture

### Complete Pipeline Structure (AWS-Based)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Launch                          │
│  ./train_pipeline.sh                                                 │
│  • AWS EC2 Instance: g4dn.xlarge (GPU)                              │
│  • Region: us-east-1                                                 │
│  • Log all version numbers (data/model/config)                       │
│  • Generate experiment_id (timestamp + git_commit)                   │
│  • Create experiment tracking record                                 │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 0-1: Data Preprocessing (Cached)                                │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Input: AWS S3 Bucket                                             │ │
│ │ • S3 Bucket: s3://rad-linter-data/                               │ │
│ │ • Path: input/indiana_cxr_v1.0/                                 │ │
│ │ • Format: JSON (indiana_cxr_dataset_v1.0.json)                   │ │
│ │ • Hash: sha256:abc123... (immutable)                             │ │
│ │ • Size: 10,000 cases                                             │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • Intermediate Results: AWS S3                                       │
│   - Path: s3://rad-linter-data/cache/step0-1/                        │
│   - Cache Key: data_v1.0_config_v1.0_hash_abc123/                    │
│ • Metadata: S3 (preprocessing_stats.json)                           │
│                                                                       │
│ Operations:                                                           │
│ • Download from S3 (if not local)                                    │
│ • Check cache (S3 or local cache)                                    │
│ • If cached: skip processing, load from cache                        │
│ • If not cached: execute processing, save to S3                      │
│                                                                       │
│ Output:                                                               │
│ • Standardized dataset (versioned)                                   │
│ • S3 Path: s3://rad-linter-data/processed/standardized_v1.0/         │
│ • Cache: s3://rad-linter-data/cache/step0-1/                        │
│ • Metadata: preprocessing_stats.json (S3)                           │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: Visual Feature Extraction (GPU-Intensive, Cached)            │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Model: TorchXRayVision v1.0 (Docker fixed version)               │ │
│ │ • Container: torchxrayvision:v1.0                                │ │
│ │ • GPU Required: Yes (AWS EC2 GPU instance)                       │ │
│ │ • Instance Type: g4dn.xlarge (NVIDIA T4 GPU)                     │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • FactStore: AWS S3 (versioned fact storage)                         │
│   - Path: s3://rad-linter-data/factstore/visual_facts/              │
│   - Version: visual_facts_v1.0/                                      │
│   - Cache: s3://rad-linter-data/cache/visual_facts/                  │
│                                                                       │
│ Operations:                                                           │
│ • Check FactStore cache in S3 (based on image hash)                  │
│ • Batch processing (parallel extraction on GPU)                      │
│ • Extract: lesions / effusion / fractures / measurements             │
│ • Save to FactStore (S3, versioned)                                  │
│                                                                       │
│ Output:                                                               │
│ • visual_facts_v1.0.jsonl                                            │
│ • S3 Path: s3://rad-linter-data/factstore/visual_facts/v1.0/         │
│ • Cache Key: visual_facts_{image_hash}_{model_v1.0} (S3)             │
│ • Statistics: extraction_stats.json (S3)                             │
│ • Timing: ~5s/image (P95)                                            │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3: Rule-Based Label Generation (Fast, Deterministic)            │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Rules: rules_v1.0.py                                             │ │
│ │ • Laterality checks                                              │ │
│ │ • Measurement consistency                                        │ │
│ │ • Required fields                                                │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • Rule Labels: AWS S3                                                │
│   - Path: s3://rad-linter-data/labels/rule_labels/                   │
│   - Version: rule_labels_v1.0.jsonl                                  │
│                                                                       │
│ Operations:                                                           │
│ • Download visual_facts from S3                                      │
│ • Execute rule engine (CPU, unit-testable)                           │
│ • Generate deterministic labels                                      │
│ • Track rule coverage statistics                                     │
│ • Upload results to S3                                               │
│                                                                       │
│ Output:                                                               │
│ • rule_labels_v1.0.jsonl                                             │
│ • S3 Path: s3://rad-linter-data/labels/rule_labels/v1.0/             │
│ • Rule coverage: coverage_stats.json (S3)                            │
│ • Timing: ~50ms/case (fast)                                          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3.5: LLM Judge Label Generation (GPU, High Cost)                │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Judge Server: SGLang Judge v1.0 (Docker fixed version)           │ │
│ │ • Container: sglang-judge:v1.0                                   │ │
│ │ • Model: qwen2.5-vl-32b                                          │ │
│ │ • Prompt: prompt_v1.2.jinja                                      │ │
│ │ • Instance: AWS EC2 GPU (g4dn.xlarge or g5.xlarge)              │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • Judge Labels: AWS S3                                               │
│   - Path: s3://rad-linter-data/labels/judge_labels/                  │
│   - Version: judge_labels_v1.0.jsonl                                 │
│ • Cache: s3://rad-linter-data/cache/judge_labels/                    │
│                                                                       │
│ Operations:                                                           │
│ • Docker start Judge Server on AWS GPU instance                      │
│ • Download visual_facts and rule_labels from S3                      │
│ • Batch inference (batching for efficiency)                          │
│ • Schema-constrained JSON output                                     │
│ • Bounded retry on parse failure / low confidence                    │
│ • Record cost and token usage                                        │
│ • Identify controversial cases (Judge ≠ Rule)                        │
│ • Upload results to S3                                               │
│                                                                       │
│ Output:                                                               │
│ • judge_labels_v1.0.jsonl                                            │
│ • S3 Path: s3://rad-linter-data/labels/judge_labels/v1.0/            │
│ • Cost tracking: judge_cost_stats.json (S3)                          │
│ • Token usage: ~1000 tokens/case                                     │
│ • Timing: ~30s/case (P95)                                            │
│ • Controversial cases: 43 cases (Judge ≠ Rule)                       │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 4: LoRA Training (GPU, Longest Duration)                        │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Training Config: train_config_v1.0.yaml                          │ │
│ │ • Base Model: qwen2.5-vl-32b (fixed version)                     │ │
│ │ • LoRA rank: 16                                                  │ │
│ │ • Learning rate: 2e-4                                            │ │
│ │ • Batch size: 4                                                  │ │
│ │ • Epochs: 5                                                      │ │
│ │ • Random seed: 42 (fixed)                                        │ │
│ │ • Instance: AWS EC2 GPU (g4dn.xlarge / p3.2xlarge)              │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • Training Data: Downloaded from S3 to local (EBS volume)            │
│ • Model Checkpoints: AWS S3                                          │
│   - Path: s3://rad-linter-data/models/checkpoints/                   │
│   - Format: checkpoints/epoch_{N}/lora_model_epoch_{N}.pt           │
│ • Training Logs: S3                                                  │
│   - Path: s3://rad-linter-data/models/training_logs/                 │
│ • TensorBoard Logs: S3                                               │
│   - Path: s3://rad-linter-data/models/tensorboard/                   │
│                                                                       │
│ Operations:                                                           │
│ • Download training data from S3 to local EBS volume                 │
│ • Load pre-trained base model (from S3 or local cache)               │
│ • Initialize LoRA adapters                                          │
│ • Training loop with monitoring:                                     │
│   - Loss tracking (TensorBoard)                                      │
│   - Accuracy tracking                                               │
│   - Learning rate scheduling                                         │
│ • Checkpoint saving (every N epochs) → Upload to S3                  │
│ • Early stopping (if validation loss plateaus)                       │
│ • Upload final model and logs to S3                                  │
│                                                                       │
│ Output:                                                               │
│ • lora_model_v1.0.pt                                                │
│ • S3 Path: s3://rad-linter-data/models/lora_model_v1.0/              │
│ • Training logs: training_logs_v1.0.json (S3)                        │
│ • Checkpoints: s3://rad-linter-data/models/checkpoints/epoch_*/      │
│ • TensorBoard logs: s3://rad-linter-data/models/tensorboard/         │
│ • Timing: ~2 hours (depends on dataset size)                         │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 5: Evaluation (Golden Set + Release Gate)                       │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Golden Set: golden_set_v1.0.jsonl (fixed)                        │ │
│ │ • S3 Path: s3://rad-linter-data/eval/golden_set_v1.0/            │ │
│ │ • Size: 200 cases                                                │ │
│ │ • Coverage: All issue types, all departments                     │ │
│ │ • Hash: sha256:def456... (immutable)                             │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Storage:                                                              │
│ • Evaluation Results: AWS S3                                         │
│   - Path: s3://rad-linter-data/eval/results/                         │
│   - Version: eval_results_v1.0/                                      │
│                                                                       │
│ Three-Panel Evaluation:                                               │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ 1. Rule Adherence: Model vs Rule Labels                         │ │
│ │    • Metric: % agreement                                         │ │
│ │    • Result: 100% ✓ (perfect rule learning)                     │ │
│ │                                                                   │ │
│ │ 2. Silver Agreement: Model vs Judge Labels                      │ │
│ │    • Metric: Accuracy / F1                                       │ │
│ │    • Result: 88.74% accuracy, 80.0% F1                          │ │
│ │                                                                   │ │
│ │ 3. Judge-Rule Gap: Judge vs Rule Labels                         │ │
│ │    • Metric: % agreement                                         │ │
│ │    • Result: 88.74% agreement (43 controversial cases)          │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Release Gate Checks:                                                  │
│ • Rule Adherence > 99% ✓                                            │
│ • Silver Agreement > 85% ✓                                          │
│ • Judge-Rule Gap not significantly worse ✓                          │
│ • Performance regression < 20% ✓                                    │
│                                                                       │
│ Output:                                                               │
│ • eval_report_v1.0.md (S3)                                           │
│ • S3 Path: s3://rad-linter-data/eval/results/v1.0/                   │
│ • Evaluation metrics: eval_metrics_v1.0.json (S3)                    │
│ • Error analysis: error_analysis_v1.0.json (S3)                      │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Release Gate Decision                                                 │
│                                                                       │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ If ALL gates pass:                                                │ │
│ │ • Release model version: lora_model_v1.0                         │ │
│ │ • Tag: v1.0                                                      │ │
│ │ • S3 Model Registry: s3://rad-linter-data/models/registry/       │ │
│ │ • Deploy to production                                            │ │
│ │ • Generate release notes                                          │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ If ANY gate fails:                                                │ │
│ │ • Block release                                                   │ │
│ │ • Generate failure analysis report (S3)                           │ │
│ │ • Return to iteration (modify config/rules/prompt)               │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### AWS Infrastructure Details

#### Data Storage: AWS S3

**S3 Bucket Structure**:
```
s3://rad-linter-data/
├── input/
│   └── indiana_cxr_v1.0/
│       └── indiana_cxr_dataset_v1.0.json          # Input JSON data
├── cache/
│   ├── step0-1/                                   # Preprocessing cache
│   ├── visual_facts/                              # Visual facts cache
│   └── judge_labels/                              # Judge labels cache
├── processed/
│   └── standardized_v1.0/                         # Standardized dataset
├── factstore/
│   └── visual_facts/
│       └── v1.0/                                  # Versioned visual facts
├── labels/
│   ├── rule_labels/
│   │   └── v1.0/
│   │       └── rule_labels_v1.0.jsonl
│   └── judge_labels/
│       └── v1.0/
│           └── judge_labels_v1.0.jsonl
├── models/
│   ├── checkpoints/
│   │   └── epoch_*/
│   ├── training_logs/
│   │   └── training_logs_v1.0.json
│   ├── tensorboard/
│   │   └── runs/
│   ├── lora_model_v1.0/
│   │   └── lora_model_v1.0.pt
│   └── registry/
│       └── v1.0/                                  # Model registry
└── eval/
    ├── golden_set_v1.0/
    │   └── golden_set_v1.0.jsonl
    └── results/
        └── v1.0/
            ├── eval_report_v1.0.md
            ├── eval_metrics_v1.0.json
            └── error_analysis_v1.0.json
```

#### Compute Infrastructure: AWS EC2

**GPU Instance Types**:

| Step | Instance Type | GPU | Use Case |
|------|--------------|-----|----------|
| Step 2: Visual Feature Extraction | g4dn.xlarge | NVIDIA T4 (16GB) | CV model inference |
| Step 3.5: LLM Judge | g4dn.xlarge or g5.xlarge | NVIDIA T4/T4G (16-24GB) | LLM inference |
| Step 4: LoRA Training | g4dn.xlarge / p3.2xlarge | NVIDIA T4/V100 (16-32GB) | Model training |

**Instance Configuration**:
- **Region**: us-east-1 (or your preferred region)
- **EBS Volume**: 100GB+ for local cache and model storage
- **Security Groups**: Allow S3 access, Docker registry access
- **IAM Role**: S3 read/write permissions

#### Data Flow: S3 ↔ EC2

**Download Pattern**:
```
1. Check local cache (EBS volume)
2. If not cached: Download from S3
3. Process on EC2 (GPU/CPU)
4. Upload results to S3
5. Save to local cache (optional)
```

**Upload Pattern**:
```
1. Save intermediate results locally
2. Upload to S3 (versioned path)
3. Update metadata (S3)
4. Optionally keep local copy for next step
```

---

### AWS S3 Bucket Structure

**Complete S3 Bucket Organization**:

```
s3://rad-linter-data/
├── input/
│   └── indiana_cxr_v1.0/
│       └── indiana_cxr_dataset_v1.0.json          # Input JSON data (10K cases)
│
├── cache/
│   ├── step0-1/                                   # Preprocessing cache
│   │   └── data_v1.0_config_v1.0_hash_abc123/
│   ├── visual_facts/                              # Visual facts cache
│   │   └── visual_facts_{image_hash}_{model_v1.0}/
│   └── judge_labels/                              # Judge labels cache
│       └── judge_labels_{facts_hash}_{prompt_v1.2}/
│
├── processed/
│   └── standardized_v1.0/                         # Standardized dataset
│       └── standardized_dataset_v1.0.jsonl
│
├── factstore/
│   └── visual_facts/
│       └── v1.0/
│           ├── visual_facts_v1.0.jsonl
│           ├── metadata.json
│           └── extraction_stats.json
│
├── labels/
│   ├── rule_labels/
│   │   └── v1.0/
│   │       ├── rule_labels_v1.0.jsonl
│   │       └── coverage_stats.json
│   └── judge_labels/
│       └── v1.0/
│           ├── judge_labels_v1.0.jsonl
│           ├── judge_cost_stats.json
│           └── controversial_cases.json
│
├── models/
│   ├── checkpoints/
│   │   └── epoch_*/
│   │       └── lora_model_epoch_{N}.pt
│   ├── training_logs/
│   │   └── v1.0/
│   │       ├── training_logs_v1.0.json
│   │       └── hyperparams_v1.0.json
│   ├── tensorboard/
│   │   └── runs/
│   │       └── exp_20250101_abc123/
│   ├── lora_model_v1.0/
│   │   └── lora_model_v1.0.pt
│   └── registry/
│       └── v1.0/
│           ├── lora_model_v1.0.pt
│           ├── model_metadata.json
│           └── release_notes.md
│
└── eval/
    ├── golden_set_v1.0/
    │   └── golden_set_v1.0.jsonl
    └── results/
        └── v1.0/
            ├── eval_report_v1.0.md
            ├── eval_metrics_v1.0.json
            └── error_analysis_v1.0.json
```

### AWS EC2 Infrastructure

#### GPU Instance Configuration

**Instance Types by Step**:

| Step | Instance Type | GPU | GPU Memory | Use Case |
|------|--------------|-----|------------|----------|
| **Step 2: Visual Feature Extraction** | g4dn.xlarge | NVIDIA T4 | 16GB | CV model inference |
| **Step 3.5: LLM Judge** | g4dn.xlarge or g5.xlarge | NVIDIA T4/T4G | 16-24GB | LLM inference |
| **Step 4: LoRA Training** | g4dn.xlarge / p3.2xlarge | NVIDIA T4/V100 | 16-32GB | Model training |

**Instance Details**:

```yaml
Step 2 - Visual Feature Extraction:
  instance_type: g4dn.xlarge
  vCPU: 4
  GPU: 1x NVIDIA T4 (16GB)
  Memory: 16GB
  EBS Storage: 100GB (for cache)
  Region: us-east-1

Step 3.5 - LLM Judge:
  instance_type: g5.xlarge
  vCPU: 4
  GPU: 1x NVIDIA A10G (24GB)
  Memory: 16GB
  EBS Storage: 100GB (for cache)
  Region: us-east-1

Step 4 - LoRA Training:
  instance_type: p3.2xlarge
  vCPU: 8
  GPU: 1x NVIDIA V100 (16GB)
  Memory: 61GB
  EBS Storage: 500GB (for model checkpoints)
  Region: us-east-1
```

#### Data Flow: S3 ↔ EC2

**Download Pattern (S3 → EC2)**:

```
For each step:
1. Check local cache (EBS volume)
   ├─ If cached: Skip download
   └─ If not cached: Continue to step 2
2. Download from S3 to local EBS volume
   ├─ Input data (JSON format)
   ├─ Visual facts (JSONL format)
   └─ Labels (JSONL format)
3. Process on EC2 (GPU/CPU)
4. Save intermediate results locally
5. Upload results to S3 (versioned path)
6. Optionally keep local copy for next step
```

**Upload Pattern (EC2 → S3)**:

```
For each step:
1. Save intermediate results locally (EBS volume)
2. Upload to S3 with versioned path:
   ├─ s3://rad-linter-data/{step_name}/v{version}/
   └─ Format: {output_name}_v{version}.{ext}
3. Update metadata (S3):
   ├─ metadata.json (version info, timestamps)
   └─ stats.json (processing statistics)
4. Update cache entry (S3):
   └─ cache/{step_name}/{cache_key}/
```

#### AWS Services Integration

**S3 Configuration**:
```yaml
Bucket: rad-linter-data
Region: us-east-1
Storage Class: Standard (for active data), Glacier (for archives)
Encryption: AES-256
Versioning: Enabled (for model registry)
Lifecycle Policy:
  - Delete incomplete multipart uploads after 7 days
  - Move old versions to Glacier after 90 days
  - Delete old versions after 1 year
```

**IAM Role Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::rad-linter-data/*",
        "arn:aws:s3:::rad-linter-data"
      ]
    }
  ]
}
```

**EBS Volume Configuration**:
```yaml
Volume Type: gp3 (General Purpose SSD)
Size: 500GB (for training instance)
IOPS: 3000 (baseline)
Throughput: 125 MB/s
Encryption: Enabled
Backup: Daily snapshots
```

#### Cost Optimization

**S3 Cost Optimization**:
- Use **S3 Intelligent-Tiering** for variable access patterns
- Use **S3 Lifecycle Policies** to move old data to Glacier
- Enable **S3 Transfer Acceleration** for large file transfers
- Use **S3 Multipart Upload** for large files (>100MB)

**EC2 Cost Optimization**:
- Use **Spot Instances** for Step 2 and Step 3.5 (interruptible)
- Use **On-Demand Instances** for Step 4 (training, cannot interrupt)
- Enable **Auto-Stop** for instances when not in use
- Use **Reserved Instances** for long-term training

**Cache Strategy for Cost Savings**:
- High cache hit rate reduces S3 download costs
- Local EBS cache reduces S3 API calls
- Estimated cost savings: 40-60% with cache

---

## Version Management & Reproducibility

### Versioning Strategy

**Every Step Must Be Versioned**:

```
Step 0-1: Data Versioning
├─ Input Data: Indiana_CXR_v1.0 (fixed hash)
├─ Subset Config: subset_config_v1.0.yaml
└─ Format Alignment: openi_format_v1.0.json

Step 2: Visual Feature Versioning
├─ Model Version: TorchXRayVision_v1.0 (Docker fixed)
├─ Extract Config: extract_config_v1.0.yaml
└─ Output Version: visual_facts_v1.0.jsonl

Step 3: Rule Versioning
├─ Rule File: rules_v1.0.py
├─ Rule Config: rule_config_v1.0.yaml
└─ Label Version: rule_labels_v1.0.jsonl

Step 3.5: Judge Versioning
├─ Judge Server: sglang_judge_v1.0 (Docker fixed version)
├─ Prompt Version: prompt_v1.2.jinja
└─ Judge Labels: judge_labels_v1.0.jsonl

Step 4: Training Versioning
├─ Training Config: train_config_v1.0.yaml
├─ Hyperparameters: hyperparams_v1.0.json
├─ Random Seed: seed=42 (fixed)
└─ Model Output: lora_model_v1.0.pt

Step 5: Evaluation Versioning
├─ Eval Config: eval_config_v1.0.yaml
├─ Golden Set: golden_set_v1.0.jsonl (fixed)
└─ Eval Report: eval_report_v1.0.md
```

### Reproducibility Guarantees

1. **Fixed Random Seeds**: `seed=42` for all random operations
2. **Docker Fixed Versions**: All containers use fixed tags
3. **Input Data Hashing**: Immutable input data with hash verification
4. **Config Versioning**: All configs are versioned and tracked
5. **Environment Locking**: Exact Python packages, CUDA versions, etc.

---

## Key Production Practices

### A. Experiment Tracking

**Track Every Experiment**:

```python
{
  "experiment_id": "exp_20250101_abc123",
  "git_commit": "abc123def456",
  "timestamp": "2025-01-01T10:00:00Z",
  "versions": {
    "data": "v1.0",
    "torchxrayvision": "v1.0",
    "rules": "v1.0",
    "judge_server": "v1.0",
    "prompt": "v1.2",
    "train_config": "v1.0"
  },
  "metrics": {
    "rule_adherence": 1.0,
    "silver_agreement": 0.8874,
    "judge_rule_gap": 0.8874
  }
}
```

**Tracking Tools**:
- TensorBoard: Training metrics visualization
- MLflow: Experiment tracking and model registry
- Custom tracking: Experiment database

### B. Caching Mechanism

**Avoid Redundant Computation**:

| Step | Cache Key | Cache Hit Rate | Speedup |
|------|-----------|----------------|---------|
| Step 0-1 | `hash(input) + hash(config)` | ~80% | 10x |
| Step 2 | `hash(image) + model_version` | ~60% | 20x |
| Step 3.5 | `hash(visual_facts) + hash(report_facts) + prompt_version` | ~40% | 5x |

**Cache Strategy**:
- Input hash + config hash → Output
- If cache hit, skip computation
- If cache miss, compute and save

### C. Release Gate Mechanism

**All Gates Must Pass**:

```python
release_gates = {
    "rule_adherence": {
        "threshold": 0.99,
        "current": 1.0,
        "status": "PASS"
    },
    "silver_agreement": {
        "threshold": 0.85,
        "current": 0.8874,
        "status": "PASS"
    },
    "judge_rule_gap": {
        "threshold": "not_significantly_worse",
        "current": 0.8874,
        "baseline": 0.8874,
        "status": "PASS"
    },
    "performance_regression": {
        "threshold": 0.20,  # < 20% regression
        "current": 0.0,      # No regression
        "status": "PASS"
    }
}

# All gates must pass
if all(gate["status"] == "PASS" for gate in release_gates.values()):
    release_model()
else:
    block_release()
```

**Gate Enforcement**:
- Automatic: No manual override
- Documented: Failure reasons recorded
- Actionable: Clear next steps on failure

### D. Monitoring & Alerting

**Real-Time Monitoring**:

```
GPU Metrics:
├─ GPU Utilization: > 80% target
├─ GPU Memory: < 90% threshold
└─ GPU Temperature: < 85°C

Training Metrics:
├─ Loss trending: Monitor for anomalies
├─ Learning rate: Track schedule
└─ Gradient norms: Check for explosion/vanish

Service Metrics:
├─ Judge Service: Health checks
├─ Cache hit rate: Performance indicator
└─ Error rate: < 1% threshold
```

**Alerting Rules**:
- GPU OOM → Immediate alert
- Training loss anomaly → Warning
- Judge service failure → Critical alert
- Evaluation regression → Block release

### E. Version Rollback

**Rollback Capability**:

```
Model Registry:
├─ lora_model_v1.0 (current)
├─ lora_model_v0.9 (previous)
└─ lora_model_v0.8 (archive)

Rollback Process:
1. Identify problematic version
2. Verify rollback target
3. Deploy previous version
4. Verify functionality
5. Document rollback reason
```

**Version Compatibility**:
- Maintain compatibility matrix
- Support A/B testing
- Track version dependencies

---

## Iteration Workflow

### Standard Iteration Flow

```
1. Modify Config/Rules/Prompt
   ↓
2. Update Version Numbers
   ↓
3. Re-run Pipeline (Leverage Cache for Speedup)
   ↓
4. Evaluate on Golden Set
   ↓
5. Release Gate Check
   ├─ Pass → Release New Version
   └─ Fail → Analyze Failure → Return to Step 1
```

### Example Iteration

**Scenario**: Improve Silver Agreement from 88.74% to 90%

```
1. Modify:
   - Update prompt: prompt_v1.2 → prompt_v1.3
   - Adjust training config: increase epochs from 5 to 7

2. Update Versions:
   - prompt_version: v1.3
   - train_config: v1.1
   - expected_output: lora_model_v1.1

3. Re-run Pipeline:
   - Step 0-1: Cache hit (skip)
   - Step 2: Cache hit (skip)
   - Step 3: Execute (fast, ~50ms/case)
   - Step 3.5: Execute with new prompt (GPU, ~30s/case)
   - Step 4: Train with new config (GPU, ~3 hours)
   - Step 5: Evaluate on Golden Set

4. Results:
   - Silver Agreement: 90.5% ✓
   - Other metrics: Within thresholds ✓

5. Release Gate: PASS → Deploy v1.1
```

---

## Summary: Key Production Training Principles

1. ✅ **Version Everything**: Data/Model/Config must be versioned
2. ✅ **Reproducibility**: Fixed random seed, Docker fixed versions
3. ✅ **Caching Mechanism**: Avoid redundant computation, accelerate iteration
4. ✅ **Evaluation Gates**: Must pass gates to release
5. ✅ **Experiment Tracking**: Record all experiments and results
6. ✅ **Monitoring & Alerting**: Real-time monitoring of training process
7. ✅ **Version Rollback**: Support rollback and A/B testing

---

## Quick Reference

### Training Launch Command (AWS)

```bash
# On AWS EC2 GPU instance
./train_pipeline.sh \
  --s3_bucket rad-linter-data \
  --s3_region us-east-1 \
  --data_version v1.0 \
  --rules_version v1.0 \
  --judge_version v1.0 \
  --prompt_version v1.2 \
  --train_config v1.0 \
  --golden_set v1.0 \
  --experiment_id exp_$(date +%Y%m%d)_$(git rev-parse --short HEAD) \
  --instance_type p3.2xlarge \
  --local_cache_dir /mnt/data/cache
```

### S3 Data Operations

**Download from S3**:
```bash
# Step 0-1: Download input JSON data
aws s3 cp s3://rad-linter-data/input/indiana_cxr_v1.0/indiana_cxr_dataset_v1.0.json \
  ./data/input/ --region us-east-1

# Step 2: Download visual facts (if cached)
aws s3 sync s3://rad-linter-data/cache/visual_facts/ \
  ./cache/visual_facts/ --region us-east-1

# Step 3.5: Download rule labels
aws s3 cp s3://rad-linter-data/labels/rule_labels/v1.0/rule_labels_v1.0.jsonl \
  ./data/labels/rule_labels/ --region us-east-1
```

**Upload to S3**:
```bash
# Step 0-1: Upload processed data
aws s3 cp ./data/processed/standardized_dataset_v1.0.jsonl \
  s3://rad-linter-data/processed/standardized_v1.0/ --region us-east-1

# Step 2: Upload visual facts (JSONL)
aws s3 cp ./data/factstore/visual_facts_v1.0.jsonl \
  s3://rad-linter-data/factstore/visual_facts/v1.0/ --region us-east-1

# Step 4: Upload model checkpoints
aws s3 sync ./checkpoints/epoch_*/ \
  s3://rad-linter-data/models/checkpoints/ --region us-east-1

# Step 4: Upload final model
aws s3 cp ./models/lora_model_v1.0.pt \
  s3://rad-linter-data/models/lora_model_v1.0/ --region us-east-1
```

### Expected Timeline (AWS)

| Step | Duration | Cached? | S3 Transfer | Total Time |
|------|----------|---------|-------------|------------|
| Step 0-1 | 10 min | Yes (80% hit) | 2 min (download JSON) | 10-12 min |
| Step 2 | 2 hours | Yes (60% hit) | 5 min (download) | 2-2.5 hours |
| Step 3 | 5 min | No | 1 min (download) | 6 min |
| Step 3.5 | 4 hours | Yes (40% hit) | 10 min (download) | 4-4.5 hours |
| Step 4 | 2-3 hours | No | 15 min (upload checkpoints) | 2.25-3.25 hours |
| Step 5 | 30 min | No | 2 min (download) | 32 min |
| **Total** | **8-10 hours** | **With cache: 4-6 hours** | **~35 min** | **8.5-10.5 hours** |

### Storage Details by Step

**Step 0-1: Data Preprocessing**
- **Input**: S3 (`s3://rad-linter-data/input/indiana_cxr_v1.0/indiana_cxr_dataset_v1.0.json`)
- **Format**: JSON (10,000 cases, ~500MB)
- **Output**: S3 (`s3://rad-linter-data/processed/standardized_v1.0/`)
- **Cache**: S3 (`s3://rad-linter-data/cache/step0-1/`)

**Step 2: Visual Feature Extraction**
- **Input**: S3 (processed data from Step 0-1)
- **Model**: TorchXRayVision v1.0 (Docker on AWS GPU instance)
- **Output**: S3 (`s3://rad-linter-data/factstore/visual_facts/v1.0/`)
- **Format**: JSONL (~1GB for 10K cases)
- **Cache**: S3 (`s3://rad-linter-data/cache/visual_facts/`)

**Step 3: Rule-Based Label Generation**
- **Input**: S3 (visual_facts from Step 2)
- **Output**: S3 (`s3://rad-linter-data/labels/rule_labels/v1.0/`)
- **Format**: JSONL (~100MB for 10K cases)
- **Processing**: CPU only (no GPU needed)

**Step 3.5: LLM Judge Label Generation**
- **Input**: S3 (visual_facts + rule_labels)
- **Model**: SGLang Judge Server (Docker on AWS GPU instance)
- **Output**: S3 (`s3://rad-linter-data/labels/judge_labels/v1.0/`)
- **Format**: JSONL (~200MB for 10K cases)
- **Cache**: S3 (`s3://rad-linter-data/cache/judge_labels/`)

**Step 4: LoRA Training**
- **Input**: S3 (labels from Step 3 + 3.5)
- **Model**: LoRA on base model (qwen2.5-vl-32b)
- **Checkpoints**: S3 (`s3://rad-linter-data/models/checkpoints/`)
- **Output**: S3 (`s3://rad-linter-data/models/lora_model_v1.0/`)
- **Format**: PyTorch model file (~500MB)
- **Logs**: S3 (`s3://rad-linter-data/models/training_logs/v1.0/`)

**Step 5: Evaluation**
- **Input**: S3 (Golden Set + trained model)
- **Output**: S3 (`s3://rad-linter-data/eval/results/v1.0/`)
- **Format**: Markdown report + JSON metrics (~10MB)

### AWS Cost Estimate

**S3 Storage Costs** (Monthly):
- Input data (JSON): 500MB × $0.023/GB = $0.01
- Processed data: 500MB × $0.023/GB = $0.01
- Visual facts (JSONL): 1GB × $0.023/GB = $0.02
- Labels (JSONL): 300MB × $0.023/GB = $0.01
- Model checkpoints: 5GB × $0.023/GB = $0.12
- Final model: 500MB × $0.023/GB = $0.01
- **Total Storage**: ~8GB × $0.023/GB = **~$0.18/month**

**S3 Transfer Costs**:
- Data transfer IN: Free (within same region)
- Data transfer OUT: ~100GB/month × $0.09/GB = **~$9/month**
- API requests: Negligible (< $0.01/month)

**EC2 Costs** (On-Demand per Training Run):
- g4dn.xlarge (Step 2): $0.526/hour × 2 hours = **$1.05**
- g5.xlarge (Step 3.5): $1.006/hour × 4 hours = **$4.02**
- p3.2xlarge (Step 4): $3.06/hour × 2.5 hours = **$7.65**
- **Total per Training Run**: **~$12.72**

**EBS Storage Costs**:
- 500GB gp3: 500GB × $0.08/GB/month = **$40/month**
- Snapshots: ~$5/month