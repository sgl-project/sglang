# A03: Training Pipeline Server Configuration
# A03: 训练流程服务器配置

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：Detailed server configuration for each training pipeline step

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Server Configuration by Step](#server-configuration-by-step)
3. [Data Flow Between Servers](#data-flow-between-servers)
4. [Infrastructure Summary](#infrastructure-summary)

---

## Overview

### Training Pipeline Steps and Servers

Each step of the training pipeline runs on specific servers with optimized configurations:

| Step | Server Type | Instance Type | GPU Required | Data Storage |
|------|------------|--------------|--------------|--------------|
| **Training Launch** | AWS EC2 | t3.medium | No | S3 |
| **Step 0-1: Data Preprocessing** | AWS EC2 | t3.large | No | S3 + Local Cache |
| **Step 2: Visual Feature Extraction** | AWS EC2 GPU | g4dn.xlarge | Yes (NVIDIA T4) | S3 + FactStore |
| **Step 3: Rule-Based Label Generation** | AWS EC2 | t3.large | No | S3 |
| **Step 3.5: LLM Judge Label Generation** | AWS EC2 GPU | g5.xlarge | Yes (NVIDIA A10G) | S3 + Local Cache |
| **Step 4: LoRA Training** | AWS EC2 GPU | p3.2xlarge | Yes (NVIDIA V100) | S3 + Local Checkpoints |
| **Step 5: Evaluation** | AWS EC2 | t3.large | No | S3 |
| **Release Gate Decision** | AWS EC2 | t3.medium | No | S3 |

---

## Server Configuration by Step

### Training Launch

**Server**: AWS EC2 (t3.medium)

**Configuration**:
```yaml
Instance Type: t3.medium
vCPU: 2
Memory: 4GB
Storage: 20GB EBS (gp3)
GPU: No
Region: us-east-1
```

**Operations**:
- Orchestrate the entire training pipeline
- Log all version numbers (data/model/config)
- Generate experiment_id (timestamp + git_commit)
- Create experiment tracking record in S3
- Monitor pipeline progress

**Storage**:
- **S3**: Experiment tracking records
  - Path: `s3://rad-linter-data/experiments/exp_{id}/`
  - Files: experiment_config.json, experiment_logs.json

**Network**:
- Access to S3 bucket
- IAM role for S3 read/write permissions

---

### Step 0-1: Data Preprocessing

**Server**: AWS EC2 (t3.large)

**Configuration**:
```yaml
Instance Type: t3.large
vCPU: 2
Memory: 8GB
Storage: 100GB EBS (gp3) for local cache
GPU: No
Region: us-east-1
```

**Operations**:
- Download input JSON data from S3
  - Source: `s3://rad-linter-data/input/indiana_cxr_v1.0/indiana_cxr_dataset_v1.0.json`
- Check local cache (EBS volume)
- If not cached: execute data preprocessing
- Upload processed data to S3
- Save to local cache

**Storage**:
- **Input**: S3 (`s3://rad-linter-data/input/indiana_cxr_v1.0/`)
  - Format: JSON (10,000 cases, ~500MB)
- **Output**: S3 (`s3://rad-linter-data/processed/standardized_v1.0/`)
  - Format: JSONL (standardized dataset)
- **Cache**: 
  - S3: `s3://rad-linter-data/cache/step0-1/`
  - Local: `/mnt/data/cache/step0-1/` (EBS volume)

**Key Features**:
- CPU-only processing (no GPU needed)
- Large EBS volume for local cache (reduce S3 downloads)
- Cache hit rate: ~80%

---

### Step 2: Visual Feature Extraction

**Server**: AWS EC2 GPU (g4dn.xlarge)

**Configuration**:
```yaml
Instance Type: g4dn.xlarge
vCPU: 4
Memory: 16GB
GPU: 1x NVIDIA T4 (16GB VRAM)
Storage: 100GB EBS (gp3) for cache
GPU Required: Yes
Region: us-east-1
```

**Operations**:
- Download processed data from S3 (Step 0-1 output)
- Check FactStore cache in S3 (based on image hash)
- If not cached: Run TorchXRayVision model (Docker container)
- Extract visual features: lesions / effusion / fractures / measurements
- Save visual facts to FactStore (S3, versioned)
- Update cache

**Storage**:
- **Input**: S3 (`s3://rad-linter-data/processed/standardized_v1.0/`)
- **Output**: S3 (`s3://rad-linter-data/factstore/visual_facts/v1.0/`)
  - Format: JSONL (`visual_facts_v1.0.jsonl`, ~1GB for 10K cases)
- **Cache**: 
  - S3: `s3://rad-linter-data/cache/visual_facts/`
  - Local: `/mnt/data/cache/visual_facts/` (EBS volume)

**Docker Container**:
- Image: `torchxrayvision:v1.0` (fixed version)
- GPU access: Yes (CUDA enabled)
- Mount volumes: `/mnt/data` (EBS volume)

**Key Features**:
- GPU-accelerated CV model inference
- Batch processing (parallel extraction)
- Cache hit rate: ~60%
- Timing: ~5s/image (P95)

---

### Step 3: Rule-Based Label Generation

**Server**: AWS EC2 (t3.large)

**Configuration**:
```yaml
Instance Type: t3.large
vCPU: 2
Memory: 8GB
Storage: 50GB EBS (gp3)
GPU: No
Region: us-east-1
```

**Operations**:
- Download visual_facts from S3 (Step 2 output)
- Execute rule engine (CPU-only, deterministic)
- Generate rule-based labels
- Track rule coverage statistics
- Upload rule labels to S3

**Storage**:
- **Input**: S3 (`s3://rad-linter-data/factstore/visual_facts/v1.0/`)
- **Output**: S3 (`s3://rad-linter-data/labels/rule_labels/v1.0/`)
  - Format: JSONL (`rule_labels_v1.0.jsonl`, ~100MB for 10K cases)
- **Metadata**: S3 (`coverage_stats.json`)

**Key Features**:
- Fast, deterministic processing (CPU-only)
- Unit-testable rule engine
- Timing: ~50ms/case (fast)
- No GPU required

---

### Step 3.5: LLM Judge Label Generation

**Server**: AWS EC2 GPU (g5.xlarge)

**Configuration**:
```yaml
Instance Type: g5.xlarge
vCPU: 4
Memory: 16GB
GPU: 1x NVIDIA A10G (24GB VRAM)
Storage: 100GB EBS (gp3) for cache
GPU Required: Yes
Region: us-east-1
```

**Operations**:
- Download visual_facts and rule_labels from S3
- Check Judge result cache in S3
- If not cached: Start SGLang Judge Server (Docker container)
- Batch inference (batching for efficiency)
- Schema-constrained JSON output
- Bounded retry on parse failure / low confidence
- Record cost and token usage
- Identify controversial cases (Judge ≠ Rule)
- Upload judge labels to S3

**Storage**:
- **Input**: S3
  - `s3://rad-linter-data/factstore/visual_facts/v1.0/`
  - `s3://rad-linter-data/labels/rule_labels/v1.0/`
- **Output**: S3 (`s3://rad-linter-data/labels/judge_labels/v1.0/`)
  - Format: JSONL (`judge_labels_v1.0.jsonl`, ~200MB for 10K cases)
- **Cache**: 
  - S3: `s3://rad-linter-data/cache/judge_labels/`
  - Local: `/mnt/data/cache/judge_labels/` (EBS volume)
- **Metadata**: S3 (`judge_cost_stats.json`)

**Docker Container**:
- Image: `sglang-judge:v1.0` (fixed version)
- Model: qwen2.5-vl-32b
- Prompt: prompt_v1.2.jinja
- GPU access: Yes (CUDA enabled)
- Mount volumes: `/mnt/data` (EBS volume)

**Key Features**:
- GPU-accelerated LLM inference
- High cost (token usage tracking)
- Cache hit rate: ~40%
- Timing: ~30s/case (P95)
- Token usage: ~1000 tokens/case

---

### Step 4: LoRA Training

**Server**: AWS EC2 GPU (p3.2xlarge)

**Configuration**:
```yaml
Instance Type: p3.2xlarge
vCPU: 8
Memory: 61GB
GPU: 1x NVIDIA V100 (16GB VRAM)
Storage: 500GB EBS (gp3) for checkpoints
GPU Required: Yes
Region: us-east-1
```

**Operations**:
- Download training data from S3 (all labels from Step 3 + 3.5)
- Load pre-trained base model (from S3 or local cache)
- Initialize LoRA adapters
- Training loop with monitoring:
  - Loss tracking (TensorBoard)
  - Accuracy tracking
  - Learning rate scheduling
- Checkpoint saving (every N epochs) → Upload to S3
- Early stopping (if validation loss plateaus)
- Upload final model and logs to S3

**Storage**:
- **Input**: S3
  - `s3://rad-linter-data/labels/rule_labels/v1.0/`
  - `s3://rad-linter-data/labels/judge_labels/v1.0/`
- **Output**: S3 (`s3://rad-linter-data/models/lora_model_v1.0/`)
  - Format: PyTorch model file (`lora_model_v1.0.pt`, ~500MB)
- **Checkpoints**: S3 (`s3://rad-linter-data/models/checkpoints/epoch_*/`)
  - Format: PyTorch checkpoint files (`lora_model_epoch_{N}.pt`)
- **Training Logs**: S3 (`s3://rad-linter-data/models/training_logs/v1.0/`)
  - Files: `training_logs_v1.0.json`, `hyperparams_v1.0.json`
- **TensorBoard Logs**: S3 (`s3://rad-linter-data/models/tensorboard/runs/`)
- **Local Cache**: `/mnt/data/checkpoints/` (EBS volume, 500GB)

**Training Configuration**:
- Base Model: qwen2.5-vl-32b (fixed version)
- LoRA rank: 16
- Learning rate: 2e-4
- Batch size: 4
- Epochs: 5
- Random seed: 42 (fixed)

**Key Features**:
- Longest duration step (~2-3 hours)
- GPU-intensive training
- Large EBS volume for checkpoints (500GB)
- Checkpoint uploads to S3 for backup
- Timing: ~2-3 hours (depends on dataset size)

---

### Step 5: Evaluation

**Server**: AWS EC2 (t3.large)

**Configuration**:
```yaml
Instance Type: t3.large
vCPU: 2
Memory: 8GB
Storage: 50GB EBS (gp3)
GPU: No (or optional for model inference)
Region: us-east-1
```

**Operations**:
- Download Golden Set from S3 (fixed test set)
- Download trained model from S3 (Step 4 output)
- Run three-panel evaluation:
  1. Rule Adherence: Model vs Rule Labels
  2. Silver Agreement: Model vs Judge Labels
  3. Judge-Rule Gap: Judge vs Rule Labels
- Execute release gate checks
- Generate evaluation report
- Upload results to S3

**Storage**:
- **Input**: S3
  - `s3://rad-linter-data/eval/golden_set_v1.0/golden_set_v1.0.jsonl` (200 cases, fixed)
  - `s3://rad-linter-data/models/lora_model_v1.0/lora_model_v1.0.pt`
- **Output**: S3 (`s3://rad-linter-data/eval/results/v1.0/`)
  - Files: 
    - `eval_report_v1.0.md`
    - `eval_metrics_v1.0.json`
    - `error_analysis_v1.0.json`

**Key Features**:
- CPU-based evaluation (no GPU needed, or optional for faster inference)
- Golden Set: Fixed test set (200 cases, immutable hash)
- Three-panel evaluation metrics
- Release gate checks
- Timing: ~30 min

---

### Release Gate Decision

**Server**: AWS EC2 (t3.medium)

**Configuration**:
```yaml
Instance Type: t3.medium
vCPU: 2
Memory: 4GB
Storage: 20GB EBS (gp3)
GPU: No
Region: us-east-1
```

**Operations**:
- Download evaluation results from S3 (Step 5 output)
- Check release gate conditions:
  - Rule Adherence > 99% ✓
  - Silver Agreement > 85% ✓
  - Judge-Rule Gap not significantly worse ✓
  - Performance regression < 20% ✓
- If ALL gates pass:
  - Release model version (tag v1.0)
  - Upload to model registry: `s3://rad-linter-data/models/registry/v1.0/`
  - Generate release notes
  - Deploy to production (trigger deployment)
- If ANY gate fails:
  - Block release
  - Generate failure analysis report
  - Upload failure report to S3
  - Return to iteration (modify config/rules/prompt)

**Storage**:
- **Input**: S3 (`s3://rad-linter-data/eval/results/v1.0/`)
- **Output**: S3 (`s3://rad-linter-data/models/registry/v1.0/`)
  - Files:
    - `lora_model_v1.0.pt`
    - `model_metadata.json`
    - `release_notes.md`
    - `failure_analysis_report.md` (if failed)

**Key Features**:
- Automated decision making
- No manual override (enforced by system)
- Records all decisions in S3

---

## Data Flow Between Servers

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Launch Server (t3.medium)                             │
│  • Orchestrates pipeline                                         │
│  • Monitors progress                                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Triggers
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 0-1 Server (t3.large)                                     │
│  • Downloads: S3 input JSON                                      │
│  • Processes: Data preprocessing                                 │
│  • Uploads: S3 processed data                                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2 Server (g4dn.xlarge GPU)                                │
│  • Downloads: S3 processed data                                  │
│  • Processes: Visual feature extraction (GPU)                    │
│  • Uploads: S3 visual_facts JSONL                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3 Server (t3.large)                                       │
│  • Downloads: S3 visual_facts                                    │
│  • Processes: Rule-based label generation (CPU)                  │
│  • Uploads: S3 rule_labels JSONL                                 │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3.5 Server (g5.xlarge GPU)                                │
│  • Downloads: S3 visual_facts + rule_labels                      │
│  • Processes: LLM Judge label generation (GPU)                   │
│  • Uploads: S3 judge_labels JSONL                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4 Server (p3.2xlarge GPU)                                 │
│  • Downloads: S3 all labels                                      │
│  • Processes: LoRA training (GPU, 2-3 hours)                     │
│  • Uploads: S3 model + checkpoints + logs                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5 Server (t3.large)                                       │
│  • Downloads: S3 Golden Set + trained model                      │
│  • Processes: Evaluation (CPU)                                   │
│  • Uploads: S3 evaluation results                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Output → S3
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Release Gate Server (t3.medium)                                │
│  • Downloads: S3 evaluation results                              │
│  • Processes: Gate decision logic                                │
│  • Uploads: S3 model registry (if pass)                          │
└─────────────────────────────────────────────────────────────────┘
```

### S3 as Central Storage Hub

**All data flows through S3**:

```
S3 Bucket: s3://rad-linter-data/

Step 0-1 → S3: processed/standardized_v1.0/
           ↓
Step 2 → S3: factstore/visual_facts/v1.0/
           ↓
Step 3 → S3: labels/rule_labels/v1.0/
           ↓
Step 3.5 → S3: labels/judge_labels/v1.0/
           ↓
Step 4 → S3: models/lora_model_v1.0/
           ↓
Step 5 → S3: eval/results/v1.0/
           ↓
Release Gate → S3: models/registry/v1.0/
```

---

## Infrastructure Summary

### Server Inventory

| Server Role | Instance Type | GPU | Count | Use Case |
|-------------|--------------|-----|-------|----------|
| **Orchestration** | t3.medium | No | 1 | Pipeline orchestration |
| **Data Processing** | t3.large | No | 1 | Data preprocessing |
| **CV Model Inference** | g4dn.xlarge | Yes (T4) | 1 | Visual feature extraction |
| **Rule Engine** | t3.large | No | 1 | Rule-based label generation |
| **LLM Judge** | g5.xlarge | Yes (A10G) | 1 | LLM label generation |
| **Model Training** | p3.2xlarge | Yes (V100) | 1 | LoRA training |
| **Evaluation** | t3.large | No | 1 | Model evaluation |
| **Release Gate** | t3.medium | No | 1 | Release decision |

### Resource Requirements Summary

**CPU Servers** (t3.medium/t3.large):
- Total vCPU: 2 + 2 + 2 + 2 + 2 = 10 vCPU
- Total Memory: 4GB + 8GB + 8GB + 8GB + 4GB = 32GB
- Total Storage: 20GB + 100GB + 50GB + 50GB + 20GB = 240GB

**GPU Servers** (g4dn.xlarge + g5.xlarge + p3.2xlarge):
- Total vCPU: 4 + 4 + 8 = 16 vCPU
- Total Memory: 16GB + 16GB + 61GB = 93GB
- Total GPU Memory: 16GB (T4) + 24GB (A10G) + 16GB (V100) = 56GB VRAM
- Total Storage: 100GB + 100GB + 500GB = 700GB

**Storage Summary**:
- **S3 Storage**: ~8GB (data + models + results)
- **EBS Storage**: ~940GB (240GB CPU + 700GB GPU)
- **Total Storage**: ~950GB

### Cost Summary (per Training Run)

**EC2 Costs** (On-Demand):
- t3.medium (orchestration + gate): $0.0416/hour × 1 hour = **$0.04**
- t3.large (preprocessing + rules + eval): $0.0832/hour × 1 hour = **$0.08**
- g4dn.xlarge (CV): $0.526/hour × 2 hours = **$1.05**
- g5.xlarge (LLM Judge): $1.006/hour × 4 hours = **$4.02**
- p3.2xlarge (training): $3.06/hour × 2.5 hours = **$7.65**

**Total EC2 Cost per Run**: **~$12.84**

**S3 Costs** (Monthly):
- Storage: ~8GB × $0.023/GB = **~$0.18/month**
- Transfer: ~100GB/month × $0.09/GB = **~$9/month**

**EBS Costs** (Monthly):
- Storage: ~940GB × $0.08/GB = **~$75/month**

---

## Deployment Strategy

### Option 1: Single EC2 Instance (Sequential)

**Approach**: All steps run sequentially on one instance

**Pros**:
- Simple deployment
- No network overhead between steps
- Lower complexity

**Cons**:
- Long total time (sequential execution)
- GPU instance needed for entire pipeline
- Higher cost (GPU running for non-GPU steps)

### Option 2: Multiple EC2 Instances (Parallel)

**Approach**: Each step runs on dedicated instance

**Pros**:
- Steps can run in parallel (if no dependencies)
- Optimized instance types per step
- Faster overall execution

**Cons**:
- More complex orchestration
- S3 transfer overhead
- Higher cost (multiple instances)

### Option 3: Hybrid (Recommended)

**Approach**: 
- **CPU steps** run on one t3.large instance (sequential or parallel)
- **GPU steps** run on dedicated GPU instances
- Use S3 as intermediate storage

**Pros**:
- Optimized cost (CPU steps don't use GPU)
- Parallel GPU steps (Step 2 and Step 3.5 can run in parallel)
- Balanced complexity and cost

**Cons**:
- Requires orchestration
- S3 transfer overhead

**Recommended Configuration**:
- 1x t3.large: Step 0-1, Step 3, Step 5 (sequential)
- 1x g4dn.xlarge: Step 2 (parallel with Step 3.5)
- 1x g5.xlarge: Step 3.5 (parallel with Step 2)
- 1x p3.2xlarge: Step 4 (after Step 3.5 completes)
- 1x t3.medium: Orchestration + Release Gate

---

## Summary

### Key Points

1. **Data Storage**: All data stored in **AWS S3** (JSON/JSONL format)
2. **Compute**: Mix of **CPU instances** (t3) and **GPU instances** (g4dn/g5/p3)
3. **GPU Usage**: Only Step 2, Step 3.5, and Step 4 require GPU
4. **Cache Strategy**: Local EBS cache + S3 cache for cost optimization
5. **Versioning**: All outputs versioned in S3 paths

### Server Allocation Summary

- **Training Launch**: t3.medium (orchestration)
- **Step 0-1**: t3.large (data preprocessing, CPU)
- **Step 2**: g4dn.xlarge (visual features, GPU)
- **Step 3**: t3.large (rules, CPU)
- **Step 3.5**: g5.xlarge (LLM Judge, GPU)
- **Step 4**: p3.2xlarge (training, GPU)
- **Step 5**: t3.large (evaluation, CPU)
- **Release Gate**: t3.medium (decision, CPU)