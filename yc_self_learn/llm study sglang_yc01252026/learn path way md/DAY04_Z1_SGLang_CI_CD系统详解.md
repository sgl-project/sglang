# DAY04_Z1: SGLang CI/CD 系统详解

**Author**：Yanda Cheng  
**Project**：SGLang  
**Purpose**：详解 SGLang 的 CI/CD 系统，包括测试类型、流程、架构  
**Related Documents**：
- `00_SGLang工程师成长路径.md` (Section: 测试要求)
- `02_SGLang生态系统关系详解.md`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [CI/CD 架构](#cicd-架构)
3. [测试类型详解](#测试类型详解)
4. [测试流程](#测试流程)
5. [测试套件组织](#测试套件组织)
6. [CI/CD 触发机制](#cicd-触发机制)
7. [性能监控与分析](#性能监控与分析)
8. [最佳实践](#最佳实践)

---

## Overview

### SGLang CI/CD 系统概览

SGLang 使用 **GitHub Actions** 作为 CI/CD 平台，包含以下核心组件：

1. **Build Tests（构建测试）**：验证代码能够正确编译和构建
2. **Unit Tests（单元测试）**：验证功能正确性
3. **Performance Tests（性能测试）**：验证性能指标（延迟、吞吐量）
4. **Quantization Tests（量化测试）**：验证量化功能正确性

### 核心特点

- **多阶段测试**：Stage A → Stage B → Stage C（顺序执行）
- **并行执行**：同一阶段内的测试可以并行运行
- **智能过滤**：根据文件变更自动决定运行哪些测试
- **资源管理**：限制并发数，避免资源耗尽
- **权限控制**：需要 `run-ci` 标签才能触发完整 CI

---

## CI/CD 架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│              PR / Schedule Trigger                       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              PR Gate (pr-gate.yml)                       │
│  - 检查 PR 状态（draft/ready）                           │
│  - 检查 run-ci 标签                                      │
│  - 速率限制（cooldown）                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│          Check Changes (check-changes)                   │
│  - 检测文件变更（main_package/sgl_kernel/jit_kernel）   │
│  - 决定运行模式（filtered vs all tests）                 │
│  - 设置并行度（max_parallel）                            │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  Stage A     │      │  Build       │
│  (快速测试)   │      │  (构建)       │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌─────────────────────────────────────────┐
│         Wait for Stage A                │
│  (PR 模式：顺序执行，等待 Stage A 完成)   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Stage B                                      │
│  - stage-b-test-small-1-gpu (5090)                       │
│  - stage-b-test-large-1-gpu (H100)                       │
│  - stage-b-test-large-2-gpu                              │
│  - stage-b-test-4-gpu-b200                               │
│  - unit-test-backend-* (多 GPU)                          │
│  - performance-test-* (性能测试)                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│         Wait for Stage B                                 │
│  (PR 模式：顺序执行，等待 Stage B 完成)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Stage C                                      │
│  - stage-c-test-large-4-gpu                              │
│  - stage-c-test-large-4-gpu-b200                         │
│  - stage-c-test-large-8-gpu-b200                         │
└─────────────────────────────────────────────────────────┘
```

### 关键组件

#### 1. PR Gate (`pr-gate.yml`)

**功能**：
- 检查 PR 是否为 draft（draft PR 不运行 CI）
- 检查是否有 `run-ci` 标签（低权限用户需要标签）
- 速率限制（cooldown period，防止滥用）

**权限控制**：
```yaml
# CI_PERMISSIONS.json 控制权限
{
  "username": {
    "cooldown_interval_minutes": 120  # 用户特定的 cooldown
  }
}
```

#### 2. Check Changes (`check-changes`)

**功能**：
- 检测文件变更（使用 `dorny/paths-filter`）
- 决定运行哪些测试
- 设置并行度

**变更检测**：
```yaml
filters:
  main_package:
    - "python/sglang/!(multimodal_gen)/**"
    - "python/pyproject.toml"
    - "test/**"
  sgl_kernel:
    - "sgl-kernel/**"
  jit_kernel:
    - "python/sglang/jit_kernel/**"
  multimodal_gen:
    - "python/sglang/multimodal_gen/**"
```

#### 3. Wait Jobs（顺序执行控制）

**功能**：
- PR 模式：顺序执行（Stage A → Stage B → Stage C）
- Scheduled 模式：并行执行（所有阶段同时运行）

**实现**：
- 使用 GitHub API 轮询检查前一个阶段是否完成
- 最大等待时间：Stage A (240分钟)，Stage B (480分钟)

---

## 测试类型详解

### 1. Build Tests（构建测试）

#### 1.1 sgl-kernel-build-wheels

**目的**：构建 sgl-kernel 的 Python wheel 包

**触发条件**：
- `sgl-kernel/**` 文件变更
- 或 `run_all_tests == true`

**流程**：
```yaml
sgl-kernel-build-wheels:
  steps:
    - Checkout code
    - Install dependencies
    - Build wheels for different Python versions
    - Upload artifacts (wheel files)
```

**输出**：
- `wheel-python3.10-cuda12.9` (artifact)
- 供后续测试使用

#### 1.2 build-test

**目的**：验证主包能够正确构建

**位置**：Stage A (`stage-a-test-1`)

**测试内容**：
- Python 包安装
- 依赖检查
- 基本导入测试

---

### 2. Unit Tests（单元测试）

#### 2.1 Stage A Tests（快速测试）

**目的**：快速验证基本功能，在 Stage B 之前运行

**测试套件**：
- `stage-a-test-1`：基础功能测试（锁定，必须通过）
- `stage-a-test-2`：扩展功能测试
- `stage-a-test-cpu`：CPU 后端测试

**特点**：
- 运行时间短（< 30分钟）
- 覆盖核心功能
- 必须全部通过才能进入 Stage B

#### 2.2 Stage B Tests（主要测试）

**目的**：全面的功能测试

**测试套件**：

**a) stage-b-test-small-1-gpu (RTX 5090)**
- **Runner**: `1-gpu-5090`
- **GPU**: RTX 5090 (32GB, SM120)
- **Partitions**: 8 个并行分区
- **用途**：5090 兼容的测试（优先使用）

**b) stage-b-test-large-1-gpu (H100)**
- **Runner**: `1-gpu-runner`
- **GPU**: H100 (80GB, SM90)
- **Partitions**: 14 个并行分区
- **用途**：
  - 5090 不兼容的测试（FA3, FP8, 大模型）
  - 需要 >32GB VRAM 的测试

**c) stage-b-test-large-2-gpu**
- **Runner**: `2-gpu-runner`
- **Partitions**: 4 个并行分区
- **用途**：需要 2 GPU 的测试（Tensor Parallelism）

**d) stage-b-test-4-gpu-b200**
- **Runner**: `4-gpu-b200`
- **用途**：Blackwell 架构测试

**测试内容**：
- 后端运行时测试（`test/srt/`）
- 前端语言测试（`test/lang/`）
- 多 GPU 并行测试
- LoRA 测试
- 结构化输出测试

#### 2.3 Stage C Tests（大规模测试）

**目的**：大规模、多 GPU 测试

**测试套件**：
- `stage-c-test-large-4-gpu`：4 GPU 测试
- `stage-c-test-large-4-gpu-b200`：4 GPU B200 测试
- `stage-c-test-large-8-gpu-b200`：8 GPU B200 测试

**特点**：
- 运行时间长
- 需要大量 GPU 资源
- 主要用于验证大规模部署场景

#### 2.4 Unit Test Backend（后端单元测试）

**多 GPU 配置**：
- `unit-test-backend-4-gpu`：4 GPU 测试
- `unit-test-backend-8-gpu-h200`：8 GPU H200 测试
- `unit-test-backend-8-gpu-h20`：8 GPU H20 测试
- `unit-test-backend-4-gpu-b200`：4 GPU B200 测试
- `unit-test-backend-4-gpu-gb200`：4 GPU GB200 测试

**测试内容**：
- Tensor Parallelism
- Pipeline Parallelism
- Expert Parallelism
- 大规模模型推理

#### 2.5 sgl-kernel-unit-test

**目的**：测试 sgl-kernel 的功能

**触发条件**：
- `sgl-kernel/**` 文件变更

**测试内容**：
- Kernel 功能测试
- 数值精度测试
- 边界情况测试

#### 2.6 jit-kernel-unit-test

**目的**：测试 JIT kernel 的功能

**触发条件**：
- `python/sglang/jit_kernel/**` 文件变更

---

### 3. Performance Tests（性能测试）

#### 3.1 performance-test-1-gpu-part-1

**目的**：单 GPU 性能测试（第一部分）

**测试内容**：
```yaml
test_bs1_default:
  - output_throughput_token_s  # 吞吐量测试

test_online_latency_default:
  - median_e2e_latency_ms  # 端到端延迟

test_offline_throughput_default:
  - output_throughput_token_s  # 离线吞吐量

test_offline_throughput_non_stream_small_batch_size:
  - output_throughput_token_s  # 非流式小批次吞吐量

test_online_latency_eagle:
  - median_e2e_latency_ms  # Eagle 延迟
  - accept_length  # 接受长度

test_lora_online_latency:
  - median_e2e_latency_ms  # LoRA 延迟
  - median_ttft_ms  # TTFT

test_lora_online_latency_with_concurrent_adapter_updates:
  - median_e2e_latency_ms  # 并发适配器更新延迟
  - median_ttft_ms
```

#### 3.2 performance-test-1-gpu-part-2

**目的**：单 GPU 性能测试（第二部分）

**测试内容**：
```yaml
test_offline_throughput_without_radix_cache:
  - output_throughput_token_s  # 无 Radix Cache 吞吐量

test_offline_throughput_with_triton_attention_backend:
  - output_throughput_token_s  # Triton Attention 后端吞吐量

test_offline_throughput_default_fp8:
  - output_throughput_token_s  # FP8 量化吞吐量

test_vlm_offline_throughput:
  - output_throughput_token_s  # VLM 离线吞吐量

test_vlm_online_latency:
  - median_e2e_latency_ms  # VLM 在线延迟
```

#### 3.3 performance-test-2-gpu

**目的**：2 GPU 性能测试

**测试内容**：
- 多 GPU 并行性能
- Tensor Parallelism 性能
- 负载均衡性能

#### 3.4 sgl-kernel-benchmark-test

**目的**：Kernel 级别的性能基准测试

**位置**：`sgl-kernel/benchmark/`

**测试内容**：
- `bench_activation.py`：激活函数性能
- `bench_dsv3_fused_a_gemm.py`：DeepSeek V3 GEMM 性能
- `bench_moe_topk_softmax.py`：MoE TopK Softmax 性能
- `bench_per_tensor_quant_fp8.py`：FP8 量化性能
- `bench_per_token_quant_fp8.py`：Per-token FP8 量化性能
- `bench_top_k_top_p_sampling.py`：采样性能

**CI 模式**：
```python
# CI 环境使用简化参数
if IS_CI:
    default_batch_sizes = [1]  # 单个批次大小
    default_seq_lens = [1]  # 单个序列长度
    default_dims = [1024]  # 单个维度
else:
    # 完整测试模式
    default_batch_sizes = [1, 4, 16]
    default_seq_lens = [1, 4, 16, 64]
    default_dims = [1024, 2048, 4096, 8192, 16384]
```

---

### 4. Quantization Tests（量化测试）

#### 4.1 FP8 量化测试

**位置**：
- `performance-test-1-gpu-part-2` 中的 `test_offline_throughput_default_fp8`
- `sgl-kernel-benchmark-test` 中的量化 benchmark

**测试内容**：
- FP8 量化正确性
- FP8 量化性能
- Per-tensor vs Per-token FP8

#### 4.2 量化功能测试

**位置**：Unit tests 中

**测试内容**：
- FP4/FP8/INT4 量化
- AWQ/GPTQ 量化
- 量化模型推理正确性

#### 4.3 Nightly 量化测试

**位置**：`nightly-test-nvidia.yml`

**测试内容**：
- 各种量化配置的组合测试
- 量化模型在不同 GPU 上的性能

---

## 测试流程

### PR 模式（顺序执行）

```
PR 创建/更新
    ↓
PR Gate 检查
    ↓
Check Changes（检测变更）
    ↓
Stage A Tests（快速测试）
    ↓
Wait for Stage A（等待完成）
    ↓
Stage B Tests（主要测试）
    ├─ stage-b-test-small-1-gpu (8 partitions)
    ├─ stage-b-test-large-1-gpu (14 partitions)
    ├─ stage-b-test-large-2-gpu (4 partitions)
    ├─ unit-test-backend-* (多 GPU)
    └─ performance-test-* (性能测试)
    ↓
Wait for Stage B（等待完成）
    ↓
Stage C Tests（大规模测试）
    ├─ stage-c-test-large-4-gpu
    ├─ stage-c-test-large-4-gpu-b200
    └─ stage-c-test-large-8-gpu-b200
    ↓
All Tests Pass → PR Ready to Merge
```

### Scheduled 模式（并行执行）

```
Schedule Trigger (每 6 小时)
    ↓
Check Changes（检测变更）
    ↓
所有阶段并行执行
    ├─ Stage A Tests
    ├─ Stage B Tests
    └─ Stage C Tests
    ↓
生成测试报告
```

### 关键差异

| 维度 | PR 模式 | Scheduled 模式 |
|------|---------|---------------|
| **执行方式** | 顺序执行（Stage A → B → C） | 并行执行（所有阶段同时） |
| **等待机制** | 有 Wait Jobs | 无 Wait Jobs |
| **并行度** | 默认 3（PR），14（high priority） | 14（全并行） |
| **continue-on-error** | false（失败即停止） | true（继续运行所有测试） |
| **用途** | PR 审查，快速反馈 | 完整测试，回归检测 |

---

## 测试套件组织

### 测试套件分类

#### Per-Commit Tests（每次提交测试）

**Stage A**：
- `stage-a-test-1`：基础测试（锁定，必须通过）
- `stage-a-test-2`：扩展测试
- `stage-a-test-cpu`：CPU 后端测试

**Stage B**：
- `stage-b-test-small-1-gpu`：5090 测试（8 partitions）
- `stage-b-test-large-1-gpu`：H100 测试（14 partitions）
- `stage-b-test-large-2-gpu`：2 GPU 测试（4 partitions）
- `stage-b-test-4-gpu-b200`：B200 测试

**Stage C**：
- `stage-c-test-large-4-gpu`：4 GPU 测试
- `stage-c-test-large-4-gpu-b200`：4 GPU B200 测试
- `stage-c-test-large-8-gpu-b200`：8 GPU B200 测试

#### Nightly Tests（夜间测试）

**位置**：`test/srt/nightly/`

**测试套件**：
- `nightly-1-gpu`：单 GPU 夜间测试
- `nightly-2-gpu`：2 GPU 夜间测试
- `nightly-4-gpu`：4 GPU 夜间测试
- `nightly-8-gpu`：8 GPU 夜间测试

**特点**：
- 运行时间长
- 覆盖更多场景
- 使用 `NightlyBenchmarkRunner` 进行性能基准测试

### 测试套件选择指南

#### 何时使用 5090 (stage-b-test-small-1-gpu)？

✅ **优先使用 5090**（如果满足）：
- 大多数 1-GPU 测试
- 标准功能测试
- 不需要特殊架构的测试

#### 何时使用 H100 (stage-b-test-large-1-gpu)？

⚠️ **必须使用 H100**（如果满足以下任一条件）：

1. **架构不兼容（SM120/Blackwell）**：
   - FA3 attention backend（需要 SM≤90）
   - MLA with FA3 backend
   - FP8/MXFP4 量化（SM120 不支持）
   - 某些 Triton kernels（共享内存限制）

2. **内存需求**：
   - 模型 >30B 参数或大型 MoE
   - 测试需要 >32GB VRAM

3. **已知 5090 失败**：
   - Weight update/sync 测试
   - 某些 spec decoding 测试

### CI Registry System（注册系统）

**位置**：`test/registered/`

**注册函数**：
```python
from sglang.test.ci.ci_register import (
    register_cuda_ci,
    register_amd_ci,
    register_cpu_ci,
    register_npu_ci,
)

# Per-commit 测试（小 1-GPU，运行在 5090）
register_cuda_ci(
    est_time=80,
    suite="stage-b-test-small-1-gpu"
)

# Per-commit 测试（大 1-GPU，运行在 H100）
register_cuda_ci(
    est_time=120,
    suite="stage-b-test-large-1-gpu"
)

# Per-commit 测试（2-GPU）
register_cuda_ci(
    est_time=200,
    suite="stage-b-test-large-2-gpu"
)

# Nightly-only 测试
register_cuda_ci(
    est_time=200,
    suite="nightly-1-gpu",
    nightly=True
)

# 多后端测试
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=120, suite="stage-a-test-1")

# 临时禁用测试
register_cuda_ci(
    est_time=80,
    suite="stage-b-test-small-1-gpu",
    disabled="flaky - see #12345"
)
```

---

## CI/CD 触发机制

### 触发方式

#### 1. Pull Request

**触发条件**：
- PR 创建或更新
- 目标分支：`main`

**流程**：
1. PR Gate 检查（draft/run-ci 标签）
2. Check Changes（检测文件变更）
3. 顺序执行测试（Stage A → B → C）

**权限要求**：
- 低权限用户：需要 `run-ci` 标签
- 高权限用户（write/maintain/admin）：自动触发

#### 2. Schedule（定时触发）

**触发条件**：
- 每 6 小时自动运行（`cron: '0 */6 * * *'`）

**流程**：
1. Check Changes（检测所有变更）
2. 并行执行所有测试
3. 生成完整测试报告

**特点**：
- 运行所有测试（`run_all_tests=true`）
- 并行执行（无 Wait Jobs）
- `continue-on-error=true`（继续运行所有测试）

#### 3. Workflow Dispatch（手动触发）

**触发条件**：
- 手动触发（需要 write 权限）

**输入参数**：
```yaml
inputs:
  version: "release" | "nightly"  # FlashInfer 版本
  target_stage: ""  # 特定阶段（可选）
  force_continue_on_error: false  # 强制 continue-on-error
  pr_head_sha: ""  # PR head SHA（用于 fork PR）
  test_parallel_dispatch: false  # 测试并行分发
```

#### 4. Workflow Call（被其他 workflow 调用）

**触发条件**：
- 被其他 workflow 调用（如 release workflow）

**输入参数**：
```yaml
inputs:
  ref: ""  # Git ref (branch, tag, or SHA)
  run_all_tests: false  # 是否运行所有测试
```

### 速率限制（Rate Limiting）

#### Cooldown 机制

**目的**：防止 CI 资源滥用

**规则**：
- **默认 cooldown**：120 分钟
- **用户特定 cooldown**：在 `CI_PERMISSIONS.json` 中配置
- **高权限用户**：无 cooldown 限制

**实现**：
```javascript
// 检查最近 N 分钟内的运行
const cutoff = new Date(Date.now() - cooldownMinutes * 60 * 1000);
const recentRuns = await listWorkflowRuns({ since: cutoff });

// 只计算实际消耗资源的运行（通过 gate 的运行）
if (await didRunPassGate(run)) {
  // 计算 cooldown
}
```

#### CI_PERMISSIONS.json

**位置**：`.github/CI_PERMISSIONS.json`

**格式**：
```json
{
  "username": {
    "cooldown_interval_minutes": 60  // 用户特定的 cooldown
  }
}
```

---

## 性能监控与分析

### CI Performance Analyzer

**位置**：`scripts/ci_monitor/ci_analyzer_perf.py`

**功能**：
- 分析性能测试结果
- 检测性能回归
- 生成性能报告

**监控的测试和指标**：

```python
target_tests_and_metrics = {
    "performance-test-1-gpu-part-1": {
        "test_bs1_default": ["output_throughput_token_s"],
        "test_online_latency_default": ["median_e2e_latency_ms"],
        "test_offline_throughput_default": ["output_throughput_token_s"],
        "test_online_latency_eagle": ["median_e2e_latency_ms", "accept_length"],
        "test_lora_online_latency": ["median_e2e_latency_ms", "median_ttft_ms"],
    },
    "performance-test-1-gpu-part-2": {
        "test_offline_throughput_without_radix_cache": ["output_throughput_token_s"],
        "test_offline_throughput_default_fp8": ["output_throughput_token_s"],
        "test_vlm_offline_throughput": ["output_throughput_token_s"],
        "test_vlm_online_latency": ["median_e2e_latency_ms"],
    },
}
```

### CI Failure Monitor

**位置**：`scripts/ci_monitor/ci_analyzer.py`

**功能**：
- 分析 CI 失败原因
- 分类失败类型（build/unit-test/performance）
- 生成失败报告

**失败分类**：
```python
job_categories = {
    "build": [
        "build-test",
        "sgl-kernel-build-wheels",
    ],
    "unit-test": [
        "stage-a-test-1",
        "unit-test-backend-1-gpu",
        "unit-test-backend-2-gpu",
        "stage-b-test-4-gpu-b200",
        "unit-test-backend-4-gpu",
        "unit-test-backend-8-gpu",
    ],
    "performance": [
        "performance-test-1-gpu-part-1",
        "performance-test-1-gpu-part-2",
        "performance-test-2-gpu",
    ],
}
```

### 性能基准测试

#### Nightly Benchmark Runner

**位置**：`test/srt/nightly/nightly_utils.py`

**功能**：
- 运行性能基准测试
- 记录性能指标
- 检测性能回归

**使用示例**：
```python
from sglang.test.srt.nightly.nightly_utils import NightlyBenchmarkRunner

class TestMyFeature(NightlyBenchmarkRunner):
    def test_my_feature_performance(self):
        # 运行基准测试
        result = self.run_benchmark(
            model="Qwen/Qwen2-7B-Instruct",
            test_name="my_feature",
            metrics=["output_throughput_token_s", "median_e2e_latency_ms"]
        )
        
        # 检查性能阈值
        assert result["output_throughput_token_s"] > 100.0
        assert result["median_e2e_latency_ms"] < 100.0
```

---

## 最佳实践

### 1. 添加新测试

#### 步骤 1：创建测试文件

**位置选择**：
- 后端运行时测试：`test/srt/`
- 前端语言测试：`test/lang/`
- 夜间测试：`test/srt/nightly/`

#### 步骤 2：注册到测试套件

**在 `run_suite.py` 中添加**：
```python
# test/srt/run_suite.py

SUITES = {
    "stage-b-test-small-1-gpu": [
        # ... existing tests ...
        "test_my_new_feature",  # 按字母顺序排序
    ],
}
```

#### 步骤 3：选择正确的测试套件

**决策树**：
```
需要多少 GPU？
├─ 1 GPU
│  ├─ 5090 兼容？ → stage-b-test-small-1-gpu
│  └─ 需要 H100？ → stage-b-test-large-1-gpu
├─ 2 GPU → stage-b-test-large-2-gpu
├─ 4 GPU → unit-test-backend-4-gpu
└─ 8 GPU → unit-test-backend-8-gpu-*
```

#### 步骤 4：添加测试入口

**确保有测试入口**：
```python
# unittest
if __name__ == "__main__":
    unittest.main()

# pytest
if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
```

### 2. 性能测试最佳实践

#### 使用 NightlyBenchmarkRunner

```python
from sglang.test.srt.nightly.nightly_utils import NightlyBenchmarkRunner

class TestMyPerformance(NightlyBenchmarkRunner):
    def test_my_performance(self):
        result = self.run_benchmark(
            model="Qwen/Qwen2-7B-Instruct",
            test_name="my_performance_test",
            metrics=["output_throughput_token_s", "median_e2e_latency_ms"]
        )
        
        # 设置性能阈值
        self.assert_performance(
            result,
            thresholds={
                "output_throughput_token_s": 100.0,  # 最小吞吐量
                "median_e2e_latency_ms": 100.0,  # 最大延迟
            }
        )
```

#### 性能测试注意事项

1. **使用小模型**：减少测试时间
2. **复用服务器**：避免重复启动
3. **设置超时**：防止测试卡死
4. **记录指标**：输出性能数据

### 3. 测试优化建议

#### 减少测试时间

1. **使用小模型**：
   ```python
   # 使用 7B 模型而不是 70B
   model = "Qwen/Qwen2-7B-Instruct"
   ```

2. **复用服务器**：
   ```python
   @classmethod
   def setUpClass(cls):
       # 启动一次服务器，多个测试复用
       cls.runtime = ...
   
   @classmethod
   def tearDownClass(cls):
       # 测试结束后关闭
       cls.runtime.shutdown()
   ```

3. **减少测试数据**：
   ```python
   # 使用少量测试用例
   num_questions = 10  # 而不是 200
   ```

#### 选择合适的测试套件

1. **优先使用 5090**：如果测试兼容
2. **避免 8-GPU 测试**：除非必要
3. **使用 nightly 测试**：对于长时间测试

### 4. CI/CD 调试

#### 本地运行测试

```bash
# 运行单个测试文件
cd test/srt
python3 test_my_feature.py

# 运行测试套件
python3 run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# 运行夜间测试
python3 run_suite.py --hw cuda --suite nightly-1-gpu --nightly
```

#### 查看 CI 日志

1. **GitHub Actions 页面**：查看完整日志
2. **CI Monitor**：使用 `scripts/ci_monitor/` 分析
3. **性能分析**：使用 `ci_analyzer_perf.py`

#### 调试失败测试

1. **检查日志**：查看错误信息
2. **本地复现**：在相同环境运行
3. **简化测试**：减少变量，定位问题
4. **添加调试输出**：打印中间状态

---

## 总结

### 关键要点

1. ✅ **四类测试**：Build、Unit、Performance、Quantization
2. ✅ **三阶段执行**：Stage A（快速）→ Stage B（主要）→ Stage C（大规模）
3. ✅ **智能过滤**：根据文件变更自动决定运行哪些测试
4. ✅ **权限控制**：需要 `run-ci` 标签才能触发完整 CI
5. ✅ **性能监控**：自动检测性能回归

### 测试类型总结表

| 测试类型 | 主要测试 | 位置 | 目的 |
|---------|---------|------|------|
| **Build Tests** | sgl-kernel-build-wheels | Stage A | 验证构建正确性 |
| **Unit Tests** | stage-b-test-* | Stage B | 验证功能正确性 |
| **Performance Tests** | performance-test-* | Stage B | 验证性能指标 |
| **Quantization Tests** | FP8/FP4 tests | Stage B/Nightly | 验证量化功能 |

### 快速参考

**运行测试**：
```bash
# 本地运行
cd test/srt
python3 run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# 运行单个测试
python3 test_my_feature.py TestMyFeature.test_something
```

**触发 CI**：
- PR 模式：创建 PR，添加 `run-ci` 标签
- Scheduled 模式：每 6 小时自动运行
- Manual 模式：GitHub Actions → Run workflow

**查看结果**：
- GitHub Actions 页面
- CI Monitor (`scripts/ci_monitor/`)
- Performance Analyzer (`ci_analyzer_perf.py`)

---

**Remember**: SGLang 的 CI/CD 系统设计用于确保代码质量和性能，理解其架构和流程对于贡献代码至关重要。
