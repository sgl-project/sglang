# A2_B2_C2_D1_E3：KYC Pipeline 的测试设计策略

---
doc_type: glossary
layer: L3
scope_in:  KYC Pipeline（per-file isolation、batch processing）的测试设计策略、测试层级、测试应该跑到第几层
scope_out: 具体测试代码编写（见 howto）；性能测试、压力测试（见 L4）
inputs:   (读者) 疑问：KYC Pipeline 应该设计哪些测试？测试应该覆盖到哪个层级？
outputs:  KYC Pipeline 测试设计策略 + 测试层级 + 测试覆盖范围 + trade-off
entrypoints: [ 核心问题 ]
children: []
related: [ KYC Pipeline, per-file isolation, batch processing, 分层校验, 单元测试, 集成测试, E2E 测试, KYC_Day01_A2_B2_C2_D1_测试的设计原理与分层策略.md, KYC_Day01_A3_B1_good_batch.md, KYC_Day01_A1_B5_validation_tradeoff.md ]
---

## Definition（定义）

**核心问题**：**KYC Pipeline 的测试应该设计到第几层？**

**KYC Pipeline**：
- ✅ **Per-File Isolation**：每个文件独立处理，一个文件失败不影响其他文件
- ✅ **Batch Processing**：批量处理多个文件，处理完成后计算指标
- ✅ **错误处理**：处理各种错误（IMAGE_FORMAT_UNSUPPORTED、API_TIMEOUT 等）

**测试层级**：
- ✅ **单元测试**：测试单个函数/类（如 `process_file()`、`calculate_metrics()`）
- ✅ **集成测试**：测试多个组件交互（如 Pipeline → API → Database）
- ✅ **E2E 测试**：测试完整流程（如 输入文件 → Pipeline → 输出结果）

---

## 🎯 核心问题：测试应该跑到第几层？

### 你的理解（需要确认）

**问题**：对于 KYC Pipeline，测试应该设计到第几层？

**可能的答案**：
- ✅ **单元测试**：测试 `process_file()`、`calculate_metrics()` 等函数
- ✅ **集成测试**：测试 Pipeline 与 API、Database 的交互
- ✅ **E2E 测试**：测试完整的 batch 处理流程

**Trade-off**：
- ✅ **小公司**：可能只做到单元测试或集成测试
- ✅ **大公司**：需要做到 E2E 测试

---

## 📊 KYC Pipeline 的测试设计策略

### 测试层级 1：单元测试（必须）

**测试什么**：
- ✅ **单个函数**：`process_file()`、`calculate_metrics()`、`validate_document()`
- ✅ **业务逻辑**：错误处理、状态转换、指标计算

**例子**：
```python
# 单元测试：测试 process_file() 函数
def test_process_file_success():
    """测试单个文件处理成功"""
    file_path = "test_data/doc_001.jpg"
    result = process_file(file_path)
    assert result.status == "success"
    assert result.file_id == "doc_001.jpg"

def test_process_file_failure():
    """测试单个文件处理失败"""
    file_path = "test_data/invalid.webp"
    result = process_file(file_path)
    assert result.status == "fail"
    assert result.error_code == "IMAGE_FORMAT_UNSUPPORTED"

def test_calculate_metrics():
    """测试指标计算"""
    results = [
        {"status": "success", "latency_ms": 100},
        {"status": "success", "latency_ms": 200},
        {"status": "fail", "error_code": "API_TIMEOUT"}
    ]
    metrics = calculate_metrics(results)
    assert metrics.success_rate == 2/3
    assert metrics.p95_latency == 200
```

**特点**：
- ✅ **快速**：< 10ms 每个测试
- ✅ **稳定**：不依赖外部服务
- ✅ **覆盖细节**：测试函数内部逻辑

---

### 测试层级 2：集成测试（推荐）

**测试什么**：
- ✅ **组件交互**：Pipeline → API → Database
- ✅ **Per-File Isolation**：一个文件失败不影响其他文件
- ✅ **Batch Processing**：批量处理多个文件

**例子**：
```python
# 集成测试：测试 per-file isolation
def test_per_file_isolation():
    """测试一个文件失败不影响其他文件"""
    files = [
        "test_data/doc_001.jpg",  # 成功
        "test_data/invalid.webp",  # 失败
        "test_data/doc_003.jpg"   # 成功
    ]
    results = process_batch(files)
    
    # 验证：所有文件都被处理了
    assert len(results) == 3
    
    # 验证：失败的文件不影响其他文件
    assert results[0].status == "success"
    assert results[1].status == "fail"
    assert results[2].status == "success"

# 集成测试：测试 batch processing
def test_batch_processing():
    """测试批量处理"""
    files = ["test_data/doc_001.jpg", "test_data/doc_002.jpg"]
    batch_result = process_batch(files)
    
    # 验证：生成了 _summary.json
    assert batch_result.summary_path.exists()
    
    # 验证：_summary.json 包含所有结果
    summary = json.load(batch_result.summary_path)
    assert len(summary["results"]) == 2
```

**特点**：
- ✅ **中等速度**：1-10s 每个测试
- ✅ **真实性高**：使用真实 API、Database
- ✅ **覆盖交互**：测试组件之间的交互

---

### 测试层级 3：E2E 测试（大公司必须）

**测试什么**：
- ✅ **完整流程**：从输入文件到输出结果的完整流程
- ✅ **错误处理**：各种错误场景的处理
- ✅ **指标计算**：批量处理后的指标计算

**例子**：
```python
# E2E 测试：测试完整流程
def test_kyc_pipeline_end_to_end():
    """测试完整的 KYC Pipeline 流程"""
    # 1. 准备输入文件
    input_dir = Path("test_data/batch_001")
    input_dir.mkdir(exist_ok=True)
    (input_dir / "doc_001.jpg").write_bytes(b"fake_image_data")
    
    # 2. 运行 Pipeline
    output_dir = Path("test_output/batch_001")
    process_batch(input_dir, output_dir)
    
    # 3. 验证输出
    summary_path = output_dir / "_summary.json"
    assert summary_path.exists()
    
    summary = json.load(summary_path)
    assert len(summary["results"]) == 1
    assert summary["results"][0]["status"] == "success"
    
    # 4. 验证指标计算
    metrics = calculate_metrics_from_summary(summary_path)
    assert metrics.success_rate == 1.0
    assert metrics.p95_latency > 0

# E2E 测试：测试错误处理
def test_kyc_pipeline_error_handling():
    """测试错误处理流程"""
    # 1. 准备包含错误文件的 batch
    input_dir = Path("test_data/batch_002")
    input_dir.mkdir(exist_ok=True)
    (input_dir / "invalid.webp").write_bytes(b"invalid_data")
    
    # 2. 运行 Pipeline
    output_dir = Path("test_output/batch_002")
    process_batch(input_dir, output_dir)
    
    # 3. 验证错误处理
    summary_path = output_dir / "_summary.json"
    summary = json.load(summary_path)
    
    # 验证：错误文件被正确处理
    assert summary["results"][0]["status"] == "fail"
    assert summary["results"][0]["error_code"] == "IMAGE_FORMAT_UNSUPPORTED"
    
    # 验证：Pipeline 没有崩溃
    assert summary["batch_id"] is not None
```

**特点**：
- ✅ **慢速**：10-60s 每个测试
- ✅ **真实性最高**：使用真实环境
- ✅ **覆盖全面**：测试完整流程

---

## 🔍 KYC Pipeline 的分层设计与测试

### Pipeline 的分层结构（来自 A1_B5）

**KYC Pipeline 的分层设计**（参考 `KYC_Day01_A1_B5_validation_tradeoff.md`）：

```
Preprocessing（入口校验）
    ↓
Post-inference（Schema 验证）
    ↓
Post-processing（业务规则）
```

**各层的作用**：
- ✅ **Preprocessing**：严格校验（格式、大小）→ 避免无效数据进入 pipeline
- ✅ **Post-inference**：可配置校验（Schema 验证）→ 验证 LLM 输出
- ✅ **Post-processing**：宽松校验（业务规则）→ 标记 `needs_review`，不直接失败

**测试策略**：
- ✅ **单元测试**：测试每一层的函数（`preprocess_image()`、`validate_output()`、`apply_deterministic_rules()`）
- ✅ **集成测试**：测试各层之间的交互
- ✅ **E2E 测试**：测试完整的分层流程

---

## 🔍 KYC Pipeline 测试的关键场景

### 场景 1：分层校验测试（基于 A1_B5 的设计）

**为什么重要**：
- ✅ **核心设计**：分层校验是 KYC Pipeline 的核心设计（Preprocessing → Post-inference → Post-processing）
- ✅ **平衡点**：通过分层校验平衡 Error Rate 和 output 质量

**测试策略**：
- ✅ **单元测试**：测试每一层的校验逻辑
- ✅ **集成测试**：测试各层之间的交互
- ✅ **E2E 测试**：测试完整的分层流程

**例子**：
```python
# 单元测试：测试 Preprocessing 层
def test_preprocessing_strict_validation():
    """测试 Preprocessing 层的严格校验"""
    # 测试格式不支持
    with pytest.raises(ImageFormatError):
        preprocess_image("test_data/invalid.webp")
    
    # 测试文件过大
    with pytest.raises(ImageFormatError):
        preprocess_image("test_data/too_large.jpg")

# 单元测试：测试 Post-inference 层
def test_post_inference_configurable_validation():
    """测试 Post-inference 层的可配置校验"""
    llm_output = {"full_name": "John Doe"}
    
    # 严格模式：应该失败（缺少必填字段）
    with pytest.raises(ValidationError):
        validate_output(llm_output, strictness="high")
    
    # 宽松模式：应该通过（只要求 name）
    result = validate_output(llm_output, strictness="low")
    assert result == llm_output

# 单元测试：测试 Post-processing 层
def test_post_processing_loose_validation():
    """测试 Post-processing 层的宽松校验（不直接失败）"""
    extracted_data = {"full_name": "John Doe", "confidence": 0.7}
    
    # 置信度低，但不直接失败，而是标记 needs_review
    result = apply_deterministic_rules(extracted_data, confidence=0.7)
    assert result["status"] == "success"  # ✅ 不失败
    assert result["needs_review"] == True  # ⚠️ 标记需要 review
    assert "low_confidence" in result["review_reasons"]
```

---

### 场景 2：Per-File Isolation 测试

**为什么重要**：
- ✅ **核心功能**：Per-File Isolation 是 KYC Pipeline 的核心特性
- ✅ **风险控制**：一个文件失败不应该影响整个 batch

**测试策略**：
- ✅ **单元测试**：测试 `process_file()` 函数的错误处理
- ✅ **集成测试**：测试多个文件处理时的隔离性
- ✅ **E2E 测试**：测试完整 batch 处理时的隔离性

**例子**：
```python
# 集成测试：测试 per-file isolation
def test_per_file_isolation_integration():
    """测试集成层面的 per-file isolation"""
    files = [
        "test_data/doc_001.jpg",  # 成功
        "test_data/invalid.webp",  # 失败（格式不支持）
        "test_data/doc_003.jpg"   # 成功
    ]
    
    # 运行 Pipeline
    results = process_batch(files)
    
    # 验证：所有文件都被处理了
    assert len(results) == 3
    
    # 验证：失败的文件不影响其他文件
    assert results[0].status == "success"
    assert results[1].status == "fail"
    assert results[1].error_code == "IMAGE_FORMAT_UNSUPPORTED"
    assert results[2].status == "success"
    
    # 验证：成功的文件正常处理
    assert results[0].file_id == "doc_001.jpg"
    assert results[2].file_id == "doc_003.jpg"
```

---

### 场景 2：Batch Processing 测试

**为什么重要**：
- ✅ **核心功能**：Batch Processing 是 KYC Pipeline 的核心特性
- ✅ **指标计算**：批量处理完成后需要计算指标

**测试策略**：
- ✅ **单元测试**：测试 `calculate_metrics()` 函数
- ✅ **集成测试**：测试批量处理流程
- ✅ **E2E 测试**：测试完整的 batch 处理流程

**例子**：
```python
# E2E 测试：测试 batch processing
def test_batch_processing_end_to_end():
    """测试完整的 batch processing 流程"""
    # 1. 准备输入文件
    input_dir = Path("test_data/batch_003")
    input_dir.mkdir(exist_ok=True)
    for i in range(10):
        (input_dir / f"doc_{i:03d}.jpg").write_bytes(b"fake_image_data")
    
    # 2. 运行 Pipeline
    output_dir = Path("test_output/batch_003")
    process_batch(input_dir, output_dir)
    
    # 3. 验证输出
    summary_path = output_dir / "_summary.json"
    assert summary_path.exists()
    
    summary = json.load(summary_path)
    assert len(summary["results"]) == 10
    
    # 4. 验证指标计算
    metrics = calculate_metrics_from_summary(summary_path)
    assert metrics.success_rate == 1.0
    assert metrics.total_files == 10
    assert metrics.successful_files == 10
```

---

### 场景 3：错误处理测试

**为什么重要**：
- ✅ **稳定性**：错误处理是系统稳定的关键
- ✅ **用户体验**：错误处理影响用户体验

**测试策略**：
- ✅ **单元测试**：测试各种错误码的处理
- ✅ **集成测试**：测试错误场景下的组件交互
- ✅ **E2E 测试**：测试完整流程中的错误处理

**例子**：
```python
# 单元测试：测试错误处理
def test_error_handling():
    """测试各种错误场景"""
    # 测试格式不支持错误
    result = process_file("test_data/invalid.webp")
    assert result.status == "fail"
    assert result.error_code == "IMAGE_FORMAT_UNSUPPORTED"
    
    # 测试 API 超时错误
    with mock.patch('api_client.call_api', side_effect=TimeoutError()):
        result = process_file("test_data/doc_001.jpg")
        assert result.status == "fail"
        assert result.error_code == "API_TIMEOUT"
    
    # 测试 Schema 验证失败错误
    with mock.patch('validator.validate', return_value=False):
        result = process_file("test_data/doc_001.jpg")
        assert result.status == "fail"
        assert result.error_code == "SCHEMA_VALIDATION_FAILED"
```

---

## ⚖️ Trade-off 分析

### Trade-off 1：测试层级 vs 开发速度

**小公司选择**：
- ✅ **单元测试为主**：快速开发，快速迭代
- ⚠️ **集成测试少**：可能只有关键的集成测试
- ❌ **E2E 测试很少或没有**：E2E 测试成本高，可能不做

**大公司选择**：
- ✅ **测试金字塔完整**：单元测试 + 集成测试 + E2E 测试
- ✅ **E2E 测试必须**：确保完整流程正确

**Trade-off**：
- ✅ **小公司**：开发速度 > 测试覆盖
- ✅ **大公司**：测试覆盖 > 开发速度

---

### Trade-off 2：测试成本 vs 风险控制

**小公司选择**：
- ✅ **测试成本低**：只做单元测试，成本低
- ⚠️ **风险控制中等**：可能有一些集成问题无法发现

**大公司选择**：
- ✅ **测试成本高**：完整的测试金字塔，成本高
- ✅ **风险控制高**：多层测试，确保质量

**Trade-off**：
- ✅ **小公司**：测试成本低 > 风险控制
- ✅ **大公司**：风险控制 > 测试成本高

---

### Trade-off 3：测试速度 vs 测试覆盖

**小公司选择**：
- ✅ **测试速度快**：只有单元测试，运行快
- ⚠️ **测试覆盖中等**：可能无法发现集成问题

**大公司选择**：
- ✅ **测试速度慢**：完整的测试金字塔，运行慢
- ✅ **测试覆盖高**：多层测试，覆盖全面

**Trade-off**：
- ✅ **小公司**：测试速度快 > 测试覆盖
- ✅ **大公司**：测试覆盖 > 测试速度慢

---

## 🎯 推荐策略

### 小公司（MVP 阶段）

**测试层级**：
- ✅ **单元测试**：必须（测试核心函数）
- ✅ **集成测试**：推荐（测试 per-file isolation、batch processing）
- ❌ **E2E 测试**：可选（如果时间允许）

**测试重点**：
- ✅ **Per-File Isolation**：必须测试（核心功能）
- ✅ **Batch Processing**：必须测试（核心功能）
- ✅ **错误处理**：推荐测试（稳定性）

**例子**：
```python
# 小公司测试策略
tests/
├── unit/
│   ├── test_process_file.py      # 单元测试：process_file()
│   ├── test_calculate_metrics.py  # 单元测试：calculate_metrics()
│   └── test_error_handling.py    # 单元测试：错误处理
└── integration/
    ├── test_per_file_isolation.py # 集成测试：per-file isolation
    └── test_batch_processing.py   # 集成测试：batch processing
```

---

### 大公司（Production 阶段）

**测试层级**：
- ✅ **单元测试**：必须（70%）
- ✅ **集成测试**：必须（20%）
- ✅ **E2E 测试**：必须（10%）

**测试重点**：
- ✅ **Per-File Isolation**：必须测试（单元 + 集成 + E2E）
- ✅ **Batch Processing**：必须测试（单元 + 集成 + E2E）
- ✅ **错误处理**：必须测试（单元 + 集成 + E2E）
- ✅ **性能测试**：推荐测试（E2E）

**例子**：
```python
# 大公司测试策略
tests/
├── unit/
│   ├── test_process_file.py
│   ├── test_calculate_metrics.py
│   └── test_error_handling.py
├── integration/
│   ├── test_per_file_isolation.py
│   ├── test_batch_processing.py
│   └── test_api_integration.py
└── e2e/
    ├── test_kyc_pipeline_end_to_end.py
    ├── test_error_scenarios.py
    └── test_performance.py
```

---

## 📊 总结

### 核心问题答案

**KYC Pipeline 的测试应该设计到第几层？**

**答案**：
- ✅ **小公司**：单元测试 + 集成测试（推荐）
- ✅ **大公司**：单元测试 + 集成测试 + E2E 测试（必须）

### 测试重点

1. **Per-File Isolation**：必须测试（核心功能）
2. **Batch Processing**：必须测试（核心功能）
3. **错误处理**：必须测试（稳定性）

### Trade-off

1. **测试层级 vs 开发速度**：
   - ✅ **小公司**：开发速度 > 测试覆盖
   - ✅ **大公司**：测试覆盖 > 开发速度

2. **测试成本 vs 风险控制**：
   - ✅ **小公司**：测试成本低 > 风险控制
   - ✅ **大公司**：风险控制 > 测试成本高

3. **测试速度 vs 测试覆盖**：
   - ✅ **小公司**：测试速度快 > 测试覆盖
   - ✅ **大公司**：测试覆盖 > 测试速度慢

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2_B2_C2_D1 测试的设计原理与分层策略（[KYC_Day01_A2_B2_C2_D1_测试的设计原理与分层策略.md](./KYC_Day01_A2_B2_C2_D1_测试的设计原理与分层策略.md)） |
| **Related** | KYC Pipeline、per-file isolation、batch processing、单元测试、集成测试、E2E 测试、[KYC_Day01_A3_B1_good_batch.md](./KYC_Day01_A3_B1_good_batch.md) |
