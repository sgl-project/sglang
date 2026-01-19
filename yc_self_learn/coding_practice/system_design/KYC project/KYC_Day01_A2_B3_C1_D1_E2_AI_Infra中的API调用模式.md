# A2_B3_C1_D1_E2：AI Infra 中的 API 调用模式

---
doc_type: glossary
layer: L2
scope_in:  AI Infra 中什么时候用别人的 API、什么时候自己搭建、API 调用的常见模式
scope_out: 具体 API 的实现细节（见 howto）；API 网关的设计（见 L4）
inputs:   (读者) 疑问：AI Infra 是不是大部分都是调用别人的 API？他们做什么？
outputs:  AI Infra 的常见架构模式 + 什么时候用 API vs 自己搭建 + 实际例子
entrypoints: [ 核心观点 ]
children: []
related: [ API 调用, AI Infra, LLM Serving, API Gateway, KYC_Day01_A2_B3_C1_D1_API调用的底层细节与开发者视角.md ]
---

## Definition（定义）

**核心观点**：**是的，大部分 AI Infra 都是调用别人的 API，然后做"使用、搭建、测试、优化"的工作。**

**类比**：
- **调用 API** = **点外卖**（用别人的服务）
- **自己搭建** = **自己做饭**（从零开始）

**AI Infra 的工作**：
- ✅ **不是**从零开始训练模型
- ✅ **而是**调用别人的 API，然后做**集成、优化、监控、测试**

---

## 🎯 核心观点：AI Infra 的工作是什么？

### 你的理解是对的！

**AI Infra 的工作**：
1. ✅ **调用别人的 API**（如 Fireworks、OpenAI、Anthropic）
2. ✅ **搭建系统**（API Gateway、负载均衡、缓存）
3. ✅ **测试和优化**（性能测试、成本优化、监控告警）
4. ✅ **集成和编排**（把多个 API 组合起来，做成完整的功能）

**不是**：
- ❌ 从零开始训练模型（那是 ML 团队的工作）
- ❌ 从零开始实现 LLM（那是模型团队的工作）

---

## 📊 AI Infra 的常见架构模式

### 模式 1：直接调用 API（最简单）

```
你的应用
    ↓
直接调用 Fireworks API / OpenAI API / Anthropic API
    ↓
返回结果
```

**特点**：
- ✅ **最简单**：直接调用，不需要搭建基础设施
- ✅ **快速上线**：可以快速开发功能
- ⚠️ **依赖外部**：依赖第三方服务的稳定性
- ⚠️ **成本较高**：按调用次数付费

**例子**：
```python
# KYC 项目：直接调用 Fireworks API
import requests

response = requests.post(
    "https://api.fireworks.ai/inference/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "qwen2.5-vl-32b", "messages": [...]}
)
```

---

### 模式 2：API Gateway + 多个 API（常见）

```
你的应用
    ↓
API Gateway（你搭建的）
    ↓
    ├─ Fireworks API
    ├─ OpenAI API
    ├─ Anthropic API
    └─ 自己的模型（可选）
```

**特点**：
- ✅ **统一入口**：所有 API 调用都通过 Gateway
- ✅ **负载均衡**：可以切换不同的 API 提供商
- ✅ **监控和限流**：统一监控、统一限流
- ✅ **故障转移**：一个 API 挂了，自动切换到另一个

**AI Infra 的工作**：
1. ✅ **搭建 API Gateway**（统一入口）
2. ✅ **集成多个 API**（Fireworks、OpenAI、Anthropic）
3. ✅ **实现负载均衡**（根据延迟、成本选择 API）
4. ✅ **实现故障转移**（一个挂了，切到另一个）
5. ✅ **监控和告警**（错误率、延迟、成本）

**例子**：
```python
# API Gateway：统一入口
@app.post("/v1/llm/generate")
def generate(prompt: str):
    # 1. 选择 API（根据延迟、成本、可用性）
    api_provider = select_best_provider()
    
    # 2. 调用选中的 API
    if api_provider == "fireworks":
        return call_fireworks_api(prompt)
    elif api_provider == "openai":
        return call_openai_api(prompt)
    elif api_provider == "anthropic":
        return call_anthropic_api(prompt)
    
    # 3. 监控和记录
    log_metrics(api_provider, latency, cost)
```

---

### 模式 3：自己的 Serving + 调用 API（混合）

```
你的应用
    ↓
你自己的 LLM Serving（SGLang/vLLM）
    ↓
    ├─ 自己的模型（主要）
    └─ 外部 API（备用/fallback）
```

**特点**：
- ✅ **主要用自己的模型**：成本更低、可控性更强
- ✅ **外部 API 作为备用**：自己的模型挂了，切到外部 API
- ✅ **需要自己搭建 Serving**：需要 GPU、需要运维

**AI Infra 的工作**：
1. ✅ **搭建自己的 LLM Serving**（SGLang、vLLM）
2. ✅ **部署自己的模型**（在自己的 GPU 上运行）
3. ✅ **集成外部 API**（作为备用）
4. ✅ **实现故障转移**（自己的模型挂了，切到外部 API）
5. ✅ **监控和优化**（性能、成本、可用性）

**例子**：
```python
# 混合模式：主要用自己的模型，外部 API 作为备用
def generate(prompt: str):
    try:
        # 1. 先用自己的模型（SGLang）
        return call_sglang_api(prompt)
    except Exception as e:
        # 2. 如果自己的模型挂了，切到外部 API
        log_error("SGLang failed, fallback to Fireworks")
        return call_fireworks_api(prompt)
```

---

## 💡 实际例子：KYC 项目 vs SGLang 项目

### KYC 项目：直接调用 API（模式 1）

**架构**：
```
KYC 应用
    ↓
直接调用 Fireworks API
    ↓
返回结果
```

**AI Infra 的工作**：
1. ✅ **调用 Fireworks API**（直接调用，不需要搭建）
2. ✅ **错误处理**（重试、超时、错误码）
3. ✅ **监控和告警**（错误率、延迟、成本）
4. ✅ **测试和优化**（性能测试、成本优化）

**代码**：
```python
# KYC 项目：直接调用 Fireworks API
def call_fireworks_api(prompt):
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "qwen2.5-vl-32b", "messages": [...]},
        timeout=30
    )
    return response.json()
```

**AI Infra 做了什么**：
- ✅ **不是**：从零开始训练模型
- ✅ **而是**：调用 Fireworks API，然后做**错误处理、监控、测试、优化**

---

### SGLang 项目：自己的 Serving（模式 3）

**架构**：
```
SGLang Router（你搭建的）
    ↓
SGLang Workers（你自己的模型）
    ↓
    ├─ Worker 1（GPU 1）
    ├─ Worker 2（GPU 2）
    └─ Worker N（GPU N）
```

**AI Infra 的工作**：
1. ✅ **搭建 SGLang Serving**（部署 SGLang 到 GPU 服务器）
2. ✅ **部署自己的模型**（在自己的 GPU 上运行模型）
3. ✅ **实现负载均衡**（Router 分发请求到不同的 Worker）
4. ✅ **实现故障转移**（Worker 挂了，切到其他 Worker）
5. ✅ **监控和优化**（性能、成本、可用性）

**代码**：
```python
# SGLang Router：负载均衡和故障转移
class SGLangRouter:
    def __init__(self):
        self.workers = [
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000"
        ]
    
    def generate(self, prompt: str):
        # 1. 选择 Worker（负载均衡）
        worker = self.select_worker()
        
        # 2. 调用 Worker
        try:
            return self.call_worker(worker, prompt)
        except Exception as e:
            # 3. 如果 Worker 挂了，切到其他 Worker
            return self.fallback_to_other_worker(prompt)
```

**AI Infra 做了什么**：
- ✅ **不是**：从零开始实现 LLM（那是模型团队的工作）
- ✅ **而是**：使用 SGLang 框架，然后做**部署、负载均衡、故障转移、监控、优化**

---

## 🔧 AI Infra 的常见工作内容

### 1. 调用别人的 API（最常见）

**做什么**：
- ✅ **集成 API**：调用 Fireworks、OpenAI、Anthropic 等 API
- ✅ **错误处理**：重试、超时、错误码处理
- ✅ **监控和告警**：错误率、延迟、成本监控
- ✅ **测试和优化**：性能测试、成本优化

**例子**：
```python
# KYC 项目：调用 Fireworks API
def call_fireworks_api(prompt):
    # 错误处理
    try:
        response = requests.post(..., timeout=30)
        return response.json()
    except TimeoutError:
        # 重试
        return retry_call()
    except Exception as e:
        # 记录错误
        log_error(e)
        raise
```

---

### 2. 搭建 API Gateway（常见）

**做什么**：
- ✅ **统一入口**：所有 API 调用都通过 Gateway
- ✅ **负载均衡**：根据延迟、成本选择 API
- ✅ **故障转移**：一个 API 挂了，切到另一个
- ✅ **限流和缓存**：控制请求频率、缓存结果

**例子**：
```python
# API Gateway：统一入口
class LLMGateway:
    def __init__(self):
        self.providers = [
            {"name": "fireworks", "url": "...", "cost": 0.001},
            {"name": "openai", "url": "...", "cost": 0.002},
            {"name": "anthropic", "url": "...", "cost": 0.003}
        ]
    
    def generate(self, prompt: str):
        # 1. 选择 API（根据成本、延迟）
        provider = self.select_best_provider()
        
        # 2. 调用 API
        return self.call_provider(provider, prompt)
```

---

### 3. 搭建自己的 Serving（高级）

**做什么**：
- ✅ **部署模型**：在自己的 GPU 上运行模型（SGLang、vLLM）
- ✅ **负载均衡**：Router 分发请求到不同的 Worker
- ✅ **故障转移**：Worker 挂了，切到其他 Worker
- ✅ **监控和优化**：性能、成本、可用性

**例子**：
```python
# SGLang Router：自己的 Serving
class SGLangRouter:
    def generate(self, prompt: str):
        # 1. 选择 Worker（负载均衡）
        worker = self.select_worker()
        
        # 2. 调用 Worker
        return self.call_worker(worker, prompt)
```

---

### 4. 测试和优化（必须）

**做什么**：
- ✅ **性能测试**：延迟、吞吐量测试
- ✅ **成本优化**：选择成本更低的 API
- ✅ **监控告警**：错误率、延迟、成本监控
- ✅ **A/B 测试**：测试不同 API 的效果

**例子**：
```python
# 性能测试
def test_api_performance():
    latencies = []
    for i in range(100):
        start = time.time()
        call_api(prompt)
        latencies.append(time.time() - start)
    
    p95 = np.percentile(latencies, 95)
    print(f"p95 latency: {p95}s")
```

---

## 📊 什么时候用别人的 API vs 自己搭建？

### 用别人的 API（推荐，大部分情况）

**什么时候用**：
- ✅ **快速开发**：需要快速上线功能
- ✅ **成本可控**：按调用次数付费，成本可控
- ✅ **不需要 GPU**：不需要自己买 GPU
- ✅ **小规模使用**：每天调用量 < 100 万次

**例子**：
- ✅ **KYC 项目**：直接调用 Fireworks API
- ✅ **小公司**：直接调用 OpenAI API
- ✅ **PoC（概念验证）**：快速验证想法

---

### 自己搭建 Serving（高级，大规模）

**什么时候用**：
- ✅ **大规模使用**：每天调用量 > 1000 万次
- ✅ **成本敏感**：自己的 GPU 成本更低
- ✅ **可控性要求高**：需要完全控制模型和性能
- ✅ **有 GPU 资源**：有 GPU 服务器和运维团队

**例子**：
- ✅ **SGLang 项目**：自己的 Serving，部署在自己的 GPU 上
- ✅ **大公司**：有自己的 GPU 集群
- ✅ **生产环境**：需要高可用、高性能

---

### 混合模式（推荐，生产环境）

**架构**：
```
API Gateway
    ↓
    ├─ 自己的 Serving（主要，成本低）
    └─ 外部 API（备用，高可用）
```

**什么时候用**：
- ✅ **生产环境**：需要高可用
- ✅ **成本优化**：主要用自己的模型，降低成本
- ✅ **故障转移**：自己的模型挂了，切到外部 API

**例子**：
```python
# 混合模式：主要用自己的模型，外部 API 作为备用
def generate(prompt: str):
    try:
        # 1. 先用自己的模型（成本低）
        return call_sglang_api(prompt)
    except Exception as e:
        # 2. 如果自己的模型挂了，切到外部 API（高可用）
        log_error("SGLang failed, fallback to Fireworks")
        return call_fireworks_api(prompt)
```

---

## 🎯 AI Infra 的工作总结

### 你的理解是对的！

**AI Infra 的工作**：
1. ✅ **调用别人的 API**（如 Fireworks、OpenAI、Anthropic）
2. ✅ **搭建系统**（API Gateway、负载均衡、缓存）
3. ✅ **测试和优化**（性能测试、成本优化、监控告警）
4. ✅ **集成和编排**（把多个 API 组合起来，做成完整的功能）

**不是**：
- ❌ 从零开始训练模型（那是 ML 团队的工作）
- ❌ 从零开始实现 LLM（那是模型团队的工作）

---

### 实际例子对比

| 项目 | 模式 | AI Infra 的工作 |
|------|------|----------------|
| **KYC 项目** | 直接调用 API | 调用 Fireworks API + 错误处理 + 监控 + 测试 |
| **SGLang 项目** | 自己的 Serving | 部署 SGLang + 负载均衡 + 故障转移 + 监控 |
| **混合模式** | 自己的 Serving + 外部 API | 部署自己的模型 + 集成外部 API + 故障转移 |

---

## 💡 总结

### 核心观点

**是的，大部分 AI Infra 都是调用别人的 API，然后做"使用、搭建、测试、优化"的工作。**

### AI Infra 的工作

1. ✅ **调用别人的 API**（Fireworks、OpenAI、Anthropic）
2. ✅ **搭建系统**（API Gateway、负载均衡、缓存）
3. ✅ **测试和优化**（性能测试、成本优化、监控告警）
4. ✅ **集成和编排**（把多个 API 组合起来）

### 什么时候用 API vs 自己搭建

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **快速开发** | 用别人的 API | 快速上线，不需要 GPU |
| **小规模使用** | 用别人的 API | 成本可控，按调用次数付费 |
| **大规模使用** | 自己搭建 Serving | 成本更低，可控性更强 |
| **生产环境** | 混合模式 | 高可用 + 成本优化 |

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A2_B3_C1_D1 API 调用的底层细节与开发者视角（[KYC_Day01_A2_B3_C1_D1_API调用的底层细节与开发者视角.md](./KYC_Day01_A2_B3_C1_D1_API调用的底层细节与开发者视角.md)） |
| **Related** | API 调用、AI Infra、LLM Serving、API Gateway、负载均衡、故障转移 |
