# AI Infra + LLM：7天冲刺 + 30天体系化计划

**目标**：几乎纯小白 + 只冲 LLM/AI Infra & Pipeline + 先能面试讲清楚  
**标准**：30天后能把系统讲成闭环的人，而不是"背 SD 八股"  
**载体**：每天产出一个可复用的"组件/文档/小 demo"

---

## 🎯 核心定位：AI Infra + LLM

**两条主线**：
1. **API/服务设计骨架**（资源建模、版本、幂等、错误码）——直接按 Google/Azure 的公开规范学，省掉瞎猜
2. **LLM 推理与 MLOps 平台骨架**（Serving、缓存/批处理、评估/监控、特征/数据链路）——参考 vLLM、SGLang、SRE、Uber/Doordash 的平台文章

---

## 🚀 7天冲刺计划（目标：能讲 2 个"端到端模块"并落地一个最小可跑 demo）

**每天固定 4 件事**：读 30–45min → 写 1 页 doc → 画 1 张图 → 写/改 50–150 行代码或伪码。  
**代码不求生产级，求"闭环 + 可讲清"。**

---

### Day 1：System Design 最小骨架（面试开场就用这套）

**学什么**：需求澄清 + SLO + 关键路径 + 数据流图（MVP 版本）。用 SRE 的"可用性/延迟/容量/风险"思维定指标。

**产出**：1 页模板（以后每题直接套）

**模板内容**：
- **功能/非功能需求**
- **SLO**（p95 延迟、成功率）
- **数据流**（入口→异步→存储→worker→回写）
- **风险清单**（幂等/重试/限流/审计）

**实战练习**：
- 用 KYC 项目填充模板
- 用 LLM Gateway 项目填充模板

**参考资源**：
- Google SRE Book: SLO, SLI, SLAs
- 《Designing Data-Intensive Applications》第 1-2 章

---

### Day 2：API 设计（把"系统说清楚"的第一步）

**学什么**：资源建模、命名、分页、错误码、版本策略。直接照 Google Cloud API Design Guide + Azure API best practices。

**练习**：为一个 `kyc_case` 设计 6 个接口（create/get/list/cancel/retry/webhook）

**产出**：
- OpenAPI 伪规范（不用真跑）
- 错误码表（`INVALID_DOC`, `PROVIDER_TIMEOUT`, `RATE_LIMITED`）

**实战动作**：
1. **学习 Google Cloud API Design Guide**
   - 资源命名规范（`kyc_cases/{case_id}`）
   - HTTP 方法选择（GET/POST/PUT/DELETE）
   - 分页策略（cursor-based pagination）
   - 错误码设计（4xx/5xx 分类）

2. **设计 KYC Case API**
   ```
   POST   /v1/kyc/cases              # 创建 case
   GET    /v1/kyc/cases/{case_id}    # 查询 case
   GET    /v1/kyc/cases              # 列表（支持分页）
   POST   /v1/kyc/cases/{case_id}:cancel  # 取消
   POST   /v1/kyc/cases/{case_id}:retry   # 重试
   POST   /v1/kyc/webhooks           # Webhook 回调
   ```

3. **设计错误码表**
   | 错误码 | HTTP 状态码 | 说明 |
   |--------|------------|------|
   | `INVALID_DOC` | 400 | 文档格式不支持 |
   | `PROVIDER_TIMEOUT` | 504 | 第三方服务超时 |
   | `RATE_LIMITED` | 429 | 请求频率超限 |
   | `CASE_NOT_FOUND` | 404 | Case 不存在 |
   | `CASE_ALREADY_PROCESSING` | 409 | Case 正在处理中 |

**参考资源**：
- [Google Cloud API Design Guide](https://cloud.google.com/apis/design)
- [Azure REST API Guidelines](https://github.com/microsoft/api-guidelines)

---

### Day 3：队列与"至少一次"交付（KYC/LLM pipeline 都绕不开）

**学什么**：异步、重试、DLQ、幂等键（`request_id`、`dedupe_key`）

**练习**：把 KYC 流程做成事件驱动：`case_created` → `check_requested` → `check_done` → `decision_made`

**产出**：
1. **1 张状态机图**（QUEUED/RUNNING/NEEDS_REVIEW/APPROVED/REJECTED/FAILED）
2. **1 页"重试策略 + DLQ 处理手册（runbook）"**（SRE 的 oncall/runbook 思路就是加分项）

**实战动作**：

1. **设计状态机**
   ```
   QUEUED → RUNNING → (APPROVED | REJECTED | NEEDS_REVIEW | FAILED)
                    ↓
                  CANCELED
   ```

2. **设计事件驱动流程**
   - `case_created`：创建 case，写入 `kyc_cases` 表，发送消息到队列
   - `check_requested`：从队列消费，开始执行检查
   - `check_done`：单个检查完成，更新 `kyc_checks` 表
   - `decision_made`：所有检查完成，计算 risk_score，做出决策

3. **设计重试策略**
   - 可恢复错误（`PROVIDER_TIMEOUT`）：指数退避重试（1s, 2s, 4s, 8s）
   - 不可恢复错误（`INVALID_DOC`）：直接失败，不重试
   - 超过 N 次（默认 3 次）进入 DLQ

4. **设计幂等机制**
   - `request_id UNIQUE`：创建 case 时插入，冲突就返回已有 case_id
   - `(case_id, check_type, provider, version) UNIQUE`：避免重复调用第三方

**参考资源**：
- 《Designing Data-Intensive Applications》第 11 章（流处理）
- AWS SQS Dead Letter Queues 文档

---

### Day 4：LLM Serving 入门：吞吐/延迟到底怎么来的

**学什么**：prefill vs decode、连续批处理、KV cache、为什么要"PagedAttention/continuous batching"。看 vLLM 文档即可。

**练习**：写一个"LLM Gateway"设计（不必真跑大模型）：
- `/generate` 同步
- `/jobs:submit` 异步（长文本）
- 缓存：prefix cache（命中就少算）

**产出**：1 页讲清 TTFT / tokens/s / p95 延迟怎么优化（batching、缓存、并发上限）

**实战动作**：

1. **理解 LLM Serving 核心概念**
   - **Prefill**：处理用户输入 prompt，生成 KV cache
   - **Decode**：基于 KV cache 逐个生成 token
   - **Continuous Batching**：动态批处理，不同请求可以同时 prefilling 或 decoding
   - **PagedAttention**：KV cache 分页管理，避免内存碎片

2. **设计 LLM Gateway API**
   ```
   POST /v1/llm/generate          # 同步生成（短文本）
   POST /v1/llm/jobs:submit       # 异步提交（长文本）
   GET  /v1/llm/jobs/{job_id}     # 查询任务状态
   ```

3. **设计性能指标**
   - **TTFT（Time To First Token）**：第一个 token 的延迟，主要受 prefill 影响
   - **tokens/s（Throughput）**：每秒生成的 token 数，主要受 decode 速度影响
   - **p95 延迟**：95% 请求的端到端延迟

4. **设计优化策略**
   - **Batching**：连续批处理，提高 GPU 利用率
   - **Cache**：Prefix cache（相同前缀复用 KV cache），减少 prefill 时间
   - **并发上限**：根据 GPU 显存和 latency 要求设置最大并发数

**参考资源**：
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sglang.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)

---

### Day 5：结构化输出（让 LLM 变成"可审计的系统组件"）

**学什么**：Schema-constrained 输出、JSON/Pydantic、为什么对审计/日志/回放很关键（你做 KYC/医院质控天然用得上）。SGLang 的定位就是"结构化程序 + 高性能服务"。

**练习**：定义 `KYCDecisionSchema`（`risk_score`、`reasons`、`evidence`、`next_action`）

**产出**：一份"输入→输出 JSON 契约" + 日志字段规范（`case_id`、`model_version`、`prompt_hash`）

**实战动作**：

1. **设计 Schema**
   ```python
   class KYCDecisionSchema(BaseModel):
       risk_score: int = Field(ge=0, le=100)
       decision: Literal["APPROVED", "REJECTED", "NEEDS_REVIEW"]
       reasons: List[str]
       evidence: Dict[str, Any]
       next_action: Optional[str]
   ```

2. **设计输入/输出契约**
   ```json
   // 输入
   {
     "case_id": "c_001",
     "documents": [...],
     "user_info": {...}
   }
   
   // 输出
   {
     "case_id": "c_001",
     "risk_score": 35,
     "decision": "NEEDS_REVIEW",
     "reasons": ["ID photo quality low", "Address verification pending"],
     "evidence": {
       "id_confidence": 0.85,
       "face_match_score": 0.92
     },
     "next_action": "Request additional documents"
   }
   ```

3. **设计日志字段规范**
   - `case_id`：关联所有相关日志
   - `model_version`：记录使用的模型版本（便于回滚）
   - `prompt_hash`：记录 prompt 的 hash（便于追踪 prompt 变更）
   - `latency_ms`：记录每个步骤的耗时
   - `tokens_used`：记录 token 使用量（成本追踪）

**参考资源**：
- SGLang Structured Generation 文档
- Pydantic 文档

---

### Day 6：ML/AI Pipeline 视角（训练/特征/部署/监控）

**学什么**：平台化组件长啥样：数据→训练→部署→在线推理→监控。看 Uber Michelangelo 的端到端视角（以及后来走向 GenAI 的演进）。

**练习**：给 KYC 或 LLM 服务加"离线评估 + 线上监控"的设计：
- 离线：golden set、回放（replay）
- 线上：漂移/错误码分布/拒绝率

**产出**：Dashboard 指标清单（10 个以内）

**实战动作**：

1. **设计离线评估流程**
   - **Golden Set**：50-200 条标注好的测试用例
   - **Replay**：用新模型/新 prompt 在 Golden Set 上重新推理，对比结果
   - **指标**：准确率、召回率、F1-score、成本（tokens）

2. **设计线上监控指标**
   - **稳定性指标**：错误率、延迟（p95/p99）、可用性
   - **业务指标**：通过率、拒绝率、人工审核率
   - **成本指标**：tokens/s、API 调用成本、GPU 使用率
   - **漂移检测**：输入分布变化、输出分布变化

3. **设计 Dashboard**
   | 指标类型 | 指标名称 | 阈值 |
   |---------|---------|------|
   | 稳定性 | Error Rate | < 2% |
   | 稳定性 | p95 Latency | < 15s |
   | 业务 | Approval Rate | > 80% |
   | 业务 | Manual Review Rate | < 20% |
   | 成本 | Tokens/Request | < 5000 |

**参考资源**：
- [Uber Michelangelo Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
- [DoorDash Feature Store](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/)

---

### Day 7：做一次"面试版演练"（15–20 分钟讲完）

**你要能把下面两题讲顺**：

1. **Design a KYC case processing system**（事件驱动 + 审计 + 人工兜底）
2. **Design an LLM inference gateway**（batching/caching/限流/灰度）

**产出**：每题 1 页：架构图 + API + 数据模型 + SLO + 风险与取舍 + runbook

**实战动作**：

#### 题目 1：KYC Case Processing System

**架构图**：
```
Client → API Gateway → KYC Intake Service
                           ↓
                       Queue (RabbitMQ/Redis)
                           ↓
                    Orchestrator Worker
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    OCR Check      Face Match Check    Watchlist Check
        ↓                  ↓                  ↓
    Rule Engine → Risk Score Calculator → Decision Engine
        ↓
    Audit Log + Database
```

**API**：见 Day 2

**数据模型**：见 Day 3

**SLO**：
- Intake API p95 < 100ms（只入库+入队）
- 大多数 case 在 30–120s 内出结果
- 可用性 99.9%

**风险与取舍**：
- 风险：第三方服务超时、数据丢失、重复处理
- 取舍：同步 vs 异步（选择异步，提高吞吐）、强一致 vs 最终一致（选择最终一致，提高性能）

**Runbook**：见 Day 3

---

#### 题目 2：LLM Inference Gateway

**架构图**：
```
Client → API Gateway → LLM Gateway
                           ↓
                    Load Balancer
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    vLLM Worker 1    vLLM Worker 2    vLLM Worker N
        ↓                  ↓                  ↓
    Prefix Cache    Prefix Cache    Prefix Cache
```

**API**：见 Day 4

**数据模型**：
- `llm_jobs`：job_id, status, request_json, response_json, latency_ms
- `llm_cache`：prefix_hash, kv_cache, hit_count

**SLO**：
- TTFT < 2s（p95）
- Throughput > 100 tokens/s（p50）
- 可用性 99.9%

**风险与取舍**：
- 风险：GPU OOM、延迟飙升、成本超限
- 取舍：延迟 vs 吞吐（根据场景选择，实时对话优先延迟，批量处理优先吞吐）、成本 vs 质量（根据预算选择模型大小）

**Runbook**：
- GPU OOM：降低 batch size 或并发数
- 延迟飙升：检查 queue 长度，扩容 worker
- 成本超限：启用 cache、降级到更小的模型

---

## 📅 30天体系化计划（目标：形成"组件库 + 题库覆盖 + 能打 mock"）

**建议按 4 周走，每周一个主题 + 2 个"代表题"。每周末做一次 45min mock（你自己录音复盘也行）。**

---

### Week 1：基础骨架与分布式常识（不追求大全，追求能讲）

**学习资源**：DDIA 用来建立"存储/一致性/复制/事务/流处理"的直觉（不需要通读，抓你用得上的章节）。

**目标产物**：组件库 v1

1. **幂等**（`request_id` + unique constraint）
2. **重试**（指数退避）+ DLQ
3. **限流**（token bucket 思路即可）
4. **可观测**（4 golden signals + 日志规范）——SRE 的监控/告警框架能直接套

**实战动作**：

#### Day 8-9：幂等组件

- **学习**：幂等的定义、实现方式（unique constraint、idempotency key）
- **实现**：为 KYC API 添加幂等检查（`request_id UNIQUE`）
- **产出**：`src/middleware/idempotency.py` + 单元测试

#### Day 10-11：重试 + DLQ 组件

- **学习**：指数退避、重试策略、DLQ 处理
- **实现**：为第三方 API 调用添加重试机制（3 次重试，超过进入 DLQ）
- **产出**：`src/utils/retry.py` + `src/workers/dlq_worker.py` + runbook

#### Day 12-13：限流组件

- **学习**：Token Bucket、Leaky Bucket、Rate Limiting 策略
- **实现**：为 LLM Gateway 添加限流（每个用户 10 req/s）
- **产出**：`src/middleware/rate_limiter.py` + 单元测试

#### Day 14：可观测组件

- **学习**：4 Golden Signals（Latency、Traffic、Errors、Saturation）、日志规范
- **实现**：为 KYC 系统添加结构化日志（`case_id`、`trace_id`、`latency_ms`）
- **产出**：`src/observability/logger.py` + Dashboard 配置

---

### Week 2：LLM 推理系统设计（面试最热的一条线）

**代表题 A**：LLM Serving Platform / Gateway（vLLM/SGLang 取舍、KV cache、batching、路由、多模型）

**代表题 B**：RAG Pipeline（索引构建、增量更新、检索/重排、缓存、评估）

**目标产物**：
1. **1 份"LLM latency 分解模板"**（TTFT、decode、并发、队列）
2. **1 份"RAG 评估清单"**（检索命中率、答案一致性、延迟/成本）

**实战动作**：

#### Day 15-17：LLM Serving Platform 设计

- **学习**：vLLM vs SGLang 的取舍、PagedAttention、Continuous Batching
- **设计**：多模型路由、负载均衡、故障转移
- **产出**：架构图 + API 设计 + 数据模型 + SLO + Runbook

#### Day 18-21：RAG Pipeline 设计

- **学习**：向量索引（FAISS、Pinecone）、检索策略（BM25 + 向量混合）、重排序
- **设计**：索引构建流程、增量更新、检索/重排/生成流程
- **产出**：架构图 + API 设计 + 评估指标 + Runbook

---

### Week 3：AI/ML Pipeline 平台化（数据→特征→训练→部署）

**参考**：Uber 平台演进、Feature platform 文章、DoorDash feature store 思路（为什么要离线/在线一致、为什么要多存储组合）。

**代表题**：
1. **Feature Store / Online Features**
2. **Model Deployment & Safety**（灰度、shadow、回滚、监控）

**目标产物**：
1. **1 张"训练链路 vs 在线链路"对照图**
2. **1 页"上线策略（shadow→1%→10%→100%）+ 回滚开关"**

**实战动作**：

#### Day 22-24：Feature Store 设计

- **学习**：离线特征 vs 在线特征、特征版本管理、特征一致性
- **设计**：Feature Store 架构（离线存储 + 在线存储）、特征服务 API
- **产出**：架构图 + API 设计 + 数据模型

#### Day 25-28：Model Deployment & Safety 设计

- **学习**：Shadow mode、Canary deployment、Blue-Green deployment、回滚策略
- **设计**：模型上线流程（shadow→1%→10%→100%）、监控告警、自动回滚
- **产出**：部署流程图 + 监控指标 + Runbook

---

### Week 4：把 KYC 做成你的"招牌系统设计题"（与你背景强绑定）

**KYC 非常适合体现**：合规/审计/风控/人工兜底/多 provider 容错

**目标产物**：KYC 端到端 design doc v1（2–3 页）

- API（Google/Azure 风格）
- event-driven pipeline（重试/DLQ）
- LLM 结构化审核（可选模块）
- SLO + oncall runbook（SRE 味儿）

**实战动作**：

#### Day 29-30：KYC 端到端 Design Doc

- **整合前 3 周的所有内容**：API、数据模型、架构、部署、监控
- **写完整的 Design Doc**：功能需求、非功能需求、架构设计、数据模型、API 设计、部署策略、监控告警、Runbook
- **准备面试表达**：30 秒 / 2 分钟 / 5 分钟三个版本

**产出**：
- `docs/KYC_SYSTEM_DESIGN.md`（完整的 Design Doc）
- `docs/KYC_INTERVIEW_SCRIPT.md`（面试表达脚本）

---

## 🎯 30 天里你只需要"吃透的 6 道题"（其余都能类比）

1. **LLM Inference Gateway / Serving Platform**（vLLM/SGLang）
2. **RAG System**（indexing + retrieval + eval）
3. **Event-driven Case Processing**（KYC/风控/工单系统通用）
4. **Feature Store / Feature Platform**（离线/在线一致）
5. **Model Deployment Safety**（shadow/canary/rollback）
6. **Observability & SLO-driven ops**（SRE）

---

## ⚠️ 你现在最容易走偏的坑（我直接帮你避开）

1. **别补"分布式大全"**：你 30 天补不完，也不需要。用 DDIA/SRE 做框架即可。
2. **别只看帖子不产出**：system design 是"输出型技能"。每天必须留下一页 doc/一张图/一个组件。
3. **别追"最优解"**：面试看的是"取舍逻辑 + 风险控制 + 可落地"，不是神架构。

---

## 💡 下一步行动

**如果你想立即开始**，我可以为你提供：

1. **Day 1 的"一页 system design 模板"**，按你最关心的两题（KYC + LLM Gateway）各填一份"标准答案版"
2. **你照着读 3 遍、自己复述 3 遍**，就能开始进入正循环

**记住**：
- 不是"学完所有内容"，而是"用项目实战技术"
- 不是"全部"，而是"最小闭环"——把核心模块练到"闭眼也能讲"
- 不是"知道概念"，而是"知道如何设计、如何验证、如何上线"
- **所有训练都基于 KYC 和 SGLang 的实际项目内容，不是假设的场景**

**加油！** 🎉

---

## 🎯 AI Infra + LLM：6 个经典 Case（背、讲、画）

**核心定位**：AI Infra + LLM + 一点点传统 ML 的经典 case，够你"背、讲、画"。最重要的 3 个优先（优先级=最适合你现在的方向、最像面试、最能迁移到任何传统行业）。

**学习方法**：背、讲、画，至少对自己的系统很清楚。基于你现有的项目（KYC、SGLang）来理解，大部分是你做过的项目，这样能更好更快地理解。

---

## 🔥 Top 3（最优先背到滚瓜烂熟）

### ① LLM Serving Platform（推理网关 / Inference Gateway）

**一句话背诵**：把"用户请求→GPU 生成→流式返回"做成低延迟、高吞吐、可控成本、可观测的平台。

**画图盒子**：
```
Client → API Gateway(Auth/RateLimit) → Router → Request Queue 
    → Inference Workers(GPU) → Streaming Response

旁路：KV Cache / Prefix Cache、Metrics/Logs/Tracing、Fallback(多模型/多provider)
```

**必须讲清的 3 个机制**：
1. **Continuous Batching**（动态拼 batch 提吞吐）
2. **Paged KV Cache / PagedAttention**（KV cache 分页管理，缓解显存碎片与 OOM）
3. **Prefix Cache / RadixAttention**（共享前缀复用，加速重复提示词）

**关键指标**（你每次都报）：TTFT、tokens/s、p95/p99、GPU 利用率、cache hit rate、$/1k tokens。

**最常见坑**：OOM/碎片、尾延迟爆炸、队列堆积、缓存失效导致吞吐抖动、无成本护栏。

**基于你的项目**：参考 SGLang 的实际实现（`sgl-router/`、`python/sglang/srt/`）

---

### ② RAG QA System（检索增强问答 + 可信评估）

**一句话背诵**：用"检索 + 生成"把 LLM 接上外部知识库，降低幻觉，并能评估与回放。

**画图盒子**：
```
Ingest(Docs) → Chunk → Embed → Vector Index(+BM25可选)
Query → Retrieve(topK) → (Rerank可选) → Prompt Compose → LLM → Answer + Citations

旁路：Eval/Replay、Feedback、Guardrails
```

**必须讲清的 3 个点**：
1. **RAG 的基本范式**（retriever + generator）
2. **"怎么评估"**：离线回放 + 在线监控（LangSmith/RAGAS 这种思路）
3. **失败模式**：检索不到/检索错、文档过期、提示词注入、引用不一致

**关键指标**：Recall@K、Answer faithfulness、引用覆盖率、延迟、成本。

**基于你的项目**：可以基于 KYC 项目扩展（知识库检索 + LLM 结构化审核）

---

### ③ ML Platform / Feature Store（把 ML 变成可规模化交付）

**一句话背诵**：把"数据→训练→部署→在线预测→监控→再训练"做成可复用流水线的平台。

**画图盒子**（两层）：
```
平台层：
Data Lake/Warehouse → Feature Pipelines → Feature Store(offline/online) 
    → Training → Model Registry → Serving → Monitoring

产品层：
业务服务调用在线特征 + 模型预测
```

**权威参考**（背名字就加分）：
- **Uber Michelangelo**：端到端 ML 平台（训练/部署/预测/监控闭环）
- **DoorDash Feature Store (Redis)**：在线特征低延迟、大规模读取优化
- **Feast**：开源 feature store 的标准抽象

**关键指标**：训练-服务一致性、线上特征延迟、数据新鲜度、模型漂移、线上业务 KPI（转化/拒付/投诉等）。

**基于你的项目**：可以基于 KYC 项目扩展（特征工程 + 模型部署 + 监控）

---

## 📚 另外 3 个（第二梯队，面试也常见）

### ④ 实时风控 / KYC Decisioning（规则 + ML + 审计链）

**一句话背诵**：把"事件流→特征→打分→决策→审计"做成可追责系统（误杀/漏放都有成本）。

**参考**：PayPal 也强调 ML 在支付欺诈检测的作用与形态（监督/半监督等）。

**基于你的项目**：这就是你的 KYC 项目！你已经掌握得差不多了。

---

### ⑤ 推荐/排序系统（Feed / Recsys）

**一句话背诵**：两阶段（candidate generation + ranking）+ 在线特征 + A/B。YouTube 推荐是经典论文级案例。

**核心组件**：
- **Candidate Generation**：粗排（召回大量候选）
- **Ranking**：精排（排序打分）
- **在线特征**：实时特征计算
- **A/B Testing**：效果验证

**基于你的项目**：可以参考 KYC 的风险打分流程（多 check 聚合 → 规则引擎打分）

---

### ⑥ Smart Compose（实时写邮件/补全）

**一句话背诵**：按键级实时生成，核心是极低延迟 + 大规模 serving。论文里就强调 90% 延迟门槛级别要求。

**核心特点**：
- **极低延迟**：p90 < 100ms
- **大规模 serving**：支持百万级用户
- **Prefix Cache**：缓存常见前缀

**基于你的项目**：参考 SGLang 的 Prefix Cache（RadixAttention）

---

## 📋 "背、讲、画"的统一模板（每个 case 都照这个说）

**你每次开口就按这 8 行走（强制闭环）**：

1. **目标 & 用户**：做什么、给谁用
2. **SLO**：延迟/可用性/成本目标（先定"能接受的失败"）
3. **核心数据流**：请求从哪来，怎么处理，结果去哪
4. **存储**：在线/离线分别放什么
5. **扩展**：怎么水平扩容（worker、分片、队列）
6. **可靠性**：幂等、重试、DLQ、灰度/回滚
7. **观测**：4 golden signals + 业务指标
8. **风险 & 取舍**：你主动说"我选择 A 而不是 B 的原因"

---

## 💡 基于你现有项目的实战建议

**如果你说"就按我现在的 KYC 项目来练"**，我可以把 Top1/Top2/Top3 直接套成同一张"可面试的端到端图"（API + 队列事件 + DB schema + 指标 + 失败策略），你照着背讲画，一周内就能把自己的系统讲得非常像"做过的人"。

**学习顺序建议**：
1. **KYC 项目**（你已经掌握得差不多了）→ 作为基础，理解事件驱动 + 审计链
2. **LLM Serving Platform**（基于 SGLang）→ 理解 batching、caching、routing
3. **RAG QA System**（可以基于 KYC 扩展）→ 理解检索 + 生成 + 评估
4. **ML Platform / Feature Store**（可以基于 KYC 扩展）→ 理解特征工程 + 模型部署

**对于 KYC**：你已经掌握得差不多了，现在主要是把知识系统化，用"背、讲、画"的模板讲清楚。

---

## 🎯 下一步行动

**现在就开始**：

1. **从 KYC 项目开始**（你已经掌握了）
   - 用"背、讲、画"的模板重新讲一遍
   - 画一张端到端的架构图
   - 准备 30 秒 / 2 分钟 / 5 分钟三个版本

2. **然后学习 LLM Serving Platform**（基于 SGLang）
   - 理解 Continuous Batching、PagedAttention、RadixAttention
   - 设计一个 LLM Gateway（API + 数据模型 + SLO）
   - 用"背、讲、画"的模板讲清楚

3. **最后学习 RAG QA System 和 ML Platform**（基于你的项目扩展）
   - 理解检索 + 生成 + 评估
   - 理解特征工程 + 模型部署
   - 用"背、讲、画"的模板讲清楚

**记住**：
- 不是"学完所有内容"，而是"用项目实战技术"
- 不是"全部"，而是"最小闭环"——把核心模块练到"闭眼也能讲"
- 不是"知道概念"，而是"知道如何设计、如何验证、如何上线"
- **所有训练都基于你的实际项目（KYC、SGLang），不是假设的场景**

**加油！** 🎉