# System Design 基础知识

**核心定位**：AI Infra + LLM（聚焦 LLM Serving、ML Pipeline、API 设计）

---

## 🎯 什么是 System Design？

System Design 是设计一个能够在现实世界中运行的系统的过程。

**System Design 可以覆盖任何 topic**：从建大桥到做手机到做网站。

**局限到软件，尤其是面试场景**，大体局限于以下几种：

---

## 📋 System Design 主要类型

### 1. Web Application（Web 应用）

**典型例子**：
- 设计一个 e-commerce 商品的 detail page（参考 Amazon 商品 detail page）
- 设计一个 online 小游戏
- 设计一个论坛

**特点**：前端展示 + 后端服务 + 数据库

---

### 2. 简单的 Web Service（Web 服务）

**典型例子**：
- 设计一个 short URL service

**特点**：轻量级 API + 数据存储

---

### 3. 实时/半实时的消息 Update（消息系统）

**典型例子**：
- Messenger
- News Feed

**特点**：实时推送 + 消息队列 + 数据同步

---

### 4. 数据系统（Data System）

**典型例子**：
- Top URL hits
- Unique URL hits

**特点**：数据聚合 + 统计分析 + 实时/批量处理

---

### 5. 内容分发系统（CDN）

**典型例子**：
- 设计 Netflix
- CDN

**特点**：内容缓存 + 地理分布 + 负载均衡

---

### 6. 专业知识领域（Domain-Specific）

**典型例子**：
- 推荐系统
- 分布式系统基础架构
- 搜索系统
- **LLM Serving Platform**（AI Infra）
- **ML Pipeline**（MLOps）

**特点**：特定领域的深度知识 + 系统化思维

---

## ⚠️ 关于 System Design 的认知

**个人认为**，要想在 System Design 上面达到像刷题那样的熟练程度，对于刚刚入行一两年甚至三四年的朋友来说是不可能的。因为这些问题要想准确答出已经远远超出了对 junior engineer 的要求，甚至已经是 senior engineer 的水准。

**但是**，即使面试官并没有期望你达到 senior engineer 的水准，至少他还是想要通过这个问题来摸清你的工程经验。如果所答完全非所问，那么给面试官留下的印象将是非常糟糕的。

**你的方案可以不是最优的，甚至可能离最优有一段距离，但是你的方案一定不能是 ridiculous 的，起码在小的 scale 上要能有可行性，可以展现你的一些基本 design sense。**

---

## 📚 基础知识准备（AI Infra + LLM 重点）

### 1. 数据库（Database）

#### Relational Database（关系型数据库）
- **了解**：Oracle/PostgreSQL 的基本知识
- **了解**：数据库的 partition（分区）
- **了解**：查询优化
- **了解**：数据库的 replication（复制）

#### NoSQL Databases（非关系型数据库）
- **Key-Value Database**：Riak / DynamoDB
- **Document-based**：MongoDB
- **Graph-based**：Neo4j
- **BigTable (Column-based)**：HBase

**重要概念**：
- **Eventually Consistency**（最终一致性）：DynamoDB 的 paper 需要明白
- **CAP 定理**：Consistency、Availability、Partition tolerance，三者不能同时满足

**AI Infra 中的应用**：
- KYC 系统的 case 数据存储（PostgreSQL）
- Feature Store 的特征存储（Redis/DynamoDB）
- LLM 的 KV Cache 管理（内存数据库）

---

### 2. 队列服务（Queue Service）

- **了解**：Kafka 或者 Kinesis
- **明白**：队列服务的应用场景

**AI Infra 中的应用**：
- KYC 的异步处理流程（`case_created` → `check_requested` → `check_done`）
- LLM 的请求队列（异步生成任务）
- 重试队列 + DLQ（Dead Letter Queue）

**参考**：RabbitMQ、Redis Streams、AWS SQS

---

### 3. Web 层（Web Layer）

- **了解**：MVC（Model-View-Controller）
- **具体技术**：Spring / Node.js / FastAPI

**AI Infra 中的应用**：
- API Gateway（统一入口、认证、限流）
- RESTful API 设计（Google/Azure 规范）

---

### 4. 前端（Frontend）

- **了解**：JavaScript / HTML5
- **了解**：SOAP 和 RESTful

**AI Infra 中的应用**：
- LLM Gateway 的前端界面（任务状态查询）
- Dashboard（监控指标可视化）

---

### 5. 缓存（Cache）

- **理解**：如何以及在何种情况下运用 cache 降低 latency

**AI Infra 中的应用**：
- **Prefix Cache**：LLM 推理中的前缀缓存（RadixAttention）
- **Feature Cache**：Feature Store 的在线特征缓存
- **Response Cache**：相同请求的结果缓存

**参考**：Redis、Memcached、SGLang 的 RadixCache

---

### 6. 监控与日志（Monitoring & Logging）

- **理解**：现代分布式系统需要大量 monitor 以及 log analysis

**AI Infra 中的应用**：
- **4 Golden Signals**：Latency、Traffic、Errors、Saturation
- **结构化日志**：`case_id`、`trace_id`、`model_version`、`latency_ms`
- **Metrics Dashboard**：错误率、延迟（p95/p99）、成本（tokens）

**参考**：Prometheus、Grafana、ELK Stack

---

### 7. 故障恢复（Failure Recovery）

- **理解**：系统中不能有 single point of failure（单点故障）
- **从 failure 的角度出发设计系统**
- **运用 Write-Ahead Log（WAL）进行故障恢复**
- **充分 replicate 你的 service**，所以任何一个机器、集群、机房的灾难都不会对你的整体服务造成不可挽回的影响

**AI Infra 中的应用**：
- **幂等机制**：`request_id UNIQUE`，避免重复处理
- **重试策略**：指数退避重试（1s, 2s, 4s, 8s）
- **故障转移**：多 provider fallback（VendorA 挂了切 VendorB）
- **数据备份**：定期 snapshot + WAL 日志

---

### 8. 高并发（High Concurrency）

- **明白**：资源共享是影响并发的主要原因之一（另一个原因是进程间通信）
- **如何 decouple 共享资源提高并发效率**

**AI Infra 中的应用**：
- **Continuous Batching**：动态批处理，提高 GPU 利用率
- **Prefill-Decode 分离**：独立扩展 Prefill 和 Decode 阶段
- **异步处理**：API → Queue → Worker，提高吞吐量

**参考**：vLLM、SGLang 的连续批处理

---

### 9. 效率评定标准（Performance Metrics）

- **明白**：基本的效率评定标准，如 **TPS（Transactions Per Second）**

**AI Infra 中的应用**：
- **QPS（Queries Per Second）**：每秒查询数
- **TTFT（Time To First Token）**：第一个 token 的延迟
- **Throughput（tokens/s）**：每秒生成的 token 数
- **p95/p99 Latency**：95%/99% 请求的延迟

---

### 10. 分布式系统基本概念（Distributed Systems Concepts）

- **CAP 定理**：Consistency、Availability、Partition tolerance
- **Consistent Hashing（一致性哈希）**：用于负载均衡、数据分片
- **Vector Clock（向量时钟）**：用于事件排序和因果关系

**AI Infra 中的应用**：
- **Consistent Hashing**：LLM 的负载均衡（多 worker 路由）
- **CAP 选择**：KYC 系统选择 AP（可用性 + 分区容错），最终一致性

**参考**：《Designing Data-Intensive Applications》第 6-8 章

---

### 11. 经典论文（Important Papers）

**建议阅读**：
- **Akamai 的 CDN**：内容分发网络的设计
- **Amazon 的 Dynamo**：分布式键值存储（CAP 定理、最终一致性）
- **Google 的 Map-Reduce**：大数据处理框架（虽然很老，但基础思想很重要）

**AI Infra 相关**：
- **PagedAttention**：vLLM 的 KV Cache 分页管理
- **RadixAttention**：SGLang 的前缀缓存技术
- **Uber Michelangelo**：ML 平台的设计思路

---

### 12. 实践项目（Hands-on Projects）

- **亲手实现一个简单的网站**，从前端到数据库都接触一些

**AI Infra 推荐项目**：
- **KYC 系统**：API + Queue + Worker + Database（端到端实战）
- **LLM Gateway**：多模型路由 + 批处理 + 缓存（基于 SGLang）
- **Feature Store**：离线特征 + 在线特征服务（简化版）

---

### 13. 大数据框架（Big Data Frameworks）

#### Hadoop
- **了解**：Hadoop 的基本功能
  - **HDFS**：分布式文件系统
  - **Map-Reduce**：分布式计算框架

**AI Infra 中的应用**：
- 训练数据的分布式存储（HDFS）
- 离线特征工程（Map-Reduce）

---

#### Apache Storm
- **了解**：Apache Storm 的基本功能（实时流处理）

**AI Infra 中的应用**：
- 实时特征计算
- 实时监控数据流

**参考**：Apache Flink、Apache Spark Streaming

---

## 🎯 设计系统时要考虑的三个维度

**结合所学的基础知识，考虑你所设计的系统的**：

1. **Availability（可用性）**：系统能够持续提供服务的能力（SLO：99.9%）
2. **Scalability（可扩展性）**：系统能够处理增长的工作负载的能力（水平扩展、垂直扩展）
3. **Performance（性能）**：系统的响应时间和吞吐量（p95 延迟、QPS）

**AI Infra 中的应用**：
- **Availability**：多 provider fallback、故障转移、冗余部署
- **Scalability**：水平扩展（多 worker）、垂直扩展（更大 GPU）、缓存层
- **Performance**：Continuous Batching、Prefix Cache、异步处理

---

## 💡 学习建议

### 对于 AI Infra + LLM 方向的 System Design

**重点学习**：
1. **API 设计规范**：Google Cloud API Design Guide、Azure API best practices
2. **LLM Serving 技术**：vLLM、SGLang 的文档和论文
3. **MLOps 平台**：Uber Michelangelo、DoorDash Feature Store 的文章
4. **SRE 实践**：Google SRE Book（SLO、Error Budget、Runbook）

**不需要过度学习**：
- 不需要通读 DDIA（抓你用得上的章节即可）
- 不需要研究所有 NoSQL 数据库（了解 Redis/DynamoDB 即可）
- 不需要深入 Hadoop/Storm（了解概念即可，除非你做大数据）

**核心目标**：
- 能把 LLM Serving 系统讲清楚（batching、caching、routing）
- 能把 ML Pipeline 讲清楚（data → feature → train → deploy → monitor）
- 能把 API 设计讲清楚（资源建模、版本、幂等、错误码）

---

## 📚 推荐学习资源

### 经典书籍
- **《Designing Data-Intensive Applications》**：存储、一致性、复制、事务、流处理
- **《Google SRE Book》**：SLO、Error Budget、Runbook、监控告警

### 技术文档
- **[Google Cloud API Design Guide](https://cloud.google.com/apis/design)**：API 设计规范
- **[Azure REST API Guidelines](https://github.com/microsoft/api-guidelines)**：API 设计最佳实践
- **[vLLM Documentation](https://docs.vllm.ai/)**：LLM Serving 技术
- **[SGLang Documentation](https://sglang.ai/)**：LLM Serving 框架

### 参考文章
- **[Uber Michelangelo Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)**：ML 平台设计
- **[DoorDash Feature Store](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/)**：特征存储设计

### 经典论文
- **DynamoDB Paper**：最终一致性、CAP 定理
- **PagedAttention Paper**：KV Cache 分页管理
- **RadixAttention**：前缀缓存技术

---

## 🎯 总结

System Design 不是"背八股"，而是**把系统讲成闭环**的能力。

**对于 AI Infra + LLM 方向**：
- **核心能力**：API 设计 + LLM Serving + ML Pipeline + 可观测性
- **核心方法**：需求澄清 + SLO + 数据流 + 风险控制
- **核心产出**：架构图 + API 设计 + 数据模型 + Runbook

**记住**：
- 不是"全部"，而是"最小闭环"
- 不是"最优解"，而是"可落地 + 有取舍"
- 不是"知道概念"，而是"能设计、能验证、能上线"

**加油！** 🎉
