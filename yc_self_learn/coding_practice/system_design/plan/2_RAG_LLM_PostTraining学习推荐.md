# RAG / LLM / Post-Training 学习推荐（AI Infra）

**目的**：学习 RAG（Retrieval-Augmented Generation）、LLM 推理和 Post-Training 相关的系统设计

**要求**：
- 工业界标准、文档完善
- RAG pipeline 设计、LLM 推理基础设施
- Post-training 和评估框架
- 适合 AI Infra 学习

---

## 🎯 精选 3 个最佳项目（AI Infra 标准）

### 1. **Haystack (Deepset)** ⭐⭐⭐⭐⭐（RAG Pipeline 标准）

**为什么选它**：
- ✅ **RAG 领域标准**：最成熟的端到端 RAG 框架（Deepset 开源，工业界广泛使用）
- ✅ **Pipeline 设计经典**：Pipeline Graph 模式，清晰的任务编排
- ✅ **文档完善**：官方文档详细，示例丰富，社区活跃
- ✅ **生产就绪**：支持多种文档存储、检索策略、LLM 集成

**GitHub**: https://github.com/deepset-ai/haystack  
**文档**: https://docs.haystack.deepset.ai/

**核心设计点（值得学习）**：

1. **RAG Pipeline 架构**
   ```python
   # 示例：典型的 RAG pipeline
   from haystack import Pipeline
   from haystack.components.builders import PromptBuilder
   from haystack.components.generators import OpenAIGenerator
   from haystack.components.retrievers import InMemoryEmbeddingRetriever
   
   # 构建 RAG pipeline
   rag_pipeline = Pipeline()
   rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store))
   rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
   rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-4"))
   
   rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
   rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
   
   # 执行 pipeline
   result = rag_pipeline.run({"retriever": {"query": "What is radiology report quality?"}})
   ```
   - **学习点**：Pipeline Graph 编排、组件可插拔、数据流管理

2. **检索系统设计**
   - **多种检索器**：BM25、Dense Embedding、Hybrid Search
   - **文档存储**：Elasticsearch、OpenSearch、Milvus、Qdrant、Pinecone、Weaviate
   - **Reranking**：支持多级检索和重排序
   - **学习点**：检索架构设计、混合检索策略、性能优化

3. **LLM 集成和生成**
   - **多种 LLM**：OpenAI、Anthropic、本地模型（Ollama、vLLM、SGLang）
   - **Prompt 管理**：PromptBuilder、模板管理、变量替换
   - **Citation 支持**：自动引用检索到的文档
   - **学习点**：LLM 抽象层、Prompt 工程、Citation 机制

4. **评估和监控**
   - **评估指标**：Retrieval 准确率、生成质量、Citation 准确性、Hallucination 检测
   - **评估 Pipeline**：自动评估流程、A/B 测试
   - **监控**：Pipeline 执行监控、性能指标、成本追踪
   - **学习点**：RAG 评估框架、监控系统设计、成本优化

**与 rad-linter 的对应关系**：
- RAG 文档处理 → rad-linter 的 Step 0-1（数据子集、格式对齐）
- 检索系统 → rad-linter 的 Step 2（视觉特征提取，类似检索）
- LLM 生成 → rad-linter 的 Step 3.5（Judge 标签生成）
- 评估系统 → rad-linter 的 Step 5（三面板评估）

**学习路径**：
1. 阅读快速开始：https://docs.haystack.deepset.ai/docs/quick-start
2. 运行示例：https://github.com/deepset-ai/haystack/tree/main/examples
3. 理解架构：https://docs.haystack.deepset.ai/docs/pipelines
4. 应用到 rad-linter：设计 rad-linter 的 RAG 评估 pipeline

---

### 2. **vLLM / SGLang** ⭐⭐⭐⭐⭐（LLM 推理基础设施）

**为什么选它**：
- ✅ **LLM 推理标准**：高性能 LLM 推理服务框架（vLLM 是 UC Berkeley 开源，SGLang 是 Stanford 开源）
- ✅ **性能优化**：PagedAttention（vLLM）、RadixAttention（SGLang）、高吞吐量、低延迟
- ✅ **生产就绪**：支持 OpenAI 兼容 API、多 GPU 分布式、自动批处理
- ✅ **开源活跃**：社区活跃，被众多公司采用

**vLLM GitHub**: https://github.com/vllm-project/vllm  
**vLLM 文档**: https://docs.vllm.ai/

**SGLang GitHub**: https://github.com/sgl-project/sglang  
**SGLang 文档**: https://sglang.readthedocs.io/

**核心设计点（值得学习）**：

1. **推理服务架构**（以 SGLang 为例，你的项目在用）
   ```python
   # 示例：SGLang 推理服务
   from sglang import function, system, user, assistant, gen, select, image
   import sglang as sgl
   
   @function
   def radiology_qa(s, question, image_data):
       s += system("You are a radiology expert.")
       s += user("Question:", question)
       s += user("Image:", image(image_data))
       s += assistant("Answer:", gen("answer", max_tokens=200))
       return s["answer"]
   
   # 启动服务
   runtime = sgl.Runtime(model_path="Qwen/Qwen2-VL-7B-Instruct")
   runtime.start_server(port=30000)
   
   # 调用服务
   result = radiology_qa.run(question="What is wrong?", image_data="path/to/image.jpg")
   ```
   - **学习点**：推理服务架构、批量处理、性能优化、API 设计

2. **性能优化技术**
   - **Attention 优化**：PagedAttention（vLLM）、RadixAttention（SGLang）
   - **KV Cache 管理**：高效的内存管理、连续批处理
   - **并行推理**：Tensor Parallelism、Pipeline Parallelism
   - **学习点**：高性能系统设计、内存优化、并行计算

3. **可扩展性设计**
   - **多 GPU 支持**：分布式推理、负载均衡
   - **自动批处理**：动态批处理、连续批处理
   - **模型部署**：支持多种模型格式（HuggingFace、Llama.cpp）
   - **学习点**：分布式系统设计、资源管理、负载均衡

4. **监控和可观测性**
   - **性能指标**：吞吐量（tokens/s）、延迟（p50/p95/p99）、GPU 利用率
   - **日志和追踪**：请求日志、错误追踪
   - **健康检查**：服务健康检查、自动恢复
   - **学习点**：性能监控、可观测性设计、故障处理

**与 rad-linter 的对应关系**：
- Judge Server → SGLang/vLLM 推理服务
- LoRA 推理 → SGLang 的 LoRA 支持
- 批量评估 → vLLM 的批量推理
- 性能优化 → Attention 优化、KV Cache 管理

**学习路径**：
1. 阅读 SGLang 文档：https://sglang.readthedocs.io/en/latest/
2. 阅读 vLLM 文档：https://docs.vllm.ai/
3. 运行示例：启动推理服务、测试性能
4. 应用到 rad-linter：优化 Judge Server 性能

---

### 3. **RAG Foundry (IntelLabs)** ⭐⭐⭐⭐⭐（端到端 RAG + Post-Training）

**为什么选它**：
- ✅ **端到端 RAG 框架**：涵盖数据创建、训练、推理、评估全流程
- ✅ **Post-Training 支持**：支持 fine-tuning、评估、迭代
- ✅ **研究导向**：Intel Labs 开源，适合研究型项目
- ✅ **模块化设计**：组件可替换，便于实验

**论文**: [RAG Foundry: A Foundational Model to Do Anything with RAG](https://arxiv.org/abs/2408.02545)  
**GitHub**: 搜索 "RAG Foundry IntelLabs" 或访问 Intel Labs 相关页面

**核心设计点（值得学习）**：

1. **端到端 RAG Pipeline**
   ```
   数据采集 → 预处理 → 索引构建 → 检索 → 生成 → 评估 → 迭代
   ```
   - **学习点**：完整的 RAG 生命周期管理、Post-training 流程

2. **Post-Training 和评估**
   - **训练数据构造**：QA 对生成、检索增强训练
   - **模型 Fine-tuning**：基于检索结果 fine-tune LLM（RAFT 方法）
   - **评估体系**：检索质量、生成质量、整体 RAG 性能
   - **学习点**：Post-training 流程、评估框架设计、迭代改进

3. **RAFT 方法（Retrieval-Augmented Fine-Tuning）**
   - **干扰文档处理**：训练模型识别有用文档、忽略干扰文档
   - **直接引用训练**：训练模型直接引用正确序列
   - **Chain-of-Thought**：结合推理过程输出
   - **学习点**：Domain-specific fine-tuning、检索增强训练

4. **实验管理**
   - **配置管理**：检索策略、重排序方式、LLM 模型
   - **Ablation Study**：快速对比不同配置组合
   - **结果追踪**：实验记录、性能对比
   - **学习点**：实验管理、配置化设计、科学实验方法

**与 rad-linter 的对应关系**：
- 数据采集/预处理 → rad-linter 的 Step 0-1
- 索引/检索 → rad-linter 的 Step 2（视觉特征提取）
- 生成/推理 → rad-linter 的 Step 3.5（Judge）+ Step 4（LoRA 训练）
- 评估 → rad-linter 的 Step 5（三面板评估）
- Post-training → rad-linter 的 Step 4（LoRA 微调）+ 迭代改进

**学习路径**：
1. 阅读论文：https://arxiv.org/abs/2408.02545
2. 查找 GitHub 代码（如果公开）
3. 理解端到端流程：数据 → 训练 → 推理 → 评估
4. 应用到 rad-linter：设计 rad-linter 的 post-training 流程

---

## 💡 核心设计模式总结

从这 3 个项目中，你学到的最重要的设计模式：

### 1. **RAG Pipeline 模式**（从 Haystack 学）
- Pipeline Graph 编排
- 检索系统设计（BM25、Dense、Hybrid）
- LLM 集成和生成（多种模型支持）
- Citation 机制和评估框架

### 2. **LLM 推理基础设施模式**（从 vLLM/SGLang 学）
- 高性能推理服务架构
- Attention 优化（PagedAttention、RadixAttention）
- KV Cache 管理和内存优化
- 批量处理和并行推理

### 3. **Post-Training 流程模式**（从 RAG Foundry 学）
- 端到端 RAG 流程（数据 → 训练 → 推理 → 评估）
- Post-training 和 Fine-tuning 流程
- 评估框架设计（检索质量 + 生成质量）
- 实验管理和配置化设计

---

## 🔗 快速链接

### 代码仓库
- **Haystack**: https://github.com/deepset-ai/haystack
- **vLLM**: https://github.com/vllm-project/vllm
- **SGLang**: https://github.com/sgl-project/sglang

### 文档和教程
- **Haystack Docs**: https://docs.haystack.deepset.ai/
- **vLLM Docs**: https://docs.vllm.ai/
- **SGLang Docs**: https://sglang.readthedocs.io/

### 论文
- **RAG Foundry**: https://arxiv.org/abs/2408.02545

---

**最后更新**：2025-01-19
