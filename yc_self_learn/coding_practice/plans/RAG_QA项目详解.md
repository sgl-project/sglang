# RAG QA 系统项目详解

> **项目背景**：构建一个端到端的检索增强生成（RAG）问答系统，用于技术文档的智能问答

---

## 📋 项目概览

**职位**：Machine Learning Engineer  
**地点**：Buffalo, NY  
**时间**：May 2024 - Aug 2024  
**核心成果**：
- 将幻觉率从 30% 降至 15%
- 将 Recall@5 提升至 90%

---

## 🎯 项目目标

构建一个**检索增强生成（RAG）问答系统**，能够：
1. 从技术文档中检索相关信息
2. 基于检索到的信息生成准确的答案
3. 减少幻觉（hallucination）
4. 提高检索召回率

---

## 📊 项目步骤详解

### 第一步：数据集构建（Data Collection）

#### 1.1 核心任务
> **"Collaborated with domain experts to build a 2k+ QA dataset from technical documents"**

#### 1.2 具体工作
```python
# 数据集构建流程
技术文档
    ↓
领域专家标注
    ↓
QA 数据集（2000+ 对）
    ↓
训练/验证/测试集
```

**关键步骤**：
1. **文档收集**
   - 收集技术文档（可能是公司内部文档、产品文档等）
   - 文档格式：PDF、Markdown、HTML 等

2. **QA 对生成**
   - 与领域专家协作
   - 专家根据文档内容生成问题
   - 专家标注正确答案

3. **数据集整理**
   - 格式：`(question, answer, context)`
   - 规模：2000+ QA 对
   - 划分：训练集（80%）、验证集（10%）、测试集（10%）

#### 1.3 技术要点
- **领域专家协作**：确保问题质量和答案准确性
- **数据质量**：高质量的数据集是系统的基础
- **覆盖范围**：确保数据集覆盖不同主题和难度

---

### 第二步：模型微调（Model Fine-tuning）

#### 2.1 核心任务
> **"Fine-tuned a LLaMA-based model (LoRA)"**

#### 2.2 具体工作
```python
# 模型微调流程
LLaMA 基础模型
    ↓
LoRA 适配器（Low-Rank Adaptation）
    ↓
微调后的模型（用于生成答案）
```

**技术栈**：
- **基础模型**：LLaMA（Large Language Model）
- **微调方法**：LoRA（Low-Rank Adaptation）
- **微调目标**：让模型更好地理解技术文档问答

#### 2.3 LoRA 的数学原理

```python
# LoRA 核心思想：低秩分解
# 传统微调：更新所有参数 W (d × d 矩阵)
# W 有 d² 个参数，例如 d=4096，参数数量 = 16M

# LoRA：将权重更新 ΔW 分解为两个低秩矩阵的乘积
# ΔW = A @ B，其中 A (d × r), B (r × d)，r << d
# 参数数量 = 2 × d × r，例如 r=8，参数数量 = 2 × 4096 × 8 = 65K

# 最终权重：W_new = W + α * (A @ B)
# α 是缩放因子，通常 α = r 或 α = 2r
```

**数学公式**：
```
原始权重：W (d × d)
权重更新：ΔW = A @ B，其中 A (d × r), B (r × d)
最终权重：W_new = W + (α/r) * (A @ B)
```

**参数对比**（以 LLaMA-7B 的 Attention 层为例）：
- **全量微调**：W (4096 × 4096) = 16M 参数
- **LoRA (r=8)**：A (4096 × 8) + B (8 × 4096) = 65K 参数
- **参数减少**：99.6% 的参数不需要训练

#### 2.4 LoRA 代码实现（使用 PEFT 库）

**完整的训练代码**：

```python
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ============================================================
# 1. 加载基础模型和分词器
# ============================================================
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型（可以量化以节省内存）
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    # load_in_8bit=True,  # 8-bit 量化（可选）
)

# ============================================================
# 2. 配置 LoRA
# ============================================================
lora_config = LoraConfig(
    r=8,                      # LoRA 秩（rank），控制适配器的容量
    lora_alpha=16,            # 缩放因子（通常设为 r 的 2 倍）
    target_modules=[          # 要应用 LoRA 的模块
        "q_proj",             # Query 投影层
        "k_proj",             # Key 投影层
        "v_proj",             # Value 投影层
        "o_proj",             # Output 投影层
        "gate_proj",          # Gate 投影层（MLP）
        "up_proj",            # Up 投影层（MLP）
        "down_proj",          # Down 投影层（MLP）
    ],
    lora_dropout=0.1,         # LoRA dropout 率
    bias="none",              # 不训练 bias
    task_type="CAUSAL_LM",    # 因果语言模型任务
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# 输出示例：
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06

# ============================================================
# 3. 准备训练数据
# ============================================================
def format_qa_prompt(context, question, answer=None):
    """格式化 QA 提示"""
    prompt = f"""Context: {context}

Question: {question}

Answer:"""
    if answer:
        prompt += f" {answer}"
    return prompt

# 示例数据集
train_data = [
    {
        "context": "SGLang is a fast serving framework for LLMs...",
        "question": "What is SGLang?",
        "answer": "SGLang is a fast serving framework for large language models."
    },
    # ... 更多数据
]

# 构建训练数据集
def preprocess_function(examples):
    """预处理函数"""
    prompts = [
        format_qa_prompt(ctx, q, a)
        for ctx, q, a in zip(
            examples["context"],
            examples["question"],
            examples["answer"]
        )
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    # 设置 labels（用于计算 loss）
    labels = model_inputs["input_ids"].copy()
    # 将 prompt 部分的 label 设为 -100（不计算 loss）
    # 只对 answer 部分计算 loss
    for i, prompt in enumerate(prompts):
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels[i][:prompt_len] = [-100] * prompt_len
    
    model_inputs["labels"] = labels
    return model_inputs

# 创建 Dataset
train_dataset = Dataset.from_dict({
    "context": [item["context"] for item in train_data],
    "question": [item["question"] for item in train_data],
    "answer": [item["answer"] for item in train_data],
})

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

# ============================================================
# 4. 配置训练参数
# ============================================================
training_args = TrainingArguments(
    output_dir="./llama-lora-qa",      # 输出目录
    num_train_epochs=3,                 # 训练轮数
    per_device_train_batch_size=4,      # 批次大小
    gradient_accumulation_steps=4,      # 梯度累积
    learning_rate=2e-4,                 # 学习率
    fp16=True,                          # 混合精度训练
    logging_steps=10,                   # 日志记录频率
    save_steps=100,                     # 保存检查点频率
    save_total_limit=3,                 # 最多保存的检查点数量
    optim="adamw_torch",                # 优化器
    warmup_steps=100,                   # 预热步数
    report_to="tensorboard",            # 日志工具
)

# ============================================================
# 5. 创建 Trainer 并训练
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言模型，不是 MLM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# ============================================================
# 6. 保存 LoRA 适配器
# ============================================================
# 只保存 LoRA 权重（非常小，通常只有几十 MB）
model.save_pretrained("./llama-lora-qa-adapter")

# 保存的目录结构：
# llama-lora-qa-adapter/
#   ├── adapter_config.json    # LoRA 配置
#   ├── adapter_model.bin      # LoRA 权重（只有几 MB）
#   └── ...
```

#### 2.5 LoRA 推理代码（加载和使用）

```python
from peft import PeftModel

# ============================================================
# 1. 加载基础模型
# ============================================================
base_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# ============================================================
# 2. 加载 LoRA 适配器
# ============================================================
# 方式 1：使用 PeftModel（推荐）
model = PeftModel.from_pretrained(
    base_model,
    "./llama-lora-qa-adapter",
    torch_dtype=torch.float16,
)

# 方式 2：合并权重（可选，但会增加内存）
# model = model.merge_and_unload()

# ============================================================
# 3. 推理
# ============================================================
def generate_answer(context, question):
    """生成答案"""
    # 构建 prompt
    prompt = format_qa_prompt(context, question)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,      # 最大生成长度
            temperature=0.7,          # 温度（控制随机性）
            do_sample=True,           # 采样
            top_p=0.9,                # Top-p 采样
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案部分
    answer = generated_text.split("Answer:")[-1].strip()
    return answer

# 使用示例
context = "SGLang is a fast serving framework for LLMs..."
question = "What is SGLang?"
answer = generate_answer(context, question)
print(answer)
```

#### 2.6 LoRA 的内部实现原理（简化版）

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 线性层（简化实现）"""
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.base_layer = base_layer  # 原始线性层（冻结）
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA 适配器（可训练）
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # 低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 1. 原始输出（冻结）
        base_output = self.base_layer(x)
        
        # 2. LoRA 输出（可训练）
        # x: (batch_size, seq_len, in_features)
        # lora_A: (r, in_features)
        # lora_B: (out_features, r)
        x = self.lora_dropout(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # 3. 合并输出
        return base_output + lora_output

# 使用示例
# 原始线性层
base_linear = nn.Linear(4096, 4096)

# 替换为 LoRA 线性层
lora_linear = LoRALinear(base_linear, r=8, lora_alpha=16)

# 前向传播
x = torch.randn(1, 128, 4096)
output = lora_linear(x)  # 形状: (1, 128, 4096)
```

**LoRA 的关键点**：
1. **原始权重冻结**：`base_layer.requires_grad = False`
2. **只训练适配器**：`lora_A` 和 `lora_B` 是可训练参数
3. **低秩分解**：`ΔW = A @ B`，其中 A (r × in), B (out × r)
4. **缩放因子**：`scaling = lora_alpha / r`，通常 α = 2r

#### 2.7 LoRA 的优势总结

**参数效率对比**（LLaMA-7B 示例）：
- **全量微调**：7B 参数，需要 14GB+ GPU 内存（FP16）
- **LoRA (r=8)**：4M 参数，只需要 8GB GPU 内存（FP16）
- **参数减少**：99.94% 的参数不需要训练

**实际优势**：
- **内存效率**：只需要训练 ~0.06% 的参数
- **训练速度**：更快收敛，因为参数更少
- **可移植性**：适配器只有几 MB，可以单独保存和加载
- **多任务支持**：可以为不同任务训练不同的适配器

#### 2.8 超参数选择

```python
# LoRA 超参数建议
lora_config = LoraConfig(
    r=8,              # 秩：控制适配器容量
                     # 太小（r=1,2）：表达能力不足
                     # 太大（r=64,128）：接近全量微调，失去优势
                     # 推荐：r=8 或 r=16
    
    lora_alpha=16,    # 缩放因子：通常设为 r 的 2 倍
                     # 控制适配器的影响强度
    
    target_modules=[  # 目标模块：通常选择 Attention 和 MLP
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    
    lora_dropout=0.1, # Dropout：防止过拟合
                     # 推荐：0.05-0.2
)
```

#### 2.4 微调数据
- **输入格式**：
  ```
  Context: [检索到的文档片段]
  Question: [用户问题]
  Answer: [期望的答案]
  ```
- **训练目标**：让模型学会基于上下文生成准确答案

---

### 第三步：检索基准测试（Retrieval Baselines）

#### 3.1 核心任务
> **"Benchmarked retrieval baselines (BM25, MiniLM)"**

#### 3.2 具体工作
```python
# 检索方法对比
用户问题
    ↓
检索方法 1: BM25（关键词匹配）
检索方法 2: MiniLM（语义搜索）
    ↓
Top-K 文档片段
    ↓
性能对比（Recall@K, Precision@K）
```

#### 3.3 BM25（关键词检索）
```python
# BM25 原理（简化版）
# 基于 TF-IDF 的改进
# 优点：速度快、不需要训练
# 缺点：无法理解语义

def bm25_retrieval(query, documents):
    """BM25 检索"""
    # 1. 计算查询词在文档中的 TF-IDF 分数
    scores = []
    for doc in documents:
        score = calculate_bm25_score(query, doc)
        scores.append((doc, score))
    
    # 2. 按分数排序，返回 Top-K
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
```

**特点**：
- **基于关键词**：传统信息检索方法
- **速度快**：不需要深度学习模型
- **可解释性强**：结果容易理解

#### 3.4 MiniLM（语义检索）
```python
# MiniLM 检索原理（简化版）
# 基于 Transformer 的语义编码器
# 优点：理解语义、效果好
# 缺点：需要模型推理

def minilm_retrieval(query, documents):
    """MiniLM 语义检索"""
    # 1. 编码查询和文档
    query_embedding = minilm_model.encode(query)
    doc_embeddings = minilm_model.encode(documents)
    
    # 2. 计算余弦相似度
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    
    # 3. 返回 Top-K
    return get_top_k(similarities, k)
```

**特点**：
- **语义理解**：基于 Transformer 编码器
- **效果好**：能够理解同义词和上下文
- **需要模型**：需要加载预训练模型

#### 3.5 基准测试结果
- **评估指标**：Recall@K、Precision@K、MRR
- **结论**：MiniLM 在语义理解上更优，BM25 在速度上更优
- **最终选择**：可能采用混合检索（BM25 + MiniLM）

---

### 第四步：评估系统设计（Evaluation Harness）

#### 4.1 核心任务
> **"Designed an end-to-end evaluation harness with a label schema (factuality, coverage, style), automatic metrics, and Python error-analysis scripts"**

#### 4.2 核心成果
- **幻觉率**：从 30% 降至 15%
- **Recall@5**：提升至 90%

#### 4.3 评估框架设计

```python
# 评估框架结构
生成答案
    ↓
评估维度 1: Factuality（事实性）
评估维度 2: Coverage（覆盖率）
评估维度 3: Style（风格）
    ↓
自动指标计算
    ↓
错误分析报告
```

#### 4.4 标签模式（Label Schema）

```python
# 评估标签定义
evaluation_schema = {
    "factuality": {
        "accurate": 1,      # 答案事实正确
        "hallucination": 0, # 包含幻觉
        "partial": 0.5      # 部分正确
    },
    "coverage": {
        "complete": 1,      # 完整覆盖问题
        "partial": 0.5,     # 部分覆盖
        "incomplete": 0     # 未覆盖
    },
    "style": {
        "professional": 1,  # 专业风格
        "casual": 0.5,      # 随意风格
        "inappropriate": 0  # 不合适风格
    }
}
```

**三个维度**：
1. **Factuality（事实性）**：答案是否准确，是否有幻觉
2. **Coverage（覆盖率）**：答案是否完整回答了问题
3. **Style（风格）**：答案的风格是否合适

#### 4.5 自动指标（Automatic Metrics）

```python
# 自动指标计算
def calculate_metrics(predictions, ground_truth, schema):
    """计算评估指标"""
    metrics = {
        "factuality_score": calculate_factuality(predictions, ground_truth),
        "coverage_score": calculate_coverage(predictions, ground_truth),
        "style_score": calculate_style(predictions, ground_truth),
        "hallucination_rate": calculate_hallucination_rate(predictions),
        "recall_at_k": calculate_recall_at_k(predictions, ground_truth, k=5)
    }
    return metrics
```

**关键指标**：
- **Hallucination Rate（幻觉率）**：答案中错误信息的比例
- **Recall@K**：检索到的相关文档占所有相关文档的比例
- **Factuality Score**：答案的事实准确性
- **Coverage Score**：答案的完整度

#### 4.6 错误分析脚本（Error Analysis Scripts）

```python
# 错误分析脚本示例
def error_analysis(predictions, ground_truth, schema):
    """错误分析"""
    errors = {
        "hallucination_cases": [],  # 幻觉案例
        "coverage_gaps": [],        # 覆盖率不足案例
        "style_issues": []          # 风格问题案例
    }
    
    for pred, gt in zip(predictions, ground_truth):
        # 1. 检测幻觉
        if detect_hallucination(pred, gt):
            errors["hallucination_cases"].append({
                "question": gt["question"],
                "predicted": pred["answer"],
                "ground_truth": gt["answer"]
            })
        
        # 2. 检测覆盖率不足
        if calculate_coverage(pred, gt) < 0.7:
            errors["coverage_gaps"].append(...)
        
        # 3. 检测风格问题
        if calculate_style(pred, gt) < 0.8:
            errors["style_issues"].append(...)
    
    # 4. 生成错误分析报告
    generate_error_report(errors)
    return errors
```

**错误分析的作用**：
- **识别问题模式**：找出常见错误类型
- **改进方向**：指导模型和系统改进
- **数据洞察**：发现数据集质量问题

---

### 第五步：RAG 后端实现（RAG Backend）

#### 5.1 核心任务
> **"Implemented a modular RAG backend: chunking and ingestion jobs, FAISS-based vector store, retriever / re-ranker layer, and LLM orchestration"**

#### 5.2 RAG 系统架构

```python
# RAG 系统架构
用户问题
    ↓
检索层（Retrieval Layer）
    ├─ Chunking（文档分块）
    ├─ Vector Store（FAISS）
    ├─ Retriever（检索器）
    └─ Re-ranker（重排序）
    ↓
Top-K 文档片段
    ↓
生成层（Generation Layer）
    ├─ LLM Orchestration（LLM 编排）
    └─ Answer Generation（答案生成）
    ↓
最终答案
```

#### 5.3 模块 1：Chunking and Ingestion（文档分块和导入）

```python
# 文档分块和导入流程
原始文档
    ↓
文档解析（PDF/Markdown/HTML）
    ↓
文本分块（Chunking）
    ↓
向量化（Embedding）
    ↓
存储到向量数据库（FAISS）
```

**分块策略**：
```python
def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """文档分块"""
    chunks = []
    for doc in documents:
        # 1. 按字符/句子/段落分割
        text = doc.text
        
        # 2. 滑动窗口分块（保留重叠）
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc.source,
                    "chunk_id": len(chunks),
                    "start_pos": i,
                    "end_pos": i + len(chunk)
                }
            })
    
    return chunks
```

**关键点**：
- **分块大小**：通常 256-512 tokens
- **重叠策略**：保留上下文连续性
- **元数据**：记录来源、位置等信息

#### 5.4 模块 2：FAISS-based Vector Store（向量存储）

```python
# FAISS 向量存储
import faiss
import numpy as np

class FAISSVectorStore:
    """基于 FAISS 的向量存储"""
    def __init__(self, dimension=384):  # MiniLM 维度
        # 1. 创建 FAISS 索引
        self.index = faiss.IndexFlatL2(dimension)  # L2 距离
        self.embeddings = []
        self.metadata = []
    
    def add(self, embeddings, metadata):
        """添加向量"""
        # 1. 存储向量
        self.index.add(embeddings.astype('float32'))
        
        # 2. 存储元数据
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, k=5):
        """检索 Top-K"""
        # 1. FAISS 检索
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # 2. 返回结果和元数据
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "chunk": self.metadata[idx],
                "score": float(dist),
                "index": int(idx)
            })
        
        return results
```

**FAISS 的优势**：
- **快速检索**：GPU 加速，毫秒级响应
- **可扩展性**：支持大规模向量库
- **多种索引**：支持 L2、内积、余弦相似度等

#### 5.5 模块 3：Retriever / Re-ranker Layer（检索/重排序层）

```python
# 检索和重排序流程
用户问题
    ↓
Retriever（粗检索）
    ├─ BM25 检索（Top-20）
    └─ MiniLM 检索（Top-20）
    ↓
合并候选（Top-40）
    ↓
Re-ranker（精排序）
    └─ Cross-Encoder（Top-5）
    ↓
最终 Top-5 文档片段
```

**检索器（Retriever）**：
```python
class HybridRetriever:
    """混合检索器"""
    def __init__(self, bm25_index, vector_store, minilm_model):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.minilm_model = minilm_model
    
    def retrieve(self, query, top_k=20):
        """检索 Top-K 候选"""
        # 1. BM25 检索
        bm25_results = self.bm25_index.search(query, top_k)
        
        # 2. MiniLM 检索
        query_embedding = self.minilm_model.encode(query)
        vector_results = self.vector_store.search(query_embedding, top_k)
        
        # 3. 合并和去重
        candidates = merge_and_deduplicate(bm25_results, vector_results)
        
        return candidates
```

**重排序器（Re-ranker）**：
```python
class ReRanker:
    """重排序器（Cross-Encoder）"""
    def __init__(self, cross_encoder_model):
        self.model = cross_encoder_model
    
    def rerank(self, query, candidates, top_k=5):
        """重排序"""
        # 1. 对每个候选计算相关性分数
        scores = []
        for candidate in candidates:
            # Cross-Encoder 同时编码 query 和 candidate
            score = self.model.predict([query, candidate["text"]])
            scores.append(score)
        
        # 2. 按分数排序
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 3. 返回 Top-K
        return [item[0] for item in ranked[:top_k]]
```

**两阶段检索的优势**：
- **速度**：Retriever 快速过滤
- **精度**：Re-ranker 精确排序
- **平衡**：速度和精度的平衡

#### 5.6 模块 4：LLM Orchestration（LLM 编排）

```python
# LLM 编排流程
Top-5 文档片段
    ↓
Prompt 构建
    ↓
LLM 生成（LoRA 微调模型）
    ↓
答案后处理
    ↓
最终答案
```

**LLM 编排代码**：
```python
class LLMOrchestrator:
    """LLM 编排器"""
    def __init__(self, llm_model, tokenizer):
        self.model = llm_model
        self.tokenizer = tokenizer
    
    def generate_answer(self, query, retrieved_contexts):
        """生成答案"""
        # 1. 构建 Prompt
        prompt = self._build_prompt(query, retrieved_contexts)
        
        # 2. 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 3. 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        # 4. 解码输出
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. 后处理
        answer = self._post_process(answer)
        
        return answer
    
    def _build_prompt(self, query, contexts):
        """构建 Prompt"""
        context_text = "\n\n".join([
            f"Document {i+1}: {ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Based on the following documents, answer the question.

Documents:
{context_text}

Question: {query}

Answer:"""
        return prompt
```

**关键点**：
- **Prompt 设计**：清晰的指令和上下文格式
- **上下文注入**：将检索到的文档注入 Prompt
- **生成控制**：temperature、max_length 等参数

---

### 第六步：服务部署（FastAPI + Docker）

#### 6.1 核心任务
> **"Served via FastAPI and containerized with Docker for on-prem deployment"**

#### 6.2 FastAPI 服务

```python
# FastAPI 服务代码
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG QA System")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """问答接口"""
    try:
        # 1. 检索
        contexts = retriever.retrieve(request.question, top_k=20)
        reranked = reranker.rerank(request.question, contexts, top_k=5)
        
        # 2. 生成答案
        answer = llm_orchestrator.generate_answer(
            request.question,
            reranked
        )
        
        # 3. 返回结果
        return QueryResponse(
            answer=answer,
            sources=[ctx["metadata"] for ctx in reranked],
            confidence=calculate_confidence(answer, reranked)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**FastAPI 的优势**：
- **自动文档**：Swagger UI
- **类型检查**：Pydantic 模型
- **异步支持**：高并发性能
- **易于部署**：标准 WSGI/ASGI

#### 6.3 Docker 容器化

```dockerfile
# Dockerfile 示例
FROM python:3.9-slim

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制应用代码
COPY . .

# 5. 下载模型（或从外部挂载）
# RUN python download_models.py

# 6. 暴露端口
EXPOSE 8000

# 7. 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose 配置**：
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models
      - FAISS_INDEX_PATH=/app/data/faiss_index
    deploy:
      resources:
        limits:
          gpus: 1
```

**容器化的优势**：
- **环境一致性**：开发、测试、生产环境一致
- **易于部署**：一键部署
- **资源隔离**：独立的运行环境
- **可扩展性**：支持 Kubernetes 等编排工具

---

### 第七步：工程化（Git + Observability + Deployment）

#### 7.1 核心任务
> **"Setup Git-based workflows and basic observability (structured logs, latency / error-rate dashboards) and shipped reproducible Docker images plus deployment runbooks for downstream teams"**

#### 7.2 Git 工作流

```bash
# Git 工作流
# 1. 特性分支开发
git checkout -b feature/rag-retrieval
git commit -m "Add retrieval layer"
git push origin feature/rag-retrieval

# 2. Pull Request 审查
# - Code Review
# - CI/CD 测试
# - 合并到 main

# 3. 版本标签
git tag v1.0.0
git push origin v1.0.0
```

**Git 工作流的优势**：
- **代码审查**：保证代码质量
- **版本控制**：可追溯的变更历史
- **协作**：团队协作的基础

#### 7.3 可观测性（Observability）

```python
# 结构化日志
import logging
import structlog

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 日志记录
def query_handler(question):
    logger.info(
        "query_received",
        question=question,
        timestamp=datetime.now().isoformat()
    )
    
    start_time = time.time()
    result = process_query(question)
    latency = time.time() - start_time
    
    logger.info(
        "query_completed",
        question=question,
        latency_ms=latency * 1000,
        answer_length=len(result["answer"])
    )
```

**监控指标**：
```python
# 监控指标（Prometheus 格式）
metrics = {
    "query_latency_seconds": Histogram("query_latency_seconds"),
    "query_error_rate": Counter("query_error_total"),
    "retrieval_recall_at_5": Gauge("retrieval_recall_at_5"),
    "hallucination_rate": Gauge("hallucination_rate")
}
```

**Dashboard（Grafana）**：
- **延迟监控**：P50、P95、P99 延迟
- **错误率**：错误请求占比
- **吞吐量**：QPS（Queries Per Second）
- **业务指标**：Recall@5、Hallucination Rate

#### 7.4 部署文档（Deployment Runbooks）

```markdown
# 部署文档示例
## RAG QA System 部署指南

### 前置要求
- Docker 20.10+
- NVIDIA Docker (GPU 支持)
- 16GB+ RAM
- 50GB+ 磁盘空间

### 部署步骤

1. **拉取代码**
   ```bash
   git clone https://github.com/company/rag-qa-system.git
   cd rag-qa-system
   ```

2. **下载模型**
   ```bash
   python scripts/download_models.py
   ```

3. **构建 Docker 镜像**
   ```bash
   docker build -t rag-qa-system:latest .
   ```

4. **启动服务**
   ```bash
   docker-compose up -d
   ```

5. **验证部署**
   ```bash
   curl http://localhost:8000/health
   ```

### 故障排查
- 查看日志：`docker-compose logs -f`
- 检查 GPU：`nvidia-smi`
- 验证模型：`python scripts/verify_models.py`
```

**部署文档的重要性**：
- **可重现性**：确保团队能够一致部署
- **故障排查**：快速定位问题
- **知识传递**：新成员能够快速上手

---

## 🎯 项目成果总结

### 性能提升
- **幻觉率**：30% → 15%（降低 50%）
- **Recall@5**：提升至 90%
- **系统可用性**：生产环境稳定运行

### 技术栈总结
- **模型**：LLaMA + LoRA
- **检索**：BM25 + MiniLM + FAISS
- **服务**：FastAPI
- **部署**：Docker
- **监控**：结构化日志 + Grafana

### 项目亮点
1. **端到端系统**：从数据到部署的完整流程
2. **模块化设计**：易于维护和扩展
3. **工程化**：Git、Docker、监控等最佳实践
4. **可衡量**：完善的评估系统和监控

---

## 📚 技术要点总结

### 1. RAG 系统核心组件
- **文档处理**：Chunking、Embedding
- **检索**：BM25、向量检索、重排序
- **生成**：LLM 编排、Prompt 工程
- **评估**：多维度评估框架

### 2. 工程化最佳实践
- **代码管理**：Git 工作流
- **容器化**：Docker 部署
- **可观测性**：日志、监控、Dashboard
- **文档**：部署指南、API 文档

### 3. 性能优化
- **检索优化**：两阶段检索（Retriever + Re-ranker）
- **模型优化**：LoRA 微调（内存和速度优化）
- **系统优化**：异步处理、GPU 加速

---

**这个项目展示了从研究到生产的完整 ML 工程能力！** 🚀
