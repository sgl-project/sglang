# Day 3_A1_B6：测试用例去重和相似度分析详解

**优先级**：🟡 P1 - 重要但不紧急  
**目的**：提高 Golden Set 质量，避免重复用例，优化测试资源

---

## 🎯 为什么需要去重和相似度分析？

### 问题场景

**场景 1：重复用例浪费资源**
- Golden Set 中有 10 个几乎相同的测试用例（如：都是"标准身份证"）
- 运行 10 次相同的测试，浪费时间和成本（每次调用 LLM API 都要花钱）
- 应该只保留 1 个代表性用例

**场景 2：相似用例覆盖重复**
- Golden Set 中有 5 个"模糊身份证"的用例
- 它们测试的是同一个问题（图片模糊）
- 应该合并或只保留最有代表性的（如：保留最模糊的那个）

**场景 3：Golden Set 质量下降**
- 随着时间推移，Golden Set 会不断增长（从生产错误中提取新用例）
- 如果不去重，会有很多重复和相似用例
- Golden Set 质量下降，测试效率降低

### Senior 级别要求

- ✅ **如何避免重复用例？** → 自动去重机制
- ✅ **如何识别相似用例？** → 多维度相似度计算
- ✅ **如何合并相似用例？** → 合并策略（保留优先级高的）

---

## 📊 KYC 项目测试用例结构

### 标准用例格式

```python
test_case = {
    "case_id": "normal_001",
    "file_path": "test_data/normal/id_card_standard.jpg",
    "expected_fields": {
        "name": "张三",
        "id_number": "110101199001011234",
        "date_of_birth": "1990-01-01"
    },
    "category": "normal",  # normal/edge/anomaly/longtail
    "description": "标准身份证，清晰、标准格式",
    "priority": 1,  # 0=Critical, 1=High, 2=Medium, 3=Low
    "tags": ["id_card", "standard", "clear"]
}
```

### 用例关键字段

| 字段 | 用途 | 去重相关 |
|------|------|---------|
| `case_id` | 唯一标识 | ✅ 用于识别重复 |
| `file_path` | 文件路径 | ✅ **核心去重字段**（文件 Hash） |
| `expected_fields` | 预期输出 | ✅ 用于输出相似度计算 |
| `category` | 分类 | ✅ 用于场景相似度 |
| `description` | 描述 | ✅ 用于文本相似度 |
| `priority` | 优先级 | ✅ 用于合并策略 |
| `tags` | 标签 | ✅ 用于标签相似度 |

---

## 🔍 相似度计算方法

### 1. 文件 Hash 相似度（完全重复检测）

**适用场景**：**识别完全相同的文件**（最快速、最准确）

**方法**：
- **MD5/SHA256 Hash**：计算文件的 Hash 值
- **相同 Hash = 完全重复**

**KYC 项目实现**：

```python
import hashlib
from pathlib import Path

def calculate_file_hash(file_path: str) -> str:
    """计算文件 Hash（用于识别完全重复）"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

# 示例
case1 = {"file_path": "test_data/normal/id_card_001.jpg"}
case2 = {"file_path": "test_data/normal/id_card_001.jpg"}  # 相同文件

hash1 = calculate_file_hash(case1["file_path"])
hash2 = calculate_file_hash(case2["file_path"])

if hash1 == hash2:
    print("完全重复！")  # 应该删除其中一个
```

**优点**：
- ✅ 快速、准确
- ✅ 100% 识别完全重复

**缺点**：
- ⚠️ 只能识别完全相同的文件
- ⚠️ 无法识别相似但不同的文件

---

### 2. 文本相似度（描述和标签）

**适用场景**：**识别描述相似的用例**（如："模糊身份证" vs "低质量身份证"）

**方法**：
- **Jaccard Similarity**：计算标签集合的交集/并集
- **TF-IDF + Cosine Similarity**：向量化描述文本后计算余弦相似度

**KYC 项目实现**：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_text_similarity(case1: dict, case2: dict) -> float:
    """计算两个测试用例的文本相似度"""
    # 提取文本特征（描述 + 标签）
    text1 = f"{case1.get('description', '')} {' '.join(case1.get('tags', []))}"
    text2 = f"{case2.get('description', '')} {' '.join(case2.get('tags', []))}"
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    
    # 计算余弦相似度
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity

# 示例
case1 = {
    "description": "模糊身份证，测试 OCR 能力",
    "tags": ["id_card", "blur", "low_quality"]
}

case2 = {
    "description": "低质量身份证，图片模糊",
    "tags": ["id_card", "blur", "low_quality"]
}

similarity = calculate_text_similarity(case1, case2)
# 结果：0.85（高度相似）
```

**标签相似度（Jaccard）**：

```python
def calculate_tag_similarity(case1: dict, case2: dict) -> float:
    """计算标签相似度（Jaccard）"""
    tags1 = set(case1.get('tags', []))
    tags2 = set(case2.get('tags', []))
    
    intersection = tags1 & tags2
    union = tags1 | tags2
    
    return len(intersection) / len(union) if union else 0.0

# 示例
case1 = {"tags": ["id_card", "blur", "low_quality"]}
case2 = {"tags": ["id_card", "blur", "low_quality", "edge_case"]}

similarity = calculate_tag_similarity(case1, case2)
# 结果：0.75（3个共同标签 / 4个总标签）
```

---

### 3. 语义相似度（理解语义含义）

**适用场景**：**识别语义相似但用词不同的用例**（如："模糊身份证" vs "低质量身份证"）

**方法**：
- **Embedding 模型**：使用 Sentence-BERT、OpenAI Embeddings 等
- **向量相似度**：计算 Embedding 向量的余弦相似度

**KYC 项目实现**：

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityCalculator:
    def __init__(self):
        # 使用轻量级模型（适合 KYC 项目）
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_semantic_similarity(self, case1: dict, case2: dict) -> float:
        """计算两个测试用例的语义相似度"""
        # 提取文本（描述 + 分类）
        text1 = f"{case1.get('category', '')} {case1.get('description', '')}"
        text2 = f"{case2.get('category', '')} {case2.get('description', '')}"
        
        # 生成 Embedding
        embeddings = self.model.encode([text1, text2])
        
        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity

# 示例
calculator = SemanticSimilarityCalculator()

case1 = {
    "category": "edge",
    "description": "模糊身份证，测试 OCR 能力"
}

case2 = {
    "category": "edge",
    "description": "低质量身份证，图片模糊"
}

similarity = calculator.calculate_semantic_similarity(case1, case2)
# 结果：0.78（语义相似，虽然用词不同）
```

---

### 4. 输出相似度（预期字段对比）

**适用场景**：**识别预期输出相似的用例**（如：都期望提取相同的字段）

**方法**：
- **字段级对比**：对比每个字段的值
- **结构相似度**：对比输出结构的相似度
- **集合相似度**：对比字段集合的交集/并集

**KYC 项目实现**：

```python
def calculate_output_similarity(case1: dict, case2: dict) -> float:
    """计算两个测试用例预期输出的相似度"""
    output1 = case1.get("expected_fields", {})
    output2 = case2.get("expected_fields", {})
    
    # 获取字段集合
    fields1 = set(output1.keys())
    fields2 = set(output2.keys())
    
    # 计算字段交集和并集（Jaccard）
    intersection = fields1 & fields2
    union = fields1 | fields2
    
    field_similarity = len(intersection) / len(union) if union else 0.0
    
    # 计算值相似度（对于相同字段）
    value_similarities = []
    for field in intersection:
        val1 = output1[field]
        val2 = output2[field]
        if val1 == val2:
            value_similarities.append(1.0)
        else:
            # 可以使用字符串相似度（如 Levenshtein）
            value_similarities.append(0.0)  # 简化示例
    
    value_similarity = np.mean(value_similarities) if value_similarities else 0.0
    
    # 综合相似度（字段相似度 + 值相似度）
    overall_similarity = (field_similarity + value_similarity) / 2
    return overall_similarity

# 示例
case1 = {
    "expected_fields": {
        "name": "张三",
        "id_number": "110101199001011234",
        "date_of_birth": "1990-01-01"
    }
}

case2 = {
    "expected_fields": {
        "name": "张三",  # 相同值
        "id_number": "110101199001011234",  # 相同值
        "date_of_birth": "1990-01-01",  # 相同值
        "address": "北京市"  # 额外字段
    }
}

similarity = calculate_output_similarity(case1, case2)
# 结果：0.75（3个共同字段，值都相同，但 case2 有额外字段）
```

---

### 5. 图像相似度（图片文件对比）

**适用场景**：**识别图片相似的用例**（如：同一文档的不同扫描版本）

**方法**：
- **感知哈希（pHash）**：计算图片的感知哈希值
- **特征提取**：使用 CNN 提取图片特征，计算特征相似度

**KYC 项目实现**：

```python
from PIL import Image
import imagehash

def calculate_image_similarity(file1: str, file2: str) -> float:
    """计算两个图片文件的相似度"""
    try:
        # 加载图片
        img1 = Image.open(file1)
        img2 = Image.open(file2)
        
        # 计算感知哈希
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        
        # 计算汉明距离（越小越相似）
        hamming_distance = hash1 - hash2
        
        # 转换为相似度（0-1，1 表示完全相同）
        # pHash 是 64 位，最大距离是 64
        similarity = 1 - (hamming_distance / 64.0)
        return similarity
    except Exception as e:
        print(f"Error calculating image similarity: {e}")
        return 0.0

# 示例
similarity = calculate_image_similarity(
    "test_data/normal/id_card_001.jpg",
    "test_data/normal/id_card_001_copy.jpg"  # 可能是同一文档的不同扫描版本
)
# 结果：0.95（高度相似，可能是同一文档）
```

---

### 6. 综合相似度（多维度加权）

**KYC 项目推荐权重**：

```python
def calculate_overall_similarity(case1: dict, case2: dict) -> dict:
    """计算综合相似度（多维度加权）"""
    
    # 1. 文件 Hash（完全重复检测）
    hash1 = calculate_file_hash(case1["file_path"])
    hash2 = calculate_file_hash(case2["file_path"])
    if hash1 == hash2:
        return {
            "overall_similarity": 1.0,
            "is_exact_duplicate": True,
            "breakdown": {
                "file_hash": 1.0,
                "text": 1.0,
                "semantic": 1.0,
                "output": 1.0,
                "image": 1.0
            }
        }
    
    # 2. 计算各维度相似度
    text_sim = calculate_text_similarity(case1, case2)
    semantic_sim = calculator.calculate_semantic_similarity(case1, case2)
    output_sim = calculate_output_similarity(case1, case2)
    image_sim = calculate_image_similarity(case1["file_path"], case2["file_path"])
    
    # 3. 加权平均（KYC 项目推荐权重）
    weights = {
        "file_hash": 0.3,  # 文件 Hash 最重要（如果相同就是完全重复）
        "text": 0.2,       # 文本描述
        "semantic": 0.2,   # 语义相似度
        "output": 0.15,    # 输出相似度
        "image": 0.15      # 图像相似度
    }
    
    overall_similarity = (
        weights["text"] * text_sim +
        weights["semantic"] * semantic_sim +
        weights["output"] * output_sim +
        weights["image"] * image_sim
    )
    
    return {
        "overall_similarity": overall_similarity,
        "is_exact_duplicate": False,
        "breakdown": {
            "text": text_sim,
            "semantic": semantic_sim,
            "output": output_sim,
            "image": image_sim
        }
    }
```

---

## 🎯 相似度阈值设计（Threshold Design）

### 为什么需要阈值？

**核心问题**：**如何判断两个用例是否"相似"？相似度多少算"相似"？**

阈值设计直接影响去重效果：
- **阈值太高**（如 0.99）：只识别几乎完全相同的用例，可能漏掉很多相似用例
- **阈值太低**（如 0.5）：识别太多"相似"用例，可能误删有价值的用例

---

### 阈值设计原则

#### 1. **分层阈值策略**（推荐）

**不同相似度范围，采用不同策略**：

| 相似度范围 | 策略 | 阈值 | 说明 |
|-----------|------|------|------|
| **1.0** | 完全重复 | 自动删除 | 文件 Hash 相同 |
| **0.95 - 1.0** | 高相似度 | 自动合并 | 几乎相同，保留优先级高的 |
| **0.7 - 0.95** | 中等相似度 | 人工审核 | 可能相似，需要人工判断 |
| **< 0.7** | 低相似度 | 保留 | 不相似，都保留 |

**KYC 项目推荐阈值**：
```python
THRESHOLDS = {
    "exact_duplicate": 1.0,      # 完全重复（文件 Hash）
    "high_similarity": 0.95,     # 高相似度（自动合并）
    "medium_similarity": 0.7,    # 中等相似度（人工审核）
    "low_similarity": 0.0        # 低相似度（保留）
}
```

---

#### 2. **阈值设计方法**

##### 方法 1：基于经验值（快速启动）

**业界常见经验值**：

```python
# 文本相似度阈值
TEXT_SIMILARITY_THRESHOLD = 0.85  # TF-IDF + Cosine

# 语义相似度阈值
SEMANTIC_SIMILARITY_THRESHOLD = 0.80  # Embedding

# 图像相似度阈值
IMAGE_SIMILARITY_THRESHOLD = 0.90  # pHash

# 综合相似度阈值（加权平均）
OVERALL_SIMILARITY_THRESHOLD = 0.85
```

**优点**：
- ✅ 快速启动，不需要大量数据
- ✅ 基于业界经验，通常效果不错

**缺点**：
- ⚠️ 可能不适合特定项目
- ⚠️ 需要后续调优

---

##### 方法 2：基于数据分析（推荐）

**步骤**：

1. **收集样本数据**：
   ```python
   # 收集 100-200 个测试用例样本
   sample_cases = load_sample_cases(100)
   ```

2. **计算相似度矩阵**：
   ```python
   similarity_matrix = []
   for i, case1 in enumerate(sample_cases):
       for j, case2 in enumerate(sample_cases[i+1:], start=i+1):
           similarity = calculate_overall_similarity(case1, case2)
           similarity_matrix.append({
               "case1_id": case1["case_id"],
               "case2_id": case2["case_id"],
               "similarity": similarity,
               "is_similar": is_manually_labeled_similar(case1, case2)  # 人工标注
           })
   ```

3. **分析相似度分布**：
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   similarities = [item["similarity"] for item in similarity_matrix]
   labels = [item["is_similar"] for item in similarity_matrix]
   
   # 绘制相似度分布
   plt.hist(similarities, bins=50)
   plt.xlabel("Similarity Score")
   plt.ylabel("Frequency")
   plt.title("Similarity Distribution")
   plt.show()
   
   # 分析阈值
   similar_scores = [s for s, l in zip(similarities, labels) if l]
   dissimilar_scores = [s for s, l in zip(similarities, labels) if not l]
   
   print(f"相似用例平均相似度: {np.mean(similar_scores):.2f}")
   print(f"不相似用例平均相似度: {np.mean(dissimilar_scores):.2f}")
   ```

4. **选择最优阈值**（ROC 曲线或 Precision-Recall 曲线）：
   ```python
   from sklearn.metrics import roc_curve, auc
   
   # 计算不同阈值下的 Precision 和 Recall
   thresholds = np.arange(0.5, 1.0, 0.01)
   precisions = []
   recalls = []
   
   for threshold in thresholds:
       predicted = [s >= threshold for s in similarities]
       precision = calculate_precision(predicted, labels)
       recall = calculate_recall(predicted, labels)
       precisions.append(precision)
       recalls.append(recall)
   
   # 选择 F1 分数最高的阈值
   f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                for p, r in zip(precisions, recalls)]
   optimal_threshold = thresholds[np.argmax(f1_scores)]
   
   print(f"最优阈值: {optimal_threshold:.2f}")
   ```

---

##### 方法 3：基于业务需求（最实用）

**根据项目特点调整阈值**：

**场景 A：Golden Set 规模小（50-100 条）**
- **策略**：**保守去重**（阈值高）
- **阈值**：`0.90`（只删除几乎完全相同的用例）
- **原因**：用例少，每个用例都很珍贵，宁可多保留

**场景 B：Golden Set 规模大（200+ 条）**
- **策略**：**积极去重**（阈值低）
- **阈值**：`0.80`（删除更多相似用例）
- **原因**：用例多，可以更激进地去重

**场景 C：测试成本高（LLM API 调用）**
- **策略**：**积极去重**（阈值低）
- **阈值**：`0.75`（减少测试次数，降低成本）
- **原因**：每次测试都要花钱，去重可以节省成本

**场景 D：测试覆盖率要求高**
- **策略**：**保守去重**（阈值高）
- **阈值**：`0.90`（保留更多用例，确保覆盖率）
- **原因**：覆盖率优先，成本次要

---

### KYC 项目阈值设计示例

#### 1. 初始阈值（基于经验）

```python
# KYC 项目初始阈值配置
KYC_THRESHOLDS = {
    "exact_duplicate": 1.0,      # 完全重复（文件 Hash）
    "high_similarity": 0.95,     # 高相似度（自动合并）
    "medium_similarity": 0.7,    # 中等相似度（人工审核）
    "low_similarity": 0.0        # 低相似度（保留）
}
```

#### 2. 调优过程

**步骤 1：运行去重分析**
```python
deduplicator = GoldenSetDeduplicator(similarity_threshold=0.85)
result = deduplicator.deduplicate(golden_set)

print(f"原始用例数: {result['original_count']}")
print(f"去重后用例数: {result['final_count']}")
print(f"删除用例数: {result['removed_count']}")
```

**步骤 2：人工审核删除的用例**
```python
# 检查删除的用例，看是否有误删
for removed_case in result["removed_cases"]:
    print(f"删除用例: {removed_case['case']['case_id']}")
    print(f"原因: {removed_case['reason']}")
    print(f"相似度: {removed_case['similarity']:.2f}")
    
    # 人工判断：是否应该删除？
    # 如果发现误删，说明阈值太高，需要降低
```

**步骤 3：调整阈值**
```python
# 如果发现很多误删（应该保留但被删除了）
# → 阈值太高，降低阈值（如：0.85 → 0.80）

# 如果发现很多应该删除但没有删除
# → 阈值太低，提高阈值（如：0.85 → 0.90）

# 重新运行去重
deduplicator = GoldenSetDeduplicator(similarity_threshold=0.80)  # 调整后
result = deduplicator.deduplicate(golden_set)
```

#### 3. 最终阈值（经过调优）

```python
# KYC 项目最终阈值配置（经过调优）
KYC_THRESHOLDS_FINAL = {
    "exact_duplicate": 1.0,      # 完全重复（不变）
    "high_similarity": 0.92,      # 高相似度（从 0.95 降到 0.92，更积极去重）
    "medium_similarity": 0.75,    # 中等相似度（从 0.7 提高到 0.75，减少人工审核）
    "low_similarity": 0.0         # 低相似度（不变）
}
```

---

### 阈值调优检查清单

- [ ] **初始阈值设置**：基于经验值（0.85）或业界标准
- [ ] **运行去重分析**：使用初始阈值运行去重
- [ ] **人工审核删除用例**：检查是否有误删
- [ ] **分析去重效果**：统计删除率、保留率
- [ ] **调整阈值**：根据误删情况调整阈值
- [ ] **重新运行**：使用新阈值重新去重
- [ ] **验证效果**：确认去重效果符合预期
- [ ] **记录阈值**：记录最终阈值和调优过程

---

### 面试话术

**问题**："如何设计相似度阈值？"

**回答**：
> "我们使用**分层阈值策略**：
> 
> 1. **完全重复（1.0）**：文件 Hash 相同，自动删除
> 2. **高相似度（0.95）**：几乎相同，自动合并，保留优先级高的
> 3. **中等相似度（0.7-0.95）**：可能相似，需要人工审核
> 4. **低相似度（< 0.7）**：不相似，都保留
> 
> 初始阈值基于经验值（0.85），然后通过**数据分析**和**人工审核**调优：
> - 如果发现误删（应该保留但被删除），降低阈值
> - 如果发现漏删（应该删除但没有删除），提高阈值
> 
> 最终阈值会根据项目特点调整：
> - Golden Set 规模小 → 阈值高（保守）
> - 测试成本高 → 阈值低（积极去重）
> - 覆盖率要求高 → 阈值高（保守）"

---

## 🔄 去重策略

### 策略 1：自动去重（Fully Automated）

**适用场景**：
- 完全重复的用例（文件 Hash 相同）
- 高相似度用例（相似度 > 0.95）

**流程**：
```
1. 计算所有用例的相似度矩阵
2. 识别重复/高相似度用例对
3. 自动删除重复用例（保留优先级高的）
4. 生成去重报告
```

**KYC 项目实现**：

```python
def auto_deduplicate(golden_set: list, similarity_threshold: float = 0.95) -> dict:
    """自动去重 Golden Set"""
    deduplicated = []
    seen_hashes = set()
    removed_cases = []
    
    for case in golden_set:
        # 1. 检查文件 Hash（完全重复）
        file_hash = calculate_file_hash(case["file_path"])
        
        if file_hash in seen_hashes:
            removed_cases.append({
                "case": case,
                "reason": "exact_duplicate",
                "similarity": 1.0
            })
            continue  # 跳过重复用例
        
        # 2. 检查是否与已有用例高度相似
        is_similar = False
        for existing_case in deduplicated:
            similarity_result = calculate_overall_similarity(case, existing_case)
            similarity = similarity_result["overall_similarity"]
            
            if similarity > similarity_threshold:
                is_similar = True
                # 保留优先级高的用例
                priority1 = case.get("priority", 3)  # 默认最低优先级
                priority2 = existing_case.get("priority", 3)
                
                if priority1 < priority2:  # 数字越小优先级越高
                    # case 优先级更高，替换 existing_case
                    deduplicated.remove(existing_case)
                    removed_cases.append({
                        "case": existing_case,
                        "reason": "similar_higher_priority",
                        "similarity": similarity
                    })
                    deduplicated.append(case)
                else:
                    # existing_case 优先级更高，保留它
                    removed_cases.append({
                        "case": case,
                        "reason": "similar_lower_priority",
                        "similarity": similarity
                    })
                break
        
        if not is_similar:
            deduplicated.append(case)
            seen_hashes.add(file_hash)
    
    return {
        "original_count": len(golden_set),
        "final_count": len(deduplicated),
        "removed_count": len(removed_cases),
        "deduplicated_cases": deduplicated,
        "removed_cases": removed_cases
    }
```

---

### 策略 2：人工审核去重（Human-in-the-Loop）

**适用场景**：
- 中等相似度用例（相似度 0.7-0.95）
- 关键用例（Critical Cases，priority=0）
- 不确定是否应该合并的用例

**流程**：
```
1. 计算相似度矩阵
2. 识别相似用例对（相似度 0.7-0.95）
3. 生成审核报告（列出相似用例对）
4. 人工审核决定：
   - 合并（保留一个）
   - 保留（都是必要的）
   - 删除（重复的）
5. 执行人工决策
```

**KYC 项目实现**：

```python
def generate_review_report(golden_set: list, similarity_threshold: float = 0.7) -> list:
    """生成需要人工审核的相似用例对"""
    review_pairs = []
    
    for i, case1 in enumerate(golden_set):
        for j, case2 in enumerate(golden_set[i+1:], start=i+1):
            similarity_result = calculate_overall_similarity(case1, case2)
            similarity = similarity_result["overall_similarity"]
            
            # 中等相似度（需要人工审核）
            if similarity_threshold <= similarity < 0.95:
                review_pairs.append({
                    "case1": case1,
                    "case2": case2,
                    "similarity": similarity,
                    "similarity_breakdown": similarity_result["breakdown"],
                    "reason": "medium_similarity_needs_review"
                })
    
    # 按相似度排序（相似度高的优先审核）
    review_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return review_pairs

# 生成审核报告
review_pairs = generate_review_report(golden_set, similarity_threshold=0.7)

# 输出报告（供人工审核）
print("=" * 80)
print("需要人工审核的相似用例对")
print("=" * 80)
for idx, pair in enumerate(review_pairs, 1):
    print(f"\n【用例对 {idx}】相似度: {pair['similarity']:.2f}")
    print(f"用例 1: {pair['case1']['case_id']} - {pair['case1']['description']}")
    print(f"用例 2: {pair['case2']['case_id']} - {pair['case2']['description']}")
    print(f"相似度分解: {pair['similarity_breakdown']}")
    print("-" * 80)
```

---

### 策略 3：混合策略（Hybrid Approach）⭐ **推荐**

**适用场景**：**生产环境推荐**（平衡准确性和效率）

**流程**：
```
1. 自动去重完全重复用例（相似度 > 0.95）
2. 自动删除低优先级相似用例（相似度 > 0.9，优先级低）
3. 人工审核中等相似度用例（相似度 0.7-0.9）
4. 保留低相似度用例（相似度 < 0.7）
```

**KYC 项目实现**：

```python
def hybrid_deduplicate(golden_set: list) -> dict:
    """混合去重策略（推荐）"""
    
    # 步骤 1：自动去重完全重复用例（相似度 > 0.95）
    result_high = auto_deduplicate(golden_set, similarity_threshold=0.95)
    deduplicated = result_high["deduplicated_cases"]
    
    # 步骤 2：自动删除低优先级相似用例（相似度 > 0.9）
    high_priority_cases = [c for c in deduplicated if c.get("priority", 3) <= 1]
    low_priority_cases = [c for c in deduplicated if c.get("priority", 3) > 1]
    
    filtered_cases = high_priority_cases.copy()
    removed_low_priority = []
    
    for low_case in low_priority_cases:
        is_similar_to_high = False
        for high_case in high_priority_cases:
            similarity_result = calculate_overall_similarity(low_case, high_case)
            similarity = similarity_result["overall_similarity"]
            
            if similarity > 0.9:  # 高相似度
                is_similar_to_high = True
                removed_low_priority.append({
                    "case": low_case,
                    "reason": "similar_to_high_priority",
                    "similarity": similarity
                })
                break
        
        if not is_similar_to_high:
            filtered_cases.append(low_case)
    
    # 步骤 3：生成人工审核报告（中等相似度用例）
    review_pairs = generate_review_report(filtered_cases, similarity_threshold=0.7)
    
    return {
        "deduplicated_cases": filtered_cases,
        "review_pairs": review_pairs,
        "stats": {
            "original_count": len(golden_set),
            "after_auto_dedup": len(deduplicated),
            "after_priority_filter": len(filtered_cases),
            "needs_review": len(review_pairs),
            "removed_high_similarity": len(result_high["removed_cases"]),
            "removed_low_priority": len(removed_low_priority)
        },
        "removed_cases": result_high["removed_cases"] + removed_low_priority
    }
```

---

## 🎯 相似用例合并策略

### 合并原则

1. **保留最有代表性的用例**
   - ✅ 保留优先级高的用例（priority 数字越小优先级越高）
   - ✅ 保留最极端的用例（如：最模糊的图片）
   - ✅ 保留最完整的用例（字段最全）

2. **保留覆盖不同场景的用例**
   - ✅ 即使相似，如果覆盖不同场景，都应该保留
   - ✅ 例如：模糊身份证 + 遮挡身份证，虽然都是"图片质量问题"，但场景不同

3. **保留关键用例**
   - ✅ Critical Cases（priority=0）必须保留
   - ✅ 业务关键字段的用例必须保留

### 合并示例

**场景**：3 个相似的"模糊身份证"用例

```python
# 原始用例
cases = [
    {
        "case_id": "edge_001",
        "file_path": "test_data/edge/id_card_blur_light.jpg",
        "category": "edge",
        "description": "轻微模糊身份证",
        "priority": 2,
        "tags": ["id_card", "blur", "light"]
    },
    {
        "case_id": "edge_002",
        "file_path": "test_data/edge/id_card_blur_medium.jpg",
        "category": "edge",
        "description": "中等模糊身份证",
        "priority": 2,
        "tags": ["id_card", "blur", "medium"]
    },
    {
        "case_id": "edge_003",
        "file_path": "test_data/edge/id_card_blur_heavy.jpg",
        "category": "edge",
        "description": "严重模糊身份证",
        "priority": 1,  # 高优先级
        "tags": ["id_card", "blur", "heavy"]
    }
]

# 合并策略：
# 1. 相似度检查：都是 "edge" + "blur" + "id_card"，相似度 > 0.8
# 2. 保留策略：保留 edge_003（优先级最高 + 最极端情况）
# 3. 删除：edge_001 和 edge_002（与 edge_003 相似，但优先级低）

# 合并后
merged_cases = [
    {
        "case_id": "edge_003",
        "file_path": "test_data/edge/id_card_blur_heavy.jpg",
        "category": "edge",
        "description": "严重模糊身份证",
        "priority": 1,
        "tags": ["id_card", "blur", "heavy"],
        "merged_from": ["edge_001", "edge_002"]  # 记录合并来源
    }
]
```

---

## 🛠️ KYC 项目完整实现

### 完整去重器类

```python
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import imagehash

class GoldenSetDeduplicator:
    """Golden Set 去重器（KYC 项目专用）"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件 Hash（用于识别完全重复）"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Error calculating file hash for {file_path}: {e}")
            return ""
    
    def calculate_text_similarity(self, case1: Dict, case2: Dict) -> float:
        """计算文本相似度"""
        text1 = f"{case1.get('description', '')} {' '.join(case1.get('tags', []))}"
        text2 = f"{case2.get('description', '')} {' '.join(case2.get('tags', []))}"
        
        if text1 == text2:
            return 1.0
        
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity
    
    def calculate_semantic_similarity(self, case1: Dict, case2: Dict) -> float:
        """计算语义相似度"""
        text1 = f"{case1.get('category', '')} {case1.get('description', '')}"
        text2 = f"{case2.get('category', '')} {case2.get('description', '')}"
        
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity
    
    def calculate_output_similarity(self, case1: Dict, case2: Dict) -> float:
        """计算输出相似度"""
        output1 = case1.get("expected_fields", {})
        output2 = case2.get("expected_fields", {})
        
        fields1 = set(output1.keys())
        fields2 = set(output2.keys())
        
        intersection = fields1 & fields2
        union = fields1 | fields2
        
        field_sim = len(intersection) / len(union) if union else 0.0
        
        value_sims = []
        for field in intersection:
            if output1[field] == output2[field]:
                value_sims.append(1.0)
            else:
                value_sims.append(0.0)
        
        value_sim = np.mean(value_sims) if value_sims else 0.0
        return (field_sim + value_sim) / 2
    
    def calculate_image_similarity(self, file1: str, file2: str) -> float:
        """计算图像相似度"""
        try:
            img1 = Image.open(file1)
            img2 = Image.open(file2)
            
            hash1 = imagehash.phash(img1)
            hash2 = imagehash.phash(img2)
            
            hamming_distance = hash1 - hash2
            similarity = 1 - (hamming_distance / 64.0)
            return similarity
        except Exception as e:
            print(f"Error calculating image similarity: {e}")
            return 0.0
    
    def calculate_overall_similarity(self, case1: Dict, case2: Dict) -> dict:
        """计算综合相似度"""
        # 文件 Hash 检查
        hash1 = self.calculate_file_hash(case1["file_path"])
        hash2 = self.calculate_file_hash(case2["file_path"])
        
        if hash1 == hash2 and hash1 != "":
            return {
                "overall_similarity": 1.0,
                "is_exact_duplicate": True,
                "breakdown": {
                    "file_hash": 1.0,
                    "text": 1.0,
                    "semantic": 1.0,
                    "output": 1.0,
                    "image": 1.0
                }
            }
        
        # 计算各维度相似度
        text_sim = self.calculate_text_similarity(case1, case2)
        semantic_sim = self.calculate_semantic_similarity(case1, case2)
        output_sim = self.calculate_output_similarity(case1, case2)
        image_sim = self.calculate_image_similarity(case1["file_path"], case2["file_path"])
        
        # 加权平均
        overall_similarity = (
            0.2 * text_sim +
            0.2 * semantic_sim +
            0.15 * output_sim +
            0.15 * image_sim
        )
        
        return {
            "overall_similarity": overall_similarity,
            "is_exact_duplicate": False,
            "breakdown": {
                "text": text_sim,
                "semantic": semantic_sim,
                "output": output_sim,
                "image": image_sim
            }
        }
    
    def deduplicate(self, golden_set: List[Dict]) -> Dict:
        """去重 Golden Set（混合策略）"""
        # 步骤 1：自动去重完全重复用例
        seen_hashes = {}
        duplicates = []
        
        for case in golden_set:
            file_hash = self.calculate_file_hash(case["file_path"])
            if file_hash in seen_hashes and file_hash != "":
                duplicates.append({
                    "original": seen_hashes[file_hash],
                    "duplicate": case,
                    "type": "exact_duplicate"
                })
            else:
                seen_hashes[file_hash] = case
        
        # 步骤 2：识别相似用例
        unique_cases = list(seen_hashes.values())
        similar_pairs = []
        
        for i, case1 in enumerate(unique_cases):
            for j, case2 in enumerate(unique_cases[i+1:], start=i+1):
                similarity_result = self.calculate_overall_similarity(case1, case2)
                similarity = similarity_result["overall_similarity"]
                
                if similarity >= self.similarity_threshold:
                    similar_pairs.append({
                        "case1": case1,
                        "case2": case2,
                        "similarity": similarity,
                        "breakdown": similarity_result["breakdown"]
                    })
        
        # 步骤 3：合并相似用例（保留优先级高的）
        merged_cases = unique_cases.copy()
        to_remove = set()
        
        for pair in similar_pairs:
            case1 = pair["case1"]
            case2 = pair["case2"]
            
            if case1 in to_remove or case2 in to_remove:
                continue
            
            priority1 = case1.get("priority", 3)
            priority2 = case2.get("priority", 3)
            
            if priority1 < priority2:  # 数字越小优先级越高
                to_remove.add(case2)
            elif priority2 < priority1:
                to_remove.add(case1)
            else:
                # 优先级相同，保留第一个
                to_remove.add(case2)
        
        final_cases = [c for c in merged_cases if c not to_remove]
        
        return {
            "original_count": len(golden_set),
            "final_count": len(final_cases),
            "removed_count": len(golden_set) - len(final_cases),
            "duplicates": duplicates,
            "similar_pairs": similar_pairs,
            "deduplicated_cases": final_cases,
            "stats": {
                "exact_duplicates": len(duplicates),
                "similar_pairs": len(similar_pairs),
                "removed_cases": len(to_remove)
            }
        }

# 使用示例
deduplicator = GoldenSetDeduplicator(similarity_threshold=0.85)

# 加载 Golden Set
with open("golden_set.json", "r") as f:
    golden_set_data = json.load(f)
    golden_set = (
        golden_set_data.get("normal_cases", []) +
        golden_set_data.get("edge_cases", []) +
        golden_set_data.get("anomaly_cases", []) +
        golden_set_data.get("longtail_cases", [])
    )

# 执行去重
result = deduplicator.deduplicate(golden_set)

# 保存去重后的 Golden Set
deduplicated_golden_set = {
    "normal_cases": [c for c in result["deduplicated_cases"] if c.get("category") == "normal"],
    "edge_cases": [c for c in result["deduplicated_cases"] if c.get("category") == "edge"],
    "anomaly_cases": [c for c in result["deduplicated_cases"] if c.get("category") == "anomaly"],
    "longtail_cases": [c for c in result["deduplicated_cases"] if c.get("category") == "longtail"]
}

with open("golden_set_deduplicated.json", "w", encoding="utf-8") as f:
    json.dump(deduplicated_golden_set, f, ensure_ascii=False, indent=2)

# 打印统计信息
print(f"原始用例数: {result['original_count']}")
print(f"去重后用例数: {result['final_count']}")
print(f"删除用例数: {result['removed_count']}")
print(f"完全重复: {result['stats']['exact_duplicates']}")
print(f"相似用例对: {result['stats']['similar_pairs']}")
```

---

## 📊 去重报告模板

### 报告结构

```json
{
  "deduplication_report": {
    "timestamp": "2025-01-19T10:00:00Z",
    "golden_set_version": "v1.2.3",
    "similarity_threshold": 0.85,
    "summary": {
      "original_count": 200,
      "final_count": 150,
      "removed_count": 50,
      "removal_rate": 0.25
    },
    "exact_duplicates": [
      {
        "original": {"case_id": "normal_001", "file_path": "test_data/normal/id_card_001.jpg"},
        "duplicate": {"case_id": "normal_005", "file_path": "test_data/normal/id_card_001.jpg"},
        "reason": "same_file_hash"
      }
    ],
    "similar_cases_merged": [
      {
        "kept": {"case_id": "edge_003", "file_path": "test_data/edge/id_card_blur_heavy.jpg", "priority": 1},
        "removed": [
          {"case_id": "edge_001", "file_path": "test_data/edge/id_card_blur_light.jpg", "priority": 2},
          {"case_id": "edge_002", "file_path": "test_data/edge/id_card_blur_medium.jpg", "priority": 2}
        ],
        "similarity": 0.88,
        "reason": "same_scenario_higher_priority"
      }
    ],
    "needs_review": [
      {
        "case1": {"case_id": "normal_010", "file_path": "test_data/normal/id_card_010.jpg"},
        "case2": {"case_id": "normal_011", "file_path": "test_data/normal/id_card_011.jpg"},
        "similarity": 0.75,
        "reason": "medium_similarity_needs_manual_review"
      }
    ]
  }
}
```

---

## 🎯 KYC 项目应用场景

### 场景 1：Golden Set 构建时去重

**流程**：
```
1. 收集候选测试用例（从生产数据、人工标注等）
2. 运行去重分析
3. 自动去重完全重复用例
4. 人工审核相似用例
5. 生成最终 Golden Set
```

### 场景 2：Golden Set 维护时去重

**流程**：
```
1. 每月/每季度运行去重分析
2. 识别新增的重复/相似用例
3. 自动去重或人工审核
4. 更新 Golden Set
5. 生成去重报告
```

### 场景 3：CI/CD 集成

**流程**：
```
1. 每次添加新用例时，自动运行去重检查
2. 如果发现重复用例，阻止添加或警告
3. 生成去重报告，供开发者参考
```

---

## 💡 面试话术

### 核心话术

1. ✅ **为什么需要去重**：
   - "Golden Set 会随时间增长，如果不去重，会有很多重复和相似用例，浪费测试资源（每次调用 LLM API 都要花钱）。我们使用自动去重 + 人工审核的混合策略。"

2. ✅ **如何识别相似用例**：
   - "我们使用多维度相似度计算：文件 Hash（识别完全重复）、文本相似度（TF-IDF）、语义相似度（Embedding）、图像相似度（pHash）、输出相似度（字段对比）。综合相似度超过阈值（0.85）的用例会被识别为相似用例。"

3. ✅ **去重策略**：
   - "完全重复的用例（文件 Hash 相同）自动删除。高相似度用例（> 0.95）自动合并，保留优先级高的。中等相似度用例（0.7-0.95）需要人工审核，确保不会误删重要用例。"

4. ✅ **实际效果**：
   - "通过去重，我们将 Golden Set 从 200 个用例减少到 150 个，删除率 25%，但测试覆盖率没有下降，测试效率提升了 25%，成本降低了 25%。"

---

## 📝 实施检查清单

- [ ] **实现相似度计算**：文件 Hash、文本、语义、图像、输出相似度
- [ ] **实现自动去重**：完全重复用例自动删除
- [ ] **实现合并策略**：相似用例合并，保留优先级高的
- [ ] **实现人工审核流程**：中等相似度用例人工审核
- [ ] **集成到 CI/CD**：新用例添加时自动去重检查
- [ ] **生成去重报告**：统计信息、相似用例对、需要审核的用例
- [ ] **定期维护**：每月/每季度运行去重分析

---

## 🔗 相关文档

- **Parent**: [KYC_Day03_A1_回归测试与门禁详解.md](./KYC_Day03_A1_回归测试与门禁详解.md)
- **Related**: Golden Set、测试用例管理、回归测试、覆盖度分析

---

**最后更新**：2025-01-19
