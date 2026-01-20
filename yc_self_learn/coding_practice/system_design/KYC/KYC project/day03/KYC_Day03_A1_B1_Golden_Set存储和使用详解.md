# Golden Set 存储和使用详解

---
doc_type: tutorial
layer: L2
scope_in:  Golden Set 存储方案（本地 vs 云端 vs 数据库）、使用方式（CI/CD 集成、定期回归测试）、版本管理
scope_out: 具体存储实现代码（见 howto）；CI/CD 平台配置（见 reference）
inputs:  (读者) 需求：理解 Golden Set 构建后如何存储和使用，确保团队能共享和持续使用
outputs:  Golden Set 存储方案对比 + 使用方式详解 + CI/CD 集成方案 + KYC 项目实际案例
entrypoints: [ Golden Set 存储, CI/CD 集成, 版本管理 ]
children: []
related: [ Golden Set, 回归测试, CI/CD, KYC_Day03_A1_回归测试与门禁详解.md ]
---

## Definition（定义）

**核心问题**：**Golden Set 构建完成后，应该存储在哪里？如何使用？**

**核心答案**：
- ✅ **存储方案**：Git 仓库（代码）+ 对象存储（测试数据文件）+ 数据库（元数据）
- ✅ **使用方式**：CI/CD 自动触发 + 定期回归测试 + Before/After 对比
- ✅ **版本管理**：Git 版本控制 + 数据快照 + 变更历史追踪

---

## 📦 Golden Set 存储方案

### 1. 存储架构设计

**业界常见做法（三层存储）**：

```
┌─────────────────────────────────────────────┐
│  Golden Set 完整存储架构                    │
├─────────────────────────────────────────────┤
│                                             │
│  1. Git 仓库（代码和配置）                  │
│     ├── golden_set.json（用例元数据）       │
│     ├── test_cases_config.yaml（配置）      │
│     └── regression_test.py（测试脚本）      │
│                                             │
│  2. 对象存储（测试数据文件）                │
│     ├── S3 / GCS / Azure Blob              │
│     ├── 图片文件（身份证、护照等）          │
│     └── PDF 文件（多页文档）                │
│                                             │
│  3. 数据库（元数据和结果）                  │
│     ├── PostgreSQL / MongoDB               │
│     ├── 测试用例元数据                      │
│     └── 历史测试结果                        │
│                                             │
└─────────────────────────────────────────────┘
```

---

### 2. 数据流程图：Golden Set 完整生命周期

**从构建到使用的完整数据流**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Golden Set 数据流程图                                 │
└─────────────────────────────────────────────────────────────────────────────┘

【阶段 1：构建阶段 - 数据收集和存储】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  生产环境                自动采样脚本            Git 仓库                  │
│  ┌──────────┐         ┌──────────────┐        ┌─────────────┐             │
│  │ 用户数据 │────────>│ 数据采样     │───────>│ golden_set  │             │
│  │ (1000+)  │         │ 脚本         │        │ .json       │             │
│  └──────────┘         └──────────────┘        │ (元数据)    │             │
│         │                     │                └─────────────┘             │
│         │                     │                        │                    │
│         │                     │                        │                    │
│         └─────────────────────┴────────────────────────┼                    │
│                         │                               │                    │
│                         ▼                               │                    │
│                  ┌──────────────┐                      │                    │
│                  │ 人工审核筛选  │                      │                    │
│                  │ (选择100条)   │                      │                    │
│                  └──────────────┘                      │                    │
│                         │                               │                    │
│                         │                               ▼                    │
│                         │                    ┌─────────────────┐             │
│                         │                    │ test_data/      │             │
│                         │                    │  normal/        │             │
│                         │                    │  edge/          │             │
│                         │                    │  anomaly/       │             │
│                         │                    └─────────────────┘             │
│                         │                               │                    │
│                         └───────────────────────────────┘                    │
│                                         │                                    │
│                                         ▼                                    │
│                              ┌──────────────────┐                            │
│                              │  对象存储 (S3)    │                            │
│                              │  test_data/      │                            │
│                              │  ├── normal/     │                            │
│                              │  ├── edge/       │                            │
│                              │  └── anomaly/    │                            │
│                              └──────────────────┘                            │
│                                         │                                    │
│                                         │                                    │
│                                         ▼                                    │
│                              ┌──────────────────┐                            │
│                              │  数据库 (可选)    │                            │
│                              │  golden_set_cases│                            │
│                              │  (元数据表)       │                            │
│                              └──────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【阶段 2：使用阶段 - CI/CD 自动触发回归测试】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  GitHub/GitLab          CI/CD Runner       回归测试脚本                     │
│  ┌──────────┐         ┌──────────────┐    ┌──────────────┐                │
│  │ PR/Commit│────────>│ GitHub       │───>│ regression_  │                │
│  │ 触发     │         │ Actions      │    │ test.py      │                │
│  └──────────┘         └──────────────┘    └──────────────┘                │
│                              │                     │                         │
│                              │                     │                         │
│                              │  ① 下载配置         │                         │
│                              │  (从 Git)           │                         │
│                              │<────────────────────┘                         │
│                              │                     │                         │
│                              │  ② 下载测试数据     │                         │
│                              │  (从 S3)            │                         │
│                              │<────────────────────┘                         │
│                              │                     │                         │
│                              │                     ▼                         │
│                              │            ┌──────────────────┐               │
│                              │            │ 读取 golden_set  │               │
│                              │            │ .json            │               │
│                              │            │ (包含 S3 URL)    │               │
│                              │            └──────────────────┘               │
│                              │                     │                         │
│                              │                     ▼                         │
│                              │            ┌──────────────────┐               │
│                              │            │ 遍历每个用例     │               │
│                              │            │ ┌──────────────┐ │               │
│                              │            │ │ case_001     │ │               │
│                              │            │ │ case_002     │ │               │
│                              │            │ │ ...          │ │               │
│                              │            │ └──────────────┘ │               │
│                              │            └──────────────────┘               │
│                              │                     │                         │
│                              │                     ▼                         │
│                              │         ┌──────────────────────┐              │
│                              │         │ 对于每个用例:         │              │
│                              │         │  1. 从 S3 下载文件   │              │
│                              │         │  2. 运行 OCR 处理    │              │
│                              │         │  3. 运行 Validator   │              │
│                              │         │  4. 对比结果         │              │
│                              │         └──────────────────────┘              │
│                              │                     │                         │
│                              │                     ▼                         │
│                              │         ┌──────────────────────┐              │
│                              │         │ 生成测试结果         │              │
│                              │         │ {                    │              │
│                              │         │   "total": 100,      │              │
│                              │         │   "passed": 95,      │              │
│                              │         │   "failed": 5,       │              │
│                              │         │   "results": [...]   │              │
│                              │         │ }                    │              │
│                              │         └──────────────────────┘              │
│                              │                     │                         │
│                              │                     │                         │
└──────────────────────────────┼─────────────────────┼─────────────────────────┘
                               │                     │
                               │                     ▼
                               │         ┌──────────────────────┐
                               │         │ 保存结果到本地        │
                               │         │ regression_results   │
                               │         │ .json                │
                               │         └──────────────────────┘
                               │                     │
                               │                     │
                               ▼                     ▼

【阶段 3：结果处理和监控 - 数据上报和告警】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  回归测试脚本           Datadog             数据库 (可选)                   │
│  ┌──────────┐         ┌──────────────┐    ┌──────────────┐                │
│  │ 测试结果 │────────>│ Metrics API  │    │ test_results │                │
│  │ JSON     │         │              │    │ (历史结果表)  │                │
│  └──────────┘         │ ┌──────────┐ │    └──────────────┘                │
│         │             │ │ 指标上传 │ │              ▲                      │
│         │             │ │ accuracy │ │              │                      │
│         │             │ │ passed   │ │              │                      │
│         │             │ │ failed   │ │              │                      │
│         │             │ └──────────┘ │              │                      │
│         │             │              │              │                      │
│         │             │ ┌──────────┐ │              │                      │
│         │             │ │ Logs API │ │              │                      │
│         │             │ │ (失败用例)│ │              │                      │
│         │             │ └──────────┘ │              │                      │
│         │             │              │              │                      │
│         │             │ ┌──────────┐ │              │                      │
│         │             │ │ Events   │ │              │                      │
│         │             │ │ (告警)   │ │              │                      │
│         │             │ └──────────┘ │              │                      │
│         │             └──────────────┘              │                      │
│         │                     │                     │                      │
│         │                     │                     │                      │
│         │                     ▼                     │                      │
│         │            ┌──────────────────┐           │                      │
│         │            │ Datadog Dashboard│           │                      │
│         │            │ 可视化监控        │           │                      │
│         │            └──────────────────┘           │                      │
│         │                     │                     │                      │
│         │                     │                     │                      │
│         └─────────────────────┴─────────────────────┘                      │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌────────────────────────┐                               │
│                    │ 告警规则                │                               │
│                    │ if accuracy < 95%:     │                               │
│                    │   send_alert()         │                               │
│                    └────────────────────────┘                               │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌────────────────────────┐                               │
│                    │ 通知团队                │                               │
│                    │ - Slack                │                               │
│                    │ - Email                │                               │
│                    │ - PagerDuty            │                               │
│                    └────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【阶段 4：反馈循环 - 更新 Golden Set】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  生产环境错误         分析脚本           更新 Golden Set                    │
│  ┌──────────┐       ┌──────────────┐    ┌──────────────┐                  │
│  │ 新错误   │──────>│ 提取失败案例 │───>│ 添加到       │                  │
│  │ 案例     │       │              │    │ Golden Set   │                  │
│  └──────────┘       └──────────────┘    └──────────────┘                  │
│         │                    │                    │                         │
│         │                    │                    ▼                         │
│         │                    │          ┌──────────────────┐                │
│         │                    │          │ 1. 更新 JSON     │                │
│         │                    │          │ 2. 上传文件到 S3 │                │
│         │                    │          │ 3. Git Commit    │                │
│         │                    │          └──────────────────┘                │
│         │                    │                    │                         │
│         └────────────────────┴────────────────────┘                         │
│                                    │                                         │
│                                    ▼                                         │
│                         ┌──────────────────────┐                            │
│                         │ 下次回归测试生效      │                            │
│                         └──────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

**数据流关键路径说明**：

1. **构建阶段**：
   ```
   生产数据 → 采样脚本 → 人工审核 → Git (元数据) + S3 (文件) + 数据库 (可选)
   ```

2. **使用阶段（CI/CD）**：
   ```
   PR/Commit 触发 → CI/CD Runner → 
     ① 从 Git 下载配置 (golden_set.json) → 
     ② 从 S3 下载测试数据文件 → 
     ③ 运行回归测试 → 
     ④ 生成结果 JSON
   ```

3. **结果处理**：
   ```
   测试结果 → Datadog (Metrics/Logs/Events) → 数据库 (可选) → Dashboard/告警
   ```

4. **反馈循环**：
   ```
   生产错误 → 提取失败案例 → 更新 Golden Set → 下次测试生效
   ```

---

### 3. 详细数据交互时序图

**单次回归测试的数据交互流程**：

```
时间轴 →

开发者                Git 仓库          CI/CD Runner       S3 存储          KYC 服务        Datadog
  │                    │                  │                  │                │                │
  │  ① Push PR         │                  │                  │                │                │
  ├───────────────────>│                  │                  │                │                │
  │                    │                  │                  │                │                │
  │                    │  ② 触发 CI/CD    │                  │                │                │
  │                    ├─────────────────>│                  │                │                │
  │                    │                  │                  │                │                │
  │                    │  ③ 下载配置      │                  │                │                │
  │                    │<─────────────────┤                  │                │                │
  │                    │  (golden_set.json)                  │                │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ④ 读取用例列表  │                │                │
  │                    │                  │  (从 JSON)       │                │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ⑤ 下载测试文件  │                │                │
  │                    │                  ├─────────────────>│                │                │
  │                    │                  │  (test_data/...) │                │                │
  │                    │                  │<─────────────────┤                │                │
  │                    │                  │  (文件下载完成)   │                │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ⑥ 运行测试      │                │                │
  │                    │                  ├──────────────────────────────────>│                │
  │                    │                  │  OCR + Validator │                │                │
  │                    │                  │<──────────────────────────────────┤                │
  │                    │                  │  (处理结果)      │                │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ⑦ 对比结果      │                │                │
  │                    │                  │  (expected vs actual)             │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ⑧ 生成测试报告  │                │                │
  │                    │                  │  (regression_results.json)        │                │
  │                    │                  │                  │                │                │
  │                    │                  │  ⑨ 上传结果      │                │                │
  │                    │                  ├──────────────────────────────────────────────────>│
  │                    │                  │  Metrics + Logs  │                │                │
  │                    │                  │<──────────────────────────────────────────────────┤
  │                    │                  │  (上传成功)      │                │                │
  │                    │                  │                  │                │                │
  │                    │  ⑩ 更新 PR 状态  │                  │                │                │
  │                    │<─────────────────┤                  │                │                │
  │                    │  (测试结果评论)   │                  │                │                │
  │  ⑪ 查看结果        │                  │                  │                │                │
  │<───────────────────┤                  │                  │                │                │
  │                    │                  │                  │                │                │
```

**关键数据交互点**：

| 步骤 | 数据源 | 数据目标 | 数据类型 | 大小 |
|------|--------|---------|---------|------|
| ① | 开发者 | Git 仓库 | 代码变更 | ~KB |
| ② | Git 仓库 | CI/CD | Webhook | ~KB |
| ③ | Git 仓库 | CI/CD Runner | JSON 配置文件 | ~10-100KB |
| ④ | JSON 配置 | CI/CD Runner | 用例元数据 | ~10-50KB |
| ⑤ | S3 | CI/CD Runner | 测试文件（图片/PDF） | ~10-100MB |
| ⑥ | CI/CD Runner | KYC 服务 | 文件 + 请求 | ~10MB |
| ⑦ | KYC 服务 | CI/CD Runner | 处理结果 | ~KB |
| ⑧ | CI/CD Runner | 本地文件 | 测试报告 JSON | ~10-100KB |
| ⑨ | CI/CD Runner | Datadog | Metrics + Logs | ~KB |
| ⑩ | CI/CD Runner | Git 仓库 | PR 评论 | ~KB |

---

### 2. 存储方案对比

| 存储位置 | 优点 | 缺点 | 适用场景 | 业界使用率 |
|---------|------|------|---------|-----------|
| **Git 仓库** | ✅ 版本控制<br>✅ 团队协作<br>✅ 免费（GitHub/GitLab） | ⚠️ 大文件限制（>100MB）<br>⚠️ 不适合二进制文件 | 配置文件、元数据、测试脚本 | 🔥🔥🔥🔥🔥 极高 |
| **对象存储**（S3/GCS） | ✅ 适合大文件<br>✅ 版本控制（版本化存储）<br>✅ CDN 加速 | ⚠️ 需要付费<br>⚠️ 需要额外配置 | 测试数据文件（图片、PDF） | 🔥🔥🔥🔥 高 |
| **数据库**（PostgreSQL） | ✅ 查询方便<br>✅ 元数据管理<br>✅ 关联历史结果 | ⚠️ 不适合存储大文件<br>⚠️ 需要数据库维护 | 测试用例元数据、历史结果 | 🔥🔥🔥 中等 |
| **本地存储** | ✅ 简单直接<br>✅ 无网络依赖 | ❌ 无法团队共享<br>❌ 无版本控制<br>❌ 容易丢失 | ❌ 不推荐生产使用 | ⭐ 仅开发测试 |
| **Datadog** | ✅ 监控和告警<br>✅ 可视化 | ❌ 不是数据存储平台<br>❌ 不适合存储测试数据 | ❌ 不适合存储 Golden Set | ❌ 不适用 |

**❌ 为什么不用 Datadog？**
- Datadog 是**监控和可观测性平台**，不是数据存储平台
- Datadog 适合存储**Metrics、Logs、Traces**，不适合存储**测试数据文件**
- Golden Set 的测试结果可以**发送到 Datadog** 进行监控，但数据本身需要存在其他地方

---

### 3. 推荐存储方案（业界最佳实践）

#### 方案 A：小型团队（推荐）

**架构**：
```
Git 仓库（GitHub/GitLab）
  ├── golden_set/
  │   ├── golden_set.json（用例元数据）
  │   ├── test_config.yaml（测试配置）
  │   └── regression_test.py（测试脚本）
  └── test_data/（小文件，<10MB/文件）
      ├── normal/
      ├── edge/
      └── anomaly/
```

**优点**：
- ✅ 简单，无需额外配置
- ✅ 免费（GitHub/GitLab）
- ✅ 版本控制完善

**适用场景**：
- 团队规模：<10 人
- 测试文件大小：<10MB/文件
- 总存储：<1GB

**代码示例**：

```python
# 项目结构
project/
├── golden_set/
│   ├── golden_set.json          # 用例元数据
│   ├── test_config.yaml         # 测试配置
│   └── regression_test.py       # 测试脚本
├── test_data/                   # 测试数据文件
│   ├── normal/
│   │   ├── id_card_001.jpg
│   │   └── passport_001.pdf
│   ├── edge/
│   │   └── id_card_blurry_001.jpg
│   └── anomaly/
│       └── passport_multipage_001.pdf
└── .gitignore                   # 大文件不上传（如果使用方案 B）
```

```python
# golden_set.json 结构
{
  "version": "1.0.0",
  "last_updated": "2024-01-15",
  "total_cases": 100,
  "cases": [
    {
      "case_id": "normal_001",
      "file_path": "test_data/normal/id_card_001.jpg",
      "file_url": null,  # 本地文件，不需要 URL
      "category": "normal",
      "expected_fields": {
        "name": "张三",
        "id_number": "110101199001011234",
        "date_of_birth": "1990-01-01"
      },
      "metadata": {
        "description": "标准身份证，清晰、标准格式",
        "added_date": "2024-01-10",
        "source": "production_sample"
      }
    }
  ]
}
```

---

#### 方案 B：中大型团队（推荐）

**架构**：
```
Git 仓库（GitHub/GitLab）
  ├── golden_set/
  │   ├── golden_set.json（用例元数据，包含 S3 URL）
  │   ├── test_config.yaml
  │   └── regression_test.py
  └── .gitignore（排除 test_data/）

对象存储（AWS S3 / GCS）
  └── test_data/
      ├── normal/
      ├── edge/
      └── anomaly/

数据库（PostgreSQL，可选）
  └── golden_set_metadata（元数据表）
```

**优点**：
- ✅ 适合大文件（无大小限制）
- ✅ 团队协作方便
- ✅ 版本控制（对象存储版本化）
- ✅ 可以 CDN 加速

**适用场景**：
- 团队规模：>10 人
- 测试文件大小：>10MB/文件
- 总存储：>1GB

**代码示例**：

```python
# golden_set.json 结构（包含 S3 URL）
{
  "version": "1.0.0",
  "last_updated": "2024-01-15",
  "total_cases": 100,
  "storage_config": {
    "type": "s3",
    "bucket": "kyc-golden-set",
    "region": "us-east-1",
    "base_url": "s3://kyc-golden-set/test_data"
  },
  "cases": [
    {
      "case_id": "normal_001",
      "file_path": "test_data/normal/id_card_001.jpg",
      "file_url": "s3://kyc-golden-set/test_data/normal/id_card_001.jpg",
      "category": "normal",
      "expected_fields": {
        "name": "张三",
        "id_number": "110101199001011234",
        "date_of_birth": "1990-01-01"
      },
      "metadata": {
        "description": "标准身份证，清晰、标准格式",
        "added_date": "2024-01-10",
        "source": "production_sample",
        "file_size_mb": 2.5,
        "s3_version": "v1.0.0"
      }
    }
  ]
}
```

```python
# 上传测试数据到 S3
import boto3
from pathlib import Path

def upload_test_data_to_s3(local_path: str, s3_bucket: str, s3_key: str):
    """上传测试数据文件到 S3"""
    s3_client = boto3.client('s3')
    
    s3_client.upload_file(
        local_path,
        s3_bucket,
        s3_key,
        ExtraArgs={
            'Metadata': {
                'upload_date': str(datetime.now()),
                'source': 'golden_set_upload'
            }
        }
    )
    
    print(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")

# 使用示例
upload_test_data_to_s3(
    local_path="test_data/normal/id_card_001.jpg",
    s3_bucket="kyc-golden-set",
    s3_key="test_data/normal/id_card_001.jpg"
)
```

```python
# 从 S3 下载测试数据
def download_test_data_from_s3(s3_bucket: str, s3_key: str, local_path: str):
    """从 S3 下载测试数据文件"""
    s3_client = boto3.client('s3')
    
    s3_client.download_file(
        s3_bucket,
        s3_key,
        local_path
    )
    
    print(f"Downloaded s3://{s3_bucket}/{s3_key} to {local_path}")
```

---

#### 方案 C：企业级（可选）

**架构**：
```
Git 仓库（GitHub/GitLab Enterprise）
  ├── golden_set/
  │   ├── golden_set.json（用例元数据）
  │   ├── test_config.yaml（测试配置）
  │   └── regression_test.py（测试脚本）
  └── .gitignore（排除 test_data/）

对象存储（S3/GCS）
  └── test_data/（测试数据文件）
      ├── normal/
      ├── edge/
      └── anomaly/

数据库（PostgreSQL）
  ├── golden_set_cases（用例表）
  ├── test_results（历史结果表）
  └── test_metadata（元数据表）

Datadog（监控）
  └── 测试结果 Metrics 和 Logs
```

**✅ 是的，Git 仍然是代码管理的核心**

**Git 的作用**：
- ✅ **版本控制**：测试脚本、配置文件、用例元数据的版本管理
- ✅ **代码审查**：通过 PR 流程审查测试用例的变更
- ✅ **团队协作**：多人协作编辑测试用例和测试脚本
- ✅ **CI/CD 集成**：Git 触发 CI/CD 流程
- ✅ **变更追踪**：追踪谁在什么时候修改了什么

**各组件分工**：
- **Git 仓库**：代码、配置、元数据（JSON/YAML）→ 版本控制
- **S3/GCS**：测试数据文件（图片、PDF）→ 大文件存储
- **数据库**：元数据副本 + 历史结果 → 查询和分析
- **Datadog**：监控指标和日志 → 可观测性

**优点**：
- ✅ 完整的元数据管理（Git + 数据库双重保障）
- ✅ 历史结果追踪（数据库存储历史测试结果）
- ✅ 监控和告警（Datadog）
- ✅ 版本控制完善（Git）
- ✅ 团队协作顺畅（Git PR 流程）

**适用场景**：
- 团队规模：>50 人
- 需要完整的历史追踪
- 需要监控和告警
- 需要复杂的权限管理（Git Enterprise 权限）

**代码示例**：

```python
# 数据库表结构（PostgreSQL）
CREATE TABLE golden_set_cases (
    case_id VARCHAR(100) PRIMARY KEY,
    category VARCHAR(50),
    file_url TEXT,
    expected_fields JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE test_results (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(100) REFERENCES golden_set_cases(case_id),
    test_version VARCHAR(50),
    test_date TIMESTAMP DEFAULT NOW(),
    passed BOOLEAN,
    actual_fields JSONB,
    error_message TEXT,
    execution_time_ms INTEGER
);

# 存储用例到数据库
def save_case_to_database(case_data: dict):
    """保存用例到数据库"""
    conn = psycopg2.connect(
        host="localhost",
        database="kyc_test",
        user="postgres",
        password="password"
    )
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO golden_set_cases 
        (case_id, category, file_url, expected_fields, metadata)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        case_data['case_id'],
        case_data['category'],
        case_data['file_url'],
        json.dumps(case_data['expected_fields']),
        json.dumps(case_data['metadata'])
    ))
    
    conn.commit()
    cur.close()
    conn.close()
```

---

## 🔄 Golden Set 使用方式

### 1. 使用场景

| 使用场景 | 触发方式 | 执行频率 | 用途 |
|---------|---------|---------|------|
| **CI/CD 集成** | PR/Commit 触发 | 每次代码变更 | 防止退化（Regression） |
| **定期回归测试** | 定时任务（Cron） | 每天/每周 | 监控系统健康度 |
| **发布前检查** | 手动触发 | 每次发布前 | 确保发布安全 |
| **模型更新后** | 手动触发 | 每次模型更新 | 验证新模型性能 |

---

### 2. CI/CD 集成（业界最佳实践）

**✅ 是的，CI/CD 配置都是自己写的！**

**CI/CD 配置方式**：

| 配置方式 | 说明 | 优点 | 缺点 |
|---------|------|------|------|
| **直接写 YAML 文件**（推荐） | 在代码仓库中创建 `.github/workflows/*.yml` 或 `.gitlab-ci.yml` | ✅ 版本控制<br>✅ 可复用<br>✅ 团队协作 | 需要学习 YAML 语法 |
| **GitHub/GitLab Web UI** | 在网页界面中创建和编辑 | ✅ 可视化<br>✅ 新手友好 | ⚠️ 功能有限<br>⚠️ 不便于版本控制 |

**业界最佳实践**：
- ✅ **配置文件放在代码仓库中**：`.github/workflows/*.yml` 或 `.gitlab-ci.yml`
- ✅ **版本控制**：和其他代码一起提交到 Git
- ✅ **团队协作**：通过 PR 流程审查 CI/CD 配置变更
- ✅ **复用性强**：可以复制模板，修改参数

**如何创建 CI/CD 配置**：

1. **在本地创建配置文件**：
   ```bash
   # GitHub Actions
   mkdir -p .github/workflows
   touch .github/workflows/regression_test.yml
   
   # GitLab CI
   touch .gitlab-ci.yml
   ```

2. **编写 YAML 配置文件**（见下面的示例）

3. **提交到 Git 仓库**：
   ```bash
   git add .github/workflows/regression_test.yml
   git commit -m "Add regression test CI/CD workflow"
   git push
   ```

4. **GitHub/GitLab 自动识别并运行**：
   - GitHub：自动检测 `.github/workflows/*.yml` 文件，并在触发事件时运行
   - GitLab：自动检测 `.gitlab-ci.yml` 文件，并在触发事件时运行

**不需要手动在 Web UI 中配置**（除非使用高级功能）：
- ✅ 配置文件在代码仓库中，GitHub/GitLab 会自动识别
- ✅ 每次 Push 代码时，CI/CD 会自动运行
- ✅ 可以在 GitHub/GitLab 的 Web UI 中查看运行结果和日志

---

#### GitHub Actions 示例

```yaml
# .github/workflows/regression_test.yml
name: Regression Test

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # 每天午夜运行

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Download test data from S3
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 sync s3://kyc-golden-set/test_data ./test_data
      
      - name: Run regression test
        run: |
          python golden_set/regression_test.py \
            --golden-set golden_set/golden_set.json \
            --baseline-version v1.0.0 \
            --fail-on-regression
      
      - name: Upload results to Datadog
        env:
          DD_API_KEY: ${{ secrets.DD_API_KEY }}
        run: |
          python scripts/upload_test_results_to_datadog.py \
            --results-file regression_results.json
      
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('regression_results.json', 'utf8'));
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Regression Test Results\n\n` +
                    `✅ Passed: ${results.passed}/${results.total}\n` +
                    `❌ Failed: ${results.failed}/${results.total}\n` +
                    `📊 Accuracy: ${results.accuracy}%`
            });
```

#### GitLab CI 示例

```yaml
# .gitlab-ci.yml
stages:
  - test
  - regression

regression-test:
  stage: regression
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - aws s3 sync s3://kyc-golden-set/test_data ./test_data
    - python golden_set/regression_test.py
  artifacts:
    reports:
      junit: regression_results.xml
    paths:
      - regression_results.json
  only:
    - merge_requests
    - main
    - schedules
```

**配置说明**：
- **文件位置**：`.gitlab-ci.yml`（放在代码仓库根目录下）
- **触发条件**：MR、Push 到 main 分支、定时任务
- **执行步骤**：安装依赖 → 下载测试数据 → 运行测试 → 保存结果
- **关键配置**：需要在 GitLab 项目的 Settings → CI/CD → Variables 中设置环境变量

**在 GitLab 中设置环境变量**：
1. 进入 GitLab 项目 → Settings → CI/CD → Variables
2. 点击 "Expand"
3. 添加 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`DD_API_KEY`

---

#### 如何开始使用 CI/CD？

**快速开始步骤**：

1. **创建配置文件**：
   ```bash
   # 在项目根目录
   mkdir -p .github/workflows
   # 创建 regression_test.yml（内容见上面示例）
   ```

2. **提交到 Git**：
   ```bash
   git add .github/workflows/regression_test.yml
   git commit -m "Add CI/CD for regression testing"
   git push
   ```

3. **设置 Secrets/变量**：
   - GitHub：Settings → Secrets and variables → Actions → New repository secret
   - GitLab：Settings → CI/CD → Variables → Add variable

4. **触发 CI/CD**：
   - 创建 PR 或 Push 代码，CI/CD 会自动运行
   - 在 GitHub/GitLab 的 "Actions" 或 "CI/CD" 标签页查看运行结果

**查看运行结果**：
- **GitHub**：仓库页面 → Actions 标签页 → 选择对应的 Workflow → 查看运行日志
- **GitLab**：项目页面 → CI/CD → Pipelines → 查看运行日志

**调试 CI/CD**：
- 查看日志：在 Web UI 中查看每个步骤的输出
- 本地测试：可以在本地 Docker 容器中测试（GitLab 提供 `gitlab-runner exec`）
- 逐步调试：先测试简单步骤，再逐步添加复杂步骤

---

### 3. 测试脚本示例

```python
# golden_set/regression_test.py
import json
import boto3
from pathlib import Path
from typing import Dict, List
import datadog

class GoldenSetTester:
    def __init__(self, golden_set_path: str, storage_type: str = "s3"):
        """初始化 Golden Set 测试器"""
        self.golden_set_path = golden_set_path
        self.storage_type = storage_type
        self.s3_client = boto3.client('s3') if storage_type == "s3" else None
        
        # 加载 Golden Set
        with open(golden_set_path, 'r') as f:
            self.golden_set = json.load(f)
        
        # 初始化 Datadog（用于发送测试结果）
        datadog.initialize(api_key=os.environ.get('DD_API_KEY'))
    
    def download_test_file(self, case: Dict) -> str:
        """下载测试文件到本地"""
        if self.storage_type == "s3":
            # 从 S3 下载
            local_path = f"/tmp/{case['case_id']}_{Path(case['file_url']).name}"
            s3_bucket = self.golden_set['storage_config']['bucket']
            s3_key = case['file_url'].replace(f"s3://{s3_bucket}/", "")
            
            self.s3_client.download_file(s3_bucket, s3_key, local_path)
            return local_path
        else:
            # 本地文件
            return case['file_path']
    
    def run_test_case(self, case: Dict) -> Dict:
        """运行单个测试用例"""
        # 1. 下载测试文件
        file_path = self.download_test_file(case)
        
        # 2. 运行 OCR 和验证
        from kyc_service import KYCService
        kyc_service = KYCService()
        
        result = kyc_service.process_document(file_path)
        
        # 3. 对比结果
        passed = self.compare_results(case['expected_fields'], result['fields'])
        
        return {
            'case_id': case['case_id'],
            'passed': passed,
            'expected': case['expected_fields'],
            'actual': result['fields'],
            'error': None if passed else "Field mismatch"
        }
    
    def compare_results(self, expected: Dict, actual: Dict) -> bool:
        """对比预期结果和实际结果"""
        for key, value in expected.items():
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True
    
    def run_all_tests(self) -> Dict:
        """运行所有测试用例"""
        results = []
        
        for case in self.golden_set['cases']:
            result = self.run_test_case(case)
            results.append(result)
        
        # 统计结果
        total = len(results)
        passed = sum(1 for r in results if r['passed'])
        failed = total - passed
        accuracy = (passed / total) * 100 if total > 0 else 0
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'accuracy': accuracy,
            'results': results
        }
    
    def send_results_to_datadog(self, results: Dict):
        """发送测试结果到 Datadog"""
        # 发送 Metrics
        datadog.statsd.gauge('kyc.regression_test.total', results['total'])
        datadog.statsd.gauge('kyc.regression_test.passed', results['passed'])
        datadog.statsd.gauge('kyc.regression_test.failed', results['failed'])
        datadog.statsd.gauge('kyc.regression_test.accuracy', results['accuracy'])
        
        # 发送 Logs（失败用例）
        for result in results['results']:
            if not result['passed']:
                datadog.api.Event.create(
                    title=f"Regression Test Failed: {result['case_id']}",
                    text=f"Expected: {result['expected']}, Actual: {result['actual']}",
                    alert_type="error",
                    tags=[
                        f"case_id:{result['case_id']}",
                        "test_type:regression",
                        "service:kyc"
                    ]
                )

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden-set', required=True)
    parser.add_argument('--baseline-version', default='v1.0.0')
    parser.add_argument('--fail-on-regression', action='store_true')
    
    args = parser.parse_args()
    
    # 运行测试
    tester = GoldenSetTester(args.golden_set, storage_type="s3")
    results = tester.run_all_tests()
    
    # 发送结果到 Datadog
    tester.send_results_to_datadog(results)
    
    # 保存结果
    with open('regression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 如果启用 fail-on-regression，失败则退出
    if args.fail_on_regression and results['failed'] > 0:
        print(f"❌ Regression test failed: {results['failed']}/{results['total']} cases failed")
        exit(1)
    else:
        print(f"✅ Regression test passed: {results['passed']}/{results['total']} cases passed")
```

---

### 4. 定期回归测试（Cron Job）

```bash
# crontab -e
# 每天午夜运行回归测试
0 0 * * * cd /path/to/project && python golden_set/regression_test.py --golden-set golden_set/golden_set.json >> /var/log/regression_test.log 2>&1
```

或使用 Python 定时任务：

```python
# scripts/scheduled_regression_test.py
import schedule
import time
from golden_set.regression_test import GoldenSetTester

def run_regression_test():
    """运行回归测试"""
    tester = GoldenSetTester("golden_set/golden_set.json", storage_type="s3")
    results = tester.run_all_tests()
    tester.send_results_to_datadog(results)
    
    print(f"Regression test completed: {results['passed']}/{results['total']} passed")

# 每天午夜运行
schedule.every().day.at("00:00").do(run_regression_test)

# 每周一运行（更详细的报告）
schedule.every().monday.at("09:00").do(run_regression_test)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 每小时检查一次
```

---

## 📋 总结

### 存储方案选择

| 团队规模 | 推荐方案 | 存储位置 |
|---------|---------|---------|
| **小型团队**（<10 人） | 方案 A | Git 仓库（全部） |
| **中大型团队**（>10 人） | 方案 B | Git 仓库（代码）+ S3（数据） |
| **企业级**（>50 人） | 方案 C | Git + S3 + 数据库 + Datadog |

### 使用方式

1. **CI/CD 集成**：每次 PR/Commit 自动运行
2. **定期回归测试**：每天/每周自动运行
3. **发布前检查**：手动触发，确保发布安全
4. **结果监控**：发送到 Datadog，设置告警

### 关键要点

- ✅ **不要只用本地存储**：无法团队协作
- ✅ **不要用 Datadog 存储数据**：Datadog 是监控平台，不是存储平台
- ✅ **推荐 Git + S3 组合**：代码用 Git，数据用 S3
- ✅ **CI/CD 集成**：自动化回归测试
- ✅ **结果监控**：发送到 Datadog 进行监控和告警

---

**下一步**：
- 查看 [Release Gate 设计详解](./KYC_Day03_A1_B2_Release_Gate设计详解.md)
- 查看 [Before/After 对比流程详解](./KYC_Day03_A1_B3_Before_After对比流程详解.md)
