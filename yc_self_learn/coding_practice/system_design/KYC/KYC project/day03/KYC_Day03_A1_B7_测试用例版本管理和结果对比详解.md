# Day 3_A1_B7：测试用例版本管理和结果对比详解

**优先级**：🟡 P1 - 重要但不紧急  
**目的**：管理 Golden Set 版本变更，对比不同版本的测试结果，支持版本回滚

---

## 🎯 为什么需要版本管理？

### 问题场景

**场景 1：Golden Set 变更追踪**
- Golden Set 会不断更新（添加新用例、删除重复用例、修改用例）
- 需要知道：什么时候添加了哪些用例？为什么删除某个用例？
- 需要能够回滚到之前的版本

**场景 2：测试结果对比**
- 不同版本的 Golden Set 运行测试，结果不同
- 需要对比：v1.0.0 vs v1.1.0 的测试结果，看是否有退化
- 需要知道：哪些用例在新版本中失败了？哪些用例在新版本中通过了？

**场景 3：模型版本对比**
- 同一个 Golden Set，不同模型版本（v1.0 vs v2.0）的测试结果不同
- 需要对比：模型 v1.0 vs v2.0 的性能差异
- 需要知道：新模型是否比旧模型更好？

---

## 🏗️ 大型 Codebase 版本管理策略

### 1. 多维度版本管理架构（完整版）

**大型 AI/ML 项目的版本管理架构图**：

```mermaid
graph TB
    subgraph "大型 Codebase 版本管理架构"
        subgraph Layer1["第 1 层：代码版本管理（Code Versioning）"]
            Git["Git<br/>📍 GitHub / GitLab / Bitbucket"]
            Docker["Docker Registry<br/>📍 Docker Hub / ECR / GCR"]
            API["API Gateway<br/>📍 Kong / AWS API Gateway"]
        end
        
        subgraph Layer2["第 2 层：模型版本管理（Model Versioning）"]
            MLflow["MLflow<br/>📍 Model Registry"]
            WnB["Weights & Biases<br/>📍 W&B Model Registry"]
            S3["对象存储<br/>📍 AWS S3 / GCS / Azure Blob"]
        end
        
        subgraph Layer3["第 3 层：配置版本管理（Config Versioning）"]
            Config["配置文件<br/>📍 YAML / JSON / TOML"]
            Prompt["Prompt 模板<br/>📍 Jinja2 / Handlebars"]
            Env["环境变量<br/>📍 .env / Vault / Consul"]
        end
        
        subgraph Layer4["第 4 层：数据库版本管理（Database Versioning）"]
            Alembic["Alembic<br/>📍 Python/SQLAlchemy"]
            Flyway["Flyway<br/>📍 Java/跨语言"]
            Django["Django Migrations<br/>📍 Django 框架"]
            DB["数据库<br/>📍 PostgreSQL / MySQL / MongoDB"]
        end
        
        subgraph Layer5["第 5 层：数据版本管理（Data Versioning）"]
            GoldenSet["Golden Set<br/>📍 JSON / Parquet"]
            DVC["DVC<br/>📍 Data Version Control"]
            DataStore["数据存储<br/>📍 S3 / GCS / HDFS"]
        end
        
        subgraph Layer6["第 6 层：依赖版本管理（Dependency Versioning）"]
            Requirements["requirements.txt<br/>📍 pip / conda"]
            Dockerfile["Dockerfile<br/>📍 Docker"]
            K8s["Kubernetes<br/>📍 Helm Charts"]
        end
        
        subgraph Layer7["第 7 层：统一版本管理（Unified Versioning）"]
            VersionManager["版本管理器<br/>📍 UnifiedVersionManager"]
            Snapshot["版本快照<br/>📍 Version Snapshot"]
            Registry["版本注册表<br/>📍 Database / Redis"]
        end
    end
    
    Layer1 --> Layer7
    Layer2 --> Layer7
    Layer3 --> Layer7
    Layer4 --> Layer7
    Layer5 --> Layer7
    Layer6 --> Layer7
    
    Git --> Docker
    MLflow --> S3
    Alembic --> DB
    GoldenSet --> DataStore
    Requirements --> Dockerfile
    
    style Layer1 fill:#e1f5ff
    style Layer2 fill:#fff4e1
    style Layer3 fill:#f3e5f5
    style Layer4 fill:#e8f5e9
    style Layer5 fill:#fff9c4
    style Layer6 fill:#fce4ec
    style Layer7 fill:#ffebee
```

**架构说明**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    大型 Codebase 版本管理架构（详细版）                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 1 层：代码版本管理（Code Versioning）                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Git        │    │   Docker     │    │   API        │                  │
│  │              │    │   Registry   │    │   Gateway    │                  │
│  │  GitHub      │    │  Docker Hub  │    │   Kong       │                  │
│  │  GitLab      │    │  ECR/GCR     │    │  AWS API GW  │                  │
│  │  Bitbucket   │    │  Harbor      │    │  Azure API   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  功能：源代码版本控制、Docker 镜像版本、API 版本管理                        │
│  工具：Git Tags/Branches、Docker Tags、API Versioning                      │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C1_第1层_代码版本管理详解.md](./KYC_Day03_A1_B7_C1_第1层_代码版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 2 层：模型版本管理（Model Versioning）                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   MLflow     │    │   W&B        │    │   对象存储    │                  │
│  │              │    │              │    │              │                  │
│  │  Model       │    │  Model       │    │  AWS S3      │                  │
│  │  Registry    │    │  Registry    │    │  GCS         │                  │
│  │              │    │              │    │  Azure Blob  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  功能：模型文件版本、模型元数据版本、训练参数版本                           │
│  工具：MLflow、Weights & Biases、自定义 Model Registry                      │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C2_第2层_模型版本管理详解.md](./KYC_Day03_A1_B7_C2_第2层_模型版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 3 层：配置版本管理（Config Versioning）                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  配置文件     │    │  Prompt 模板  │    │  环境变量     │                  │
│  │              │    │              │    │              │                  │
│  │  YAML/JSON   │    │  Jinja2      │    │  .env        │                  │
│  │  TOML        │    │  Handlebars  │    │  Vault       │                  │
│  │  Properties  │    │  Template    │    │  Consul      │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  功能：配置文件版本、Prompt 模板版本、环境变量版本                          │
│  工具：Git 版本控制、配置管理工具（Vault、Consul）                         │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C3_第3层_配置版本管理详解.md](./KYC_Day03_A1_B7_C3_第3层_配置版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 4 层：数据库版本管理（Database Versioning）                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Alembic    │    │   Flyway     │    │   Django     │                  │
│  │              │    │              │    │              │                  │
│  │  Python      │    │  Java/跨语言  │    │  Django      │                  │
│  │  SQLAlchemy  │    │  通用推荐     │    │  Migrations  │                  │
│  │              │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│           ↓                  ↓                  ↓                           │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │              数据库（Database）                       │                  │
│  │                                                      │                  │
│  │  PostgreSQL  │  MySQL  │  MongoDB  │  Redis      │                  │
│  │  Oracle      │  SQLite  │  DynamoDB │  Cassandra  │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                                                                             │
│  功能：数据库 Schema 版本、迁移脚本版本、数据版本                           │
│  工具：Alembic（Python）、Flyway（跨语言）、Django Migrations              │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C4_第4层_数据库版本管理详解.md](./KYC_Day03_A1_B7_C4_第4层_数据库版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 5 层：数据版本管理（Data Versioning）                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ Golden Set   │    │     DVC      │    │   数据存储    │                  │
│  │              │    │              │    │              │                  │
│  │  JSON/Parquet│    │  Data        │    │  AWS S3      │                  │
│  │  版本化       │    │  Version     │    │  GCS         │                  │
│  │              │    │  Control     │    │  HDFS        │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  功能：Golden Set 版本、训练数据版本、测试数据版本                         │
│  工具：Git + JSON、DVC（Data Version Control）、对象存储版本化              │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C5_第5层_数据版本管理详解.md](./KYC_Day03_A1_B7_C5_第5层_数据版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 6 层：依赖版本管理（Dependency Versioning）                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ requirements │    │  Dockerfile  │    │ Kubernetes   │                  │
│  │              │    │              │    │              │                  │
│  │  pip/conda   │    │  Docker       │    │  Helm Charts │                  │
│  │  Poetry      │    │  BuildKit    │    │  Kustomize   │                  │
│  │  Pipenv      │    │  Multi-stage │    │  Operators   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  功能：Python 包版本、Docker 镜像版本、K8s 部署版本                        │
│  工具：requirements.txt、Dockerfile、Helm Charts                          │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C6_第6层_依赖版本管理详解.md](./KYC_Day03_A1_B7_C6_第6层_依赖版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第 7 层：统一版本管理（Unified Versioning）                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │           版本管理器（Version Manager）                        │          │
│  │                                                              │          │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │          │
│  │  │ 版本快照     │  │ 版本注册表    │  │ 版本关联      │      │          │
│  │  │              │  │              │  │              │      │          │
│  │  │ Version      │  │ Database     │  │ Component    │      │          │
│  │  │ Snapshot     │  │ Registry     │  │ Linking      │      │          │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │          │
│  │                                                              │          │
│  │  功能：统一版本号、版本快照、版本关联、版本回滚              │          │
│  │  工具：自定义 Version Manager、Database、Redis              │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  📖 详细文档：                                                              │
│     [KYC_Day03_A1_B7_C7_第7层_统一版本管理详解.md](./KYC_Day03_A1_B7_C7_第7层_统一版本管理详解.md) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. 各层工具选择矩阵

| 层级 | 功能 | Python 项目推荐 | Java 项目推荐 | 跨语言项目推荐 | 数据库选择 |
|------|------|----------------|--------------|--------------|-----------|
| **第 1 层** | 代码版本 | Git (GitHub/GitLab) | Git (GitHub/GitLab) | Git (GitHub/GitLab) | - |
| **第 1 层** | Docker 镜像 | Docker Hub / ECR | Docker Hub / ECR | Docker Hub / ECR | - |
| **第 2 层** | 模型版本 | MLflow / W&B | MLflow | MLflow | - |
| **第 2 层** | 模型存储 | AWS S3 / GCS | AWS S3 / GCS | AWS S3 / GCS | - |
| **第 3 层** | 配置文件 | YAML + Git | YAML + Git | YAML + Git | - |
| **第 3 层** | Prompt 模板 | Jinja2 | Handlebars | Jinja2 | - |
| **第 4 层** | 数据库版本 | **Alembic** | **Flyway** | **Flyway** | **PostgreSQL** / **MySQL** |
| **第 4 层** | 数据库 | PostgreSQL / MySQL | PostgreSQL / MySQL | PostgreSQL / MySQL | PostgreSQL / MySQL |
| **第 5 层** | 数据版本 | Git + JSON / DVC | Git + JSON | Git + JSON | - |
| **第 5 层** | 数据存储 | AWS S3 / GCS | AWS S3 / GCS | AWS S3 / GCS | - |
| **第 6 层** | 依赖版本 | requirements.txt | Maven / Gradle | requirements.txt | - |
| **第 6 层** | 容器化 | Docker | Docker | Docker | - |
| **第 7 层** | 统一管理 | 自定义 Manager | 自定义 Manager | 自定义 Manager | PostgreSQL / Redis |

---

### 3. KYC 项目推荐技术栈

**基于 Python 的 KYC 项目推荐**：

```
┌─────────────────────────────────────────────────────────────┐
│              KYC 项目版本管理技术栈（推荐）                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第 1 层：代码版本                                          │
│    ✅ Git (GitHub/GitLab)                                   │
│    ✅ Docker Hub / AWS ECR                                  │
│                                                             │
│  第 2 层：模型版本                                          │
│    ✅ MLflow (Model Registry)                              │
│    ✅ AWS S3 (模型文件存储)                                 │
│                                                             │
│  第 3 层：配置版本                                          │
│    ✅ YAML 配置文件 + Git                                   │
│    ✅ Jinja2 (Prompt 模板)                                 │
│                                                             │
│  第 4 层：数据库版本                                        │
│    ✅ Alembic (数据库迁移)                                  │
│    ✅ PostgreSQL (数据库)                                  │
│                                                             │
│  第 5 层：数据版本                                          │
│    ✅ Git + JSON (Golden Set)                              │
│    ✅ AWS S3 (测试数据存储)                                 │
│                                                             │
│  第 6 层：依赖版本                                         │
│    ✅ requirements.txt (Python 包)                         │
│    ✅ Dockerfile (容器化)                                  │
│                                                             │
│  第 7 层：统一版本                                         │
│    ✅ UnifiedVersionManager (自定义)                       │
│    ✅ PostgreSQL (版本注册表)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 版本关联策略（Version Linking）

**核心问题**：**如何确保代码、模型、配置、数据版本的一致性？**

**策略 1：统一版本号（Unified Versioning）**

```python
# 统一版本号：所有组件使用相同的版本号
RELEASE_VERSION = "v1.2.3"

# 版本关联配置
version_config = {
    "release_version": "v1.2.3",
    "code_version": "v1.2.3",           # Git Tag
    "model_version": "v1.2.3",          # Model Registry
    "config_version": "v1.2.3",          # Config File
    "prompt_version": "v1.2.3",          # Prompt Template
    "golden_set_version": "v1.2.3",      # Golden Set
    "data_version": "v1.2.3",            # Training Data
    "docker_image": "kyc-service:v1.2.3" # Docker Image
}
```

**策略 2：版本快照（Version Snapshot）**

```python
# 版本快照：记录某个时间点的所有组件版本
version_snapshot = {
    "snapshot_id": "snapshot_20250119_100000",
    "release_version": "v1.2.3",
    "timestamp": "2025-01-19T10:00:00Z",
    "components": {
        "code": {
            "git_commit": "abc123def456",
            "git_tag": "v1.2.3",
            "branch": "main"
        },
        "model": {
            "model_id": "model_v1.2.3",
            "model_path": "s3://models/v1.2.3/model.pkl",
            "training_config": "config_v1.2.3.yaml"
        },
        "config": {
            "config_file": "configs/v1.2.3/config.yaml",
            "prompt_template": "prompts/v1.2.3/prompt.jinja"
        },
        "data": {
            "golden_set": "golden_set_v1.2.3.json",
            "training_data": "data/v1.2.3/train.parquet"
        },
        "dependencies": {
            "requirements": "requirements_v1.2.3.txt",
            "docker_image": "kyc-service:v1.2.3"
        }
    }
}
```

---

### 3. 各维度版本管理详解

#### 3.1 代码版本管理（Git）

**策略**：
- ✅ **Git Tags**：每个发布版本打 Tag（`v1.2.3`）
- ✅ **Git Branches**：主分支（`main`）、开发分支（`develop`）、功能分支（`feature/*`）
- ✅ **Git Commits**：语义化提交信息（`feat:`, `fix:`, `refactor:`）

**实现**：

```bash
# 发布新版本
git tag -a v1.2.3 -m "Release v1.2.3: Add new validation rules"
git push origin v1.2.3

# 查看版本历史
git tag -l "v*"
git log --oneline v1.2.2..v1.2.3
```

**版本文件**：

```python
# version.py（代码中定义版本）
__version__ = "1.2.3"

# setup.py（Python 包版本）
setup(
    name="kyc-service",
    version="1.2.3",
    ...
)
```

---

#### 3.2 模型版本管理（Model Registry）

**工具选择**：
- ✅ **MLflow**：开源模型注册表（推荐）
- ✅ **Weights & Biases**：商业模型管理平台
- ✅ **自定义方案**：S3 + 数据库

**MLflow 实现**：

```python
import mlflow
import mlflow.sklearn

# 训练和注册模型
with mlflow.start_run():
    # 训练模型
    model = train_model(training_data)
    
    # 记录参数和指标
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", 0.95)
    
    # 注册模型
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="kyc_model"
    )
    
    # 标记版本
    mlflow.set_tag("version", "v1.2.3")

# 加载指定版本的模型
model = mlflow.sklearn.load_model(
    "models:/kyc_model/v1.2.3"
)
```

**自定义实现（S3 + 数据库）**：

```python
# 模型注册表（数据库）
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,  -- S3 path
    training_config JSONB,
    metrics JSONB,
    created_at TIMESTAMP NOT NULL,
    UNIQUE(model_name, version)
);

# 注册模型
def register_model(model_name: str, version: str, model_path: str):
    db.execute("""
        INSERT INTO model_registry (model_name, version, model_path, created_at)
        VALUES (%s, %s, %s, NOW())
    """, (model_name, version, model_path))
```

---

#### 3.3 配置版本管理（Configuration）

**策略**：
- ✅ **配置文件版本化**：`configs/v1.2.3/config.yaml`
- ✅ **Prompt 模板版本化**：`prompts/v1.2.3/prompt.jinja`
- ✅ **环境变量版本化**：`.env.v1.2.3`

**实现**：

```python
# configs/v1.2.3/config.yaml
version: "v1.2.3"
model:
  model_version: "v1.2.3"
  model_path: "s3://models/v1.2.3/model.pkl"
prompt:
  prompt_version: "v1.2.3"
  prompt_path: "prompts/v1.2.3/prompt.jinja"
validation:
  schema_version: "v1.2.3"
  rules_version: "v1.2.3"

# 加载配置
def load_config(version: str):
    config_path = f"configs/{version}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# 使用配置
config = load_config("v1.2.3")
model = load_model(config["model"]["model_path"])
prompt = load_prompt(config["prompt"]["prompt_path"])
```

**Prompt 版本管理**：

```python
# prompts/v1.2.3/prompt.jinja
{% set prompt_version = "v1.2.3" %}
You are a KYC expert. Extract the following fields from the document:
- Name: {{ name }}
- ID Number: {{ id_number }}
...

# 加载 Prompt
def load_prompt(version: str):
    prompt_path = f"prompts/{version}/prompt.jinja"
    with open(prompt_path, "r") as f:
        template = jinja2.Template(f.read())
    return template
```

---

#### 3.4 数据库 Schema 版本管理（Database Schema Versioning）

**核心问题**：**如何管理数据库表结构的变更？如何追踪数据库 schema 的版本？**

**业界常见工具**：

| 工具 | 语言生态 | 数据库支持 | 特点 | 适用场景 |
|------|---------|-----------|------|---------|
| **Flyway** | Java | ✅ 广泛支持（PostgreSQL, MySQL, Oracle, SQL Server 等） | ✅ 简单易用，SQL 文件版本化 | ✅ **通用推荐**（跨语言） |
| **Liquibase** | Java | ✅ 广泛支持 | ✅ XML/YAML/SQL 格式，功能强大 | ✅ 复杂变更场景 |
| **Alembic** | Python | ✅ PostgreSQL, MySQL, SQLite 等 | ✅ SQLAlchemy 集成，Python 原生 | ✅ Python 项目 |
| **Django Migrations** | Python | ✅ Django 支持的数据库 | ✅ Django 框架自带 | ✅ Django 项目 |
| **Rails Migrations** | Ruby | ✅ Rails 支持的数据库 | ✅ Rails 框架自带 | ✅ Rails 项目 |
| **Sequelize Migrations** | Node.js | ✅ 多种数据库 | ✅ Sequelize ORM 集成 | ✅ Node.js 项目 |
| **Prisma Migrate** | TypeScript | ✅ PostgreSQL, MySQL, SQLite 等 | ✅ Prisma 集成，类型安全 | ✅ TypeScript/Prisma 项目 |

---

##### 工具 1：Flyway（推荐 - 通用）

**特点**：
- ✅ **简单易用**：SQL 文件版本化，按文件名顺序执行
- ✅ **跨语言**：不依赖特定语言框架
- ✅ **广泛支持**：支持主流数据库
- ✅ **版本追踪**：自动记录已执行的迁移脚本

**实现**：

```bash
# 1. 安装 Flyway
# Maven
<dependency>
    <groupId>org.flywaydb</groupId>
    <artifactId>flyway-core</artifactId>
    <version>9.22.0</version>
</dependency>

# 或使用 Flyway CLI
# https://flywaydb.org/documentation/usage/commandline/
```

**目录结构**：

```
project/
├── src/
│   └── main/
│       └── resources/
│           └── db/
│               └── migration/
│                   ├── V1__Create_test_cases_table.sql
│                   ├── V2__Create_test_results_table.sql
│                   ├── V3__Add_version_column.sql
│                   └── V4__Create_version_history_table.sql
```

**迁移脚本示例**：

```sql
-- V1__Create_test_cases_table.sql
CREATE TABLE test_cases (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    file_path VARCHAR(500),
    expected_fields JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_test_cases_version ON test_cases(version);

-- V2__Create_test_results_table.sql
CREATE TABLE test_results (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    golden_set_version VARCHAR(50) NOT NULL,
    summary JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- V3__Add_version_column.sql
ALTER TABLE test_cases 
ADD COLUMN priority INTEGER DEFAULT 0;

-- V4__Create_version_history_table.sql
CREATE TABLE version_history (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    version_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    description TEXT
);
```

**Flyway 配置**：

```properties
# flyway.conf
flyway.url=jdbc:postgresql://localhost:5432/kyc_db
flyway.user=kyc_user
flyway.password=kyc_password
flyway.locations=filesystem:src/main/resources/db/migration
flyway.baselineOnMigrate=true
```

**使用方式**：

```bash
# 运行迁移
flyway migrate

# 查看迁移历史
flyway info

# 回滚（需要 Flyway Teams）
flyway undo
```

**Python 集成**：

```python
# 使用 flyway-python 或 subprocess 调用
import subprocess

def run_flyway_migrate():
    """运行 Flyway 迁移"""
    result = subprocess.run(
        ["flyway", "migrate"],
        cwd="project_root",
        capture_output=True,
        text=True
    )
    return result.returncode == 0

# 使用
run_flyway_migrate()
```

---

##### 工具 2：Alembic（Python/SQLAlchemy 推荐）

**特点**：
- ✅ **Python 原生**：与 SQLAlchemy 深度集成
- ✅ **自动生成**：根据 SQLAlchemy 模型自动生成迁移脚本
- ✅ **版本控制**：自动管理迁移版本

**实现**：

```bash
# 1. 安装
pip install alembic sqlalchemy

# 2. 初始化
alembic init alembic
```

**目录结构**：

```
project/
├── alembic/
│   ├── versions/
│   │   ├── 001_create_test_cases_table.py
│   │   ├── 002_create_test_results_table.py
│   │   └── 003_add_version_column.py
│   ├── env.py
│   └── alembic.ini
├── models.py
└── app.py
```

**Alembic 配置**：

```python
# alembic/env.py
from sqlalchemy import engine_from_config
from models import Base

target_metadata = Base.metadata

# alembic.ini
[alembic]
sqlalchemy.url = postgresql://user:password@localhost/kyc_db
```

**迁移脚本示例**：

```python
# alembic/versions/001_create_test_cases_table.py
"""Create test_cases table

Revision ID: 001
Revises: 
Create Date: 2025-01-19 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.create_table(
        'test_cases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('file_path', sa.String(500)),
        sa.Column('expected_fields', postgresql.JSONB()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_test_cases_version', 'test_cases', ['version'])

def downgrade():
    op.drop_index('idx_test_cases_version', table_name='test_cases')
    op.drop_table('test_cases')
```

**使用方式**：

```bash
# 自动生成迁移脚本（根据模型变更）
alembic revision --autogenerate -m "Add version column"

# 运行迁移
alembic upgrade head

# 回滚到上一个版本
alembic downgrade -1

# 查看迁移历史
alembic history

# 查看当前版本
alembic current
```

**Python 代码集成**：

```python
# app.py
from alembic import command
from alembic.config import Config

def run_migrations():
    """运行数据库迁移"""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

# 使用
run_migrations()
```

---

##### 工具 3：Liquibase（功能强大）

**特点**：
- ✅ **多种格式**：支持 XML、YAML、JSON、SQL
- ✅ **复杂变更**：支持数据迁移、回滚
- ✅ **变更集（Changeset）**：细粒度控制

**实现**：

```bash
# 1. 安装
# Maven
<dependency>
    <groupId>org.liquibase</groupId>
    <artifactId>liquibase-core</artifactId>
    <version>4.24.0</version>
</dependency>

# 或使用 Liquibase CLI
# https://docs.liquibase.com/start/install/home.html
```

**目录结构**：

```
project/
├── src/
│   └── main/
│       └── resources/
│           └── db/
│               └── changelog/
│                   ├── db.changelog-master.xml
│                   ├── changesets/
│                   │   ├── 001-create-tables.xml
│                   │   └── 002-add-columns.xml
```

**Changelog 文件**：

```xml
<!-- db.changelog-master.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
    http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.24.xsd">
    
    <include file="changesets/001-create-tables.xml"/>
    <include file="changesets/002-add-columns.xml"/>
</databaseChangeLog>
```

**Changeset 文件**：

```xml
<!-- changesets/001-create-tables.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
    http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.24.xsd">
    
    <changeSet id="001" author="developer">
        <createTable tableName="test_cases">
            <column name="id" type="SERIAL">
                <constraints primaryKey="true"/>
            </column>
            <column name="case_id" type="VARCHAR(100)">
                <constraints nullable="false"/>
            </column>
            <column name="version" type="VARCHAR(50)">
                <constraints nullable="false"/>
            </column>
            <column name="file_path" type="VARCHAR(500)"/>
            <column name="expected_fields" type="JSONB"/>
            <column name="created_at" type="TIMESTAMP" defaultValueComputed="NOW()">
                <constraints nullable="false"/>
            </column>
        </createTable>
        
        <createIndex indexName="idx_test_cases_version" tableName="test_cases">
            <column name="version"/>
        </createIndex>
    </changeSet>
</databaseChangeLog>
```

**使用方式**：

```bash
# 运行迁移
liquibase update

# 查看迁移历史
liquibase history

# 回滚
liquibase rollback-count 1
```

---

##### 工具 4：Django Migrations（Django 项目）

**特点**：
- ✅ **Django 原生**：Django 框架自带
- ✅ **自动生成**：根据 Model 自动生成迁移脚本
- ✅ **Python 代码**：迁移脚本是 Python 代码

**实现**：

```python
# models.py
from django.db import models

class TestCase(models.Model):
    case_id = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    file_path = models.CharField(max_length=500, null=True)
    expected_fields = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'test_cases'
        indexes = [
            models.Index(fields=['version']),
        ]
```

**使用方式**：

```bash
# 生成迁移脚本
python manage.py makemigrations

# 运行迁移
python manage.py migrate

# 查看迁移历史
python manage.py showmigrations

# 回滚
python manage.py migrate app_name migration_number
```

---

### 3.5 数据库版本管理最佳实践

#### 实践 1：版本命名规范

**Flyway 命名规范**：
```
V{version}__{description}.sql

示例：
V1__Create_test_cases_table.sql
V2__Add_version_column.sql
V3__Create_indexes.sql
```

**Alembic 命名规范**：
```
{revision_id}_{description}.py

示例：
001_create_test_cases_table.py
002_add_version_column.py
```

#### 实践 2：迁移脚本原则

**DO（应该做）**：
- ✅ **幂等性**：迁移脚本可以安全地多次执行
- ✅ **可回滚**：每个迁移都应该有对应的回滚脚本
- ✅ **小步快跑**：每次迁移只做一件事
- ✅ **测试先行**：先在测试环境验证迁移脚本

**DON'T（不应该做）**：
- ❌ **修改已执行的迁移**：已执行的迁移脚本不要修改
- ❌ **删除迁移历史**：不要删除数据库中的迁移历史记录
- ❌ **大事务**：避免在迁移中使用大事务（可能导致锁表）

#### 实践 3：CI/CD 集成

```yaml
# .github/workflows/migrate.yml
name: Database Migration

on:
  push:
    branches: [main]

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Flyway Migration
        run: |
          flyway migrate
        env:
          FLYWAY_URL: ${{ secrets.DATABASE_URL }}
          FLYWAY_USER: ${{ secrets.DATABASE_USER }}
          FLYWAY_PASSWORD: ${{ secrets.DATABASE_PASSWORD }}
```

---

### 4. KYC 项目推荐方案

**推荐工具选择**：

| 项目类型 | 推荐工具 | 原因 |
|---------|---------|------|
| **Python 项目（SQLAlchemy）** | ✅ **Alembic** | Python 原生，SQLAlchemy 集成 |
| **Python 项目（Django）** | ✅ **Django Migrations** | Django 框架自带 |
| **Java 项目** | ✅ **Flyway** | 简单易用，广泛支持 |
| **跨语言项目** | ✅ **Flyway** | 不依赖特定语言框架 |
| **复杂变更场景** | ✅ **Liquibase** | 功能强大，支持复杂变更 |

**KYC 项目实现示例（Python + Alembic）**：

```python
# models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TestCase(Base):
    __tablename__ = 'test_cases'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    file_path = Column(String(500))
    expected_fields = Column(JSON)
    created_at = Column(DateTime, nullable=False)

# 生成迁移脚本
# alembic revision --autogenerate -m "Create test_cases table"

# 运行迁移
# alembic upgrade head
```

---

#### 3.4 数据版本管理（Data Versioning）

**策略**：
- ✅ **Golden Set 版本化**：`golden_set_v1.2.3.json`
- ✅ **训练数据版本化**：`data/v1.2.3/train.parquet`
- ✅ **测试数据版本化**：`test_data/v1.2.3/`

**实现**：

```python
# 数据版本管理
class DataVersionManager:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
    
    def save_golden_set(self, golden_set: list, version: str):
        """保存 Golden Set"""
        file_path = self.base_path / f"golden_set_{version}.json"
        with open(file_path, "w") as f:
            json.dump(golden_set, f, indent=2)
    
    def load_golden_set(self, version: str):
        """加载 Golden Set"""
        file_path = self.base_path / f"golden_set_{version}.json"
        with open(file_path, "r") as f:
            return json.load(f)
```

**DVC（Data Version Control）**：

```bash
# 使用 DVC 管理数据版本
dvc add data/train.parquet
dvc push

# 版本化数据
git add data/train.parquet.dvc
git commit -m "Add training data v1.2.3"
git tag v1.2.3
```

---

#### 3.5 依赖版本管理（Dependencies）

**策略**：
- ✅ **requirements.txt 版本化**：`requirements_v1.2.3.txt`
- ✅ **Docker 镜像版本化**：`kyc-service:v1.2.3`
- ✅ **锁定依赖版本**：`requirements.lock`

**实现**：

```python
# requirements_v1.2.3.txt
# KYC Service Dependencies v1.2.3
torch==2.0.0
transformers==4.30.0
pydantic==1.10.0
fastapi==0.95.0
...

# Dockerfile
FROM python:3.9-slim

# 复制依赖文件
COPY requirements_v1.2.3.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 构建镜像
# docker build -t kyc-service:v1.2.3 .
```

---

### 4. 统一版本管理实现

#### 4.1 版本管理器（Version Manager）

```python
class UnifiedVersionManager:
    """统一版本管理器"""
    
    def __init__(self):
        self.version_registry = {}
        self.snapshots = []
    
    def create_release(
        self,
        version: str,
        code_commit: str,
        model_version: str,
        config_version: str,
        data_version: str,
        description: str = ""
    ):
        """创建发布版本"""
        release = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "code": {
                    "git_commit": code_commit,
                    "git_tag": version
                },
                "model": {
                    "model_version": model_version,
                    "model_path": f"s3://models/{model_version}/model.pkl"
                },
                "config": {
                    "config_version": config_version,
                    "config_path": f"configs/{config_version}/config.yaml"
                },
                "data": {
                    "golden_set_version": data_version,
                    "golden_set_path": f"data/golden_set_{data_version}.json"
                }
            },
            "description": description
        }
        
        self.version_registry[version] = release
        self.snapshots.append(release)
        
        return release
    
    def get_release(self, version: str):
        """获取发布版本信息"""
        return self.version_registry.get(version)
    
    def list_releases(self):
        """列出所有发布版本"""
        return sorted(self.version_registry.keys(), reverse=True)
    
    def deploy_release(self, version: str):
        """部署指定版本"""
        release = self.get_release(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        # 1. 加载配置
        config = load_config(release["components"]["config"]["config_version"])
        
        # 2. 加载模型
        model = load_model(release["components"]["model"]["model_version"])
        
        # 3. 加载数据
        golden_set = load_golden_set(release["components"]["data"]["golden_set_version"])
        
        # 4. 部署服务
        deploy_service(version, config, model, golden_set)
        
        return True

# 使用示例
version_manager = UnifiedVersionManager()

# 创建发布版本
release = version_manager.create_release(
    version="v1.2.3",
    code_commit="abc123def456",
    model_version="v1.2.3",
    config_version="v1.2.3",
    data_version="v1.2.3",
    description="Add new validation rules"
)

# 部署版本
version_manager.deploy_release("v1.2.3")
```

---

#### 4.2 版本关联数据库设计

```sql
-- 发布版本表
CREATE TABLE releases (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL,
    created_by VARCHAR(100)
);

-- 组件版本关联表
CREATE TABLE component_versions (
    id SERIAL PRIMARY KEY,
    release_version VARCHAR(50) NOT NULL,
    component_type VARCHAR(50) NOT NULL,  -- 'code', 'model', 'config', 'data'
    component_version VARCHAR(50) NOT NULL,
    component_path VARCHAR(500),
    metadata JSONB,
    FOREIGN KEY (release_version) REFERENCES releases(version)
);

-- 创建索引
CREATE INDEX idx_component_versions_release ON component_versions(release_version);
CREATE INDEX idx_component_versions_type ON component_versions(component_type);

-- 查询某个发布版本的所有组件
SELECT * FROM component_versions 
WHERE release_version = 'v1.2.3';
```

---

### 5. 版本管理最佳实践

#### 5.1 版本命名规范

**语义化版本（Semantic Versioning）**：
- `MAJOR.MINOR.PATCH`
- `v1.2.3` → Major=1, Minor=2, Patch=3

**版本号规则**：
- **MAJOR**：不兼容的 API 变更、重大功能变更
- **MINOR**：向后兼容的功能新增、模型更新
- **PATCH**：向后兼容的问题修复、配置调整

**示例**：
```
v1.0.0  → 初始版本
v1.1.0  → 添加新功能（向后兼容）
v1.1.1  → 修复 Bug
v2.0.0  → 重大变更（不兼容）
```

---

#### 5.2 版本发布流程

```
1. 开发阶段
   ├── 代码开发（feature branch）
   ├── 模型训练（model training）
   ├── 配置调整（config update）
   └── 数据更新（data update）

2. 测试阶段
   ├── 单元测试（unit tests）
   ├── 集成测试（integration tests）
   └── 回归测试（regression tests）

3. 版本创建
   ├── 打 Git Tag（v1.2.3）
   ├── 注册模型版本（MLflow）
   ├── 保存配置版本（configs/v1.2.3/）
   └── 创建版本快照（version snapshot）

4. 部署阶段
   ├── 构建 Docker 镜像（kyc-service:v1.2.3）
   ├── 部署到测试环境
   ├── 运行回归测试
   └── 部署到生产环境

5. 监控阶段
   ├── 监控指标（metrics）
   ├── 收集反馈（feedback）
   └── 记录问题（issues）
```

---

#### 5.3 版本回滚策略

**回滚场景**：
1. **代码回滚**：Git revert 或 checkout 到旧版本
2. **模型回滚**：加载旧版本的模型
3. **配置回滚**：使用旧版本的配置文件
4. **数据回滚**：使用旧版本的 Golden Set

**统一回滚**：

```python
def rollback_release(target_version: str):
    """回滚到指定版本"""
    release = version_manager.get_release(target_version)
    
    # 1. 回滚代码
    git_checkout(release["components"]["code"]["git_tag"])
    
    # 2. 回滚模型
    model = load_model(release["components"]["model"]["model_version"])
    
    # 3. 回滚配置
    config = load_config(release["components"]["config"]["config_version"])
    
    # 4. 回滚数据
    golden_set = load_golden_set(release["components"]["data"]["golden_set_version"])
    
    # 5. 重新部署
    deploy_service(target_version, config, model, golden_set)
    
    return True
```

---

## 📊 版本管理架构

### 1. 三层版本管理架构（Golden Set 专用）

```
┌─────────────────────────────────────────────────────────┐
│              版本管理三层架构                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Git 版本控制（代码和配置）                          │
│     ├── golden_set_v1.0.0.json                         │
│     ├── golden_set_v1.1.0.json                         │
│     └── regression_test.py                             │
│                                                         │
│  2. 数据库版本管理（元数据和结果）                      │
│     ├── test_cases 表（用例元数据，带版本号）           │
│     ├── test_results 表（测试结果，带版本号）            │
│     └── version_history 表（版本变更历史）              │
│                                                         │
│  3. 对象存储版本管理（测试数据文件）                    │
│     ├── test_data/v1.0.0/                              │
│     ├── test_data/v1.1.0/                              │
│     └── results/v1.0.0/（测试结果文件）                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔢 版本号设计

### 语义化版本（Semantic Versioning）

**格式**：`MAJOR.MINOR.PATCH`

- **MAJOR**：重大变更（如：用例结构改变、删除大量用例）
- **MINOR**：新增用例、修改用例（如：添加 10 个新用例）
- **PATCH**：小修复（如：修复用例描述、修复文件路径）

**示例**：
```
v1.0.0  → 初始版本（100 个用例）
v1.1.0  → 添加 20 个新用例（MINOR）
v1.1.1  → 修复用例描述错误（PATCH）
v1.2.0  → 添加 15 个新用例（MINOR）
v2.0.0  → 重大变更：用例结构改变（MAJOR）
```

---

## 💾 版本存储策略

### 策略 1：保存 Summary（推荐）

**适用场景**：**大多数情况**（节省存储空间）

**保存内容**：
- ✅ **Summary 指标**：总体准确率、Schema Pass Rate、字段级准确率等
- ✅ **用例级结果**：每个用例的通过/失败状态
- ✅ **失败用例详情**：失败用例的 ID、错误信息
- ❌ **不保存**：每个用例的详细输出（除非失败）

**存储结构**：

```python
# 版本结果 Summary（轻量级）
version_result_summary = {
    "version": "v1.1.0",
    "golden_set_version": "v1.1.0",
    "model_version": "v2.0.0",
    "timestamp": "2025-01-19T10:00:00Z",
    "summary": {
        "total_cases": 120,
        "passed_cases": 110,
        "failed_cases": 10,
        "schema_pass_rate": 0.9167,  # 110/120
        "field_accuracy": 0.95,
        "field_consistency": 0.98
    },
    "failed_cases": [
        {
            "case_id": "edge_005",
            "error_type": "SCHEMA_VALIDATION_FAILED",
            "error_message": "Missing field: date_of_birth"
        },
        # ... 其他失败用例
    ],
    "passed_cases": [
        "normal_001",
        "normal_002",
        # ... 只保存 ID，不保存详细输出
    ]
}
```

**优点**：
- ✅ 存储空间小（只保存关键信息）
- ✅ 查询速度快（数据量小）
- ✅ 满足大多数对比需求

**缺点**：
- ⚠️ 无法查看通过用例的详细输出
- ⚠️ 如果需要详细输出，需要重新运行测试

---

### 策略 2：保存详细结果（Full Results）

**适用场景**：**关键版本**（如：每个 MAJOR 版本）

**保存内容**：
- ✅ **Summary 指标**：总体准确率等
- ✅ **用例级详细结果**：每个用例的完整输入和输出
- ✅ **所有用例详情**：包括通过的用例

**存储结构**：

```python
# 版本结果详细（完整）
version_result_full = {
    "version": "v1.1.0",
    "golden_set_version": "v1.1.0",
    "model_version": "v2.0.0",
    "timestamp": "2025-01-19T10:00:00Z",
    "summary": {
        "total_cases": 120,
        "passed_cases": 110,
        "failed_cases": 10,
        "schema_pass_rate": 0.9167,
        "field_accuracy": 0.95
    },
    "case_results": [
        {
            "case_id": "normal_001",
            "status": "passed",
            "expected_fields": {
                "name": "张三",
                "id_number": "110101199001011234"
            },
            "actual_fields": {
                "name": "张三",
                "id_number": "110101199001011234"
            },
            "latency_ms": 1200,
            "cost_tokens": 500
        },
        {
            "case_id": "edge_005",
            "status": "failed",
            "error_type": "SCHEMA_VALIDATION_FAILED",
            "error_message": "Missing field: date_of_birth",
            "expected_fields": {...},
            "actual_fields": {...}
        },
        # ... 所有用例的详细结果
    ]
}
```

**优点**：
- ✅ 完整信息，可以深入分析
- ✅ 不需要重新运行测试就能查看详细结果

**缺点**：
- ⚠️ 存储空间大（每个用例都保存完整输出）
- ⚠️ 查询速度慢（数据量大）

---

### 策略 3：混合策略（推荐）

**适用场景**：**生产环境推荐**（平衡存储和需求）

**策略**：
- ✅ **所有版本**：保存 Summary（轻量级）
- ✅ **关键版本**（MAJOR）：保存详细结果（Full Results）
- ✅ **失败用例**：所有版本都保存详细结果（便于分析）

**存储结构**：

```python
# 混合策略：Summary + 失败用例详情
version_result_hybrid = {
    "version": "v1.1.0",
    "golden_set_version": "v1.1.0",
    "model_version": "v2.0.0",
    "timestamp": "2025-01-19T10:00:00Z",
    "summary": {
        "total_cases": 120,
        "passed_cases": 110,
        "failed_cases": 10,
        "schema_pass_rate": 0.9167,
        "field_accuracy": 0.95
    },
    "passed_cases": [
        "normal_001",
        "normal_002",
        # ... 只保存 ID
    ],
    "failed_cases_detail": [
        {
            "case_id": "edge_005",
            "status": "failed",
            "error_type": "SCHEMA_VALIDATION_FAILED",
            "error_message": "Missing field: date_of_birth",
            "expected_fields": {...},
            "actual_fields": {...},
            "latency_ms": 1500,
            "cost_tokens": 600
        },
        # ... 只保存失败用例的详细结果
    ],
    "is_full_result": False  # 标记是否为完整结果
}
```

**优点**：
- ✅ 平衡存储空间和需求
- ✅ 关键信息（失败用例）都有详细记录
- ✅ 满足大多数对比需求

---

## 🔍 版本对比功能

### 1. Golden Set 版本对比

**对比内容**：
- 用例数量变化（新增、删除、修改）
- 用例内容变化（字段变化、描述变化）
- 用例分类变化（normal/edge/anomaly 分布）

**实现示例**：

```python
def compare_golden_set_versions(version1: str, version2: str) -> dict:
    """对比两个版本的 Golden Set"""
    golden_set_v1 = load_golden_set(version1)
    golden_set_v2 = load_golden_set(version2)
    
    # 提取用例 ID
    cases_v1 = {case["case_id"]: case for case in golden_set_v1}
    cases_v2 = {case["case_id"]: case for case in golden_set_v2}
    
    # 找出新增、删除、修改的用例
    added_cases = [case_id for case_id in cases_v2 if case_id not in cases_v1]
    removed_cases = [case_id for case_id in cases_v1 if case_id not in cases_v2]
    modified_cases = []
    
    for case_id in cases_v1:
        if case_id in cases_v2:
            if cases_v1[case_id] != cases_v2[case_id]:
                modified_cases.append(case_id)
    
    return {
        "version1": version1,
        "version2": version2,
        "summary": {
            "v1_count": len(cases_v1),
            "v2_count": len(cases_v2),
            "added_count": len(added_cases),
            "removed_count": len(removed_cases),
            "modified_count": len(modified_cases)
        },
        "added_cases": added_cases,
        "removed_cases": removed_cases,
        "modified_cases": modified_cases,
        "details": {
            "added": [cases_v2[case_id] for case_id in added_cases],
            "removed": [cases_v1[case_id] for case_id in removed_cases],
            "modified": [
                {
                    "case_id": case_id,
                    "v1": cases_v1[case_id],
                    "v2": cases_v2[case_id]
                }
                for case_id in modified_cases
            ]
        }
    }

# 使用示例
comparison = compare_golden_set_versions("v1.0.0", "v1.1.0")
print(f"新增用例: {comparison['summary']['added_count']}")
print(f"删除用例: {comparison['summary']['removed_count']}")
print(f"修改用例: {comparison['summary']['modified_count']}")
```

---

### 2. 测试结果版本对比（Summary 对比）

**对比内容**：
- Summary 指标对比（准确率、通过率等）
- 失败用例对比（新增失败、修复失败）
- 性能指标对比（延迟、成本）

**实现示例**：

```python
def compare_test_results(result_v1: dict, result_v2: dict) -> dict:
    """对比两个版本的测试结果（Summary 对比）"""
    
    summary_v1 = result_v1["summary"]
    summary_v2 = result_v2["summary"]
    
    failed_v1 = set(result_v1.get("failed_cases", []))
    failed_v2 = set(result_v2.get("failed_cases", []))
    
    # 计算指标变化
    metrics_comparison = {
        "schema_pass_rate": {
            "v1": summary_v1["schema_pass_rate"],
            "v2": summary_v2["schema_pass_rate"],
            "delta": summary_v2["schema_pass_rate"] - summary_v1["schema_pass_rate"],
            "delta_percent": (summary_v2["schema_pass_rate"] - summary_v1["schema_pass_rate"]) * 100
        },
        "field_accuracy": {
            "v1": summary_v1["field_accuracy"],
            "v2": summary_v2["field_accuracy"],
            "delta": summary_v2["field_accuracy"] - summary_v1["field_accuracy"],
            "delta_percent": (summary_v2["field_accuracy"] - summary_v1["field_accuracy"]) * 100
        },
        "total_cases": {
            "v1": summary_v1["total_cases"],
            "v2": summary_v2["total_cases"],
            "delta": summary_v2["total_cases"] - summary_v1["total_cases"]
        }
    }
    
    # 失败用例对比
    new_failures = failed_v2 - failed_v1  # v2 新增的失败用例
    fixed_failures = failed_v1 - failed_v2  # v2 修复的失败用例
    persistent_failures = failed_v1 & failed_v2  # 持续失败的用例
    
    return {
        "version1": result_v1["version"],
        "version2": result_v2["version"],
        "metrics_comparison": metrics_comparison,
        "failure_analysis": {
            "new_failures": list(new_failures),
            "fixed_failures": list(fixed_failures),
            "persistent_failures": list(persistent_failures),
            "new_failures_count": len(new_failures),
            "fixed_failures_count": len(fixed_failures),
            "persistent_failures_count": len(persistent_failures)
        },
        "regression_detected": len(new_failures) > 0,
        "improvement_detected": len(fixed_failures) > 0
    }

# 使用示例
result_v1 = load_test_result("v1.0.0")
result_v2 = load_test_result("v1.1.0")

comparison = compare_test_results(result_v1, result_v2)

print(f"Schema Pass Rate 变化: {comparison['metrics_comparison']['schema_pass_rate']['delta_percent']:.2f}%")
print(f"新增失败用例: {comparison['failure_analysis']['new_failures_count']}")
print(f"修复失败用例: {comparison['failure_analysis']['fixed_failures_count']}")

if comparison["regression_detected"]:
    print("⚠️ 检测到回归！")
if comparison["improvement_detected"]:
    print("✅ 检测到改进！")
```

---

### 3. 模型版本对比（同一 Golden Set）

**对比内容**：
- 同一 Golden Set，不同模型版本的性能对比
- 哪些用例在新模型中通过/失败？

**实现示例**：

```python
def compare_model_versions(
    golden_set_version: str,
    model_v1: str,
    model_v2: str
) -> dict:
    """对比同一 Golden Set 下不同模型版本的性能"""
    
    result_v1 = load_test_result_by_model(golden_set_version, model_v1)
    result_v2 = load_test_result_by_model(golden_set_version, model_v2)
    
    comparison = compare_test_results(result_v1, result_v2)
    
    # 添加模型特定信息
    comparison["golden_set_version"] = golden_set_version
    comparison["model_v1"] = model_v1
    comparison["model_v2"] = model_v2
    
    return comparison

# 使用示例
comparison = compare_model_versions(
    golden_set_version="v1.0.0",
    model_v1="v1.0.0",
    model_v2="v2.0.0"
)

print(f"模型 v1.0.0 → v2.0.0 性能变化:")
print(f"  Schema Pass Rate: {comparison['metrics_comparison']['schema_pass_rate']['delta_percent']:.2f}%")
print(f"  Field Accuracy: {comparison['metrics_comparison']['field_accuracy']['delta_percent']:.2f}%")
```

---

## 📊 版本对比报告

### 报告结构

```python
version_comparison_report = {
    "report_type": "version_comparison",
    "timestamp": "2025-01-19T10:00:00Z",
    "comparison": {
        "version1": "v1.0.0",
        "version2": "v1.1.0",
        "comparison_type": "golden_set"  # or "model" or "both"
    },
    "summary": {
        "metrics_changes": {
            "schema_pass_rate": {"v1": 0.90, "v2": 0.92, "delta": +0.02},
            "field_accuracy": {"v1": 0.95, "v2": 0.96, "delta": +0.01},
            "total_cases": {"v1": 100, "v2": 120, "delta": +20}
        },
        "failure_changes": {
            "new_failures": 2,
            "fixed_failures": 5,
            "persistent_failures": 3
        }
    },
    "details": {
        "new_failures": [
            {"case_id": "edge_010", "error_type": "SCHEMA_VALIDATION_FAILED"}
        ],
        "fixed_failures": [
            {"case_id": "edge_005", "was_error": "SCHEMA_VALIDATION_FAILED", "now": "passed"}
        ],
        "persistent_failures": [
            {"case_id": "anomaly_003", "error_type": "FIELD_EXTRACTION_FAILED"}
        ]
    },
    "recommendation": "✅ 整体性能提升，建议采用 v1.1.0"
}
```

---

## 🔄 版本回滚机制

### 1. Golden Set 版本回滚

**场景**：新版本的 Golden Set 有问题，需要回滚到旧版本

**实现**：

```python
def rollback_golden_set(target_version: str) -> bool:
    """回滚 Golden Set 到指定版本"""
    # 1. 加载目标版本的 Golden Set
    target_golden_set = load_golden_set(target_version)
    
    # 2. 备份当前版本
    current_version = get_current_golden_set_version()
    backup_golden_set(current_version)
    
    # 3. 恢复目标版本
    save_golden_set(target_golden_set, version=target_version)
    set_current_golden_set_version(target_version)
    
    # 4. 记录回滚操作
    log_rollback_operation(
        from_version=current_version,
        to_version=target_version,
        reason="Manual rollback"
    )
    
    return True

# 使用示例
rollback_golden_set("v1.0.0")
```

---

### 2. 测试结果版本查询

**场景**：查看历史版本的测试结果

**实现**：

```python
def get_test_result_by_version(version: str) -> dict:
    """根据版本号获取测试结果"""
    # 从数据库或文件系统加载
    result = load_test_result(version)
    return result

def list_all_versions() -> list:
    """列出所有可用版本"""
    versions = query_all_versions()
    return sorted(versions, reverse=True)  # 最新的在前

# 使用示例
versions = list_all_versions()
print(f"可用版本: {versions}")

for version in versions[:5]:  # 显示最近 5 个版本
    result = get_test_result_by_version(version)
    print(f"{version}: Schema Pass Rate = {result['summary']['schema_pass_rate']:.2%}")
```

---

## 🗄️ 数据库设计（版本管理）

### 表结构设计

```sql
-- 版本历史表
CREATE TABLE version_history (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    version_type VARCHAR(20) NOT NULL,  -- 'golden_set' or 'model'
    created_at TIMESTAMP NOT NULL,
    created_by VARCHAR(100),
    description TEXT,
    change_summary JSONB  -- 变更摘要
);

-- 测试用例表（带版本）
CREATE TABLE test_cases (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    file_path VARCHAR(500),
    expected_fields JSONB,
    category VARCHAR(50),
    description TEXT,
    priority INTEGER,
    tags TEXT[],
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (version) REFERENCES version_history(version)
);

-- 测试结果表（带版本）
CREATE TABLE test_results (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    golden_set_version VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    summary JSONB,  -- Summary 指标
    failed_cases JSONB,  -- 失败用例详情
    passed_cases TEXT[],  -- 通过用例 ID 列表
    is_full_result BOOLEAN DEFAULT FALSE,  -- 是否为完整结果
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (version) REFERENCES version_history(version)
);

-- 创建索引
CREATE INDEX idx_test_cases_version ON test_cases(version);
CREATE INDEX idx_test_results_version ON test_results(version);
CREATE INDEX idx_test_results_timestamp ON test_results(timestamp);
```

---

## 🛠️ KYC 项目完整实现

### 版本管理器类

```python
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class GoldenSetVersionManager:
    """Golden Set 版本管理器"""
    
    def __init__(self, storage_path: str = "golden_set_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.current_version = self._load_current_version()
    
    def create_version(
        self,
        golden_set: List[Dict],
        version: str,
        description: str = "",
        created_by: str = "system"
    ) -> bool:
        """创建新版本"""
        # 1. 保存 Golden Set
        version_file = self.storage_path / f"golden_set_{version}.json"
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(golden_set, f, ensure_ascii=False, indent=2)
        
        # 2. 保存版本元数据
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "created_by": created_by,
            "description": description,
            "case_count": len(golden_set),
            "case_ids": [case["case_id"] for case in golden_set]
        }
        
        metadata_file = self.storage_path / f"metadata_{version}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 3. 更新当前版本
        self.current_version = version
        self._save_current_version()
        
        return True
    
    def load_version(self, version: str) -> List[Dict]:
        """加载指定版本的 Golden Set"""
        version_file = self.storage_path / f"golden_set_{version}.json"
        with open(version_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """对比两个版本"""
        golden_set_v1 = self.load_version(version1)
        golden_set_v2 = self.load_version(version2)
        
        cases_v1 = {case["case_id"]: case for case in golden_set_v1}
        cases_v2 = {case["case_id"]: case for case in golden_set_v2}
        
        added = [case_id for case_id in cases_v2 if case_id not in cases_v1]
        removed = [case_id for case_id in cases_v1 if case_id not in cases_v2]
        modified = [
            case_id for case_id in cases_v1
            if case_id in cases_v2 and cases_v1[case_id] != cases_v2[case_id]
        ]
        
        return {
            "version1": version1,
            "version2": version2,
            "summary": {
                "v1_count": len(cases_v1),
                "v2_count": len(cases_v2),
                "added_count": len(added),
                "removed_count": len(removed),
                "modified_count": len(modified)
            },
            "added_cases": added,
            "removed_cases": removed,
            "modified_cases": modified
        }
    
    def list_versions(self) -> List[str]:
        """列出所有版本"""
        versions = []
        for file in self.storage_path.glob("metadata_*.json"):
            version = file.stem.replace("metadata_", "")
            versions.append(version)
        return sorted(versions, reverse=True)
    
    def rollback_to_version(self, target_version: str) -> bool:
        """回滚到指定版本"""
        if target_version not in self.list_versions():
            raise ValueError(f"Version {target_version} not found")
        
        # 备份当前版本
        if self.current_version:
            backup_file = self.storage_path / f"backup_{self.current_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            current_golden_set = self.load_version(self.current_version)
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(current_golden_set, f, ensure_ascii=False, indent=2)
        
        # 恢复目标版本
        self.current_version = target_version
        self._save_current_version()
        
        return True
    
    def _load_current_version(self) -> Optional[str]:
        """加载当前版本"""
        current_file = self.storage_path / "current_version.txt"
        if current_file.exists():
            return current_file.read_text().strip()
        return None
    
    def _save_current_version(self):
        """保存当前版本"""
        current_file = self.storage_path / "current_version.txt"
        current_file.write_text(self.current_version)


class TestResultVersionManager:
    """测试结果版本管理器"""
    
    def __init__(self, storage_path: str = "test_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_result(
        self,
        result: Dict,
        save_full: bool = False
    ) -> bool:
        """保存测试结果"""
        version = result["version"]
        
        # 根据策略决定保存内容
        if save_full:
            # 保存完整结果
            result_file = self.storage_path / f"result_{version}_full.json"
        else:
            # 只保存 Summary + 失败用例详情
            result_file = self.storage_path / f"result_{version}_summary.json"
            # 移除通过用例的详细输出
            result = self._create_summary_result(result)
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return True
    
    def load_result(self, version: str) -> Dict:
        """加载测试结果"""
        # 优先加载 Summary（更小）
        summary_file = self.storage_path / f"result_{version}_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # 如果没有 Summary，加载完整结果
        full_file = self.storage_path / f"result_{version}_full.json"
        if full_file.exists():
            with open(full_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        raise FileNotFoundError(f"Result for version {version} not found")
    
    def compare_results(self, version1: str, version2: str) -> Dict:
        """对比两个版本的测试结果"""
        result_v1 = self.load_result(version1)
        result_v2 = self.load_result(version2)
        
        return self._compare_results_internal(result_v1, result_v2)
    
    def _create_summary_result(self, result: Dict) -> Dict:
        """创建 Summary 结果（移除通过用例的详细输出）"""
        summary_result = {
            "version": result["version"],
            "golden_set_version": result.get("golden_set_version"),
            "model_version": result.get("model_version"),
            "timestamp": result.get("timestamp"),
            "summary": result.get("summary", {}),
            "failed_cases": result.get("failed_cases", []),
            "passed_cases": [
                case["case_id"] if isinstance(case, dict) else case
                for case in result.get("passed_cases", [])
            ],
            "is_full_result": False
        }
        return summary_result
    
    def _compare_results_internal(self, result_v1: Dict, result_v2: Dict) -> Dict:
        """内部对比方法"""
        summary_v1 = result_v1["summary"]
        summary_v2 = result_v2["summary"]
        
        failed_v1 = set(
            case["case_id"] if isinstance(case, dict) else case
            for case in result_v1.get("failed_cases", [])
        )
        failed_v2 = set(
            case["case_id"] if isinstance(case, dict) else case
            for case in result_v2.get("failed_cases", [])
        )
        
        metrics_comparison = {
            "schema_pass_rate": {
                "v1": summary_v1["schema_pass_rate"],
                "v2": summary_v2["schema_pass_rate"],
                "delta": summary_v2["schema_pass_rate"] - summary_v1["schema_pass_rate"]
            },
            "field_accuracy": {
                "v1": summary_v1["field_accuracy"],
                "v2": summary_v2["field_accuracy"],
                "delta": summary_v2["field_accuracy"] - summary_v1["field_accuracy"]
            }
        }
        
        return {
            "version1": result_v1["version"],
            "version2": result_v2["version"],
            "metrics_comparison": metrics_comparison,
            "failure_analysis": {
                "new_failures": list(failed_v2 - failed_v1),
                "fixed_failures": list(failed_v1 - failed_v2),
                "persistent_failures": list(failed_v1 & failed_v2)
            }
        }


# 使用示例
# 1. 版本管理
version_manager = GoldenSetVersionManager()

# 创建版本
version_manager.create_version(
    golden_set=golden_set,
    version="v1.1.0",
    description="添加 20 个新用例"
)

# 对比版本
comparison = version_manager.compare_versions("v1.0.0", "v1.1.0")
print(f"新增用例: {comparison['summary']['added_count']}")

# 回滚版本
version_manager.rollback_to_version("v1.0.0")

# 2. 测试结果管理
result_manager = TestResultVersionManager()

# 保存结果（Summary）
result_manager.save_result(test_result, save_full=False)

# 对比结果
comparison = result_manager.compare_results("v1.0.0", "v1.1.0")
print(f"Schema Pass Rate 变化: {comparison['metrics_comparison']['schema_pass_rate']['delta']:.2%}")
```

---

## 💡 面试话术

### 核心话术

1. ✅ **大型 Codebase 版本管理策略**：
   - "我们使用**多维度版本管理**：代码版本（Git Tags）、模型版本（MLflow/Model Registry）、配置版本（配置文件、Prompt 模板）、数据版本（Golden Set、训练数据）、依赖版本（requirements.txt、Docker 镜像）。所有组件使用统一的版本号（v1.2.3），通过版本快照（Version Snapshot）关联所有组件版本，确保部署时的一致性。"

2. ✅ **版本关联策略**：
   - "我们使用**统一版本号**和**版本快照**两种策略：统一版本号确保所有组件使用相同的版本号（v1.2.3），版本快照记录某个时间点的所有组件版本（代码 commit、模型路径、配置路径、数据路径），这样可以精确复现某个版本的完整环境。"

3. ✅ **Golden Set 版本管理**：
   - "我们使用**三层版本管理**：Git 版本控制（代码和配置）、数据库版本管理（元数据和结果）、对象存储版本管理（测试数据文件）。版本号使用语义化版本（MAJOR.MINOR.PATCH）。"

4. ✅ **结果存储策略**：
   - "我们使用**混合策略**：所有版本保存 Summary（轻量级，包含总体指标和失败用例详情），关键版本（MAJOR）保存完整结果。这样既能满足对比需求，又能节省存储空间。"

5. ✅ **版本对比功能**：
   - "我们支持三种对比：Golden Set 版本对比（用例变化）、测试结果版本对比（性能变化）、模型版本对比（同一 Golden Set 不同模型）。对比内容包括指标变化、失败用例变化、新增/修复的失败用例。"

6. ✅ **数据库版本管理**：
   - "我们使用**Alembic**（Python 项目）或**Flyway**（跨语言项目）管理数据库 schema 版本。每次数据库结构变更都会创建迁移脚本（Migration Script），通过版本号追踪变更历史。迁移脚本支持升级（upgrade）和回滚（downgrade），确保数据库结构变更的可追溯性和可回滚性。"

7. ✅ **版本回滚机制**：
   - "如果新版本有问题，可以回滚到旧版本。回滚时会自动备份当前版本，然后恢复目标版本的所有组件（代码、模型、配置、数据、数据库 schema），并记录回滚操作历史。"

---

## 📝 实施检查清单

- [ ] **版本号设计**：使用语义化版本（MAJOR.MINOR.PATCH）
- [ ] **版本存储**：Git + 数据库 + 对象存储三层架构
- [ ] **结果存储策略**：Summary（所有版本）+ Full Results（关键版本）
- [ ] **版本对比功能**：Golden Set 对比、测试结果对比、模型对比
- [ ] **版本回滚机制**：支持回滚到任意历史版本
- [ ] **版本历史追踪**：记录所有版本变更历史
- [ ] **版本查询功能**：列出所有版本、查询指定版本

---

## 🔗 相关文档

- **Parent**: [KYC_Day03_A1_回归测试与门禁详解.md](./KYC_Day03_A1_回归测试与门禁详解.md)
- **Related**: Golden Set、测试结果、版本管理、回归测试

---

**最后更新**：2025-01-19
