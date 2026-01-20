# Day 3_A1_B7_C3：第 3 层 - 配置版本管理详解

**Parent**: [KYC_Day03_A1_B7_测试用例版本管理和结果对比详解.md](./KYC_Day03_A1_B7_测试用例版本管理和结果对比详解.md)  
**层级**: 第 3 层 - 配置版本管理（Config Versioning）  
**目的**：详细讲解配置版本管理的架构、工具和实践

---

## 🎯 第 3 层：配置版本管理概述

### 核心职责

**配置版本管理负责**：
- ✅ **配置文件版本控制**：追踪配置文件的变更历史
- ✅ **Prompt 模板版本管理**：LLM Prompt 模板的版本化
- ✅ **环境变量管理**：敏感配置的安全管理
- ✅ **配置热更新**：配置变更的动态加载

---

## 📊 第 3 层架构图（详细版）

```mermaid
graph TB
    subgraph Layer3["第 3 层：配置版本管理（Config Versioning）"]
        subgraph ConfigFiles["3.1 配置文件版本（Config Files）"]
            YAML["YAML<br/>📍 config.yaml"]
            JSON["JSON<br/>📍 config.json"]
            TOML["TOML<br/>📍 config.toml"]
            Properties["Properties<br/>📍 application.properties"]
            
            YAML --> Git1["Git 版本控制"]
            JSON --> Git2["Git 版本控制"]
            TOML --> Git3["Git 版本控制"]
        end
        
        subgraph PromptTemplates["3.2 Prompt 模板版本（Prompt Templates）"]
            Jinja2["Jinja2<br/>📍 prompt.jinja"]
            Handlebars["Handlebars<br/>📍 prompt.hbs"]
            Python["Python Template<br/>📍 prompt.py"]
            
            Jinja2 --> Git4["Git 版本控制"]
            Handlebars --> Git5["Git 版本控制"]
        end
        
        subgraph EnvVars["3.3 环境变量管理（Environment Variables）"]
            DotEnv[".env<br/>📍 本地环境变量"]
            Vault["HashiCorp Vault<br/>📍 密钥管理"]
            Consul["Consul<br/>📍 配置中心"]
            AWS["AWS Secrets Manager<br/>📍 AWS 密钥管理"]
            
            DotEnv --> Git6["Git 版本控制"]
            Vault --> SecureStore["安全存储"]
            Consul --> ConfigStore["配置存储"]
        end
        
        subgraph ConfigLoad["3.4 配置加载（Config Loading）"]
            PythonLoad["Python ConfigLoader<br/>📍 配置加载器"]
            JavaLoad["Java Config<br/>📍 Spring Config"]
            NodeLoad["Node Config<br/>📍 dotenv/config"]
            
            PythonLoad --> App1["应用"]
            JavaLoad --> App2["应用"]
            NodeLoad --> App3["应用"]
        end
    end
    
    ConfigFiles --> ConfigLoad
    PromptTemplates --> ConfigLoad
    EnvVars --> ConfigLoad
    
    style ConfigFiles fill:#f3e5f5
    style PromptTemplates fill:#e1f5ff
    style EnvVars fill:#fff4e1
    style ConfigLoad fill:#e8f5e9
```

---

## 🔧 3.1 配置文件版本（Config Files）

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          3.1 配置文件版本（Config Files）                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        配置文件结构（Config Structure）                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  configs/                                                                  │
│  ├── v1.0.0/                                                               │
│  │   ├── config.yaml                                                      │
│  │   ├── config.json                                                       │
│  │   └── config.toml                                                       │
│  ├── v1.1.0/                                                               │
│  │   ├── config.yaml                                                      │
│  │   └── config.json                                                       │
│  └── v1.2.3/                                                               │
│      ├── config.yaml                                                      │
│      └── config.json                                                       │
│                                                                             │
│  配置文件格式：                                                              │
│    • YAML：易读、支持嵌套（推荐）                                           │
│    • JSON：标准格式、易于解析                                               │
│    • TOML：简洁、适合简单配置                                               │
│    • Properties：Java 风格、键值对                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 配置文件示例

```yaml
# configs/v1.2.3/config.yaml
version: "v1.2.3"

model:
  model_name: "kyc_model"
  model_version: "v1.2.3"
  model_path: "s3://models/kyc_model/v1.2.3/model.pkl"

prompt:
  prompt_version: "v1.2.3"
  prompt_path: "prompts/v1.2.3/prompt.jinja"
  temperature: 0.7
  max_tokens: 1000

validation:
  schema_version: "v1.2.3"
  rules_version: "v1.2.3"
  strict_mode: true

api:
  version: "v1"
  timeout: 30
  retry_count: 3

database:
  url: "${DATABASE_URL}"
  pool_size: 10
  echo: false

logging:
  level: "INFO"
  format: "json"
```

---

### 配置加载器实现

```python
# config_loader.py
import yaml
import os
from pathlib import Path
from typing import Dict, Optional

class ConfigLoader:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_config(self, version: str) -> Dict:
        """加载指定版本的配置"""
        config_file = self.config_dir / version / "config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # 替换环境变量
        config = self._substitute_env_vars(config)
        
        return config
    
    def _substitute_env_vars(self, config: Dict) -> Dict:
        """替换配置中的环境变量"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, str):
            # 支持 ${VAR_NAME} 格式
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.getenv(var_name, config)
        return config

# 使用示例
loader = ConfigLoader()
config = loader.load_config("v1.2.3")
print(config["model"]["model_path"])
```

---

## 📝 3.2 Prompt 模板版本（Prompt Templates）

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│        3.2 Prompt 模板版本（Prompt Templates）                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Prompt 模板结构（Template Structure）                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  prompts/                                                                   │
│  ├── v1.0.0/                                                               │
│  │   ├── prompt.jinja                                                      │
│  │   └── prompt_variants/                                                  │
│  │       ├── prompt_short.jinja                                            │
│  │       └── prompt_long.jinja                                             │
│  ├── v1.1.0/                                                               │
│  │   └── prompt.jinja                                                      │
│  └── v1.2.3/                                                               │
│      ├── prompt.jinja                                                      │
│      └── metadata.json                                                     │
│                                                                             │
│  Prompt 模板格式：                                                           │
│    • Jinja2：Python 生态、功能强大（推荐）                                  │
│    • Handlebars：JavaScript 生态、简洁                                      │
│    • Python Template：原生支持、灵活                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Prompt 模板示例

```jinja
# prompts/v1.2.3/prompt.jinja
{% set prompt_version = "v1.2.3" %}
You are a KYC (Know Your Customer) expert. Your task is to extract structured information from identity documents.

## Document Information
- Document Type: {{ document_type }}
- Document Image: [Image provided]

## Fields to Extract
Please extract the following fields from the document:

1. **Full Name**: The complete name as shown on the document
2. **ID Number**: The identification number
3. **Date of Birth**: The date of birth in YYYY-MM-DD format
4. **Address**: The residential address (if available)
5. **Expiry Date**: The document expiry date (if applicable)

## Output Format
Please provide the extracted information in the following JSON format:
```json
{
  "full_name": "...",
  "id_number": "...",
  "date_of_birth": "YYYY-MM-DD",
  "address": "...",
  "expiry_date": "YYYY-MM-DD"
}
```

## Instructions
- Be accurate and precise
- If a field is not found, use null
- Follow the exact date format specified
- Do not make up information

{% if strict_mode %}
## Strict Mode
In strict mode, you must verify all information matches the document exactly.
{% endif %}
```

---

### Prompt 加载器实现

```python
# prompt_loader.py
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict

class PromptLoader:
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.prompt_dir)))
    
    def load_prompt(self, version: str, template_name: str = "prompt.jinja") -> str:
        """加载 Prompt 模板"""
        template_path = f"{version}/{template_name}"
        template = self.env.get_template(template_path)
        return template
    
    def render_prompt(
        self,
        version: str,
        context: Dict,
        template_name: str = "prompt.jinja"
    ) -> str:
        """渲染 Prompt"""
        template = self.load_prompt(version, template_name)
        return template.render(**context)

# 使用示例
loader = PromptLoader()
context = {
    "document_type": "ID Card",
    "strict_mode": True
}
prompt = loader.render_prompt("v1.2.3", context)
print(prompt)
```

---

## 🔐 3.3 环境变量管理（Environment Variables）

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│        3.3 环境变量管理（Environment Variables）                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        环境变量存储（Environment Storage）                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   .env       │    │   Vault      │    │   Consul     │                  │
│  │              │    │              │    │              │                  │
│  │  本地开发     │    │  密钥管理     │    │  配置中心     │                  │
│  │  不提交 Git   │    │  加密存储     │    │  分布式配置   │                  │
│  │              │    │              │    │              │                  │
│  │  .env.local  │    │  vault read  │    │  consul kv   │                  │
│  │  .env.prod   │    │  secret/...  │    │  get config/ │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  ┌──────────────┐                                                          │
│  │ AWS Secrets  │                                                          │
│  │ Manager      │                                                          │
│  │              │                                                          │
│  │  AWS 密钥管理 │                                                          │
│  │  自动加密     │                                                          │
│  │              │                                                          │
│  │  aws secrets │                                                          │
│  │  get-secret  │                                                          │
│  └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 环境变量管理实践

```python
# .env.local（本地开发，不提交 Git）
DATABASE_URL=postgresql://localhost:5432/kyc_dev
API_KEY=dev_key_12345
SECRET_KEY=dev_secret_12345

# .env.prod（生产环境，不提交 Git）
DATABASE_URL=postgresql://prod-db:5432/kyc_prod
API_KEY=prod_key_xxxxx
SECRET_KEY=prod_secret_xxxxx

# .env.example（示例文件，提交 Git）
DATABASE_URL=postgresql://localhost:5432/kyc
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

---

### Vault 集成示例

```python
# vault_client.py
import hvac

class VaultClient:
    def __init__(self, vault_url: str, token: str):
        self.client = hvac.Client(url=vault_url, token=token)
    
    def get_secret(self, path: str) -> dict:
        """获取密钥"""
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data']
    
    def set_secret(self, path: str, data: dict):
        """设置密钥"""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data
        )

# 使用示例
vault = VaultClient(
    vault_url="https://vault.example.com",
    token=os.getenv("VAULT_TOKEN")
)

# 获取数据库密码
db_secret = vault.get_secret("secret/kyc/database")
database_url = f"postgresql://{db_secret['user']}:{db_secret['password']}@..."
```

---

## 📊 第 3 层工具选择矩阵

| 功能 | Python 项目推荐 | Java 项目推荐 | 成本 |
|------|----------------|--------------|------|
| **配置文件** | YAML + Git | YAML/Properties + Git | 免费 |
| **Prompt 模板** | Jinja2 | Handlebars | 免费 |
| **环境变量** | .env / Vault | .env / Vault / Consul | 免费/付费 |

---

## 💡 面试话术

1. ✅ **配置文件版本管理**：
   - "我们使用 **YAML** 格式存储配置文件，通过 Git 进行版本控制。配置文件按版本号组织（configs/v1.2.3/config.yaml），支持环境变量替换。配置加载器支持动态加载指定版本的配置。"

2. ✅ **Prompt 模板版本管理**：
   - "我们使用 **Jinja2** 模板引擎管理 Prompt 模板。Prompt 模板按版本号组织（prompts/v1.2.3/prompt.jinja），支持变量替换和条件渲染。每次 Prompt 变更都会创建新版本，确保可追溯性。"

3. ✅ **环境变量管理**：
   - "敏感配置（如数据库密码、API 密钥）存储在 **HashiCorp Vault** 或 **AWS Secrets Manager**，不提交到 Git。本地开发使用 .env 文件，生产环境从密钥管理服务动态加载。"

---

## 📝 实施检查清单

- [ ] **配置文件**：选择 YAML/JSON/TOML 格式
- [ ] **配置版本化**：按版本号组织配置文件
- [ ] **Prompt 模板**：选择 Jinja2/Handlebars
- [ ] **环境变量**：配置 Vault/Secrets Manager
- [ ] **配置加载**：实现配置加载器

---

**最后更新**：2025-01-19
