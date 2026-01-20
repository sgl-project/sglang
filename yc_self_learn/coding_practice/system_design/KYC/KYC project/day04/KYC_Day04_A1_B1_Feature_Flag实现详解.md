# KYC_Day04_A1_B1: Feature Flag 实现详解 - 如何只开想要开的功能

## 📋 目录
1. [核心问题：如何精确控制功能开启？](#核心问题如何精确控制功能开启)
2. [Feature Flag 的粒度设计](#feature-flag-的粒度设计)
3. [精确控制实现方法](#精确控制实现方法)
4. [KYC 项目实际案例](#kyc-项目实际案例)
5. [常见场景和最佳实践](#常见场景和最佳实践)

---

## 核心问题：如何精确控制功能开启？

### 问题场景

**需求**：我只想开启"模型版本切换"功能，但不想开启"Prompt 版本切换"和"验证器严格程度调整"。

**挑战**：
- 如何独立控制每个功能？
- 如何避免功能之间的相互影响？
- 如何确保只开启想要的功能？

---

## Feature Flag 的粒度设计

### 1. 独立开关设计（推荐）

**原则**：每个功能都有独立的开关，互不影响。

#### 设计示例

```yaml
# config/feature_flags.yaml
feature_flags:
  # 功能 1：模型版本切换（独立开关）
  model_version:
    enabled: true          # ✅ 开启
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
    canary_percentage: 5
  
  # 功能 2：Prompt 版本切换（独立开关）
  prompt_version:
    enabled: false         # ❌ 关闭（不开启）
    default: "v1"
    options:
      - "v1"
      - "v2"
    canary_percentage: 0
  
  # 功能 3：验证器严格程度（独立开关）
  validator_strictness:
    enabled: false         # ❌ 关闭（不开启）
    default: "medium"
    options:
      - "high"
      - "medium"
      - "low"
    canary_percentage: 0
```

**关键点**：
- ✅ 每个功能都有独立的 `enabled` 开关
- ✅ 关闭的功能使用 `default` 值，不参与 Canary Release
- ✅ 功能之间完全独立，互不影响

---

## 精确控制实现方法

### 方法 1：基于配置的开关控制（最简单）

#### 代码实现

```python
# src/feature_flags.py
from typing import Dict, Any, Optional
import yaml
import hashlib

class FeatureFlagManager:
    """Feature Flag 管理器 - 精确控制功能开启"""
    
    def __init__(self, config_path: str = "config/feature_flags.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        检查功能是否开启
        
        Args:
            feature_name: 功能名称（如 "model_version", "prompt_version"）
        
        Returns:
            True if enabled, False otherwise
        """
        feature_config = self.config['feature_flags'].get(feature_name)
        if feature_config is None:
            return False  # 功能不存在，默认关闭
        
        return feature_config.get('enabled', False)
    
    def get_feature_value(self, feature_name: str, trace_id: str) -> Any:
        """
        获取功能的值（如果功能开启，返回 Canary 值；否则返回默认值）
        
        Args:
            feature_name: 功能名称
            trace_id: 请求的 trace_id（用于 Canary Release）
        
        Returns:
            功能的值（如模型版本、Prompt 版本等）
        """
        feature_config = self.config['feature_flags'].get(feature_name)
        if feature_config is None:
            return None
        
        # 如果功能未开启，直接返回默认值
        if not feature_config.get('enabled', False):
            return feature_config.get('default')
        
        # 如果功能开启，检查是否需要 Canary Release
        canary_percentage = feature_config.get('canary_percentage', 0)
        if canary_percentage <= 0:
            # 没有 Canary，直接返回默认值
            return feature_config.get('default')
        
        # Canary Release：根据 trace_id 决定是否使用新版本
        options = feature_config.get('options', [])
        if len(options) < 2:
            return feature_config.get('default')
        
        # 使用 trace_id 的 hash 值确保一致性
        hash_value = int(hashlib.md5(trace_id.encode()).hexdigest(), 16) % 100
        
        if hash_value < canary_percentage:
            # 使用新版本（假设最后一个选项是新版本）
            return options[-1]
        else:
            # 使用默认版本
            return feature_config.get('default')
    
    def get_model_version(self, trace_id: str) -> str:
        """获取模型版本（只开启 model_version 功能时使用）"""
        if not self.is_feature_enabled('model_version'):
            # 功能未开启，返回默认值
            return self.config['feature_flags']['model_version']['default']
        
        return self.get_feature_value('model_version', trace_id)
    
    def get_prompt_version(self, trace_id: str) -> str:
        """获取 Prompt 版本（只开启 prompt_version 功能时使用）"""
        if not self.is_feature_enabled('prompt_version'):
            # 功能未开启，返回默认值
            return self.config['feature_flags']['prompt_version']['default']
        
        return self.get_feature_value('prompt_version', trace_id)
    
    def get_validator_strictness(self, trace_id: str) -> str:
        """获取验证器严格程度（只开启 validator_strictness 功能时使用）"""
        if not self.is_feature_enabled('validator_strictness'):
            # 功能未开启，返回默认值
            return self.config['feature_flags']['validator_strictness']['default']
        
        return self.get_feature_value('validator_strictness', trace_id)
```

#### 使用示例

```python
# 使用 Feature Flag Manager
flag_manager = FeatureFlagManager("config/feature_flags.yaml")

# 场景 1：只开启模型版本切换
# config/feature_flags.yaml:
#   model_version.enabled = true
#   prompt_version.enabled = false
#   validator_strictness.enabled = false

trace_id = "req_12345"

# 获取模型版本（功能开启，可能返回新版本）
model_version = flag_manager.get_model_version(trace_id)
# 可能返回: "qwen2.5-vl-7b" (如果 trace_id 在 5% Canary 范围内)
# 或返回: "qwen2.5-vl-32b" (如果 trace_id 不在 Canary 范围内)

# 获取 Prompt 版本（功能未开启，总是返回默认值）
prompt_version = flag_manager.get_prompt_version(trace_id)
# 总是返回: "v1" (因为功能未开启)

# 获取验证器严格程度（功能未开启，总是返回默认值）
validator_strictness = flag_manager.get_validator_strictness(trace_id)
# 总是返回: "medium" (因为功能未开启)
```

---

### 方法 2：基于用户/请求的精确控制（高级）

#### 场景

**需求**：只对特定用户或特定请求开启功能。

#### 设计示例

```yaml
# config/feature_flags.yaml
feature_flags:
  model_version:
    enabled: true
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
    canary_percentage: 5
    # 精确控制：只对特定用户开启
    allow_list:
      - user_id: "user_001"
      - user_id: "user_002"
    deny_list: []  # 黑名单（优先级高于 allow_list）
```

#### 代码实现

```python
# src/feature_flags.py (扩展版)

class FeatureFlagManager:
    """Feature Flag 管理器 - 支持精确控制"""
    
    def __init__(self, config_path: str = "config/feature_flags.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def is_feature_enabled_for_request(
        self, 
        feature_name: str, 
        trace_id: str,
        user_id: Optional[str] = None,
        request_metadata: Optional[Dict] = None
    ) -> bool:
        """
        检查功能是否对特定请求开启
        
        Args:
            feature_name: 功能名称
            trace_id: 请求的 trace_id
            user_id: 用户 ID（可选）
            request_metadata: 请求元数据（可选）
        
        Returns:
            True if enabled for this request, False otherwise
        """
        feature_config = self.config['feature_flags'].get(feature_name)
        if feature_config is None:
            return False
        
        # 1. 检查全局开关
        if not feature_config.get('enabled', False):
            return False
        
        # 2. 检查黑名单（优先级最高）
        deny_list = feature_config.get('deny_list', [])
        if user_id and user_id in deny_list:
            return False
        
        # 3. 检查白名单
        allow_list = feature_config.get('allow_list', [])
        if allow_list:
            # 如果设置了白名单，只允许白名单中的用户
            if user_id and user_id in allow_list:
                return True
            else:
                return False  # 不在白名单中，不允许
        
        # 4. 检查 Canary 百分比
        canary_percentage = feature_config.get('canary_percentage', 0)
        if canary_percentage > 0:
            hash_value = int(hashlib.md5(trace_id.encode()).hexdigest(), 16) % 100
            return hash_value < canary_percentage
        
        # 5. 默认：功能开启，对所有请求可用
        return True
    
    def get_feature_value(
        self, 
        feature_name: str, 
        trace_id: str,
        user_id: Optional[str] = None,
        request_metadata: Optional[Dict] = None
    ) -> Any:
        """
        获取功能的值（考虑精确控制）
        """
        # 检查功能是否对当前请求开启
        if not self.is_feature_enabled_for_request(
            feature_name, trace_id, user_id, request_metadata
        ):
            # 功能未开启，返回默认值
            feature_config = self.config['feature_flags'].get(feature_name)
            return feature_config.get('default') if feature_config else None
        
        # 功能开启，返回 Canary 值或默认值
        feature_config = self.config['feature_flags'].get(feature_name)
        options = feature_config.get('options', [])
        
        if len(options) < 2:
            return feature_config.get('default')
        
        # Canary Release：根据 trace_id 决定版本
        canary_percentage = feature_config.get('canary_percentage', 0)
        if canary_percentage > 0:
            hash_value = int(hashlib.md5(trace_id.encode()).hexdigest(), 16) % 100
            if hash_value < canary_percentage:
                return options[-1]  # 新版本
        
        return feature_config.get('default')  # 默认版本
```

#### 使用示例

```python
# 场景：只对特定用户开启新模型版本

flag_manager = FeatureFlagManager("config/feature_flags.yaml")

# 用户 1：在白名单中，可以使用新模型
user_id_1 = "user_001"
trace_id_1 = "req_001"
model_version_1 = flag_manager.get_feature_value(
    'model_version', trace_id_1, user_id=user_id_1
)
# 可能返回: "qwen2.5-vl-7b" (新版本)

# 用户 2：不在白名单中，使用默认模型
user_id_2 = "user_999"
trace_id_2 = "req_002"
model_version_2 = flag_manager.get_feature_value(
    'model_version', trace_id_2, user_id=user_id_2
)
# 总是返回: "qwen2.5-vl-32b" (默认版本)
```

---

### 方法 3：基于环境的精确控制（多环境）

#### 场景

**需求**：开发环境开启所有功能，生产环境只开启部分功能。

#### 设计示例

```yaml
# config/feature_flags.yaml
feature_flags:
  model_version:
    enabled: true
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
    canary_percentage: 5
    # 环境控制
    environments:
      dev: true      # 开发环境：开启
      staging: true  # 预发布环境：开启
      prod: true     # 生产环境：开启
  
  prompt_version:
    enabled: true
    default: "v1"
    options:
      - "v1"
      - "v2"
    canary_percentage: 10
    # 环境控制
    environments:
      dev: true      # 开发环境：开启
      staging: true  # 预发布环境：开启
      prod: false    # 生产环境：关闭（暂时不开启）
  
  validator_strictness:
    enabled: true
    default: "medium"
    options:
      - "high"
      - "medium"
      - "low"
    canary_percentage: 0
    # 环境控制
    environments:
      dev: true      # 开发环境：开启
      staging: false # 预发布环境：关闭
      prod: false    # 生产环境：关闭
```

#### 代码实现

```python
# src/feature_flags.py (环境感知版)

import os

class FeatureFlagManager:
    """Feature Flag 管理器 - 支持环境控制"""
    
    def __init__(self, config_path: str = "config/feature_flags.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 获取当前环境（从环境变量读取）
        self.current_env = os.getenv('ENVIRONMENT', 'dev')
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        检查功能是否在当前环境开启
        """
        feature_config = self.config['feature_flags'].get(feature_name)
        if feature_config is None:
            return False
        
        # 1. 检查全局开关
        if not feature_config.get('enabled', False):
            return False
        
        # 2. 检查环境开关
        environments = feature_config.get('environments', {})
        if environments:
            # 如果设置了环境控制，检查当前环境
            return environments.get(self.current_env, False)
        
        # 3. 默认：功能开启
        return True
```

#### 使用示例

```python
# 设置环境变量
os.environ['ENVIRONMENT'] = 'prod'

flag_manager = FeatureFlagManager("config/feature_flags.yaml")

# 生产环境：只开启 model_version
# config/feature_flags.yaml:
#   model_version.environments.prod = true
#   prompt_version.environments.prod = false
#   validator_strictness.environments.prod = false

trace_id = "req_12345"

# 模型版本（生产环境开启）
model_version = flag_manager.get_model_version(trace_id)
# 可能返回新版本或默认版本

# Prompt 版本（生产环境关闭）
prompt_version = flag_manager.get_prompt_version(trace_id)
# 总是返回: "v1" (因为生产环境关闭)

# 验证器严格程度（生产环境关闭）
validator_strictness = flag_manager.get_validator_strictness(trace_id)
# 总是返回: "medium" (因为生产环境关闭)
```

---

## KYC 项目实际案例

### 案例 1：只开启模型版本切换

#### 配置

```yaml
# config/feature_flags.yaml
feature_flags:
  model_version:
    enabled: true          # ✅ 开启
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
    canary_percentage: 5   # 5% 流量使用新模型
  
  prompt_version:
    enabled: false         # ❌ 关闭
    default: "v1"
    options:
      - "v1"
      - "v2"
    canary_percentage: 0
  
  validator_strictness:
    enabled: false         # ❌ 关闭
    default: "medium"
    options:
      - "high"
      - "medium"
      - "low"
    canary_percentage: 0
```

#### 代码使用

```python
# src/kyc_pipeline.py
from feature_flags import FeatureFlagManager

class KYCPipeline:
    def __init__(self):
        self.flag_manager = FeatureFlagManager()
    
    def process_request(self, request: Dict, trace_id: str):
        """处理 KYC 请求"""
        
        # 1. 获取模型版本（功能开启，可能使用新模型）
        model_version = self.flag_manager.get_model_version(trace_id)
        # 可能返回: "qwen2.5-vl-7b" (5% Canary) 或 "qwen2.5-vl-32b" (默认)
        
        # 2. 获取 Prompt 版本（功能关闭，总是使用默认值）
        prompt_version = self.flag_manager.get_prompt_version(trace_id)
        # 总是返回: "v1" (因为功能未开启)
        
        # 3. 获取验证器严格程度（功能关闭，总是使用默认值）
        validator_strictness = self.flag_manager.get_validator_strictness(trace_id)
        # 总是返回: "medium" (因为功能未开启)
        
        # 4. 使用这些值处理请求
        result = self.run_kyc_pipeline(
            model_version=model_version,
            prompt_version=prompt_version,
            validator_strictness=validator_strictness,
            request=request
        )
        
        return result
```

### 案例 2：组合控制（开启多个功能）

#### 配置

```yaml
# config/feature_flags.yaml
feature_flags:
  model_version:
    enabled: true          # ✅ 开启
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
    canary_percentage: 5
  
  prompt_version:
    enabled: true          # ✅ 开启
    default: "v1"
    options:
      - "v1"
      - "v2"
    canary_percentage: 10
  
  validator_strictness:
    enabled: false         # ❌ 关闭
    default: "medium"
    options:
      - "high"
      - "medium"
      - "low"
    canary_percentage: 0
```

**结果**：
- ✅ 模型版本切换：开启（5% Canary）
- ✅ Prompt 版本切换：开启（10% Canary）
- ❌ 验证器严格程度：关闭（使用默认值 "medium"）

---

## 常见场景和最佳实践

### 场景 1：逐步开启功能

**需求**：先开启模型版本切换，确认稳定后再开启 Prompt 版本切换。

**步骤**：

1. **阶段 1**：只开启模型版本切换
   ```yaml
   model_version:
     enabled: true
     canary_percentage: 5
   prompt_version:
     enabled: false  # 暂时关闭
   ```

2. **阶段 2**：模型版本稳定后，开启 Prompt 版本切换
   ```yaml
   model_version:
     enabled: true
     canary_percentage: 100  # 全量发布
   prompt_version:
     enabled: true  # 现在开启
     canary_percentage: 5
   ```

### 场景 2：紧急关闭功能

**需求**：发现新模型有问题，立即关闭。

**操作**：

```yaml
# 修改配置（无需重新部署）
model_version:
  enabled: false  # 立即关闭
  default: "qwen2.5-vl-32b"  # 所有请求使用默认模型
```

**代码自动处理**：

```python
# 功能关闭后，get_model_version() 总是返回默认值
model_version = flag_manager.get_model_version(trace_id)
# 总是返回: "qwen2.5-vl-32b" (因为 enabled = false)
```

### 场景 3：A/B 测试

**需求**：同时测试两个 Prompt 版本，对比效果。

**配置**：

```yaml
prompt_version:
  enabled: true
  default: "v1"
  options:
    - "v1"  # 50% 流量
    - "v2"  # 50% 流量
  canary_percentage: 50  # 50% 使用 v2
```

**结果**：
- 50% 请求使用 v1（默认）
- 50% 请求使用 v2（Canary）
- 可以对比两个版本的效果

---

## 最佳实践总结

### 1. 独立开关原则

✅ **每个功能都有独立的 `enabled` 开关**
- 可以单独开启/关闭每个功能
- 功能之间互不影响

❌ **不要使用一个开关控制多个功能**
- 难以精确控制
- 容易产生副作用

### 2. 默认值原则

✅ **每个功能都有明确的默认值**
- 功能关闭时，使用默认值
- 确保系统始终有可用的配置

❌ **不要依赖功能开启才能运行**
- 如果功能关闭，系统应该能正常运行
- 使用默认值作为 fallback

### 3. 一致性原则

✅ **同一个 trace_id 总是使用同一个版本**
- 使用 trace_id 的 hash 值决定版本
- 确保请求的一致性

❌ **不要随机选择版本**
- 同一个请求可能得到不同的结果
- 难以追踪和调试

### 4. 渐进式开启原则

✅ **逐步开启功能**
- 先开启一个功能，确认稳定
- 再开启下一个功能

❌ **不要一次性开启所有功能**
- 难以定位问题
- 风险太大

---

## 相关文档

- [KYC_Day04_A1_发布策略与回滚详解.md](./KYC_Day04_A1_发布策略与回滚详解.md) - Feature Flag 基础概念
- [KYC_Day04_A1_B2_Canary_Release监控详解.md](./KYC_Day04_A1_B2_Canary_Release监控详解.md) - Canary Release 监控
- [KYC_Day04_A1_B3_Rollback自动化详解.md](./KYC_Day04_A1_B3_Rollback自动化详解.md) - Rollback 自动化

---

## 总结

### 核心答案

**如何只开想要开的功能？**

1. **独立开关**：每个功能都有独立的 `enabled` 开关
2. **精确控制**：通过配置精确控制哪些功能开启
3. **默认值**：功能关闭时使用默认值，确保系统正常运行
4. **一致性**：同一个请求总是使用同一个版本

### 实现方法

- **方法 1**：基于配置的开关控制（最简单）
- **方法 2**：基于用户/请求的精确控制（高级）
- **方法 3**：基于环境的精确控制（多环境）

### 关键代码

```python
# 检查功能是否开启
if flag_manager.is_feature_enabled('model_version'):
    # 功能开启，使用 Canary 值
    model_version = flag_manager.get_feature_value('model_version', trace_id)
else:
    # 功能关闭，使用默认值
    model_version = flag_manager.config['feature_flags']['model_version']['default']
```
