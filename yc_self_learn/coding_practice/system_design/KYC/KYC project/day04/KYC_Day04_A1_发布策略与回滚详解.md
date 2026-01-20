# Day 4｜发布策略与回滚：把"上线"变成可控实验

---
doc_type: tutorial
layer: L2
scope_in:  发布策略（Feature Flag、Canary Release、Rollback）、灰度发布流程、回滚机制、版本管理
scope_out: 具体 CI/CD 集成（见 reference）；具体 Feature Flag 工具配置（见 reference）
inputs:   (读者) 需求：理解发布策略设计，知道如何用 Feature Flag + Canary Release 安全上线，如何快速回滚
outputs:  发布策略完整设计 + Feature Flag 设计 + Canary Release 流程 + Rollback 机制 + KYC 项目实际案例
entrypoints: [ Feature Flag 设计, Canary Release 流程, Rollback 机制 ]
children: [ ]
related: [ 发布策略, Feature Flag, Canary Release, Rollback, KYC_Day01_A1_详细讲解_指标与测试.md, KYC_Day02_A1_可观测性详解.md, KYC_Day03_A1_回归测试与门禁详解.md ]
---

## Definition（定义）

**核心问题**：**如何安全地发布新版本？如何控制风险？如何快速回滚？**

**核心答案**：
- ✅ **Feature Flag（功能开关）**：动态控制功能开启/关闭，无需重新部署
- ✅ **Canary Release（金丝雀发布）**：逐步扩大流量，观察指标，异常立即回滚
- ✅ **Rollback（回滚）**：快速回退到稳定版本，最小化影响

**类比**：
- **Feature Flag** = **电灯开关**（随时可以开/关，不需要重新布线）
- **Canary Release** = **疫苗试验**（先小范围测试，确认安全后再扩大）
- **Rollback** = **紧急刹车**（发现问题立即停止，回到安全状态）

---

## 🎯 为什么要练发布策略？

### Senior 的价值定位

**不是"能上线"**：
- ❌ 直接全量发布（风险大）
- ❌ 没有回滚预案（出问题只能干瞪眼）
- ❌ 无法控制影响范围（一出问题影响所有用户）

**而是"能安全上线"**：
- ✅ 逐步扩大流量（Canary Release）
- ✅ 动态控制功能（Feature Flag）
- ✅ 快速回滚机制（Rollback）
- ✅ 实时监控指标（Metrics）

**面试中的价值**：
- ✅ 能讲出"Feature Flag 设计"：如何设计开关，如何管理版本
- ✅ 能设计"Canary Release 流程"：流量分配、观察指标、决策机制
- ✅ 能说明"Rollback 策略"：回滚条件、回滚流程、回滚验证

---

## 📊 Feature Flag（功能开关）详解

### 1. Feature Flag 是什么？

**定义**：**动态控制功能开启/关闭的机制，无需重新部署代码**。

**核心价值**：
- ✅ **动态控制**：随时可以开启/关闭功能
- ✅ **无需部署**：修改配置即可，不需要重新发布代码
- ✅ **风险控制**：可以快速关闭有问题的功能
- ✅ **A/B 测试**：可以同时运行多个版本，对比效果

**类比**：
- **Feature Flag** = **电灯开关**（随时可以开/关）
- **不是硬编码**：不是把功能写死在代码里（需要重新部署才能改）
- **而是配置化**：功能开关存储在配置中心（修改配置即可）

---

### 2. Feature Flag 设计原则

**KYC 项目的 Feature Flag 设计**：

#### 2.1 模型版本切换（Model Version）

**场景**：切换不同的模型（Qwen2.5-VL-32B vs 其他模型）

**设计**：
```python
# config/feature_flags.yaml
feature_flags:
  model_version:
    enabled: true
    default: "qwen2.5-vl-32b"
    options:
      - "qwen2.5-vl-32b"
      - "qwen2.5-vl-7b"
      - "gpt-4-vision"
    canary_percentage: 5  # 5% 流量使用新模型
```

**代码实现**：
```python
# src/feature_flags.py
from typing import Dict, Any
import yaml
import random

class FeatureFlagManager:
    """Feature Flag 管理器"""
    
    def __init__(self, config_path: str = "config/feature_flags.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_version(self, trace_id: str) -> str:
        """根据 trace_id 决定使用哪个模型版本"""
        model_config = self.config['feature_flags']['model_version']
        
        if not model_config['enabled']:
            return model_config['default']
        
        # Canary 发布：根据 trace_id 的 hash 值决定是否使用新版本
        canary_percentage = model_config.get('canary_percentage', 0)
        if canary_percentage > 0:
            # 使用 trace_id 的 hash 值确保一致性（同一个 trace_id 总是使用同一个版本）
            hash_value = hash(trace_id) % 100
            if hash_value < canary_percentage:
                # 使用新版本（假设第一个选项是新版本）
                return model_config['options'][1] if len(model_config['options']) > 1 else model_config['default']
        
        return model_config['default']
```

#### 2.2 Prompt 版本切换（Prompt Version）

**场景**：切换不同的 prompt 版本（优化 prompt 效果）

**设计**：
```python
# config/feature_flags.yaml
feature_flags:
  prompt_version:
    enabled: true
    default: "v1"
    options:
      - "v1"  # 旧版本
      - "v2"  # 新版本（优化后的 prompt）
    canary_percentage: 10  # 10% 流量使用新 prompt
```

#### 2.3 验证器严格程度（Validator Strictness）

**场景**：调整 Schema 验证的严格程度（high/medium/low）

**设计**：
```python
# config/feature_flags.yaml
feature_flags:
  validator_strictness:
    enabled: true
    default: "medium"
    options:
      - "high"    # 严格模式：所有字段必须通过验证
      - "medium"  # 中等模式：关键字段必须通过验证
      - "low"     # 宽松模式：允许部分字段失败
    canary_percentage: 0  # 暂时不启用 Canary
```

---

### 3. Feature Flag 管理策略

#### 3.1 配置存储

**方案 1：配置文件（适合小项目）**
- ✅ 简单直接
- ✅ 版本控制（Git）
- ⚠️ 需要重新部署才能更新

**方案 2：配置中心（适合大项目）**
- ✅ 动态更新（无需重新部署）
- ✅ 集中管理
- ✅ 支持多环境
- ⚠️ 需要额外的配置中心服务（如 Consul、Vault、AWS Parameter Store）

**KYC 项目推荐**：
- **PoV 阶段**：使用配置文件（`config/feature_flags.yaml`）
- **Production 阶段**：迁移到配置中心（AWS Parameter Store 或 Consul）

#### 3.2 版本管理

**原则**：
- ✅ 每个 Feature Flag 都有版本号
- ✅ 记录变更历史
- ✅ 支持回滚到历史版本

**示例**：
```python
# config/feature_flags.yaml
feature_flags:
  model_version:
    version: "1.0.0"
    enabled: true
    default: "qwen2.5-vl-32b"
    history:
      - version: "1.0.0"
        date: "2025-01-15"
        change: "初始版本"
      - version: "1.1.0"
        date: "2025-01-20"
        change: "添加 canary_percentage 支持"
```

---

## 🚀 Canary Release（金丝雀发布）详解

### 1. Canary Release 是什么？

**定义**：**逐步扩大新版本的流量，观察指标，确认安全后再全量发布**。

**核心价值**：
- ✅ **风险控制**：小范围测试，降低影响
- ✅ **实时监控**：每步都观察指标，异常立即停止
- ✅ **快速回滚**：发现问题立即回滚，影响最小

**类比**：
- **Canary Release** = **疫苗试验**（先小范围测试，确认安全后再扩大）
- **不是全量发布**：不是一次性把所有流量都切到新版本（风险太大）
- **而是逐步扩大**：1% → 5% → 25% → 100%（每步都观察）

---

### 2. Canary Release 流程设计

#### 2.1 流量分配策略

**KYC 项目的 Canary Release 流程**：

```
阶段 1：1% 流量（观察 1 小时）
    ↓
    ├─ 指标正常 → 进入阶段 2
    └─ 指标异常 → 立即回滚

阶段 2：5% 流量（观察 2 小时）
    ↓
    ├─ 指标正常 → 进入阶段 3
    └─ 指标异常 → 立即回滚

阶段 3：25% 流量（观察 4 小时）
    ↓
    ├─ 指标正常 → 进入阶段 4
    └─ 指标异常 → 立即回滚

阶段 4：100% 流量（全量发布）
    ↓
    ├─ 指标正常 → 发布成功
    └─ 指标异常 → 立即回滚
```

**代码实现**：
```python
# src/canary_release.py
from typing import Dict, List
from datetime import datetime, timedelta
import time

class CanaryReleaseManager:
    """Canary Release 管理器"""
    
    def __init__(self):
        self.stages = [
            {"percentage": 1, "duration_minutes": 60, "name": "Stage 1: 1%"},
            {"percentage": 5, "duration_minutes": 120, "name": "Stage 2: 5%"},
            {"percentage": 25, "duration_minutes": 240, "name": "Stage 3: 25%"},
            {"percentage": 100, "duration_minutes": 0, "name": "Stage 4: 100%"}
        ]
        self.current_stage = 0
        self.start_time = None
    
    def get_traffic_percentage(self, trace_id: str) -> int:
        """根据 trace_id 决定是否使用新版本"""
        if self.current_stage >= len(self.stages):
            return 100  # 全量发布
        
        stage = self.stages[self.current_stage]
        hash_value = hash(trace_id) % 100
        
        if hash_value < stage['percentage']:
            return stage['percentage']
        return 0  # 使用旧版本
    
    def should_advance_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        if self.current_stage >= len(self.stages) - 1:
            return False  # 已经是最后阶段
        
        stage = self.stages[self.current_stage]
        
        # 检查是否达到观察时间
        if self.start_time is None:
            self.start_time = datetime.now()
            return False
        
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        return elapsed_minutes >= stage['duration_minutes']
    
    def advance_stage(self):
        """进入下一阶段"""
        self.current_stage += 1
        self.start_time = datetime.now()
        print(f"✅ 进入 {self.stages[self.current_stage]['name']}")
    
    def rollback(self):
        """回滚到旧版本"""
        self.current_stage = 0
        self.start_time = None
        print("❌ 回滚到旧版本")
```

#### 2.2 观察指标

**KYC 项目的观察指标**：

| 指标 | 阈值 | 动作 |
|------|------|------|
| **Schema Pass Rate** | < 95% | 立即回滚 |
| **p95 Latency** | > 15s（+20%） | 立即回滚 |
| **Error Rate** | > 5% | 立即回滚 |
| **Cost per Request** | > $0.002 | 观察，不立即回滚 |

**代码实现**：
```python
# src/canary_monitor.py
from typing import Dict
from datetime import datetime

class CanaryMonitor:
    """Canary Release 监控器"""
    
    def __init__(self):
        self.thresholds = {
            "schema_pass_rate": 0.95,  # 95%
            "p95_latency_seconds": 15.0,  # 15秒
            "error_rate": 0.05,  # 5%
            "cost_per_request": 0.002  # $0.002
        }
    
    def check_metrics(self, metrics: Dict) -> Dict:
        """检查指标是否超过阈值"""
        alerts = []
        
        # 检查 Schema Pass Rate
        if metrics.get('schema_pass_rate', 1.0) < self.thresholds['schema_pass_rate']:
            alerts.append({
                "metric": "schema_pass_rate",
                "value": metrics['schema_pass_rate'],
                "threshold": self.thresholds['schema_pass_rate'],
                "action": "立即回滚"
            })
        
        # 检查 p95 Latency
        if metrics.get('p95_latency_seconds', 0) > self.thresholds['p95_latency_seconds']:
            alerts.append({
                "metric": "p95_latency_seconds",
                "value": metrics['p95_latency_seconds'],
                "threshold": self.thresholds['p95_latency_seconds'],
                "action": "立即回滚"
            })
        
        # 检查 Error Rate
        if metrics.get('error_rate', 0) > self.thresholds['error_rate']:
            alerts.append({
                "metric": "error_rate",
                "value": metrics['error_rate'],
                "threshold": self.thresholds['error_rate'],
                "action": "立即回滚"
            })
        
        return {
            "status": "healthy" if len(alerts) == 0 else "unhealthy",
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
```

---

## 🔄 Rollback（回滚）详解

### 1. Rollback 是什么？

**定义**：**快速回退到之前的稳定版本，最小化影响**。

**核心价值**：
- ✅ **快速恢复**：发现问题立即回滚，减少影响时间
- ✅ **风险控制**：有明确的回滚预案，不会手忙脚乱
- ✅ **可验证**：回滚后验证系统是否恢复正常

**类比**：
- **Rollback** = **紧急刹车**（发现问题立即停止，回到安全状态）
- **不是修复问题**：不是在新版本上修复问题（可能来不及）
- **而是快速回退**：先回滚到稳定版本，再慢慢修复问题

---

### 2. Rollback 触发条件

**KYC 项目的回滚条件**：

| 条件 | 阈值 | 优先级 |
|------|------|--------|
| **Schema Fail Rate × 2** | 旧版本 2% → 新版本 4% | 🔴 P0（立即回滚） |
| **p95 Latency + 20%** | 旧版本 10s → 新版本 12s | 🔴 P0（立即回滚） |
| **Error Rate > 5%** | 错误率 > 5% | 🔴 P0（立即回滚） |
| **Cost per Request + 50%** | 成本增加 > 50% | 🟡 P1（观察，不立即回滚） |

**代码实现**：
```python
# src/rollback_manager.py
from typing import Dict, Optional
from datetime import datetime

class RollbackManager:
    """回滚管理器"""
    
    def __init__(self):
        self.rollback_conditions = {
            "schema_fail_rate_multiplier": 2.0,  # Schema Fail Rate × 2
            "p95_latency_increase_percent": 20,  # p95 Latency + 20%
            "error_rate_threshold": 0.05,  # Error Rate > 5%
            "cost_increase_percent": 50  # Cost + 50%（观察，不立即回滚）
        }
        self.baseline_metrics = None  # 旧版本的基准指标
    
    def set_baseline(self, metrics: Dict):
        """设置基准指标（旧版本的指标）"""
        self.baseline_metrics = metrics
        print(f"✅ 设置基准指标: {metrics}")
    
    def should_rollback(self, current_metrics: Dict) -> tuple[bool, str]:
        """判断是否应该回滚"""
        if self.baseline_metrics is None:
            return False, "未设置基准指标"
        
        # 检查 Schema Fail Rate
        baseline_schema_fail_rate = 1 - self.baseline_metrics.get('schema_pass_rate', 1.0)
        current_schema_fail_rate = 1 - current_metrics.get('schema_pass_rate', 1.0)
        
        if current_schema_fail_rate >= baseline_schema_fail_rate * self.rollback_conditions['schema_fail_rate_multiplier']:
            return True, f"Schema Fail Rate 超过阈值: {current_schema_fail_rate:.2%} >= {baseline_schema_fail_rate * self.rollback_conditions['schema_fail_rate_multiplier']:.2%}"
        
        # 检查 p95 Latency
        baseline_p95 = self.baseline_metrics.get('p95_latency_seconds', 0)
        current_p95 = current_metrics.get('p95_latency_seconds', 0)
        
        if baseline_p95 > 0:
            latency_increase_percent = ((current_p95 - baseline_p95) / baseline_p95) * 100
            if latency_increase_percent >= self.rollback_conditions['p95_latency_increase_percent']:
                return True, f"p95 Latency 增加超过阈值: {latency_increase_percent:.1f}% >= {self.rollback_conditions['p95_latency_increase_percent']}%"
        
        # 检查 Error Rate
        current_error_rate = current_metrics.get('error_rate', 0)
        if current_error_rate > self.rollback_conditions['error_rate_threshold']:
            return True, f"Error Rate 超过阈值: {current_error_rate:.2%} > {self.rollback_conditions['error_rate_threshold']:.2%}"
        
        return False, "指标正常"
    
    def rollback(self, reason: str):
        """执行回滚"""
        print(f"🔄 开始回滚: {reason}")
        print(f"⏰ 回滚时间: {datetime.now().isoformat()}")
        # 实际回滚逻辑：切换 Feature Flag、更新配置等
        print("✅ 回滚完成")
```

---

### 3. Rollback 流程

**KYC 项目的回滚流程**：

```
1. 检测到异常指标
    ↓
2. 触发回滚条件检查
    ↓
3. 确认回滚（自动或手动）
    ↓
4. 执行回滚操作
    - 切换 Feature Flag（关闭新版本）
    - 更新配置（恢复到旧版本）
    - 通知团队（发送告警）
    ↓
5. 验证回滚效果
    - 检查指标是否恢复正常
    - 确认系统稳定
    ↓
6. 记录回滚事件
    - 记录回滚原因
    - 记录回滚时间
    - 记录影响范围
```

**代码实现**：
```python
# src/rollback_executor.py
from typing import Dict
from datetime import datetime
import json

class RollbackExecutor:
    """回滚执行器"""
    
    def __init__(self, feature_flag_manager, canary_manager, rollback_manager):
        self.feature_flag_manager = feature_flag_manager
        self.canary_manager = canary_manager
        self.rollback_manager = rollback_manager
        self.rollback_history = []
    
    def execute_rollback(self, reason: str, metrics: Dict):
        """执行回滚"""
        rollback_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "metrics_before_rollback": metrics,
            "actions": []
        }
        
        # 1. 关闭新版本的 Feature Flag
        self.feature_flag_manager.disable_new_version()
        rollback_event["actions"].append("关闭新版本 Feature Flag")
        
        # 2. 重置 Canary Release
        self.canary_manager.rollback()
        rollback_event["actions"].append("重置 Canary Release")
        
        # 3. 验证回滚效果（等待一段时间后检查指标）
        time.sleep(60)  # 等待 1 分钟
        post_rollback_metrics = self.get_current_metrics()
        rollback_event["metrics_after_rollback"] = post_rollback_metrics
        
        # 4. 记录回滚事件
        rollback_event["status"] = "success" if self.verify_rollback(post_rollback_metrics) else "failed"
        self.rollback_history.append(rollback_event)
        
        # 5. 保存回滚历史
        self.save_rollback_history()
        
        return rollback_event
    
    def verify_rollback(self, metrics: Dict) -> bool:
        """验证回滚是否成功"""
        # 检查指标是否恢复正常
        should_rollback, reason = self.rollback_manager.should_rollback(metrics)
        return not should_rollback
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标（模拟）"""
        # 实际应该从监控系统获取
        return {
            "schema_pass_rate": 0.98,
            "p95_latency_seconds": 10.0,
            "error_rate": 0.01
        }
    
    def save_rollback_history(self):
        """保存回滚历史"""
        with open("rollback_history.json", "w") as f:
            json.dump(self.rollback_history, f, indent=2)
```

---

## 📊 KYC 项目实际应用场景

### 场景 1：模型版本切换

**场景**：从 Qwen2.5-VL-32B 切换到 Qwen2.5-VL-7B（降低成本）

**流程**：
1. **设置 Feature Flag**：`model_version = "qwen2.5-vl-7b"`，`canary_percentage = 1%`
2. **Canary Release**：
   - 阶段 1：1% 流量使用新模型（观察 1 小时）
   - 阶段 2：5% 流量使用新模型（观察 2 小时）
   - 阶段 3：25% 流量使用新模型（观察 4 小时）
   - 阶段 4：100% 流量使用新模型（全量发布）
3. **监控指标**：
   - Schema Pass Rate（目标：> 95%）
   - p95 Latency（目标：< 15s）
   - Error Rate（目标：< 5%）
   - Cost per Request（目标：降低 50%）
4. **回滚条件**：
   - Schema Fail Rate × 2 → 立即回滚
   - p95 Latency + 20% → 立即回滚
   - Error Rate > 5% → 立即回滚

---

### 场景 2：Prompt 优化

**场景**：优化 prompt，提高字段提取准确率

**流程**：
1. **设置 Feature Flag**：`prompt_version = "v2"`，`canary_percentage = 5%`
2. **Canary Release**：
   - 阶段 1：5% 流量使用新 prompt（观察 2 小时）
   - 阶段 2：25% 流量使用新 prompt（观察 4 小时）
   - 阶段 3：100% 流量使用新 prompt（全量发布）
3. **监控指标**：
   - 字段级准确率（目标：提高 5%）
   - Schema Pass Rate（目标：> 95%）
   - p95 Latency（目标：不变）
4. **回滚条件**：
   - 字段级准确率下降 → 立即回滚
   - Schema Fail Rate × 2 → 立即回滚

---

### 场景 3：验证器严格程度调整

**场景**：调整 Schema 验证严格程度（从 medium 到 high）

**流程**：
1. **设置 Feature Flag**：`validator_strictness = "high"`，`canary_percentage = 10%`
2. **Canary Release**：
   - 阶段 1：10% 流量使用严格模式（观察 2 小时）
   - 阶段 2：50% 流量使用严格模式（观察 4 小时）
   - 阶段 3：100% 流量使用严格模式（全量发布）
3. **监控指标**：
   - Schema Pass Rate（预期：可能下降，但应该 > 90%）
   - Fallback Rate（目标：< 5%）
   - Error Rate（目标：< 5%）
4. **回滚条件**：
   - Schema Pass Rate < 90% → 立即回滚
   - Fallback Rate > 10% → 立即回滚

---

## 🎯 总结

### 核心概念

1. **Feature Flag**：动态控制功能开启/关闭，无需重新部署
2. **Canary Release**：逐步扩大流量，观察指标，异常立即回滚
3. **Rollback**：快速回退到稳定版本，最小化影响

### 设计要点

1. **Feature Flag 设计**：
   - 配置化（存储在配置中心或配置文件）
   - 版本管理（记录变更历史）
   - 一致性（同一个 trace_id 总是使用同一个版本）

2. **Canary Release 流程**：
   - 流量分配（1% → 5% → 25% → 100%）
   - 观察时间（每步观察足够长的时间）
   - 监控指标（Schema Pass Rate、Latency、Error Rate）

3. **Rollback 机制**：
   - 明确的回滚条件（阈值设定）
   - 快速回滚流程（自动化或手动）
   - 回滚验证（确认系统恢复正常）

### KYC 项目实践

- ✅ **Feature Flag**：模型版本、Prompt 版本、验证器严格程度
- ✅ **Canary Release**：1% → 5% → 25% → 100%
- ✅ **Rollback**：Schema Fail Rate × 2、p95 Latency + 20%、Error Rate > 5%

---

## 📚 相关文档

- [KYC_Day01_A1_详细讲解_指标与测试.md](./day01/KYC_Day01_A1_详细讲解_指标与测试.md) - L0/L1/L2 指标
- [KYC_Day02_A1_可观测性详解.md](./day02/KYC_Day02_A1_可观测性详解.md) - Metrics/Logs/Traces
- [KYC_Day03_A1_回归测试与门禁详解.md](./day03/KYC_Day03_A1_回归测试与门禁详解.md) - Golden Set + Release Gate
