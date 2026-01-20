# KYC_Day04_A1_B2: Canary Release 监控详解

## 📋 目录
1. [核心问题：如何监控 Canary Release？](#核心问题如何监控-canary-release)
2. [监控指标设计](#监控指标设计)
3. [自动化推进机制](#自动化推进机制)
4. [自动化回滚机制](#自动化回滚机制)
5. [监控 Dashboard 设计](#监控-dashboard-设计)
6. [KYC 项目实际案例](#kyc-项目实际案例)

---

## 核心问题：如何监控 Canary Release？

### 问题场景

**需求**：
- 如何实时监控 Canary Release 的指标？
- 如何自动判断是否应该进入下一阶段？
- 如何自动检测异常并回滚？

**挑战**：
- 需要实时收集和聚合指标
- 需要自动判断指标是否正常
- 需要自动触发推进或回滚

---

## 监控指标设计

### 1. KYC 项目核心指标

#### 1.1 稳定性指标（L0）

| 指标 | 阈值 | 动作 | 优先级 |
|------|------|------|--------|
| **Schema Pass Rate** | < 95% | 立即回滚 | 🔴 P0 |
| **p95 Latency** | > 15s（+20%） | 立即回滚 | 🔴 P0 |
| **Error Rate** | > 5% | 立即回滚 | 🔴 P0 |

#### 1.2 业务指标（L1）

| 指标 | 阈值 | 动作 | 优先级 |
|------|------|------|--------|
| **字段级准确率** | 下降 > 5% | 观察，不立即回滚 | 🟡 P1 |
| **Cost per Request** | > $0.002 | 观察，不立即回滚 | 🟡 P1 |

#### 1.3 长期健康指标（L2）

| 指标 | 阈值 | 动作 | 优先级 |
|------|------|------|--------|
| **Fallback Rate** | > 10% | 观察，不立即回滚 | 🟢 P2 |

---

## 自动化推进机制

### 1. 推进条件设计

**原则**：
- ✅ 所有关键指标（L0）必须正常
- ✅ 观察时间必须达到要求
- ✅ 样本量必须足够（至少 100 个请求）

#### 代码实现

```python
# src/canary_automation.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from collections import defaultdict

class CanaryAutomation:
    """Canary Release 自动化管理器"""
    
    def __init__(self):
        self.stages = [
            {"percentage": 1, "duration_minutes": 60, "min_samples": 100, "name": "Stage 1: 1%"},
            {"percentage": 5, "duration_minutes": 120, "min_samples": 500, "name": "Stage 2: 5%"},
            {"percentage": 25, "duration_minutes": 240, "min_samples": 2500, "name": "Stage 3: 25%"},
            {"percentage": 100, "duration_minutes": 0, "min_samples": 0, "name": "Stage 4: 100%"}
        ]
        self.current_stage = 0
        self.start_time = None
        self.metrics_history = []  # 存储历史指标
        self.request_count = 0  # 当前阶段的请求数
    
    def record_request(self, trace_id: str, metrics: Dict):
        """记录请求和指标"""
        self.request_count += 1
        self.metrics_history.append({
            "trace_id": trace_id,
            "timestamp": datetime.now(),
            "metrics": metrics
        })
    
    def should_advance_stage(self) -> tuple[bool, str]:
        """
        判断是否应该进入下一阶段
        
        Returns:
            (should_advance, reason)
        """
        if self.current_stage >= len(self.stages) - 1:
            return False, "已经是最后阶段"
        
        stage = self.stages[self.current_stage]
        
        # 1. 检查观察时间
        if self.start_time is None:
            self.start_time = datetime.now()
            return False, "阶段刚开始，等待观察"
        
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_minutes < stage['duration_minutes']:
            return False, f"观察时间不足：{elapsed_minutes:.1f}/{stage['duration_minutes']} 分钟"
        
        # 2. 检查样本量
        if self.request_count < stage['min_samples']:
            return False, f"样本量不足：{self.request_count}/{stage['min_samples']} 个请求"
        
        # 3. 检查指标是否正常
        recent_metrics = self.get_recent_metrics(minutes=stage['duration_minutes'])
        is_healthy, reason = self.check_metrics_health(recent_metrics)
        
        if not is_healthy:
            return False, f"指标异常：{reason}"
        
        return True, "所有条件满足，可以进入下一阶段"
    
    def get_recent_metrics(self, minutes: int) -> List[Dict]:
        """获取最近 N 分钟的指标"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_history
            if m['timestamp'] >= cutoff_time
        ]
    
    def check_metrics_health(self, metrics_list: List[Dict]) -> tuple[bool, str]:
        """
        检查指标是否健康
        
        Returns:
            (is_healthy, reason)
        """
        if not metrics_list:
            return False, "没有指标数据"
        
        # 聚合指标
        aggregated = self.aggregate_metrics(metrics_list)
        
        # 检查 L0 指标（关键指标）
        # 1. Schema Pass Rate
        schema_pass_rate = aggregated.get('schema_pass_rate', 1.0)
        if schema_pass_rate < 0.95:
            return False, f"Schema Pass Rate 过低：{schema_pass_rate:.2%} < 95%"
        
        # 2. p95 Latency
        p95_latency = aggregated.get('p95_latency_seconds', 0)
        if p95_latency > 15.0:
            return False, f"p95 Latency 过高：{p95_latency:.1f}s > 15s"
        
        # 3. Error Rate
        error_rate = aggregated.get('error_rate', 0)
        if error_rate > 0.05:
            return False, f"Error Rate 过高：{error_rate:.2%} > 5%"
        
        return True, "所有指标正常"
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        # 计算 Schema Pass Rate
        total_requests = len(metrics_list)
        passed_requests = sum(
            1 for m in metrics_list
            if m['metrics'].get('schema_pass', False)
        )
        schema_pass_rate = passed_requests / total_requests if total_requests > 0 else 1.0
        
        # 计算 Latency（p95）
        latencies = [
            m['metrics'].get('latency_ms', 0) / 1000.0
            for m in metrics_list
        ]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0
        
        # 计算 Error Rate
        error_count = sum(
            1 for m in metrics_list
            if m['metrics'].get('error', False)
        )
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        return {
            "schema_pass_rate": schema_pass_rate,
            "p95_latency_seconds": p95_latency,
            "error_rate": error_rate,
            "total_requests": total_requests
        }
    
    def advance_stage(self):
        """进入下一阶段"""
        self.current_stage += 1
        self.start_time = datetime.now()
        self.request_count = 0
        self.metrics_history = []  # 清空历史，开始新阶段
        print(f"✅ 进入 {self.stages[self.current_stage]['name']}")
    
    def rollback(self, reason: str):
        """回滚到旧版本"""
        self.current_stage = 0
        self.start_time = None
        self.request_count = 0
        self.metrics_history = []
        print(f"❌ 回滚到旧版本：{reason}")
```

---

## 自动化回滚机制

### 1. 回滚条件设计

**原则**：
- ✅ 任何 L0 指标异常，立即回滚
- ✅ 连续多次检查异常，才触发回滚（避免误报）
- ✅ 回滚后验证系统是否恢复正常

#### 代码实现

```python
# src/canary_rollback.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

class CanaryRollbackAutomation:
    """Canary Release 自动化回滚管理器"""
    
    def __init__(self):
        self.rollback_thresholds = {
            "schema_pass_rate": 0.95,  # < 95% 触发回滚
            "p95_latency_seconds": 15.0,  # > 15s 触发回滚
            "error_rate": 0.05,  # > 5% 触发回滚
        }
        self.consecutive_failures = 0  # 连续失败次数
        self.consecutive_failure_threshold = 3  # 连续 3 次失败才回滚
        self.check_interval_seconds = 60  # 每 60 秒检查一次
        self.last_check_time = None
        self.metrics_buffer = deque(maxlen=100)  # 存储最近 100 个指标
    
    def record_metrics(self, metrics: Dict):
        """记录指标"""
        self.metrics_buffer.append({
            "timestamp": datetime.now(),
            "metrics": metrics
        })
    
    def should_rollback(self) -> tuple[bool, str]:
        """
        判断是否应该回滚
        
        Returns:
            (should_rollback, reason)
        """
        # 检查是否到了检查时间
        if self.last_check_time is None:
            self.last_check_time = datetime.now()
            return False, "首次检查，等待数据"
        
        elapsed_seconds = (datetime.now() - self.last_check_time).total_seconds()
        if elapsed_seconds < self.check_interval_seconds:
            return False, f"距离上次检查时间不足：{elapsed_seconds:.0f}s < {self.check_interval_seconds}s"
        
        # 获取最近的指标
        recent_metrics = self.get_recent_metrics(minutes=5)
        if not recent_metrics:
            return False, "没有足够的指标数据"
        
        # 聚合指标
        aggregated = self.aggregate_metrics(recent_metrics)
        
        # 检查是否异常
        is_unhealthy, reason = self.check_if_unhealthy(aggregated)
        
        if is_unhealthy:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.consecutive_failure_threshold:
                return True, f"连续 {self.consecutive_failures} 次检查异常：{reason}"
            else:
                return False, f"指标异常但未达到阈值：{reason}（连续失败 {self.consecutive_failures}/{self.consecutive_failure_threshold}）"
        else:
            # 指标正常，重置连续失败计数
            self.consecutive_failures = 0
            self.last_check_time = datetime.now()
            return False, "指标正常"
    
    def get_recent_metrics(self, minutes: int) -> List[Dict]:
        """获取最近 N 分钟的指标"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_buffer
            if m['timestamp'] >= cutoff_time
        ]
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """聚合指标（与 CanaryAutomation 相同）"""
        if not metrics_list:
            return {}
        
        total_requests = len(metrics_list)
        passed_requests = sum(
            1 for m in metrics_list
            if m['metrics'].get('schema_pass', False)
        )
        schema_pass_rate = passed_requests / total_requests if total_requests > 0 else 1.0
        
        latencies = [
            m['metrics'].get('latency_ms', 0) / 1000.0
            for m in metrics_list
        ]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0
        
        error_count = sum(
            1 for m in metrics_list
            if m['metrics'].get('error', False)
        )
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        return {
            "schema_pass_rate": schema_pass_rate,
            "p95_latency_seconds": p95_latency,
            "error_rate": error_rate,
            "total_requests": total_requests
        }
    
    def check_if_unhealthy(self, aggregated: Dict) -> tuple[bool, str]:
        """
        检查指标是否异常
        
        Returns:
            (is_unhealthy, reason)
        """
        # 1. Schema Pass Rate
        schema_pass_rate = aggregated.get('schema_pass_rate', 1.0)
        if schema_pass_rate < self.rollback_thresholds['schema_pass_rate']:
            return True, f"Schema Pass Rate 过低：{schema_pass_rate:.2%} < {self.rollback_thresholds['schema_pass_rate']:.2%}"
        
        # 2. p95 Latency
        p95_latency = aggregated.get('p95_latency_seconds', 0)
        if p95_latency > self.rollback_thresholds['p95_latency_seconds']:
            return True, f"p95 Latency 过高：{p95_latency:.1f}s > {self.rollback_thresholds['p95_latency_seconds']:.1f}s"
        
        # 3. Error Rate
        error_rate = aggregated.get('error_rate', 0)
        if error_rate > self.rollback_thresholds['error_rate']:
            return True, f"Error Rate 过高：{error_rate:.2%} > {self.rollback_thresholds['error_rate']:.2%}"
        
        return False, "指标正常"
    
    def reset(self):
        """重置状态（回滚后调用）"""
        self.consecutive_failures = 0
        self.last_check_time = None
        self.metrics_buffer.clear()
```

---

## 监控 Dashboard 设计

### 1. Dashboard 指标展示

#### 1.1 实时指标面板

```python
# src/canary_dashboard.py
from typing import Dict, List
from datetime import datetime
import json

class CanaryDashboard:
    """Canary Release 监控 Dashboard"""
    
    def __init__(self, automation: CanaryAutomation, rollback: CanaryRollbackAutomation):
        self.automation = automation
        self.rollback = rollback
    
    def get_dashboard_data(self) -> Dict:
        """获取 Dashboard 数据"""
        current_stage = self.automation.stages[self.automation.current_stage]
        
        # 获取当前阶段的指标
        recent_metrics = self.automation.get_recent_metrics(
            minutes=current_stage['duration_minutes'] if current_stage['duration_minutes'] > 0 else 60
        )
        aggregated = self.automation.aggregate_metrics(recent_metrics)
        
        # 检查是否可以推进
        can_advance, advance_reason = self.automation.should_advance_stage()
        
        # 检查是否需要回滚
        should_rollback, rollback_reason = self.rollback.should_rollback()
        
        return {
            "current_stage": {
                "name": current_stage['name'],
                "percentage": current_stage['percentage'],
                "duration_minutes": current_stage['duration_minutes'],
                "min_samples": current_stage['min_samples'],
                "start_time": self.automation.start_time.isoformat() if self.automation.start_time else None,
                "elapsed_minutes": (
                    (datetime.now() - self.automation.start_time).total_seconds() / 60
                    if self.automation.start_time else 0
                ),
                "request_count": self.automation.request_count
            },
            "metrics": {
                "schema_pass_rate": aggregated.get('schema_pass_rate', 0),
                "p95_latency_seconds": aggregated.get('p95_latency_seconds', 0),
                "error_rate": aggregated.get('error_rate', 0),
                "total_requests": aggregated.get('total_requests', 0)
            },
            "status": {
                "can_advance": can_advance,
                "advance_reason": advance_reason,
                "should_rollback": should_rollback,
                "rollback_reason": rollback_reason,
                "health_status": "healthy" if not should_rollback else "unhealthy"
            },
            "thresholds": {
                "schema_pass_rate": 0.95,
                "p95_latency_seconds": 15.0,
                "error_rate": 0.05
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """生成监控报告"""
        data = self.get_dashboard_data()
        
        report = f"""
# Canary Release 监控报告

## 当前阶段
- **阶段**: {data['current_stage']['name']}
- **流量百分比**: {data['current_stage']['percentage']}%
- **观察时间**: {data['current_stage']['elapsed_minutes']:.1f} / {data['current_stage']['duration_minutes']} 分钟
- **请求数**: {data['current_stage']['request_count']} / {data['current_stage']['min_samples']}

## 指标状态
- **Schema Pass Rate**: {data['metrics']['schema_pass_rate']:.2%} (阈值: {data['thresholds']['schema_pass_rate']:.2%})
- **p95 Latency**: {data['metrics']['p95_latency_seconds']:.1f}s (阈值: {data['thresholds']['p95_latency_seconds']:.1f}s)
- **Error Rate**: {data['metrics']['error_rate']:.2%} (阈值: {data['thresholds']['error_rate']:.2%})

## 状态
- **健康状态**: {data['status']['health_status']}
- **可以推进**: {data['status']['can_advance']} ({data['status']['advance_reason']})
- **需要回滚**: {data['status']['should_rollback']} ({data['status']['rollback_reason']})

## 时间戳
- **报告生成时间**: {data['timestamp']}
"""
        return report
```

---

## KYC 项目实际案例

### 案例 1：自动化 Canary Release

#### 配置

```yaml
# config/canary_config.yaml
canary_release:
  stages:
    - percentage: 1
      duration_minutes: 60
      min_samples: 100
    - percentage: 5
      duration_minutes: 120
      min_samples: 500
    - percentage: 25
      duration_minutes: 240
      min_samples: 2500
    - percentage: 100
      duration_minutes: 0
      min_samples: 0
  
  rollback:
    check_interval_seconds: 60
    consecutive_failure_threshold: 3
    thresholds:
      schema_pass_rate: 0.95
      p95_latency_seconds: 15.0
      error_rate: 0.05
```

#### 使用示例

```python
# src/kyc_canary_service.py
from canary_automation import CanaryAutomation
from canary_rollback import CanaryRollbackAutomation
from canary_dashboard import CanaryDashboard

class KYCCanaryService:
    """KYC Canary Release 服务"""
    
    def __init__(self):
        self.automation = CanaryAutomation()
        self.rollback = CanaryRollbackAutomation()
        self.dashboard = CanaryDashboard(self.automation, self.rollback)
    
    def process_request(self, request: Dict, trace_id: str):
        """处理请求并记录指标"""
        # 1. 处理请求
        result = self.run_kyc_pipeline(request)
        
        # 2. 记录指标
        metrics = {
            "schema_pass": result.get('schema_pass', False),
            "latency_ms": result.get('latency_ms', 0),
            "error": result.get('error', False)
        }
        self.automation.record_request(trace_id, metrics)
        self.rollback.record_metrics(metrics)
        
        # 3. 检查是否需要回滚
        should_rollback, reason = self.rollback.should_rollback()
        if should_rollback:
            self.automation.rollback(reason)
            # 触发回滚通知
            self.notify_rollback(reason)
        
        # 4. 检查是否可以推进（定期检查，比如每 5 分钟）
        can_advance, advance_reason = self.automation.should_advance_stage()
        if can_advance:
            self.automation.advance_stage()
            # 触发推进通知
            self.notify_advance(advance_reason)
        
        return result
    
    def get_dashboard(self) -> Dict:
        """获取 Dashboard 数据"""
        return self.dashboard.get_dashboard_data()
    
    def get_report(self) -> str:
        """获取监控报告"""
        return self.dashboard.generate_report()
    
    def notify_rollback(self, reason: str):
        """通知回滚"""
        print(f"🚨 回滚通知: {reason}")
        # 实际应该发送到 Slack、邮件等
    
    def notify_advance(self, reason: str):
        """通知推进"""
        print(f"✅ 推进通知: {reason}")
        # 实际应该发送到 Slack、邮件等
```

---

## 相关文档

- [KYC_Day04_A1_发布策略与回滚详解.md](./KYC_Day04_A1_发布策略与回滚详解.md) - Canary Release 基础概念
- [KYC_Day04_A1_B3_Rollback自动化详解.md](./KYC_Day04_A1_B3_Rollback自动化详解.md) - Rollback 自动化实现
- [KYC_Day02_A1_可观测性详解.md](../day02/KYC_Day02_A1_可观测性详解.md) - Metrics、Logs、Traces

---

## 总结

### 核心要点

1. **监控指标设计**：
   - L0 指标（稳定性）：Schema Pass Rate、p95 Latency、Error Rate
   - L1 指标（业务）：字段级准确率、Cost per Request
   - L2 指标（长期健康）：Fallback Rate

2. **自动化推进机制**：
   - 检查观察时间
   - 检查样本量
   - 检查指标是否正常

3. **自动化回滚机制**：
   - 连续多次检查异常才回滚（避免误报）
   - 回滚后验证系统是否恢复正常

4. **监控 Dashboard**：
   - 实时指标展示
   - 状态监控
   - 报告生成
