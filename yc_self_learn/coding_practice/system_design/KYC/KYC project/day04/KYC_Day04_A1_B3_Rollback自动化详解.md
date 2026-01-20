# KYC_Day04_A1_B3: Rollback 自动化详解

## 📋 目录
1. [核心问题：如何自动化 Rollback？](#核心问题如何自动化-rollback)
2. [自动化检测机制](#自动化检测机制)
3. [自动化回滚执行](#自动化回滚执行)
4. [回滚验证机制](#回滚验证机制)
5. [Rollback 历史记录](#rollback-历史记录)
6. [KYC 项目实际案例](#kyc-项目实际案例)

---

## 核心问题：如何自动化 Rollback？

### 问题场景

**需求**：
- 如何自动检测异常并触发回滚？
- 如何自动执行回滚操作？
- 如何自动验证回滚是否成功？

**挑战**：
- 需要实时监控指标
- 需要自动判断回滚条件
- 需要自动执行回滚并验证

---

## 自动化检测机制

### 1. 回滚触发条件

#### 1.1 KYC 项目回滚条件

| 条件 | 阈值 | 优先级 | 动作 |
|------|------|--------|------|
| **Schema Fail Rate × 2** | 旧版本 2% → 新版本 4% | 🔴 P0 | 立即回滚 |
| **p95 Latency + 20%** | 旧版本 10s → 新版本 12s | 🔴 P0 | 立即回滚 |
| **Error Rate > 5%** | 错误率 > 5% | 🔴 P0 | 立即回滚 |
| **Cost per Request + 50%** | 成本增加 > 50% | 🟡 P1 | 观察，不立即回滚 |

#### 1.2 代码实现

```python
# src/rollback_detector.py
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import time

class RollbackDetector:
    """回滚检测器 - 自动检测异常并触发回滚"""
    
    def __init__(self):
        self.rollback_conditions = {
            "schema_fail_rate_multiplier": 2.0,  # Schema Fail Rate × 2
            "p95_latency_increase_percent": 20,  # p95 Latency + 20%
            "error_rate_threshold": 0.05,  # Error Rate > 5%
            "cost_increase_percent": 50  # Cost + 50%（观察，不立即回滚）
        }
        self.baseline_metrics = None  # 旧版本的基准指标
        self.metrics_buffer = deque(maxlen=1000)  # 存储最近 1000 个指标
        self.check_interval_seconds = 30  # 每 30 秒检查一次
        self.last_check_time = None
        self.consecutive_failures = 0  # 连续失败次数
        self.consecutive_failure_threshold = 3  # 连续 3 次失败才回滚
    
    def set_baseline(self, metrics: Dict):
        """设置基准指标（旧版本的指标）"""
        self.baseline_metrics = metrics
        print(f"✅ 设置基准指标: {metrics}")
    
    def record_metrics(self, metrics: Dict):
        """记录指标"""
        self.metrics_buffer.append({
            "timestamp": datetime.now(),
            "metrics": metrics
        })
    
    def should_rollback(self) -> Tuple[bool, str]:
        """
        判断是否应该回滚
        
        Returns:
            (should_rollback, reason)
        """
        if self.baseline_metrics is None:
            return False, "未设置基准指标"
        
        # 检查是否到了检查时间
        if self.last_check_time is None:
            self.last_check_time = datetime.now()
            return False, "首次检查，等待数据"
        
        elapsed_seconds = (datetime.now() - self.last_check_time).total_seconds()
        if elapsed_seconds < self.check_interval_seconds:
            return False, f"距离上次检查时间不足：{elapsed_seconds:.0f}s < {self.check_interval_seconds}s"
        
        # 获取最近的指标（最近 5 分钟）
        recent_metrics = self.get_recent_metrics(minutes=5)
        if not recent_metrics:
            return False, "没有足够的指标数据"
        
        # 聚合当前指标
        current_metrics = self.aggregate_metrics(recent_metrics)
        
        # 检查回滚条件
        is_unhealthy, reason = self.check_rollback_conditions(current_metrics)
        
        if is_unhealthy:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.consecutive_failure_threshold:
                self.last_check_time = datetime.now()
                return True, f"连续 {self.consecutive_failures} 次检查异常：{reason}"
            else:
                return False, f"指标异常但未达到阈值：{reason}（连续失败 {self.consecutive_failures}/{self.consecutive_failure_threshold}）"
        else:
            # 指标正常，重置连续失败计数
            self.consecutive_failures = 0
            self.last_check_time = datetime.now()
            return False, "指标正常"
    
    def get_recent_metrics(self, minutes: int) -> list:
        """获取最近 N 分钟的指标"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_buffer
            if m['timestamp'] >= cutoff_time
        ]
    
    def aggregate_metrics(self, metrics_list: list) -> Dict:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        total_requests = len(metrics_list)
        
        # Schema Pass Rate
        passed_requests = sum(
            1 for m in metrics_list
            if m['metrics'].get('schema_pass', False)
        )
        schema_pass_rate = passed_requests / total_requests if total_requests > 0 else 1.0
        
        # p95 Latency
        latencies = [
            m['metrics'].get('latency_ms', 0) / 1000.0
            for m in metrics_list
        ]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0
        
        # Error Rate
        error_count = sum(
            1 for m in metrics_list
            if m['metrics'].get('error', False)
        )
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        # Cost per Request（如果有）
        total_cost = sum(
            m['metrics'].get('cost', 0)
            for m in metrics_list
        )
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        
        return {
            "schema_pass_rate": schema_pass_rate,
            "schema_fail_rate": 1 - schema_pass_rate,
            "p95_latency_seconds": p95_latency,
            "error_rate": error_rate,
            "cost_per_request": cost_per_request,
            "total_requests": total_requests
        }
    
    def check_rollback_conditions(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        检查回滚条件
        
        Returns:
            (is_unhealthy, reason)
        """
        baseline = self.baseline_metrics
        
        # 1. 检查 Schema Fail Rate × 2
        baseline_schema_fail_rate = baseline.get('schema_fail_rate', 0)
        current_schema_fail_rate = current_metrics.get('schema_fail_rate', 0)
        
        if baseline_schema_fail_rate > 0:
            multiplier = current_schema_fail_rate / baseline_schema_fail_rate
            if multiplier >= self.rollback_conditions['schema_fail_rate_multiplier']:
                return True, (
                    f"Schema Fail Rate 超过阈值："
                    f"{current_schema_fail_rate:.2%} >= {baseline_schema_fail_rate * self.rollback_conditions['schema_fail_rate_multiplier']:.2%} "
                    f"(倍数: {multiplier:.2f}x)"
                )
        
        # 2. 检查 p95 Latency + 20%
        baseline_p95 = baseline.get('p95_latency_seconds', 0)
        current_p95 = current_metrics.get('p95_latency_seconds', 0)
        
        if baseline_p95 > 0:
            latency_increase_percent = ((current_p95 - baseline_p95) / baseline_p95) * 100
            if latency_increase_percent >= self.rollback_conditions['p95_latency_increase_percent']:
                return True, (
                    f"p95 Latency 增加超过阈值："
                    f"{latency_increase_percent:.1f}% >= {self.rollback_conditions['p95_latency_increase_percent']}% "
                    f"({baseline_p95:.1f}s → {current_p95:.1f}s)"
                )
        
        # 3. 检查 Error Rate > 5%
        current_error_rate = current_metrics.get('error_rate', 0)
        if current_error_rate > self.rollback_conditions['error_rate_threshold']:
            return True, (
                f"Error Rate 超过阈值："
                f"{current_error_rate:.2%} > {self.rollback_conditions['error_rate_threshold']:.2%}"
            )
        
        return False, "所有指标正常"
    
    def reset(self):
        """重置状态（回滚后调用）"""
        self.consecutive_failures = 0
        self.last_check_time = None
        self.metrics_buffer.clear()
```

---

## 自动化回滚执行

### 1. 回滚执行器

#### 代码实现

```python
# src/rollback_executor.py
from typing import Dict, Optional
from datetime import datetime
import json
import time

class RollbackExecutor:
    """回滚执行器 - 自动执行回滚操作"""
    
    def __init__(self, feature_flag_manager, canary_manager, rollback_detector):
        self.feature_flag_manager = feature_flag_manager
        self.canary_manager = canary_manager
        self.rollback_detector = rollback_detector
        self.rollback_history = []
    
    def execute_rollback(self, reason: str, metrics_before: Dict) -> Dict:
        """
        执行回滚
        
        Returns:
            回滚事件信息
        """
        rollback_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "metrics_before_rollback": metrics_before,
            "actions": [],
            "status": "pending"
        }
        
        try:
            # 1. 关闭新版本的 Feature Flag
            self.feature_flag_manager.disable_new_version()
            rollback_event["actions"].append("关闭新版本 Feature Flag")
            
            # 2. 重置 Canary Release
            self.canary_manager.rollback()
            rollback_event["actions"].append("重置 Canary Release")
            
            # 3. 重置回滚检测器
            self.rollback_detector.reset()
            rollback_event["actions"].append("重置回滚检测器")
            
            # 4. 等待系统稳定（等待 1 分钟）
            print("⏳ 等待系统稳定...")
            time.sleep(60)
            
            # 5. 验证回滚效果
            post_rollback_metrics = self.get_current_metrics()
            rollback_event["metrics_after_rollback"] = post_rollback_metrics
            
            # 6. 检查回滚是否成功
            is_success = self.verify_rollback(post_rollback_metrics)
            rollback_event["status"] = "success" if is_success else "failed"
            
            if is_success:
                print("✅ 回滚成功，系统已恢复正常")
            else:
                print("⚠️ 回滚后系统仍未恢复正常，需要人工介入")
            
            # 7. 记录回滚事件
            self.rollback_history.append(rollback_event)
            self.save_rollback_history()
            
            # 8. 发送通知
            self.notify_rollback(rollback_event)
            
            return rollback_event
            
        except Exception as e:
            rollback_event["status"] = "error"
            rollback_event["error"] = str(e)
            print(f"❌ 回滚执行失败: {e}")
            self.rollback_history.append(rollback_event)
            self.save_rollback_history()
            raise
    
    def verify_rollback(self, metrics: Dict) -> bool:
        """
        验证回滚是否成功
        
        Returns:
            True if rollback successful, False otherwise
        """
        # 检查指标是否恢复正常
        should_rollback, reason = self.rollback_detector.should_rollback()
        return not should_rollback
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标（模拟）"""
        # 实际应该从监控系统获取
        # 这里返回模拟数据
        return {
            "schema_pass_rate": 0.98,
            "p95_latency_seconds": 10.0,
            "error_rate": 0.01,
            "total_requests": 100
        }
    
    def save_rollback_history(self):
        """保存回滚历史"""
        with open("rollback_history.json", "w") as f:
            json.dump(self.rollback_history, f, indent=2)
    
    def notify_rollback(self, rollback_event: Dict):
        """通知回滚"""
        print(f"🚨 回滚通知:")
        print(f"   时间: {rollback_event['timestamp']}")
        print(f"   原因: {rollback_event['reason']}")
        print(f"   状态: {rollback_event['status']}")
        # 实际应该发送到 Slack、邮件等
```

---

## 回滚验证机制

### 1. 验证策略

#### 1.1 验证步骤

1. **立即验证**（回滚后 1 分钟）
   - 检查指标是否恢复正常
   - 检查系统是否稳定

2. **短期验证**（回滚后 5 分钟）
   - 检查指标趋势
   - 确认没有新的异常

3. **长期验证**（回滚后 30 分钟）
   - 确认系统完全稳定
   - 记录回滚效果

#### 1.2 代码实现

```python
# src/rollback_verifier.py
from typing import Dict, List
from datetime import datetime, timedelta
import time

class RollbackVerifier:
    """回滚验证器 - 自动验证回滚是否成功"""
    
    def __init__(self, rollback_detector):
        self.rollback_detector = rollback_detector
        self.verification_stages = [
            {"name": "立即验证", "wait_seconds": 60, "check_duration_minutes": 1},
            {"name": "短期验证", "wait_seconds": 300, "check_duration_minutes": 5},
            {"name": "长期验证", "wait_seconds": 1800, "check_duration_minutes": 30}
        ]
    
    def verify_rollback(self, rollback_timestamp: datetime) -> Dict:
        """
        验证回滚是否成功
        
        Returns:
            验证结果
        """
        verification_result = {
            "rollback_timestamp": rollback_timestamp.isoformat(),
            "stages": [],
            "overall_status": "pending"
        }
        
        for stage in self.verification_stages:
            print(f"⏳ {stage['name']}: 等待 {stage['wait_seconds']} 秒...")
            time.sleep(stage['wait_seconds'])
            
            # 获取最近 N 分钟的指标
            recent_metrics = self.rollback_detector.get_recent_metrics(
                minutes=stage['check_duration_minutes']
            )
            
            if not recent_metrics:
                verification_result["stages"].append({
                    "name": stage['name'],
                    "status": "insufficient_data",
                    "reason": "没有足够的指标数据"
                })
                continue
            
            # 聚合指标
            aggregated = self.rollback_detector.aggregate_metrics(recent_metrics)
            
            # 检查指标是否正常
            is_unhealthy, reason = self.rollback_detector.check_rollback_conditions(aggregated)
            
            stage_result = {
                "name": stage['name'],
                "status": "healthy" if not is_unhealthy else "unhealthy",
                "reason": reason if is_unhealthy else "指标正常",
                "metrics": aggregated,
                "timestamp": datetime.now().isoformat()
            }
            
            verification_result["stages"].append(stage_result)
            
            if is_unhealthy:
                print(f"⚠️ {stage['name']}失败: {reason}")
                verification_result["overall_status"] = "failed"
                return verification_result
            else:
                print(f"✅ {stage['name']}通过")
        
        # 所有阶段都通过
        verification_result["overall_status"] = "success"
        return verification_result
```

---

## Rollback 历史记录

### 1. 历史记录设计

#### 代码实现

```python
# src/rollback_history.py
from typing import Dict, List
from datetime import datetime
import json

class RollbackHistory:
    """回滚历史记录管理器"""
    
    def __init__(self, history_file: str = "rollback_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """加载历史记录"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_history(self):
        """保存历史记录"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_rollback_event(self, event: Dict):
        """添加回滚事件"""
        self.history.append(event)
        self.save_history()
    
    def get_recent_rollbacks(self, days: int = 7) -> List[Dict]:
        """获取最近 N 天的回滚记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            event for event in self.history
            if datetime.fromisoformat(event['timestamp']) >= cutoff_date
        ]
    
    def generate_report(self) -> str:
        """生成回滚报告"""
        recent_rollbacks = self.get_recent_rollbacks(days=30)
        
        if not recent_rollbacks:
            return "最近 30 天没有回滚记录"
        
        total_rollbacks = len(recent_rollbacks)
        successful_rollbacks = sum(
            1 for r in recent_rollbacks
            if r.get('status') == 'success'
        )
        failed_rollbacks = total_rollbacks - successful_rollbacks
        
        report = f"""
# Rollback 历史报告

## 统计信息
- **总回滚次数**: {total_rollbacks}
- **成功回滚**: {successful_rollbacks} ({successful_rollbacks/total_rollbacks*100:.1f}%)
- **失败回滚**: {failed_rollbacks} ({failed_rollbacks/total_rollbacks*100:.1f}%)

## 最近回滚记录
"""
        for event in recent_rollbacks[-10:]:  # 最近 10 次
            report += f"""
### {event['timestamp']}
- **原因**: {event['reason']}
- **状态**: {event['status']}
- **操作**: {', '.join(event.get('actions', []))}
"""
        
        return report
```

---

## KYC 项目实际案例

### 案例 1：自动化 Rollback 流程

#### 配置

```yaml
# config/rollback_config.yaml
rollback:
  detection:
    check_interval_seconds: 30
    consecutive_failure_threshold: 3
    thresholds:
      schema_fail_rate_multiplier: 2.0
      p95_latency_increase_percent: 20
      error_rate_threshold: 0.05
  
  verification:
    stages:
      - name: "立即验证"
        wait_seconds: 60
        check_duration_minutes: 1
      - name: "短期验证"
        wait_seconds: 300
        check_duration_minutes: 5
      - name: "长期验证"
        wait_seconds: 1800
        check_duration_minutes: 30
```

#### 使用示例

```python
# src/kyc_rollback_service.py
from rollback_detector import RollbackDetector
from rollback_executor import RollbackExecutor
from rollback_verifier import RollbackVerifier
from rollback_history import RollbackHistory

class KYCRollbackService:
    """KYC Rollback 服务"""
    
    def __init__(self, feature_flag_manager, canary_manager):
        self.detector = RollbackDetector()
        self.executor = RollbackExecutor(
            feature_flag_manager, canary_manager, self.detector
        )
        self.verifier = RollbackVerifier(self.detector)
        self.history = RollbackHistory()
    
    def process_request(self, request: Dict, trace_id: str):
        """处理请求并记录指标"""
        # 1. 处理请求
        result = self.run_kyc_pipeline(request)
        
        # 2. 记录指标
        metrics = {
            "schema_pass": result.get('schema_pass', False),
            "latency_ms": result.get('latency_ms', 0),
            "error": result.get('error', False),
            "cost": result.get('cost', 0)
        }
        self.detector.record_metrics(metrics)
        
        # 3. 检查是否需要回滚
        should_rollback, reason = self.detector.should_rollback()
        if should_rollback:
            # 获取当前指标
            current_metrics = self.detector.aggregate_metrics(
                self.detector.get_recent_metrics(minutes=5)
            )
            
            # 执行回滚
            rollback_event = self.executor.execute_rollback(reason, current_metrics)
            
            # 验证回滚
            verification_result = self.verifier.verify_rollback(
                datetime.fromisoformat(rollback_event['timestamp'])
            )
            rollback_event['verification'] = verification_result
            
            # 记录历史
            self.history.add_rollback_event(rollback_event)
        
        return result
    
    def set_baseline(self, metrics: Dict):
        """设置基准指标"""
        self.detector.set_baseline(metrics)
    
    def get_history_report(self) -> str:
        """获取历史报告"""
        return self.history.generate_report()
```

---

## 相关文档

- [KYC_Day04_A1_发布策略与回滚详解.md](./KYC_Day04_A1_发布策略与回滚详解.md) - Rollback 基础概念
- [KYC_Day04_A1_B2_Canary_Release监控详解.md](./KYC_Day04_A1_B2_Canary_Release监控详解.md) - Canary Release 监控
- [KYC_Day02_A1_可观测性详解.md](../day02/KYC_Day02_A1_可观测性详解.md) - Metrics、Logs、Traces

---

## 总结

### 核心要点

1. **自动化检测机制**：
   - 实时监控指标
   - 连续多次检查异常才回滚（避免误报）
   - 检查 Schema Fail Rate、p95 Latency、Error Rate

2. **自动化回滚执行**：
   - 关闭新版本 Feature Flag
   - 重置 Canary Release
   - 重置回滚检测器

3. **回滚验证机制**：
   - 立即验证（1 分钟）
   - 短期验证（5 分钟）
   - 长期验证（30 分钟）

4. **Rollback 历史记录**：
   - 记录所有回滚事件
   - 生成回滚报告
   - 分析回滚趋势
