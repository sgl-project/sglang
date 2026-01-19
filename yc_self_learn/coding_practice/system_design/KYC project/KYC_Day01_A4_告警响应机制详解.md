# Day 1 补充｜告警触发后的响应机制：从 Alert 到 Recovery

**目标**：理解告警触发后，系统应该做什么？是发送错误消息、自动修复、降级、跳过还是其他？

---

## 🎯 核心概念：告警响应层次（Alert Response Hierarchy）

### 告警不是"通知"，而是"触发动作"

```
告警触发 → 判断严重性 → 选择响应策略 → 执行动作 → 验证恢复
```

**关键点**：
- ❌ **不是**：告警 = 发送错误消息（这只是第一步）
- ✅ **而是**：告警 = 触发**自动化响应机制**（这才是 Senior 的设计）

---

## 📊 第一部分：告警严重性分级（Severity Levels）

### 告警分级标准

| 严重性 | 触发条件 | 响应时间 | 响应策略 |
|--------|---------|---------|---------|
| **Info** | 指标轻微波动 | 1小时内 | 记录日志，无需立即处理 |
| **Warning** | 指标接近阈值 | 15分钟内 | 自动重试/降级，通知 On-Call |
| **Critical** | 指标严重超标 | 立即（< 5分钟） | 自动熔断/回滚，立即通知 |
| **Emergency** | 系统完全不可用 | 立即（< 1分钟） | 自动回滚，紧急通知，启动应急预案 |

---

## 🔄 第二部分：告警触发后的响应策略（Response Strategies）

### 策略 1：自动重试（Auto Retry）

**适用场景**：
- 临时性错误（网络波动、API 超时）
- 可恢复的错误（`API_TIMEOUT`, `API_CONNECTION_ERROR`）

**触发条件**：
- Success Rate < 98%（但 > 95%）
- 错误类型：`API_TIMEOUT` 或 `API_SERVER_ERROR`

**响应动作**：
```python
# 自动重试机制
def on_success_rate_warning(success_rate, error_breakdown):
    """Success Rate < 98% 时触发"""
    
    if success_rate < 0.98 and success_rate > 0.95:
        # 1. 分析错误类型
        timeout_errors = error_breakdown.get("API_TIMEOUT", 0)
        connection_errors = error_breakdown.get("API_CONNECTION_ERROR", 0)
        
        # 2. 如果是可恢复错误，自动重试失败的请求
        if timeout_errors > 0 or connection_errors > 0:
            # 自动重试失败的请求（指数退避）
            retry_failed_requests(
                max_retries=3,
                backoff_multiplier=2,
                initial_delay=1.0
            )
            
            # 3. 记录日志
            logger.warning(
                f"Auto-retry triggered: {timeout_errors} timeout errors, "
                f"{connection_errors} connection errors"
            )
            
            # 4. 通知 On-Call（但不紧急）
            send_notification(
                level="warning",
                message="Success rate below 98%, auto-retry in progress"
            )
```

**验证恢复**：
- 重试后，Success Rate 恢复到 > 98% → 告警解除
- 重试后，Success Rate 仍然 < 98% → 升级到 Critical

---

### 策略 2：自动降级（Auto Fallback）

**适用场景**：
- 主服务不可用或性能严重下降
- 有备用方案（如 OCR-only fallback）

**触发条件**：
- Success Rate < 95%
- p95 > 20s（严重延迟）
- Fireworks API 失败率 > 5%

**响应动作**：
```python
# 自动降级机制
def on_critical_alert(success_rate, p95, api_failure_rate):
    """Critical 告警触发自动降级"""
    
    if success_rate < 0.95 or p95 > 20 or api_failure_rate > 0.05:
        # 1. 触发熔断器（Circuit Breaker）
        circuit_breaker.open()  # 停止调用主服务
        
        # 2. 切换到降级方案
        fallback_strategy = determine_fallback_strategy()
        
        if fallback_strategy == "ocr_only":
            # OCR-only fallback（不调用 LLM）
            logger.warning("Switching to OCR-only fallback")
            switch_to_ocr_only_mode()
            
        elif fallback_strategy == "manual_review":
            # 转人工审核
            logger.warning("Switching to manual review queue")
            route_to_manual_review_queue()
        
        # 3. 通知 On-Call（紧急）
        send_notification(
            level="critical",
            message=f"Auto-fallback triggered: Success rate {success_rate*100:.2f}%, p95 {p95:.2f}s"
        )
        
        # 4. 记录降级事件（用于后续分析）
        record_fallback_event({
            "trigger_time": datetime.now(),
            "success_rate": success_rate,
            "p95": p95,
            "fallback_strategy": fallback_strategy
        })
```

**验证恢复**：
- 降级后，Success Rate 恢复到 > 95% → 继续使用降级方案
- 主服务恢复后，逐步切回主服务（Canary 发布）

---

### 策略 3：自动熔断（Circuit Breaker）

**适用场景**：
- 外部服务（Fireworks API）持续失败
- 防止雪崩效应（避免大量请求堆积）

**触发条件**：
- Fireworks API 失败率 > 5%（持续 1 分钟）
- 连续 10 个请求失败

**响应动作**：
```python
# 熔断器机制
class CircuitBreaker:
    def __init__(self):
        self.state = "closed"  # closed / open / half-open
        self.failure_count = 0
        self.last_failure_time = None
        self.failure_threshold = 10
        self.timeout = 60  # 60秒后尝试恢复
    
    def on_request_failure(self):
        """请求失败时调用"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            # 打开熔断器
            self.state = "open"
            logger.critical("Circuit breaker OPEN: Fireworks API failure rate too high")
            
            # 立即切换到降级方案
            switch_to_fallback()
            
            # 通知 On-Call
            send_notification(
                level="critical",
                message="Circuit breaker OPEN: Fireworks API unavailable"
            )
    
    def on_request_success(self):
        """请求成功时调用"""
        if self.state == "half-open":
            # 半开状态：如果成功，关闭熔断器
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker CLOSED: Fireworks API recovered")
    
    def should_attempt_request(self):
        """判断是否应该尝试请求"""
        if self.state == "open":
            # 检查是否超时（可以尝试恢复）
            if (datetime.now() - self.last_failure_time).seconds > self.timeout:
                self.state = "half-open"  # 进入半开状态，尝试恢复
                logger.info("Circuit breaker HALF-OPEN: Attempting recovery")
                return True
            return False  # 熔断器打开，不尝试请求
        return True  # 熔断器关闭，正常请求
```

**验证恢复**：
- 熔断器打开后，所有请求走降级方案
- 60 秒后，进入半开状态，尝试少量请求
- 如果成功，关闭熔断器，恢复正常

---

### 策略 4：自动回滚（Auto Rollback）

**适用场景**：
- 新版本发布后，指标严重下降
- 必须立即恢复到上一个稳定版本

**触发条件**：
- Success Rate < 90%（严重下降）
- Error Rate > 10%
- p95 > 30s（严重延迟）

**响应动作**：
```python
# 自动回滚机制
def on_emergency_alert(success_rate, error_rate, p95):
    """Emergency 告警触发自动回滚"""
    
    if success_rate < 0.90 or error_rate > 0.10 or p95 > 30:
        # 1. 立即停止新版本流量
        logger.critical("EMERGENCY: Stopping new version traffic")
        stop_new_version_traffic()
        
        # 2. 回滚到上一个稳定版本
        previous_version = get_last_stable_version()
        logger.critical(f"Rolling back to version {previous_version}")
        rollback_to_version(previous_version)
        
        # 3. 验证回滚成功
        wait_for_rollback_completion(timeout=300)  # 5分钟超时
        
        # 4. 验证指标恢复
        new_metrics = wait_for_metrics_stabilize(timeout=600)  # 10分钟
        
        if new_metrics["success_rate"] > 0.95:
            logger.info("Rollback successful: Metrics recovered")
        else:
            logger.critical("Rollback failed: Metrics still below threshold")
            # 启动应急预案（人工介入）
            trigger_emergency_procedure()
        
        # 5. 通知所有相关人员
        send_notification(
            level="emergency",
            message=f"Auto-rollback triggered: Success rate {success_rate*100:.2f}%"
        )
        
        # 6. 记录回滚事件（用于 Postmortem）
        record_rollback_event({
            "trigger_time": datetime.now(),
            "success_rate_before": success_rate,
            "success_rate_after": new_metrics["success_rate"],
            "rollback_version": previous_version
        })
```

**验证恢复**：
- 回滚后，等待 10 分钟，验证指标恢复
- 如果恢复，记录 Postmortem
- 如果未恢复，启动应急预案

---

### 策略 5：跳过/限流（Skip/Rate Limiting）

**适用场景**：
- 系统负载过高
- 需要保护核心功能

**触发条件**：
- RPS > 阈值（系统过载）
- p95 > 15s（延迟过高）

**响应动作**：
```python
# 限流机制
def on_overload_alert(rps, p95):
    """系统过载时触发限流"""
    
    if rps > MAX_RPS or p95 > 15:
        # 1. 降低请求速率
        rate_limiter.set_rate(MAX_RPS * 0.8)  # 降低到 80%
        
        # 2. 跳过非关键请求（如果定义了优先级）
        if has_priority_queue():
            skip_low_priority_requests()
        
        # 3. 返回 429 Too Many Requests
        logger.warning(f"Rate limiting: RPS {rps}, p95 {p95:.2f}s")
        
        # 4. 通知 On-Call
        send_notification(
            level="warning",
            message=f"Rate limiting active: RPS {rps}, p95 {p95:.2f}s"
        )
```

**验证恢复**：
- 限流后，系统负载降低
- p95 恢复到 < 15s → 逐步恢复限流阈值

---

## 🎯 第三部分：KYC 项目的完整响应流程

### 场景 1：Success Rate < 98%（Warning）

**触发条件**：
```
Success Rate = 97.5% (< 98% 阈值)
Error Breakdown: API_TIMEOUT = 2%, SCHEMA_VALIDATION_FAILED = 0.5%
```

**响应流程**：

```python
def handle_success_rate_warning(metrics):
    """处理 Success Rate Warning"""
    
    # Step 1: 分析错误类型
    error_breakdown = metrics["error_breakdown"]
    timeout_errors = error_breakdown.get("API_TIMEOUT", 0)
    
    # Step 2: 判断响应策略
    if timeout_errors > 0.01:  # 超过 1% 是超时错误
        # 策略：自动重试
        logger.warning("Triggering auto-retry for timeout errors")
        retry_failed_requests(
            error_types=["API_TIMEOUT"],
            max_retries=3,
            backoff_multiplier=2
        )
        
        # Step 3: 通知 On-Call（非紧急）
        send_notification(
            level="warning",
            message="Success rate below 98%, auto-retry in progress",
            action_required=False  # 不需要立即处理
        )
    
    else:
        # 策略：记录日志，等待观察
        logger.info("Success rate below 98%, monitoring...")
        # 不触发自动动作，只记录
```

**结果**：
- ✅ 自动重试后，Success Rate 恢复到 98.5% → 告警解除
- ❌ 重试后仍然 < 98% → 升级到 Critical

---

### 场景 2：Success Rate < 95%（Critical）

**触发条件**：
```
Success Rate = 93% (< 95% 阈值)
p95 = 18s (> 15s 阈值)
Fireworks API 失败率 = 6% (> 5% 阈值)
```

**响应流程**：

```python
def handle_critical_alert(metrics):
    """处理 Critical 告警"""
    
    # Step 1: 立即触发熔断器
    circuit_breaker.open()
    logger.critical("Circuit breaker OPEN: Critical alert triggered")
    
    # Step 2: 切换到降级方案
    if can_use_ocr_fallback():
        # 使用 OCR-only fallback
        switch_to_ocr_only_mode()
        logger.warning("Switched to OCR-only fallback")
    else:
        # 转人工审核队列
        route_to_manual_review_queue()
        logger.warning("Routed to manual review queue")
    
    # Step 3: 通知 On-Call（紧急）
    send_notification(
        level="critical",
        message=f"Critical alert: Success rate {metrics['success_rate']*100:.2f}%, "
                f"p95 {metrics['p95']:.2f}s, API failure rate {metrics['api_failure_rate']*100:.2f}%",
        action_required=True,  # 需要立即处理
        oncall_phone=True  # 电话通知
    )
    
    # Step 4: 记录事件
    record_critical_event(metrics)
```

**结果**：
- ✅ 降级后，Success Rate 恢复到 96% → 继续使用降级方案
- ❌ 降级后仍然 < 95% → 升级到 Emergency（考虑回滚）

---

### 场景 3：Success Rate < 90%（Emergency）

**触发条件**：
```
Success Rate = 85% (< 90% 阈值)
Error Rate = 15% (> 10% 阈值)
p95 = 35s (> 30s 阈值)
```

**响应流程**：

```python
def handle_emergency_alert(metrics):
    """处理 Emergency 告警"""
    
    # Step 1: 立即停止新版本流量
    stop_new_version_traffic()
    logger.critical("EMERGENCY: Stopped new version traffic")
    
    # Step 2: 检查是否有新版本发布（最近 1 小时）
    recent_releases = get_recent_releases(hours=1)
    if recent_releases:
        # 自动回滚到上一个稳定版本
        previous_version = get_last_stable_version()
        logger.critical(f"Auto-rollback to version {previous_version}")
        rollback_to_version(previous_version)
        
        # Step 3: 验证回滚成功
        wait_for_rollback_completion(timeout=300)
        new_metrics = wait_for_metrics_stabilize(timeout=600)
        
        if new_metrics["success_rate"] > 0.95:
            logger.info("Rollback successful")
        else:
            # 回滚失败，启动应急预案
            trigger_emergency_procedure()
    
    # Step 4: 紧急通知所有相关人员
    send_notification(
        level="emergency",
        message=f"EMERGENCY: Success rate {metrics['success_rate']*100:.2f}%, "
                f"Auto-rollback triggered",
        action_required=True,
        oncall_phone=True,
        escalate_to_manager=True  # 升级到经理
    )
    
    # Step 5: 记录事件（用于 Postmortem）
    record_emergency_event(metrics)
```

**结果**：
- ✅ 回滚后，Success Rate 恢复到 96% → 记录 Postmortem
- ❌ 回滚后仍然 < 90% → 启动应急预案（人工介入）

---

## 📊 第四部分：响应策略决策树（Decision Tree）

### 决策流程图

```
告警触发
    ↓
判断严重性
    ↓
    ├─ Info (轻微波动)
    │   └─ 记录日志，无需动作
    │
    ├─ Warning (接近阈值)
    │   ├─ 可恢复错误？ → 自动重试
    │   └─ 不可恢复错误？ → 记录日志，通知 On-Call
    │
    ├─ Critical (严重超标)
    │   ├─ 外部服务失败？ → 自动熔断 + 降级
    │   ├─ 延迟过高？ → 限流 + 跳过非关键请求
    │   └─ 新版本问题？ → 考虑回滚
    │
    └─ Emergency (系统不可用)
        ├─ 新版本发布？ → 自动回滚
        └─ 非版本问题？ → 启动应急预案
```

---

## 🔧 第五部分：实现示例（KYC 项目）

### 完整的告警响应系统

```python
# kyc_alert_handler.py

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertMetrics:
    """告警指标"""
    success_rate: float
    error_rate: float
    p95: float
    p99: float
    error_breakdown: Dict[str, int]
    api_failure_rate: float
    rps: float


class AlertResponseHandler:
    """告警响应处理器"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.fallback_enabled = False
    
    def handle_alert(self, metrics: AlertMetrics) -> None:
        """处理告警"""
        
        # Step 1: 判断严重性
        severity = self._determine_severity(metrics)
        
        # Step 2: 选择响应策略
        if severity == AlertSeverity.INFO:
            self._handle_info(metrics)
        elif severity == AlertSeverity.WARNING:
            self._handle_warning(metrics)
        elif severity == AlertSeverity.CRITICAL:
            self._handle_critical(metrics)
        elif severity == AlertSeverity.EMERGENCY:
            self._handle_emergency(metrics)
    
    def _determine_severity(self, metrics: AlertMetrics) -> AlertSeverity:
        """判断告警严重性"""
        
        if metrics.success_rate < 0.90 or metrics.error_rate > 0.10 or metrics.p95 > 30:
            return AlertSeverity.EMERGENCY
        elif metrics.success_rate < 0.95 or metrics.p95 > 20 or metrics.api_failure_rate > 0.05:
            return AlertSeverity.CRITICAL
        elif metrics.success_rate < 0.98:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _handle_warning(self, metrics: AlertMetrics) -> None:
        """处理 Warning 告警：自动重试"""
        
        # 分析错误类型
        timeout_errors = metrics.error_breakdown.get("API_TIMEOUT", 0)
        connection_errors = metrics.error_breakdown.get("API_CONNECTION_ERROR", 0)
        
        if timeout_errors > 0 or connection_errors > 0:
            # 自动重试
            logger.warning("Triggering auto-retry for recoverable errors")
            self._retry_failed_requests(
                error_types=["API_TIMEOUT", "API_CONNECTION_ERROR"],
                max_retries=3
            )
            
            # 通知 On-Call（非紧急）
            self._send_notification(
                level="warning",
                message=f"Success rate {metrics.success_rate*100:.2f}%, auto-retry in progress",
                action_required=False
            )
        else:
            # 只记录日志
            logger.info(f"Success rate {metrics.success_rate*100:.2f}%, monitoring...")
    
    def _handle_critical(self, metrics: AlertMetrics) -> None:
        """处理 Critical 告警：自动熔断 + 降级"""
        
        # Step 1: 打开熔断器
        if metrics.api_failure_rate > 0.05:
            self.circuit_breaker.open()
            logger.critical("Circuit breaker OPEN: API failure rate too high")
        
        # Step 2: 切换到降级方案
        if not self.fallback_enabled:
            self._enable_fallback()
            logger.warning("Switched to fallback mode")
        
        # Step 3: 限流（如果延迟过高）
        if metrics.p95 > 20:
            self.rate_limiter.set_rate(MAX_RPS * 0.8)
            logger.warning(f"Rate limiting: p95 {metrics.p95:.2f}s")
        
        # Step 4: 通知 On-Call（紧急）
        self._send_notification(
            level="critical",
            message=f"Critical alert: Success rate {metrics.success_rate*100:.2f}%, "
                    f"p95 {metrics.p95:.2f}s",
            action_required=True,
            oncall_phone=True
        )
        
        # Step 5: 记录事件
        self._record_critical_event(metrics)
    
    def _handle_emergency(self, metrics: AlertMetrics) -> None:
        """处理 Emergency 告警：自动回滚"""
        
        # Step 1: 检查是否有新版本发布
        recent_releases = self._get_recent_releases(hours=1)
        
        if recent_releases:
            # 自动回滚
            logger.critical("EMERGENCY: Auto-rollback triggered")
            previous_version = self._get_last_stable_version()
            self._rollback_to_version(previous_version)
            
            # 验证回滚
            new_metrics = self._wait_for_metrics_stabilize(timeout=600)
            
            if new_metrics.success_rate > 0.95:
                logger.info("Rollback successful")
            else:
                logger.critical("Rollback failed, triggering emergency procedure")
                self._trigger_emergency_procedure()
        
        # Step 2: 紧急通知
        self._send_notification(
            level="emergency",
            message=f"EMERGENCY: Success rate {metrics.success_rate*100:.2f}%",
            action_required=True,
            oncall_phone=True,
            escalate_to_manager=True
        )
        
        # Step 3: 记录事件
        self._record_emergency_event(metrics)
    
    def _retry_failed_requests(self, error_types: list, max_retries: int) -> None:
        """自动重试失败的请求"""
        # 实现重试逻辑
        pass
    
    def _enable_fallback(self) -> None:
        """启用降级方案"""
        self.fallback_enabled = True
        # 切换到 OCR-only 或人工审核
        pass
    
    def _rollback_to_version(self, version: str) -> None:
        """回滚到指定版本"""
        # 实现回滚逻辑
        pass
    
    def _send_notification(self, level: str, message: str, 
                          action_required: bool = False,
                          oncall_phone: bool = False,
                          escalate_to_manager: bool = False) -> None:
        """发送通知"""
        # 实现通知逻辑（Slack, PagerDuty, 电话等）
        pass
```

---

## 📝 总结

### 告警触发后的响应策略

| 严重性 | 触发条件 | 响应动作 | 验证恢复 |
|--------|---------|---------|---------|
| **Warning** | Success Rate < 98% | 自动重试 | 重试后恢复 > 98% |
| **Critical** | Success Rate < 95% 或 p95 > 20s | 自动熔断 + 降级 | 降级后恢复 > 95% |
| **Emergency** | Success Rate < 90% 或 Error Rate > 10% | 自动回滚 | 回滚后恢复 > 95% |

### 关键设计原则

1. **自动化优先**：能自动修复的，不要等人工
2. **快速响应**：Critical 告警 < 5 分钟响应
3. **可验证**：每个动作都要验证是否恢复
4. **可追溯**：所有动作都要记录（用于 Postmortem）

### 面试要点

✅ **能说出**：
- "我们设计了多层次的告警响应机制：Warning 自动重试，Critical 自动熔断+降级，Emergency 自动回滚"
- "每个响应动作都有验证机制，确保系统能够恢复"
- "所有告警事件都记录，用于后续的 Postmortem 分析"

---

## 🎯 下一步

1. **实现告警响应系统**：基于 KYC 项目实现 `AlertResponseHandler`
2. **集成到监控系统**：每次 batch 完成后自动触发
3. **测试响应机制**：模拟各种告警场景，验证响应是否正确
