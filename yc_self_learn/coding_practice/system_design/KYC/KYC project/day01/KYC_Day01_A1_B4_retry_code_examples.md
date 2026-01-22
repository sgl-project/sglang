# RETRY 代码实现指南

基于 A1_B4 的设计原则，提供重试代码的常见实现方式。

---

## 1. Python 版本实现

### 1.1 基础版本（简单重试）

```python
import time
import random
from typing import Callable, Any, Optional

def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """
    基础重试函数，带指数退避
    
    Args:
        fn: 要执行的函数（无参数）
        max_retries: 最大重试次数（不包括首次尝试）
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_multiplier: 退避 multiplier（通常为 2.0）
        should_retry: 判断异常是否可重试的函数，返回 False 则不重试
    
    Returns:
        函数执行结果
    
    Raises:
        最后一次尝试的异常，或 MaxRetriesExceeded
    """
    attempt = 0
    
    while True:
        try:
            return fn()
        except Exception as e:
            # 检查是否应该重试
            if should_retry and not should_retry(e):
                raise  # 不可重试的错误直接抛出
            
            # 检查是否超过最大重试次数
            if attempt >= max_retries:
                raise Exception(f"Max retries ({max_retries}) exceeded. Last error: {e}")
            
            # 计算延迟时间（指数退避）
            delay = min(initial_delay * (backoff_multiplier ** attempt), max_delay)
            
            # 添加 jitter（随机抖动，避免 thundering herd）
            jitter = delay * 0.1 * random.random()  # 10% jitter
            delay += jitter
            
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
            
            attempt += 1


# 使用示例
def call_api():
    """模拟 API 调用"""
    import random
    if random.random() < 0.7:  # 70% 失败率
        raise ConnectionError("API connection failed")
    return {"status": "success"}

# 判断是否可重试
def is_retryable_error(e: Exception) -> bool:
    """只对可恢复错误重试"""
    retryable_errors = (
        ConnectionError,
        TimeoutError,
        OSError,  # 网络错误
    )
    return isinstance(e, retryable_errors)

# 调用
result = retry_with_backoff(
    call_api,
    max_retries=3,
    initial_delay=1.0,
    should_retry=is_retryable_error,
)
```

### 1.2 完整版本（带错误分类和指标记录）

```python
import time
import random
import logging
from typing import Callable, Any, Optional, Dict
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    RETRYABLE = "retryable"      # 可重试（网络错误、超时等）
    NON_RETRYABLE = "non_retryable"  # 不可重试（业务规则错误、格式错误等）
    FATAL = "fatal"              # 致命错误（不应重试）


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1  # 10% jitter


@dataclass
class RetryResult:
    """重试结果"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0


class RetryExecutor:
    """重试执行器（完整版）"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.metrics = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
        }
    
    def classify_error(self, e: Exception) -> ErrorType:
        """
        错误分类：判断错误是否可重试
        
        根据 KYC 设计原则：
        - RETRYABLE: API_TIMEOUT, API_CONNECTION_ERROR, API_SERVER_ERROR
        - NON_RETRYABLE: IMAGE_FORMAT_UNSUPPORTED, SCHEMA_VALIDATION_FAILED
        """
        # 可重试错误
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        
        # 不可重试错误（业务规则）
        non_retryable_errors = (
            ValueError,  # 格式错误
            TypeError,   # 类型错误
        )
        
        if isinstance(e, retryable_errors):
            return ErrorType.RETRYABLE
        elif isinstance(e, non_retryable_errors):
            return ErrorType.NON_RETRYABLE
        else:
            # 默认：根据错误消息判断
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["timeout", "connection", "network"]):
                return ErrorType.RETRYABLE
            elif any(keyword in error_msg for keyword in ["format", "validation", "invalid"]):
                return ErrorType.NON_RETRYABLE
            else:
                return ErrorType.FATAL
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避 + jitter）"""
        # 指数退避
        delay = min(
            self.config.initial_delay * (self.config.backoff_multiplier ** attempt),
            self.config.max_delay
        )
        
        # 添加 jitter（避免 thundering herd）
        jitter = delay * self.config.jitter_factor * (random.random() * 2 - 1)
        delay = max(0, delay + jitter)
        
        return delay
    
    def execute(
        self,
        fn: Callable[[], Any],
        operation_name: str = "operation",
    ) -> RetryResult:
        """
        执行带重试的操作
        
        Args:
            fn: 要执行的函数
            operation_name: 操作名称（用于日志）
        
        Returns:
            RetryResult: 重试结果
        """
        attempt = 0
        total_delay = 0.0
        
        while True:
            try:
                result = fn()
                
                # 成功：记录指标
                if attempt > 0:
                    self.metrics["successful_retries"] += 1
                    logger.info(
                        f"{operation_name} succeeded after {attempt} retries"
                    )
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                )
                
            except Exception as e:
                error_type = self.classify_error(e)
                self.metrics["total_attempts"] += 1
                
                # 不可重试的错误：直接失败
                if error_type != ErrorType.RETRYABLE:
                    logger.warning(
                        f"{operation_name} failed with non-retryable error: {e}"
                    )
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=attempt + 1,
                        total_delay=total_delay,
                    )
                
                # 检查是否超过最大重试次数
                if attempt >= self.config.max_retries:
                    self.metrics["failed_retries"] += 1
                    logger.error(
                        f"{operation_name} failed after {attempt + 1} attempts: {e}"
                    )
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=attempt + 1,
                        total_delay=total_delay,
                    )
                
                # 计算延迟并等待
                delay = self.calculate_delay(attempt)
                total_delay += delay
                
                logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s... (max_retries={self.config.max_retries})"
                )
                
                time.sleep(delay)
                attempt += 1


# 使用示例
def process_kyc_document(file_path: str):
    """处理 KYC 文档（模拟）"""
    import random
    error_type = random.choice(["timeout", "connection", "format", "success"])
    
    if error_type == "timeout":
        raise TimeoutError("API timeout")
    elif error_type == "connection":
        raise ConnectionError("API connection failed")
    elif error_type == "format":
        raise ValueError("Unsupported image format")
    else:
        return {"status": "success", "file": file_path}


# 配置
config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    jitter_factor=0.1,
)

# 执行
executor = RetryExecutor(config)
result = executor.execute(
    lambda: process_kyc_document("doc.pdf"),
    operation_name="process_kyc_document",
)

if result.success:
    print(f"Success: {result.result}")
else:
    print(f"Failed after {result.attempts} attempts: {result.error}")

# 查看指标
print(f"Metrics: {executor.metrics}")
```

### 1.3 异步版本（async/await）

```python
import asyncio
import random
from typing import Callable, Any, Optional, Awaitable

async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """
    异步重试函数
    """
    attempt = 0
    
    while True:
        try:
            return await fn()
        except Exception as e:
            if should_retry and not should_retry(e):
                raise
            
            if attempt >= max_retries:
                raise Exception(f"Max retries ({max_retries}) exceeded: {e}")
            
            delay = min(initial_delay * (backoff_multiplier ** attempt), max_delay)
            jitter = delay * 0.1 * random.random()
            delay += jitter
            
            print(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
            
            attempt += 1


# 使用示例
async def call_api_async():
    """异步 API 调用"""
    await asyncio.sleep(0.1)
    if random.random() < 0.7:
        raise ConnectionError("API failed")
    return {"status": "success"}

# 调用
result = await retry_async(
    call_api_async,
    max_retries=3,
    initial_delay=1.0,
)
```

---

## 2. Rust 版本实现

### 2.1 基础版本（基于代码库实现）

```rust
use std::time::Duration;
use std::future::Future;
use tokio::time::sleep;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

pub struct BackoffCalculator;

impl BackoffCalculator {
    /// 计算延迟时间（指数退避 + jitter）
    pub fn calculate_delay(config: &RetryConfig, attempt: u32) -> Duration {
        // 指数退避
        let pow = config.backoff_multiplier.powi(attempt as i32);
        let mut delay_ms = (config.initial_delay_ms as f32 * pow) as u64;
        
        // 限制最大延迟
        if delay_ms > config.max_delay_ms {
            delay_ms = config.max_delay_ms;
        }
        
        // 添加 jitter
        if config.jitter_factor > 0.0 {
            let mut rng = rand::thread_rng();
            let jitter_scale: f32 = rng.gen_range(-config.jitter_factor..=config.jitter_factor);
            let jitter_ms = (delay_ms as f32 * jitter_scale).round() as i64;
            let adjusted = (delay_ms as i64 + jitter_ms).max(0) as u64;
            return Duration::from_millis(adjusted);
        }
        
        Duration::from_millis(delay_ms)
    }
}

#[derive(Debug)]
pub enum RetryError {
    MaxRetriesExceeded,
    NonRetryableError(String),
}

/// 判断错误是否可重试
pub trait Retryable {
    fn is_retryable(&self) -> bool;
}

/// 重试执行器
pub struct RetryExecutor;

impl RetryExecutor {
    /// 执行带重试的异步操作
    pub async fn execute_with_retry<F, Fut, T, E>(
        config: &RetryConfig,
        mut operation: F,
        should_retry: impl Fn(&E) -> bool,
    ) -> Result<T, RetryError>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let max = config.max_retries.max(1);
        let mut attempt: u32 = 0;
        
        loop {
            match operation(attempt).await {
                Ok(val) => return Ok(val),
                Err(e) => {
                    // 检查是否可重试
                    if !should_retry(&e) {
                        return Err(RetryError::NonRetryableError(format!("{:?}", e)));
                    }
                    
                    // 检查是否超过最大重试次数
                    let is_last = attempt + 1 >= max;
                    if is_last {
                        return Err(RetryError::MaxRetriesExceeded);
                    }
                    
                    // 计算延迟并等待
                    let delay = BackoffCalculator::calculate_delay(config, attempt);
                    eprintln!("Attempt {} failed. Retrying in {:?}...", attempt + 1, delay);
                    sleep(delay).await;
                    
                    attempt += 1;
                }
            }
        }
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let config = RetryConfig::default();
    
    let result = RetryExecutor::execute_with_retry(
        &config,
        |attempt| async move {
            // 模拟 API 调用
            if attempt < 2 {
                Err("connection failed")
            } else {
                Ok("success")
            }
        },
        |e| {
            // 判断是否可重试
            !e.contains("format")  // 格式错误不可重试
        },
    ).await;
    
    match result {
        Ok(val) => println!("Success: {}", val),
        Err(e) => println!("Failed: {:?}", e),
    }
}
```

### 2.2 完整版本（带错误分类和指标）

```rust
use std::time::Duration;
use std::future::Future;
use tokio::time::sleep;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub jitter_factor: f32,
}

#[derive(Debug)]
pub enum ErrorType {
    Retryable,      // 可重试（网络错误、超时等）
    NonRetryable,   // 不可重试（业务规则错误）
    Fatal,          // 致命错误
}

pub trait ErrorClassifier {
    fn classify(&self) -> ErrorType;
}

#[derive(Debug)]
pub struct RetryResult<T> {
    pub success: bool,
    pub result: Option<T>,
    pub error: Option<String>,
    pub attempts: u32,
    pub total_delay_ms: u64,
}

pub struct RetryExecutor {
    config: RetryConfig,
    metrics: RetryMetrics,
}

#[derive(Debug, Default)]
pub struct RetryMetrics {
    pub total_attempts: u64,
    pub successful_retries: u64,
    pub failed_retries: u64,
}

impl RetryExecutor {
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            metrics: RetryMetrics::default(),
        }
    }
    
    pub async fn execute<F, Fut, T, E>(
        &mut self,
        mut operation: F,
        error_classifier: impl Fn(&E) -> ErrorType,
    ) -> RetryResult<T>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let max = self.config.max_retries.max(1);
        let mut attempt: u32 = 0;
        let mut total_delay_ms: u64 = 0;
        
        loop {
            match operation(attempt).await {
                Ok(val) => {
                    if attempt > 0 {
                        self.metrics.successful_retries += 1;
                    }
                    return RetryResult {
                        success: true,
                        result: Some(val),
                        error: None,
                        attempts: attempt + 1,
                        total_delay_ms,
                    };
                }
                Err(e) => {
                    self.metrics.total_attempts += 1;
                    let error_type = error_classifier(&e);
                    
                    // 不可重试的错误：直接失败
                    if !matches!(error_type, ErrorType::Retryable) {
                        return RetryResult {
                            success: false,
                            result: None,
                            error: Some(format!("{:?}", e)),
                            attempts: attempt + 1,
                            total_delay_ms,
                        };
                    }
                    
                    // 检查是否超过最大重试次数
                    if attempt + 1 >= max {
                        self.metrics.failed_retries += 1;
                        return RetryResult {
                            success: false,
                            result: None,
                            error: Some(format!("Max retries exceeded: {:?}", e)),
                            attempts: attempt + 1,
                            total_delay_ms,
                        };
                    }
                    
                    // 计算延迟并等待
                    let delay = BackoffCalculator::calculate_delay(&self.config, attempt);
                    total_delay_ms += delay.as_millis() as u64;
                    
                    eprintln!(
                        "Attempt {} failed: {:?}. Retrying in {:?}...",
                        attempt + 1,
                        e,
                        delay
                    );
                    
                    sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }
    
    pub fn metrics(&self) -> &RetryMetrics {
        &self.metrics
    }
}
```

---

## 3. 设计要点总结

### 3.1 核心原则（基于 A1_B4）

1. **限制重试次数**：`max_retries = 3`，避免无限重试导致概率累积
2. **指数退避**：`1s → 2s → 4s`，避免快速重试压垮上游
3. **错误分类**：只对可恢复错误重试（网络错误、超时），不可恢复错误（格式错误、业务规则）不重试
4. **Jitter（随机抖动）**：避免 thundering herd 问题
5. **幂等性保证**：确保重复执行不会产生副作用

### 3.2 Error Rate 计算

**关键**：Error Rate = **最终失败的请求数** / **原始请求数**

```python
# ❌ 错误计算
error_rate = total_failures / total_attempts  # 包含重试次数

# ✅ 正确计算
error_rate = final_failures / original_requests  # 只按最终结果
```

### 3.3 可重试错误判断

```python
def is_retryable_error(e: Exception) -> bool:
    """判断错误是否可重试"""
    # 可重试错误
    retryable = (
        ConnectionError,    # 连接错误
        TimeoutError,       # 超时
        OSError,           # 网络错误
    )
    
    # 不可重试错误
    non_retryable = (
        ValueError,        # 格式错误
        TypeError,         # 类型错误
        KeyError,          # 业务规则错误
    )
    
    if isinstance(e, retryable):
        return True
    elif isinstance(e, non_retryable):
        return False
    else:
        # 根据错误消息判断
        msg = str(e).lower()
        return any(kw in msg for kw in ["timeout", "connection", "network"])
```

---

## 4. 使用第三方库

### 4.1 Python: `tenacity`

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def call_api():
    # API 调用
    pass
```

### 4.2 Rust: `backoff`

```rust
use backoff::{ExponentialBackoff, Error};

let backoff = ExponentialBackoff::default();

backoff::future::retry(backoff, || async {
    // 操作
    Ok(())
}).await?;
```

---

## 5. 最佳实践

1. **记录指标**：记录重试次数、延迟时间、成功率
2. **日志记录**：记录每次重试的原因和延迟
3. **监控告警**：重试率过高时告警
4. **测试覆盖**：测试重试逻辑、错误分类、退避策略
5. **幂等性验证**：确保操作可以安全重试

---

## Links

- **Parent**: [A1_B4 重试如何影响错误率](./KYC_Day01_A1_B4_retry_error_rate.md)
- **Related**: Error Rate、错误预算、幂等性、指数退避
