# SGLang Engine Abort 功能

本文档介绍如何在 SGLang Engine 模式下使用 abort 功能来中止请求。

## 功能概述

SGLang Engine 现在支持 abort 功能，允许你中止正在运行的请求或所有请求。这个功能与 HTTP Server 模式下的 abort 功能保持一致。

## API 接口

```python
def abort_request(self, rid: str = "", abort_all: bool = False):
    """中止特定请求或所有请求

    Args:
        rid: 要中止的请求ID。如果为空且 abort_all 为 False，则不执行任何操作
        abort_all: 如果为 True，则中止所有正在运行的请求，忽略 rid 参数
    """
```

## 使用示例

### 基本用法

```python
from sglang.srt.entrypoints.engine import Engine

# 创建引擎
engine = Engine(
    model="meta-llama/Llama-2-7b-chat-hf",
    trust_remote_code=True
)

# 中止所有请求
engine.abort_request(abort_all=True)

# 中止特定请求
engine.abort_request(rid="specific_request_id")

# 关闭引擎
engine.shutdown()
```

### 大规模请求中止测试

这是一个更科学的测试场景：发送1000条请求，等待800条完成后中止剩余200条。

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class MassRequestAbortTest:
    def __init__(self):
        self.engine = Engine(
            model="meta-llama/Llama-2-7b-chat-hf",
            trust_remote_code=True
        )
        self.completed_count = 0
        self.total_requests = 0
        self.lock = threading.Lock()

    def single_request(self, request_id: str):
        """执行单个请求"""
        try:
            result = self.engine.generate(
                prompt=f"Request {request_id}: What is 2+2?",
                sampling_params={"max_new_tokens": 10, "temperature": 0.1}
            )

            with self.lock:
                self.completed_count += 1
                if self.completed_count % 100 == 0:
                    print(f"✅ Completed {self.completed_count}/{self.total_requests}")

            return {"request_id": request_id, "status": "success"}

        except Exception as e:
            with self.lock:
                self.completed_count += 1
            return {"request_id": request_id, "status": "error", "error": str(e)}

    def monitor_and_abort(self, target_completion: int):
        """监控进度并在达到目标时中止"""
        while self.completed_count < target_completion:
            time.sleep(0.1)

        print(f"🎯 TARGET REACHED: {self.completed_count}/{self.total_requests}")
        print("🚫 ABORTING REMAINING REQUESTS...")
        self.engine.abort_request(abort_all=True)

    def run_mass_abort_test(self, total_requests=1000, target_completion=800):
        """运行大规模abort测试"""
        self.total_requests = total_requests
        self.completed_count = 0

        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_and_abort, args=(target_completion,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        # 并发执行请求
        with ThreadPoolExecutor(max_workers=50) as executor:
            request_ids = [f"req_{i:04d}" for i in range(total_requests)]
            futures = {
                executor.submit(self.single_request, rid): rid
                for rid in request_ids
            }

            for future in as_completed(futures):
                future.result()

        # 统计结果
        final_completed = self.completed_count
        aborted_count = total_requests - final_completed

        print(f"📊 RESULTS:")
        print(f"   Completed: {final_completed}")
        print(f"   Aborted: {aborted_count}")
        print(f"   Target: {target_completion}")

        return {
            "completed": final_completed,
            "aborted": aborted_count,
            "target": target_completion
        }

    def shutdown(self):
        self.engine.shutdown()

# 运行测试
test = MassRequestAbortTest()
result = test.run_mass_abort_test(1000, 800)
test.shutdown()
```

## 与 Server 模式的对比

| 功能 | Server 模式 | Engine 模式 |
|------|-------------|-------------|
| HTTP API | ✅ `/abort_request` | ❌ 无 HTTP API |
| 程序化调用 | ❌ 需要 HTTP 请求 | ✅ 直接方法调用 |
| 中止特定请求 | ✅ | ✅ |
| 中止所有请求 | ✅ | ✅ |
| 大规模测试 | ✅ | ✅ |

## 实现细节

Engine 模式的 abort 功能通过以下方式实现：

1. **直接调用**：Engine 直接调用 `tokenizer_manager.abort_request()`
2. **参数一致**：与 Server 模式保持相同的参数接口
3. **错误处理**：安全的错误处理，不会因为中止不存在的请求而报错

## 最佳实践

1. **资源清理**：在关闭引擎前调用 abort 确保所有请求都被正确清理
2. **错误处理**：在长时间运行的任务中使用 abort 作为紧急停止机制
3. **并发安全**：abort 操作是线程安全的，可以在多线程环境中使用
4. **性能考虑**：abort 操作是轻量级的，不会影响引擎性能
5. **大规模测试**：使用并发请求来测试 abort 功能的有效性

## 注意事项

1. **立即生效**：abort 操作会立即发送到调度器，但正在运行的推理可能需要一些时间才能完全停止
2. **资源释放**：abort 会自动清理相关的 KV cache 和内存资源
3. **日志记录**：abort 操作会被记录在日志中，便于调试
4. **兼容性**：与现有代码完全兼容，不会影响其他功能

## 测试

### 基本测试

```bash
python test_engine_abort.py
```

### 大规模测试

```bash
# 测试1000条请求，800条完成后中止剩余200条
python test_abort_1000_requests.py
```

### 详细测试

```bash
# 包含多种场景的详细测试
python test_engine_abort_realistic.py
```

### 示例运行

```bash
python engine_abort_example.py
```

## 测试结果示例

运行大规模测试的典型输出：

```
🧪 MASS REQUEST ABORT TEST
Send 1000 requests, wait for 800 to complete, then abort the remaining 200
============================================================
🚀 MASS ABORT TEST
   Sending 1000 requests...
   Will abort after 800 completions
   Expected to abort 200 requests
============================================================
📤 Submitting all requests...
⏳ Waiting for requests to complete...
✅ Completed 100/1000 requests (10.0%) [25.0 req/s]
✅ Completed 200/1000 requests (20.0%) [22.2 req/s]
...
✅ Completed 800/1000 requests (80.0%) [20.5 req/s]

🎯 TARGET REACHED: 800/1000 completed!
🚫 ABORTING REMAINING REQUESTS...
✅ Abort command sent!

============================================================
📊 FINAL RESULTS:
   Total requests sent: 1000
   Completed: 800
   Aborted: 200
   Target completion: 800
   Actual completion: 800
   Completion rate: 80.0%
   Total time: 45.23s
   Average rate: 22.1 requests/second
✅ SUCCESS: Reached target (800) before abort
🚫 ABORT EFFECTIVE: 200 requests were aborted
```
