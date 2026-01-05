# process_batch 具体实现详解

## 🎯 什么是 process_batch？

`process_batch` 是一个函数，用于**并发发送多个请求**到 SGLang 服务器，并等待所有请求完成。

## 📝 完整实现示例

### 示例 1: 使用 aiohttp + asyncio.gather（推荐）

这是最常见的实现方式：

```python
import asyncio
import aiohttp
import json
from typing import List, Dict

async def process_batch(
    requests: List[Dict],
    api_url: str = "http://localhost:30000/v1/chat/completions",
    timeout: float = 300.0
) -> List[Dict]:
    """
    并发处理一批请求
    
    Args:
        requests: 请求列表，每个元素是一个请求字典
        api_url: API 端点 URL
        timeout: 超时时间（秒）
    
    Returns:
        响应列表，每个元素对应一个请求的响应
    """
    async def send_single_request(session: aiohttp.ClientSession, request: Dict) -> Dict:
        """发送单个请求"""
        try:
            async with session.post(
                url=api_url,
                json=request,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    if request.get("stream", False):
                        # 流式响应处理
                        generated_text = ""
                        async for line in response.content:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # 处理 SSE 格式: "data: {...}"
                            if line.startswith(b"data: "):
                                data_str = line[6:].decode("utf-8")
                                if data_str == "[DONE]":
                                    break
                                
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    generated_text += content
                        
                        return {
                            "success": True,
                            "text": generated_text,
                            "status": response.status
                        }
                    else:
                        # 非流式响应
                        result = await response.json()
                        return {
                            "success": True,
                            "text": result["choices"][0]["message"]["content"],
                            "status": response.status,
                            "usage": result.get("usage", {})
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": error_text,
                        "status": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": 0
            }
    
    # 创建 HTTP 会话
    async with aiohttp.ClientSession() as session:
        # 创建所有请求的任务
        tasks = [
            send_single_request(session, req)
            for req in requests
        ]
        
        # 并发执行所有任务，等待全部完成
        results = await asyncio.gather(*tasks)
        
        return results


# 使用示例
async def main():
    # 准备一批评论
    comments = [
        "这个产品很棒！",
        "质量一般，价格有点贵。",
        "非常满意，会再次购买。",
        # ... 更多评论
    ]
    
    # 构建请求
    requests = [
        {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个情感分析助手。"
                },
                {
                    "role": "user",
                    "content": f"请分析这条评论的情感倾向：{comment}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False  # 非流式，一次性返回结果
        }
        for comment in comments
    ]
    
    # 批量处理
    results = await process_batch(requests)
    
    # 处理结果
    for i, result in enumerate(results):
        if result["success"]:
            print(f"评论 {i+1}: {comments[i]}")
            print(f"分析结果: {result['text']}\n")
        else:
            print(f"请求 {i+1} 失败: {result['error']}")


# 运行
if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 2: 使用 SGLang Engine API

如果直接在代码中使用 SGLang Engine（不通过 HTTP）：

```python
import asyncio
import sglang as sgl
from typing import List, Dict

async def process_batch_engine(
    prompts: List[str],
    sampling_params: Dict = None
) -> List[Dict]:
    """
    使用 SGLang Engine 批量处理
    
    Args:
        prompts: 提示词列表
        sampling_params: 采样参数
    
    Returns:
        生成结果列表
    """
    # 初始化引擎（只初始化一次）
    engine = sgl.Engine(
        model_path="qwen/qwen2.5-0.5b-instruct",
        log_level="error"
    )
    
    # 默认采样参数
    if sampling_params is None:
        sampling_params = {
            "temperature": 0.7,
            "max_new_tokens": 100
        }
    
    async def generate_one(prompt: str) -> Dict:
        """生成单个结果"""
        result = await engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params
        )
        return {
            "prompt": prompt,
            "text": result["text"],
            "meta_info": result.get("meta_info", {})
        }
    
    # 创建所有任务
    tasks = [
        generate_one(prompt)
        for prompt in prompts
    ]
    
    # 并发执行
    results = await asyncio.gather(*tasks)
    
    return results


# 使用示例
async def main():
    prompts = [
        "介绍一下人工智能",
        "介绍一下机器学习",
        "介绍一下深度学习"
    ]
    
    results = await process_batch_engine(prompts)
    
    for result in results:
        print(f"Prompt: {result['prompt']}")
        print(f"Response: {result['text']}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 3: 流式批处理

如果使用流式响应：

```python
import asyncio
import aiohttp
import json
from typing import List, Dict, AsyncGenerator

async def process_batch_streaming(
    requests: List[Dict],
    api_url: str = "http://localhost:30000/v1/chat/completions"
) -> AsyncGenerator[List[Dict], None]:
    """
    流式批量处理
    
    这个方法会逐步返回结果，不需要等待所有请求完成
    """
    async def stream_single_request(session: aiohttp.ClientSession, request: Dict) -> AsyncGenerator:
        """流式发送单个请求"""
        async with session.post(
            url=api_url,
            json={**request, "stream": True},  # 强制流式
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    line = line.strip()
                    if not line or not line.startswith(b"data: "):
                        continue
                    
                    data_str = line[6:].decode("utf-8")
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        yield {
                            "success": True,
                            "data": data
                        }
                    except json.JSONDecodeError:
                        continue
            else:
                yield {
                    "success": False,
                    "error": await response.text(),
                    "status": response.status
                }
    
    async with aiohttp.ClientSession() as session:
        # 创建所有流式任务
        streams = [
            stream_single_request(session, req)
            for req in requests
        ]
        
        # 使用 asyncio.wait 处理多个流
        tasks = {
            asyncio.create_task(stream.__anext__()): (i, stream)
            for i, stream in enumerate(streams)
        }
        
        # 结果缓冲区
        results = [None] * len(requests)
        completed = 0
        
        # 逐步处理结果
        while tasks:
            done, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                idx, stream = tasks.pop(task)
                try:
                    chunk = await task
                    if results[idx] is None:
                        results[idx] = {"chunks": []}
                    results[idx]["chunks"].append(chunk)
                    
                    # 继续下一个 chunk
                    new_task = asyncio.create_task(stream.__anext__())
                    tasks[new_task] = (idx, stream)
                except StopAsyncIteration:
                    # 这个流完成了
                    completed += 1
                    if completed == len(requests):
                        # 所有请求完成
                        yield results
                        return


# 使用示例
async def main():
    requests = [
        {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [{"role": "user", "content": f"问题 {i}"}]
        }
        for i in range(10)
    ]
    
    async for batch_results in process_batch_streaming(requests):
        # 逐步处理结果
        print(f"收到 {len(batch_results)} 个结果")
        # ... 处理逻辑


if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 4: 带错误处理和重试

更健壮的实现：

```python
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional
import time

async def process_batch_with_retry(
    requests: List[Dict],
    api_url: str = "http://localhost:30000/v1/chat/completions",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 300.0
) -> List[Dict]:
    """
    带重试的批量处理
    """
    async def send_with_retry(
        session: aiohttp.ClientSession,
        request: Dict,
        request_id: int
    ) -> Dict:
        """发送单个请求，带重试"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url=api_url,
                    json=request,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "request_id": request_id,
                            "text": result["choices"][0]["message"]["content"],
                            "usage": result.get("usage", {}),
                            "attempts": attempt + 1
                        }
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
                        
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
            except Exception as e:
                last_error = str(e)
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))  # 指数退避
        
        # 所有重试都失败
        return {
            "success": False,
            "request_id": request_id,
            "error": last_error,
            "attempts": max_retries
        }
    
    async with aiohttp.ClientSession() as session:
        # 创建任务
        tasks = [
            send_with_retry(session, req, i)
            for i, req in enumerate(requests)
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "request_id": i,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
```

## 🔍 关键点解析

### 1. asyncio.gather() 的作用

```python
# 创建任务列表
tasks = [send_request(req) for req in requests]

# 并发执行所有任务
results = await asyncio.gather(*tasks)
```

**作用**:
- 同时启动所有任务
- 等待所有任务完成
- 返回结果列表（按输入顺序）

**优势**:
- 真正的并发，不是串行
- 自动处理异常
- 简单易用

### 2. aiohttp.ClientSession 的重要性

```python
async with aiohttp.ClientSession() as session:
    # 所有请求共享同一个 session
    tasks = [send_request(session, req) for req in requests]
```

**为什么重要**:
- **连接复用**: 复用 TCP 连接，提高性能
- **资源管理**: 自动管理连接池
- **性能提升**: 比每次创建新连接快很多

### 3. 并发控制

如果请求太多，可能需要限制并发数：

```python
import asyncio
from asyncio import Semaphore

async def process_batch_with_limit(
    requests: List[Dict],
    max_concurrent: int = 32,
    api_url: str = "http://localhost:30000/v1/chat/completions"
) -> List[Dict]:
    """限制并发数的批量处理"""
    
    semaphore = Semaphore(max_concurrent)
    
    async def send_with_limit(session, request):
        async with semaphore:  # 限制并发
            return await send_single_request(session, request)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_with_limit(session, req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        return results
```

## 📊 性能对比

### 串行处理 vs 并发处理

```python
# 串行处理（慢）
results = []
for request in requests:
    result = await send_request(request)
    results.append(result)
# 时间: 32 个请求 × 1秒 = 32秒

# 并发处理（快）
tasks = [send_request(req) for req in requests]
results = await asyncio.gather(*tasks)
# 时间: 32 个请求并发 = 约 1-2秒
```

**性能提升**: 10-30x

## 💡 实际应用示例

### 完整的批量评论分析

```python
import asyncio
import aiohttp
from typing import List

async def analyze_comments_batch(comments: List[str]) -> List[Dict]:
    """
    批量分析评论情感
    """
    api_url = "http://localhost:30000/v1/chat/completions"
    
    # 构建请求
    requests = [
        {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [
                {"role": "system", "content": "你是情感分析专家。"},
                {"role": "user", "content": f"分析这条评论的情感：{comment}"}
            ],
            "temperature": 0.3,
            "max_tokens": 50
        }
        for comment in comments
    ]
    
    # 批量处理
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(session, req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        
        # 提取结果
        analyses = []
        for i, result in enumerate(results):
            analyses.append({
                "comment": comments[i],
                "analysis": result.get("text", "") if result.get("success") else "分析失败",
                "success": result.get("success", False)
            })
        
        return analyses


# 使用
async def main():
    comments = [
        "产品很好，推荐购买！",
        "质量一般，价格偏高。",
        "非常满意，会回购。",
        # ... 更多评论
    ]
    
    # 分批处理（每批 32 个）
    batch_size = 32
    all_results = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        batch_results = await analyze_comments_batch(batch)
        all_results.extend(batch_results)
        
        print(f"完成批次 {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
    
    # 输出结果
    for result in all_results:
        print(f"评论: {result['comment']}")
        print(f"分析: {result['analysis']}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

## 🎓 总结

### process_batch 的核心模式

```python
# 1. 准备请求列表
requests = [create_request(data) for data in data_list]

# 2. 创建异步任务
async with aiohttp.ClientSession() as session:
    tasks = [send_request(session, req) for req in requests]

# 3. 并发执行
results = await asyncio.gather(*tasks)

# 4. 处理结果
for result in results:
    process(result)
```

### 关键要素

1. ✅ **asyncio.gather()**: 并发执行的核心
2. ✅ **aiohttp.ClientSession**: 连接复用
3. ✅ **异步函数**: 使用 `async def` 和 `await`
4. ✅ **错误处理**: 捕获和处理异常
5. ✅ **并发控制**: 使用 Semaphore 限制并发数

### 性能优势

- **串行**: 32 个请求 × 1秒 = 32秒
- **并发**: 32 个请求并发 ≈ 1-2秒
- **提升**: **10-30x 性能提升**

---

## 📚 相关资源

- [多请求场景与批处理](./07_多请求场景与批处理.md)
- [asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
- [aiohttp 官方文档](https://docs.aiohttp.org/)

