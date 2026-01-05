# process_batch 返回结果详解

## 🎯 返回结果的数据结构

`process_batch` 返回的是一个**列表**，每个元素对应一个请求的响应结果。

## 📊 结果格式详解

### 基本结构

```python
results = await process_batch(requests)

# results 是一个列表
# results[0] 对应 requests[0] 的响应
# results[1] 对应 requests[1] 的响应
# ...

for i, result in enumerate(results):
    print(f"请求 {i} 的结果: {result}")
```

### 结果对象结构

每个结果对象包含以下字段：

```python
{
    "success": bool,        # 请求是否成功
    "text": str,            # 生成的文本内容（如果成功）
    "status": int,          # HTTP 状态码
    "error": str,           # 错误信息（如果失败）
    "usage": dict,          # Token 使用统计（可选）
    # ... 其他字段
}
```

## 🔍 结果是如何得到的？

### 步骤 1: 发送 HTTP 请求

```python
async with session.post(
    url="http://localhost:30000/v1/chat/completions",
    json=request,
    headers={"Content-Type": "application/json"}
) as response:
    # response 是 aiohttp.ClientResponse 对象
```

### 步骤 2: 获取 HTTP 响应

#### 非流式响应（stream=False）

```python
# 获取 JSON 响应
response_json = await response.json()

# 响应格式（OpenAI 兼容格式）
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "qwen/qwen2.5-0.5b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "人工智能是..."  # 生成的文本
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,      # 输入 tokens
        "completion_tokens": 50,   # 输出 tokens
        "total_tokens": 60         # 总 tokens
    }
}
```

#### 流式响应（stream=True）

```python
# 逐行读取 SSE 格式的流
async for line in response.content:
    # 每行格式: "data: {...}\n\n"
    # 示例:
    # data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"人工"},"index":0}]}
    # data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"智能"},"index":0}]}
    # data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}
    # data: [DONE]
```

### 步骤 3: 提取和格式化结果

#### 非流式响应提取

```python
def extract_result_from_json(response_json: dict) -> dict:
    """从 JSON 响应中提取结果"""
    
    if "choices" in response_json and len(response_json["choices"]) > 0:
        # 提取生成的文本
        content = response_json["choices"][0]["message"]["content"]
        
        # 提取使用统计
        usage = response_json.get("usage", {})
        
        return {
            "success": True,
            "text": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "finish_reason": response_json["choices"][0].get("finish_reason", "unknown"),
            "model": response_json.get("model", ""),
            "id": response_json.get("id", "")
        }
    else:
        return {
            "success": False,
            "error": "No choices in response",
            "text": ""
        }
```

#### 流式响应提取

```python
async def extract_result_from_stream(response) -> dict:
    """从流式响应中提取结果"""
    
    generated_text = ""
    finish_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    
    async for line in response.content:
        line = line.strip()
        if not line:
            continue
        
        # 处理 SSE 格式
        if line.startswith(b"data: "):
            data_str = line[6:].decode("utf-8")
            
            # 检查是否结束
            if data_str == "[DONE]":
                break
            
            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        # 累积文本
                        generated_text += content
                    
                    # 检查是否完成
                    if "finish_reason" in choices[0]:
                        finish_reason = choices[0]["finish_reason"]
                    
                    # 提取使用统计（通常在最后一个 chunk）
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        
            except json.JSONDecodeError:
                continue
    
    return {
        "success": True,
        "text": generated_text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "finish_reason": finish_reason or "unknown"
    }
```

## 📝 完整示例：结果如何处理

### 完整代码示例

```python
import asyncio
import aiohttp
import json
from typing import List, Dict

async def process_batch(requests: List[Dict]) -> List[Dict]:
    """完整的 process_batch 实现"""
    
    async def send_single_request(session: aiohttp.ClientSession, request: Dict) -> Dict:
        """发送单个请求并提取结果"""
        try:
            async with session.post(
                url="http://localhost:30000/v1/chat/completions",
                json=request,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                # 检查 HTTP 状态
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "status": response.status,
                        "text": ""
                    }
                
                # 处理流式和非流式响应
                if request.get("stream", False):
                    # 流式响应
                    return await extract_streaming_result(response)
                else:
                    # 非流式响应
                    response_json = await response.json()
                    return extract_non_streaming_result(response_json)
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout",
                "status": 0,
                "text": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": 0,
                "text": ""
            }
    
    async def extract_streaming_result(response) -> Dict:
        """提取流式响应结果"""
        generated_text = ""
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        
        async for line in response.content:
            line = line.strip()
            if not line or not line.startswith(b"data: "):
                continue
            
            data_str = line[6:].decode("utf-8")
            if data_str == "[DONE]":
                break
            
            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        generated_text += content
                    
                    if "finish_reason" in choices[0]:
                        finish_reason = choices[0]["finish_reason"]
                    
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
            except json.JSONDecodeError:
                continue
        
        return {
            "success": True,
            "text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "finish_reason": finish_reason,
            "status": 200
        }
    
    def extract_non_streaming_result(response_json: Dict) -> Dict:
        """提取非流式响应结果"""
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            usage = response_json.get("usage", {})
            
            return {
                "success": True,
                "text": content,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "finish_reason": choice.get("finish_reason", "unknown"),
                "model": response_json.get("model", ""),
                "id": response_json.get("id", ""),
                "status": 200
            }
        else:
            return {
                "success": False,
                "error": "No choices in response",
                "text": "",
                "status": 200
            }
    
    # 执行批处理
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(session, req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        return results


# 使用示例
async def main():
    # 准备请求
    comments = [
        "这个产品很棒！",
        "质量一般，价格有点贵。",
        "非常满意，会再次购买。"
    ]
    
    requests = [
        {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [
                {"role": "system", "content": "你是情感分析助手。"},
                {"role": "user", "content": f"分析这条评论：{comment}"}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False  # 非流式
        }
        for comment in comments
    ]
    
    # 执行批处理
    results = await process_batch(requests)
    
    # 处理结果
    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"请求 {i+1}: 评论 '{comments[i]}'")
        print(f"{'='*60}")
        
        if result["success"]:
            print(f"✅ 成功")
            print(f"📝 生成的文本:")
            print(f"   {result['text']}")
            print(f"\n📊 Token 使用:")
            usage = result.get("usage", {})
            print(f"   - 输入: {usage.get('prompt_tokens', 0)} tokens")
            print(f"   - 输出: {usage.get('completion_tokens', 0)} tokens")
            print(f"   - 总计: {usage.get('total_tokens', 0)} tokens")
            print(f"\n🏁 完成原因: {result.get('finish_reason', 'unknown')}")
            if "model" in result:
                print(f"🤖 模型: {result['model']}")
        else:
            print(f"❌ 失败")
            print(f"   错误: {result.get('error', 'Unknown error')}")
            print(f"   状态码: {result.get('status', 0)}")
    
    # 统计信息
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_tokens = sum(
        r.get("usage", {}).get("total_tokens", 0)
        for r in results
        if r["success"]
    )
    
    print(f"\n{'='*60}")
    print(f"📈 批处理统计:")
    print(f"   - 总请求数: {len(results)}")
    print(f"   - 成功: {successful}")
    print(f"   - 失败: {failed}")
    print(f"   - 总 Token 数: {total_tokens}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 📋 实际输出示例

### 成功的结果

```python
# 输入
comments = ["这个产品很棒！", "质量一般，价格有点贵。"]

# 处理结果
results = [
    {
        "success": True,
        "text": "这条评论表达了积极的情感。用户对产品感到满意，认为产品很好，并推荐购买。",
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 32,
            "total_tokens": 57
        },
        "finish_reason": "stop",
        "model": "qwen/qwen2.5-0.5b-instruct",
        "id": "chatcmpl-abc123",
        "status": 200
    },
    {
        "success": True,
        "text": "这条评论表达了较为中性的情感。用户认为产品质量一般，但对价格不太满意。",
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 35,
            "total_tokens": 65
        },
        "finish_reason": "stop",
        "model": "qwen/qwen2.5-0.5b-instruct",
        "id": "chatcmpl-def456",
        "status": 200
    }
]
```

### 失败的结果

```python
results = [
    {
        "success": False,
        "error": "HTTP 429: Rate limit exceeded",
        "status": 429,
        "text": ""
    },
    {
        "success": False,
        "error": "Request timeout",
        "status": 0,
        "text": ""
    }
]
```

## 🔄 完整的数据流

### 步骤详解

```
1. 准备请求列表
   requests = [
       {"messages": [...], "model": "...", ...},
       {"messages": [...], "model": "...", ...},
       ...
   ]
        ↓
2. 并发发送 HTTP 请求
   session.post(url, json=request1) → HTTP 请求 1
   session.post(url, json=request2) → HTTP 请求 2
   session.post(url, json=request3) → HTTP 请求 3
   (所有请求同时发送)
        ↓
3. 等待所有响应
   HTTP 响应 1 ← 服务器响应 1
   HTTP 响应 2 ← 服务器响应 2
   HTTP 响应 3 ← 服务器响应 3
        ↓
4. 解析响应
   响应1 → JSON 解析 → 提取文本 → result1
   响应2 → JSON 解析 → 提取文本 → result2
   响应3 → JSON 解析 → 提取文本 → result3
        ↓
5. 返回结果列表
   results = [result1, result2, result3]
```

### 非流式响应的完整解析

```python
# 服务器返回的原始 JSON
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "qwen/qwen2.5-0.5b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "人工智能是模拟人类智能的技术系统..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60
    }
}

# 提取后的结果
{
    "success": True,
    "text": "人工智能是模拟人类智能的技术系统...",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60
    },
    "finish_reason": "stop",
    "model": "qwen/qwen2.5-0.5b-instruct",
    "id": "chatcmpl-abc123",
    "status": 200
}
```

### 流式响应的完整解析

```python
# 服务器返回的流式数据（SSE 格式）
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"人工"},"index":0}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"智能"},"index":0}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"是"},"index":0}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"模拟"},"index":0}]}
...
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}}
data: [DONE]

# 提取后的结果（累积所有 chunks）
{
    "success": True,
    "text": "人工智能是模拟...",  # 所有 chunks 拼接
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60
    },
    "finish_reason": "stop",
    "status": 200
}
```

## 💡 结果处理的最佳实践

### 1. 检查成功状态

```python
for i, result in enumerate(results):
    if result["success"]:
        # 处理成功的结果
        process_success_result(result)
    else:
        # 处理失败的结果
        handle_error(result)
```

### 2. 提取文本内容

```python
# 提取所有成功的文本
texts = [
    result["text"]
    for result in results
    if result["success"]
]
```

### 3. 统计 Token 使用

```python
# 计算总 Token 使用
total_tokens = sum(
    result.get("usage", {}).get("total_tokens", 0)
    for result in results
    if result["success"]
)

print(f"总 Token 数: {total_tokens}")
```

### 4. 错误处理

```python
# 收集所有错误
errors = [
    (i, result["error"])
    for i, result in enumerate(results)
    if not result["success"]
]

if errors:
    print(f"有 {len(errors)} 个请求失败:")
    for idx, error in errors:
        print(f"  请求 {idx}: {error}")
```

### 5. 保存结果到文件

```python
import json

# 保存所有结果
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 只保存成功的文本
successful_texts = [
    {"index": i, "comment": comments[i], "analysis": result["text"]}
    for i, result in enumerate(results)
    if result["success"]
]

with open("successful_results.json", "w", encoding="utf-8") as f:
    json.dump(successful_texts, f, ensure_ascii=False, indent=2)
```

## 🎓 总结

### 结果结构

```python
results = [
    {
        "success": bool,      # 是否成功
        "text": str,          # 生成的文本
        "usage": {...},       # Token 统计
        "status": int,        # HTTP 状态码
        "error": str,         # 错误信息（如果失败）
        # ... 其他字段
    },
    # ... 更多结果
]
```

### 如何得到结果

1. **发送请求**: 使用 `aiohttp.post()` 发送 HTTP 请求
2. **接收响应**: 获取 HTTP 响应（JSON 或 SSE 流）
3. **解析响应**: 从响应中提取 `choices[0].message.content`
4. **格式化**: 整理成统一的结果格式
5. **返回列表**: 所有结果组成列表返回

### 关键理解

> **每个请求对应一个结果，顺序保持一致**
> 
> `results[0]` 对应 `requests[0]`  
> `results[1]` 对应 `requests[1]`  
> `results[2]` 对应 `requests[2]`  
> ...

---

## 📚 相关资源

- [process_batch 具体实现详解](./08_process_batch具体实现详解.md)
- [多请求场景与批处理](./07_多请求场景与批处理.md)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference/chat)

