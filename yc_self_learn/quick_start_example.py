"""
SGLang 快速开始示例
这个脚本展示了如何使用 SGLang 进行基本的文本生成
"""

import openai
import requests
import json

# ============================================
# 配置
# ============================================
SERVER_URL = "http://127.0.0.1:30000"
MODEL_NAME = "qwen/qwen2.5-0.5b-instruct"  # 你可以替换为其他模型

# ============================================
# 方法 1: 使用 OpenAI 客户端（推荐）
# ============================================
def example_openai_client():
    """使用 OpenAI 兼容的客户端"""
    print("=" * 60)
    print("方法 1: 使用 OpenAI 客户端")
    print("=" * 60)
    
    client = openai.Client(
        base_url=f"{SERVER_URL}/v1",
        api_key="None"  # SGLang 不需要 API key
    )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "用一句话解释什么是人工智能。"},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    
    print(f"回答: {response.choices[0].message.content}\n")


# ============================================
# 方法 2: 使用流式输出
# ============================================
def example_streaming():
    """流式输出示例"""
    print("=" * 60)
    print("方法 2: 流式输出")
    print("=" * 60)
    
    client = openai.Client(
        base_url=f"{SERVER_URL}/v1",
        api_key="None"
    )
    
    print("回答: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "列出3个编程语言及其特点。"},
        ],
        temperature=0.7,
        max_tokens=150,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


# ============================================
# 方法 3: 使用 requests 库
# ============================================
def example_requests():
    """使用 requests 库直接调用 API"""
    print("=" * 60)
    print("方法 3: 使用 requests 库")
    print("=" * 60)
    
    url = f"{SERVER_URL}/v1/chat/completions"
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Python 和 JavaScript 的主要区别是什么？"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"回答: {result['choices'][0]['message']['content']}\n")


# ============================================
# 方法 4: 使用原生 Generate API
# ============================================
def example_native_api():
    """使用 SGLang 原生的 Generate API"""
    print("=" * 60)
    print("方法 4: 使用原生 Generate API")
    print("=" * 60)
    
    url = f"{SERVER_URL}/generate"
    
    data = {
        "text": "机器学习是",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 50,
        },
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"完整文本: {result['text']}\n")


# ============================================
# 方法 5: 多轮对话
# ============================================
def example_multi_turn():
    """多轮对话示例"""
    print("=" * 60)
    print("方法 5: 多轮对话")
    print("=" * 60)
    
    client = openai.Client(
        base_url=f"{SERVER_URL}/v1",
        api_key="None"
    )
    
    messages = [
        {"role": "system", "content": "你是一个友好的AI助手。"},
        {"role": "user", "content": "你好！"},
    ]
    
    # 第一轮
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=50,
    )
    
    assistant_reply = response.choices[0].message.content
    print(f"用户: 你好！")
    print(f"助手: {assistant_reply}")
    
    # 添加助手回复到消息历史
    messages.append({"role": "assistant", "content": assistant_reply})
    
    # 第二轮
    messages.append({"role": "user", "content": "你能帮我学习编程吗？"})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )
    
    print(f"用户: 你能帮我学习编程吗？")
    print(f"助手: {response.choices[0].message.content}\n")


# ============================================
# 方法 6: 带参数的生成
# ============================================
def example_with_params():
    """展示不同参数的效果"""
    print("=" * 60)
    print("方法 6: 不同参数的效果")
    print("=" * 60)
    
    client = openai.Client(
        base_url=f"{SERVER_URL}/v1",
        api_key="None"
    )
    
    prompt = "写一首关于春天的短诗。"
    
    # 低温度（更确定性）
    print("低温度 (temperature=0.1):")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100,
    )
    print(response.choices[0].message.content)
    print()
    
    # 高温度（更随机性）
    print("高温度 (temperature=1.5):")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5,
        max_tokens=100,
    )
    print(response.choices[0].message.content)
    print()


# ============================================
# 主函数
# ============================================
def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("SGLang 快速开始示例")
    print("=" * 60)
    print(f"\n确保 SGLang 服务器正在运行在 {SERVER_URL}")
    print(f"使用模型: {MODEL_NAME}\n")
    
    try:
        # 检查服务器是否运行
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("⚠️  警告: 无法连接到服务器，请确保服务器正在运行")
            print(f"   启动命令: python3 -m sglang.launch_server --model-path {MODEL_NAME} --host 0.0.0.0 --port 30000")
            return
    except requests.exceptions.RequestException:
        print("⚠️  警告: 无法连接到服务器，请确保服务器正在运行")
        print(f"   启动命令: python3 -m sglang.launch_server --model-path {MODEL_NAME} --host 0.0.0.0 --port 30000")
        return
    
    # 运行示例
    try:
        example_openai_client()
        example_streaming()
        example_requests()
        example_native_api()
        example_multi_turn()
        example_with_params()
        
        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. SGLang 服务器正在运行")
        print("2. 模型路径正确")
        print("3. 端口号正确（默认 30000）")


if __name__ == "__main__":
    main()


