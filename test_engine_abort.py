#!/usr/bin/env python3
"""
Test script for Engine abort functionality
"""

import asyncio
import threading
import time

from sglang.srt.entrypoints.engine import Engine


def test_engine_abort():
    """Test abort functionality in Engine mode"""

    # 创建引擎实例
    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf",
        trust_remote_code=True,
        max_new_tokens=100,
        log_level="info",
    )

    print("Engine created successfully")

    # 测试中止所有请求
    print("\n=== Testing abort_all ===")
    engine.abort_request(abort_all=True)
    print("✓ abort_all completed")

    # 测试中止特定请求
    print("\n=== Testing abort specific request ===")
    engine.abort_request(rid="test_request_id")
    print("✓ abort specific request completed")

    # 关闭引擎
    engine.shutdown()
    print("\n✓ Engine shutdown completed")


def test_engine_with_generation():
    """Test abort during generation"""

    # 创建引擎实例
    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf",
        trust_remote_code=True,
        max_new_tokens=100,
        log_level="info",
    )

    print("Engine created successfully")

    # 启动一个长时间运行的生成任务
    def long_generation():
        try:
            result = engine.generate(
                prompt="Write a very long story about a magical forest:",
                sampling_params={"max_new_tokens": 1000, "temperature": 0.7},
            )
            print("Generation completed:", result)
        except Exception as e:
            print("Generation interrupted:", e)

    # 在后台运行生成任务
    gen_thread = threading.Thread(target=long_generation)
    gen_thread.start()

    # 等待一段时间让生成开始
    time.sleep(2)

    # 中止所有请求
    print("Aborting all requests...")
    engine.abort_request(abort_all=True)

    # 等待线程结束
    gen_thread.join(timeout=5)

    # 关闭引擎
    engine.shutdown()
    print("✓ Test completed")


if __name__ == "__main__":
    print("Testing Engine abort functionality...")

    # 运行基本测试
    test_engine_abort()

    # 运行生成测试（可选，需要模型）
    # test_engine_with_generation()

    print("\nAll tests completed!")
