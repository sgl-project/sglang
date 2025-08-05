#!/usr/bin/env python3
"""
Example: Using abort functionality in Engine mode

This example demonstrates how to use the abort functionality in SGLang Engine mode.
"""

import asyncio
import threading
import time

from sglang.srt.entrypoints.engine import Engine


def example_basic_abort():
    """Basic abort functionality example"""

    # 创建引擎
    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, log_level="info"
    )

    print("=== Basic Abort Example ===")

    # 中止所有请求
    engine.abort_request(abort_all=True)
    print("✓ Aborted all requests")

    # 中止特定请求
    engine.abort_request(rid="specific_request_id")
    print("✓ Aborted specific request")

    engine.shutdown()


def example_abort_during_generation():
    """Example of aborting during generation"""

    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, log_level="info"
    )

    print("=== Abort During Generation Example ===")

    # 启动长时间生成任务
    def long_generation():
        try:
            result = engine.generate(
                prompt="Write a detailed story about space exploration:",
                sampling_params={"max_new_tokens": 500, "temperature": 0.8},
            )
            print("Generation result:", result)
        except Exception as e:
            print(f"Generation was interrupted: {e}")

    # 在后台运行生成
    gen_thread = threading.Thread(target=long_generation)
    gen_thread.start()

    # 等待生成开始
    time.sleep(1)

    # 中止所有请求
    print("Aborting generation...")
    engine.abort_request(abort_all=True)

    # 等待线程结束
    gen_thread.join(timeout=3)

    engine.shutdown()


def example_multiple_requests_abort():
    """Example of aborting specific requests among multiple requests"""

    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, log_level="info"
    )

    print("=== Multiple Requests Abort Example ===")

    # 模拟多个请求
    request_ids = ["req_1", "req_2", "req_3", "req_4"]

    # 中止特定请求
    for rid in request_ids:
        engine.abort_request(rid=rid)
        print(f"✓ Aborted request: {rid}")

    # 中止所有请求
    engine.abort_request(abort_all=True)
    print("✓ Aborted all remaining requests")

    engine.shutdown()


def example_error_handling():
    """Example with error handling"""

    engine = Engine(
        model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, log_level="info"
    )

    print("=== Error Handling Example ===")

    try:
        # 尝试中止不存在的请求
        engine.abort_request(rid="non_existent_request")
        print("✓ No error when aborting non-existent request")

        # 正常中止
        engine.abort_request(abort_all=True)
        print("✓ Successfully aborted all requests")

    except Exception as e:
        print(f"Error during abort: {e}")

    finally:
        engine.shutdown()


if __name__ == "__main__":
    print("SGLang Engine Abort Examples")
    print("=" * 40)

    # 运行所有示例
    example_basic_abort()
    print()

    example_multiple_requests_abort()
    print()

    example_error_handling()
    print()

    # 注释掉需要实际模型的示例
    # example_abort_during_generation()

    print("All examples completed!")
