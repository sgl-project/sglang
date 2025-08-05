#!/usr/bin/env python3
"""
Realistic test for Engine abort functionality
Send 1000 requests, wait for 800 to complete, then abort the remaining 200
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from sglang.srt.entrypoints.engine import Engine
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class AbortTestEngine:
    def __init__(self, model_name: str = DEFAULT_SMALL_MODEL_NAME_FOR_TEST):
        self.engine = Engine(model=model_name, trust_remote_code=True, log_level="info")
        self.completed_requests = 0
        self.total_requests = 0
        self.request_results: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def generate_single_request(self, request_id: str) -> Dict:
        """生成单个请求"""
        try:
            result = self.engine.generate(
                prompt=f"Request {request_id}: Write a short story about a robot learning to paint.",
                sampling_params={"max_new_tokens": 50, "temperature": 0.7},
            )

            with self.lock:
                self.completed_requests += 1
                self.request_results[request_id] = {
                    "status": "completed",
                    "result": result,
                    "completed_at": time.time(),
                }

            return {"request_id": request_id, "status": "completed", "result": result}

        except Exception as e:
            with self.lock:
                self.completed_requests += 1
                self.request_results[request_id] = {
                    "status": "error",
                    "error": str(e),
                    "completed_at": time.time(),
                }
            return {"request_id": request_id, "status": "error", "error": str(e)}

    def monitor_progress(self, target_completion: int):
        """监控进度，当达到目标完成数时中止剩余请求"""
        while self.completed_requests < target_completion:
            time.sleep(0.1)
            if self.completed_requests >= target_completion:
                print(
                    f"\n🎯 Target reached: {self.completed_requests}/{self.total_requests} completed"
                )
                print("🚫 Aborting remaining requests...")
                self.engine.abort_request(abort_all=True)
                break

    def run_realistic_abort_test(
        self, total_requests: int = 1000, target_completion: int = 800
    ):
        """运行真实的abort测试"""
        print(f"🚀 Starting realistic abort test:")
        print(f"   - Total requests: {total_requests}")
        print(f"   - Target completion: {target_completion}")
        print(f"   - Will abort after: {target_completion} completions")
        print("=" * 60)

        self.total_requests = total_requests
        self.completed_requests = 0
        self.request_results.clear()

        # 生成请求ID列表
        request_ids = [f"req_{i:04d}" for i in range(total_requests)]

        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_progress, args=(target_completion,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        start_time = time.time()

        # 使用线程池并发执行请求
        with ThreadPoolExecutor(max_workers=20) as executor:
            # 提交所有请求
            future_to_request_id = {
                executor.submit(self.generate_single_request, rid): rid
                for rid in request_ids
            }

            # 等待所有请求完成
            for future in as_completed(future_to_request_id):
                request_id = future_to_request_id[future]
                try:
                    result = future.result()
                    # 结果已经在generate_single_request中处理了
                except Exception as e:
                    print(f"❌ Exception for {request_id}: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        # 统计结果
        completed_count = sum(
            1 for r in self.request_results.values() if r["status"] == "completed"
        )
        error_count = sum(
            1 for r in self.request_results.values() if r["status"] == "error"
        )
        aborted_count = total_requests - len(self.request_results)

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS:")
        print(f"   - Total requests: {total_requests}")
        print(f"   - Completed: {completed_count}")
        print(f"   - Errors: {error_count}")
        print(f"   - Aborted: {aborted_count}")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Requests per second: {total_requests/total_time:.2f}")
        print(f"   - Completion rate: {completed_count/total_requests*100:.1f}%")

        # 分析完成时间分布
        if self.request_results:
            completion_times = [
                r["completed_at"] - start_time
                for r in self.request_results.values()
                if "completed_at" in r
            ]
            if completion_times:
                avg_completion_time = sum(completion_times) / len(completion_times)
                print(f"   - Average completion time: {avg_completion_time:.2f}s")

        return {
            "total_requests": total_requests,
            "completed": completed_count,
            "errors": error_count,
            "aborted": aborted_count,
            "total_time": total_time,
            "completion_rate": completed_count / total_requests * 100,
        }

    def shutdown(self):
        """关闭引擎"""
        self.engine.shutdown()


def test_with_different_scenarios():
    """测试不同场景"""
    scenarios = [
        {"total": 100, "target": 80, "name": "Small scale test"},
        {"total": 500, "target": 400, "name": "Medium scale test"},
        {"total": 1000, "target": 800, "name": "Large scale test"},
    ]

    results = []

    for scenario in scenarios:
        print(f"\n🔬 Running: {scenario['name']}")
        print(
            f"   {scenario['total']} requests, abort after {scenario['target']} completions"
        )

        test_engine = AbortTestEngine()

        try:
            result = test_engine.run_realistic_abort_test(
                total_requests=scenario["total"], target_completion=scenario["target"]
            )
            results.append((scenario["name"], result))

        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            test_engine.shutdown()

        # 等待一下再开始下一个测试
        time.sleep(2)

    # 打印总结
    print("\n" + "=" * 60)
    print("📈 SUMMARY:")
    for name, result in results:
        print(f"   {name}:")
        print(f"     - Completion rate: {result['completion_rate']:.1f}%")
        print(f"     - Aborted: {result['aborted']} requests")
        print(f"     - Time: {result['total_time']:.2f}s")


def test_abort_timing_analysis():
    """测试abort时机分析"""
    print("\n🔍 ABORT TIMING ANALYSIS")
    print("Testing different abort timing scenarios...")

    test_engine = AbortTestEngine()

    try:
        # 测试不同完成率下的abort效果
        completion_rates = [0.5, 0.7, 0.8, 0.9]

        for rate in completion_rates:
            total = 200
            target = int(total * rate)

            print(f"\n📊 Testing {rate*100:.0f}% completion rate:")
            print(f"   Total: {total}, Target: {target}")

            result = test_engine.run_realistic_abort_test(total, target)

            print(f"   Result: {result['completed']}/{total} completed")
            print(f"   Aborted: {result['aborted']} requests")
            print(f"   Time: {result['total_time']:.2f}s")

            time.sleep(1)  # 短暂休息

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    finally:
        test_engine.shutdown()


if __name__ == "__main__":
    print("🧪 REALISTIC ENGINE ABORT TEST")
    print("=" * 60)

    # 运行主要测试
    test_with_different_scenarios()

    # 运行时机分析测试
    test_abort_timing_analysis()

    print("\n✅ All tests completed!")
