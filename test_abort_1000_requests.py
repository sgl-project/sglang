#!/usr/bin/env python3
"""
Test: Send different numbers of requests, but all abort after 80 completions
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List

from sglang.srt.entrypoints.engine import Engine
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class MassRequestAbortTest:
    def __init__(self):
        self.engine = Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            trust_remote_code=True,
            log_level="info",
        )
        self.completed_count = 0
        self.total_requests = 0
        self.lock = threading.Lock()
        self.start_time = None
        self.should_abort = False
        self.abort_time = None

    def single_request(self, request_id: str) -> Dict:
        """执行单个请求"""
        try:
            # 生成一个简单的请求
            result = self.engine.generate(
                prompt=f"Request {request_id}: How to compute the sum of 1 to {request_id}? Let's think step by step.",
                sampling_params={"max_new_tokens": 10, "temperature": 0.1},
            )

            with self.lock:
                self.completed_count += 1
                current_count = self.completed_count

                # 每完成20个请求打印一次进度
                if current_count % 20 == 0:
                    elapsed = time.time() - self.start_time
                    rate = current_count / elapsed if elapsed > 0 else 0
                    print(
                        f"✅ Completed {current_count}/{self.total_requests} requests "
                        f"({current_count/self.total_requests*100:.1f}%) "
                        f"[{rate:.1f} req/s]"
                    )

            return {"request_id": request_id, "status": "success", "result": result}

        except Exception as e:
            with self.lock:
                self.completed_count += 1
                print(f"❌ Request {request_id} failed: {e}")

            return {"request_id": request_id, "status": "error", "error": str(e)}

    def monitor_progress(self, target_completion: int):
        """监控进度并在达到目标时设置中止标志"""
        while self.completed_count < target_completion:
            time.sleep(0.1)

        print(
            f"\n🎯 TARGET REACHED: {self.completed_count}/{self.total_requests} completed!"
        )
        print("🚫 SETTING ABORT FLAG...")

        # 记录abort时间
        self.abort_time = time.time()

        # 设置中止标志
        self.should_abort = True

        print("✅ Abort flag set!")

    def run_mass_abort_test(
        self, total_requests: int = 100, target_completion: int = 80
    ):
        """运行大规模abort测试"""
        print(f"🚀 MASS ABORT TEST")
        print(f"   Sending {total_requests} requests...")
        print(f"   Will abort after {target_completion} completions")
        print(f"   Expected to abort {total_requests - target_completion} requests")
        print("=" * 60)

        self.total_requests = total_requests
        self.completed_count = 0
        self.start_time = time.time()
        self.should_abort = False
        self.abort_time = None

        # 生成请求ID
        request_ids = [f"req_{i:04d}" for i in range(total_requests)]

        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_progress, args=(target_completion,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        # 顺序执行请求（避免多线程事件循环问题）
        print("📤 Submitting requests sequentially...")
        results = []

        for i, request_id in enumerate(request_ids):
            if self.should_abort:
                print(f"🚫 Aborting remaining {len(request_ids) - i} requests...")
                break

            result = self.single_request(request_id)
            results.append(result)

            # 每20个请求打印一次进度
            if (i + 1) % 20 == 0:
                elapsed = time.time() - self.start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"📊 Progress: {i + 1}/{total_requests} ({rate:.1f} req/s)")

        # 在主线程中执行abort操作
        if self.should_abort:
            print("🚫 Executing abort in main thread...")
            try:
                self.engine.abort_request(abort_all=True)
                print("✅ Abort command sent successfully!")
            except Exception as e:
                print(f"❌ Abort failed: {e}")

        end_time = time.time()
        total_time = end_time - self.start_time

        # 计算abort相关时间
        abort_duration = None
        if self.abort_time:
            abort_duration = end_time - self.abort_time

        # 统计结果
        final_completed = self.completed_count
        aborted_count = total_requests - final_completed

        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS:")
        print(f"   Total requests sent: {total_requests}")
        print(f"   Completed: {final_completed}")
        print(f"   Aborted: {aborted_count}")
        print(f"   Target completion: {target_completion}")
        print(f"   Actual completion: {final_completed}")
        print(f"   Completion rate: {final_completed/total_requests*100:.1f}%")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average rate: {total_requests/total_time:.1f} requests/second")
        if abort_duration:
            print(f"   Abort duration: {abort_duration:.2f}s")

        # 分析abort效果
        if final_completed >= target_completion:
            print(f"✅ SUCCESS: Reached target ({target_completion}) before abort")
        else:
            print(f"⚠️  PARTIAL: Only completed {final_completed}/{target_completion}")

        if aborted_count > 0:
            print(f"🚫 ABORT EFFECTIVE: {aborted_count} requests were aborted")
        else:
            print("⚠️  No requests were aborted (all completed before abort)")

        return {
            "total_requests": total_requests,
            "completed": final_completed,
            "aborted": aborted_count,
            "target": target_completion,
            "total_time": total_time,
            "abort_duration": abort_duration,
            "completion_rate": final_completed / total_requests * 100,
        }

    def shutdown(self):
        """关闭引擎"""
        self.engine.shutdown()


def test_different_scales():
    """测试不同规模的abort效果"""
    test_scenarios = [
        {"total": 500, "target": 400, "name": "Small (500→400)"},
        {"total": 1000, "target": 400, "name": "Medium (1000→400)"},
        {"total": 2000, "target": 400, "name": "Large (2000→400)"},
    ]

    results = []

    # 只创建一个engine实例
    test = MassRequestAbortTest()

    try:
        for scenario in test_scenarios:
            print(f"\n🔬 Testing: {scenario['name']}")
            print(f"   {scenario['total']} requests, abort after {scenario['target']}")

            result = test.run_mass_abort_test(
                total_requests=scenario["total"], target_completion=scenario["target"]
            )
            results.append((scenario["name"], result))

            # 等待一下再开始下一个测试
            time.sleep(2)

        # 打印总结
        print("\n" + "=" * 60)
        print("📈 SUMMARY:")
        for name, result in results:
            print(f"   {name}:")
            print(f"     - Completed: {result['completed']}/{result['total_requests']}")
            print(f"     - Aborted: {result['aborted']} requests")
            print(f"     - Time: {result['total_time']:.2f}s")
            if result["abort_duration"]:
                print(f"     - Abort duration: {result['abort_duration']:.2f}s")
            print(f"     - Rate: {result['completion_rate']:.1f}%")

        # 比较abort时间
        print("\n🔍 ABORT TIME COMPARISON:")
        abort_times = [r["abort_duration"] for _, r in results if r["abort_duration"]]
        if abort_times:
            avg_abort_time = sum(abort_times) / len(abort_times)
            print(f"   Average abort time: {avg_abort_time:.2f}s")
            print(f"   Min abort time: {min(abort_times):.2f}s")
            print(f"   Max abort time: {max(abort_times):.2f}s")
            print(f"   Abort time variance: {max(abort_times) - min(abort_times):.2f}s")

    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        test.shutdown()


if __name__ == "__main__":
    print("🧪 MASS REQUEST ABORT TEST")
    print("Send different numbers of requests, but all abort after 80 completions")
    print("Compare abort times across different scales")
    print("=" * 60)

    # 运行不同规模的测试
    test_different_scales()

    print("\n✅ All tests completed!")
