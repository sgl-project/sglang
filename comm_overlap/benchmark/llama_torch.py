import atexit
import json
import os
import signal
import statistics
import subprocess
import time

import requests


def start_server():
    print("starting server...")
    server_cmd = (
        "python -m sglang.launch_server "
        "--model-path meta-llama/Meta-Llama-3.1-8B-Instruct "
        "--port 30001 --host 0.0.0.0 "
        "--tp 4"
    )

    original_dir = os.getcwd()
    server_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "python"
    )
    os.chdir(server_dir)

    server_process = subprocess.Popen(server_cmd.split())

    os.chdir(original_dir)

    atexit.register(lambda: os.kill(server_process.pid, signal.SIGTERM))

    print("waiting for server to start...")
    time.sleep(30)
    return server_process


def test_latency(num_requests=50):
    url = "http://localhost:30001/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "text": "The president of USA,",
        "sampling_params": {"max_new_tokens": 100, "temperature": 0},
    }

    latencies = []

    print(f"start to test latency for {num_requests} requests...")

    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
                print(f"request {i+1}: {latency:.2f}ms")
                print(f"response: {response.json()}")
            else:
                print(f"request {i+1} failed: HTTP {response.status_code}")
                print(f"error: {response.text}")
        except Exception as e:
            print(f"request {i+1} failed: {str(e)}")

        time.sleep(0.5)

    if latencies:
        results = {
            "min latency": f"{min(latencies):.2f}ms",
            "max latency": f"{max(latencies):.2f}ms",
            "mean latency": f"{statistics.mean(latencies):.2f}ms",
            "median latency": f"{statistics.median(latencies):.2f}ms",
        }

        with open("latency_results_torch_{max_new_tokens}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\ntest results:")
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    server_process = start_server()
    try:
        test_latency()
    finally:
        print("closing server...")
        server_process.terminate()
        server_process.wait()
