import time
import requests
import statistics
import json
import subprocess
import os
import signal
import atexit

def start_server():
    print("启动服务器...")
    server_cmd = (
        "python -m sglang.launch_server "
        "--model-path meta-llama/Meta-Llama-3.1-8B-Instruct "
        "--port 30001 --host 0.0.0.0 "
        "--tp 4"
    )
    
    # 切换到正确的目录
    original_dir = os.getcwd()
    server_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "python")
    os.chdir(server_dir)
    
    server_process = subprocess.Popen(server_cmd.split())
    
    # 切回原来的目录
    os.chdir(original_dir)
    
    # 注册退出时清理
    atexit.register(lambda: os.kill(server_process.pid, signal.SIGTERM))
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(30)
    return server_process

def test_latency(num_requests=50):
    url = "http://localhost:30001/generate"
    headers = {"Content-Type": "application/json"}
    
    # 修改payload格式以匹配正确的API格式
    payload = {
        "text": "The president of USA,",
        "sampling_params": {
            "max_new_tokens": 100,
            "temperature": 0
        }
    }
    
    latencies = []
    
    print(f"开始测试 {num_requests} 个请求的延迟...")
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
                print(f"请求 {i+1}: {latency:.2f}ms")
                print(f"响应内容: {response.json()}")
            else:
                print(f"请求 {i+1} 失败: HTTP {response.status_code}")
                print(f"错误信息: {response.text}")
        except Exception as e:
            print(f"请求 {i+1} 失败: {str(e)}")
        
        # 请求之间稍作暂停
        time.sleep(0.5)
    
    if latencies:
        results = {
            "平均延迟": f"{statistics.mean(latencies):.2f}ms",
            "最小延迟": f"{min(latencies):.2f}ms",
            "最大延迟": f"{max(latencies):.2f}ms",
            "延迟中位数": f"{statistics.median(latencies):.2f}ms"
        }
        
        # 保存结果
        with open("latency_results_torch.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print("\n测试结果:")
        for key, value in results.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    server_process = start_server()
    try:
        test_latency()
    finally:
        print("关闭服务器...")
        server_process.terminate()
        server_process.wait()
