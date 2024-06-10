import subprocess
import time

# 启动 sglang 服务器
server_cmd = [
    'python3', '-m', 'sglang.launch_server',
    '--model-path', '/home/users/ntu/chih0001/scratch/model/llava-v1.5-7b',
    '--tokenizer-path', '/home/users/ntu/chih0001/scratch/model/llava-1.5-7b-hf',
    '--port', '30000'
]

print("Starting sglang server...")
server_process = subprocess.Popen(server_cmd)

# 等待服务器启动
print("Waiting for sglang server to start...")
time.sleep(30)  # 这里可以根据实际情况调整等待时间

# 运行 bench_sglang_llava_cot.py
benchmark_cmd = [
    'python3', 'bench_sglang_llava_cot.py',
    # '--model-path', '/home/users/ntu/chih0001/scratch/model/llava-v1.5-7b',
    '--question-file', '/home/users/ntu/chih0001/scratch/VLM/sglang_fork/benchmark/llava_bench/questions_cot.jsonl',
    '--answer-file', '/home/users/ntu/chih0001/scratch/VLM/sglang_fork/benchmark/llava_bench/answers_cot.jsonl',
    '--image-folder', '/home/users/ntu/chih0001/scratch/VLM/sglang_fork/benchmark/llava_bench',
    '--temperature', '0.0',
    '--num-questions', '10',
    '--max-tokens', '768',
    '--parallel', '1'
]

print("Running bench_sglang_llava_cot.py...")
subprocess.run(benchmark_cmd)

# 杀死服务器进程
print("Stopping sglang server...")
server_process.terminate()

print("Done.")
