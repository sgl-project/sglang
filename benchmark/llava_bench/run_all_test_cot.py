import subprocess
import time
import socket
import torch

def check_server(port, host='localhost'):
    """Check if a server is listening on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

# python3 -m sglang.launch_server --model-path /home/users/ntu/chih0001/scratch/model/llava-v1.5-7b --tokenizer-path /home/users/ntu/chih0001/scratch/model/llava-1.5-7b-hf --port 30000
def main():
    
    # If CUDA is available, clear the cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Launch the server
    server_command = [
        'python3', '-m', 'sglang.launch_server',
        '--model-path', '/home/users/ntu/chih0001/scratch/model/llava-v1.5-7b',
        '--tokenizer-path', '/home/users/ntu/chih0001/scratch/model/llava-1.5-7b-hf',
        '--port', '30000'
    ]
    print(f"Starting server with command: {' '.join(server_command)}")
    server_process = subprocess.Popen(server_command)

    # Wait for the server to start
    server_started = False
    max_attempts = 200
    attempt = 0
    while not server_started and attempt < max_attempts:
        attempt += 1  # Move attempt increment to the beginning of the loop
        if attempt % 30 == 0 or attempt == 1:  # Only print every 30 attempts, and the first one
            print(f"Checking if server is up (attempt {attempt}/{max_attempts})...")
        server_started = check_server(30000)
        if server_started:
            print("Server is up and running!")
        else:
            time.sleep(1)  # Wait for 1 second before trying again

    if not server_started:
        print("Server failed to start after several attempts.")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        return  # Exit the program if the server didn't start

    # 
    try:
        # Run the benchmark test
        bench_command = [
            # 正常的
            # 'python3', 'bench_sglang_driver.py',
            # 调试的(2024/-3/22目前是好用的)
            'python3', 'bench_sglang_driver_cot.py',
            # '--question-file', '/home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench/questions/questions_ultrasimplified.jsonl' 
            '--question-file', '/home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench/questions/test_cot.jsonl' 
            # '--question-file', '/home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench/questions/questions_detailed.jsonl' 
            # '--num-questions', '10' 
            # '--num-questions', 'None' # None代表全部
        ]
        print(f"Running benchmark with command: {' '.join(bench_command)}")
        subprocess.check_call(bench_command)
    finally:
        # Shutdown the server
        print("Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)  # Increase timeout if needed
        except subprocess.TimeoutExpired:
            print("Forcing server shutdown...")
            server_process.kill()




if __name__ == '__main__':
    main()
