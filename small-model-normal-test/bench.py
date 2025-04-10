import os
import sys
import time
import subprocess
import openai_benchmark
import logging
import random
import numpy as np
import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "20000"

REMOTE_ADDR = "127.0.0.1"
REMOTE_SSH_PORT = "2222"

# MODEL="/home/qspace/upload/luban_cache/model/luban-llm_deepseek_v3-model_path/DeepSeek-V3/"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

SGLANG_COMMON_ARGS = [
  f"--model-path", MODEL,
  "--trust-remote-code",
  # "--disable-radix-cache", # for large bs cache reuse
  "--schedule-policy", "fcfs",
  "--tp", "1",
  "--dist-init-addr", f"{MASTER_ADDR}:{MASTER_PORT}",
  "--host", "0.0.0.0",
  "--port", "8080",
  "--mem-fraction-static", "0.70",
  "--disable-overlap-schedule",
  "--chunked-prefill-size 32768",
  "--allow-auto-truncate"
]



REMOTE_OUTPUT_LOG='/tmp/sgl-remote.log'
LOCAL_OUTPUT_LOG='/tmp/sgl-local.log'
BENCH_OUTPUT_LOG='/tmp/sgl-bench.log'
os.system("rm -rf " + REMOTE_OUTPUT_LOG + " " + LOCAL_OUTPUT_LOG)

# create two stream to use in Popen
local_output_log = open(LOCAL_OUTPUT_LOG, 'w')
remote_output_log = open(REMOTE_OUTPUT_LOG, 'w')
bench_output_log = open(BENCH_OUTPUT_LOG, 'w')


def runCommand(cmd: list[str], remoteAddr: tuple[str, str] = None, outputStream = subprocess.DEVNULL) -> subprocess.Popen:
    if remoteAddr is not None:
      # source ~/.bashrc fails to alias proxy_on. Dirty hack to fix.
      PROXY_ON = "export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
        export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
        export http_proxy=http://hk-mmhttpproxy.woa.com:11113 && \
        export https_proxy=http://hk-mmhttpproxy.woa.com:11113"
      
      # PS1=[] dirtyhack to bypass ~/.bashrc checking
      remote_cmd = ' '.join(['ssh', '-p', remoteAddr[1], f'root@{remoteAddr[0]}', f'"PS1=[] source ~/.bashrc && env && (', *cmd, ')"'])
      logger.info(f"runCommand remotely: {remote_cmd}")
      proc = subprocess.Popen(remote_cmd, shell=True, stdout=outputStream, stderr=outputStream)
      return proc
    else:
      # run_func in a new process
      logger.info(f"runCommand locally: {' '.join(cmd)}")
      proc = subprocess.Popen(' '.join(cmd), shell=True, stdout=outputStream, stderr=outputStream)
      return proc


def wait_server(addr, port):
  import socket
  # poll the server until it is ready.
  logger.info(f"wait_server: {addr}:{port}")
  while True:
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # set s timeout 1s
        s.settimeout(1)
        s.connect((addr, port))
        # 构建一个简单的空 HTTP GET 请求
        http_request = b"GET / HTTP/1.1\r\nHost: %s:%d\r\n\r\n" % (addr.encode(), port)
        s.sendall(http_request)
        response = s.recv(4096)
        print(f"Received response: {response.decode('utf-8', errors='ignore')}")
        return
    except Exception as e:
      time.sleep(1)

def do_bs_exp():

  def clean_up():
    logger.info("clean up...")
    cleanup_cmd = "ps -ef | grep 'sglang' | grep -v grep | grep -v defunct | awk '{print \$2}' | xargs -r kill -SIGKILL; ps aux | grep 'sglang' | grep -v defunct; pkill -f sglang"
    a = runCommand([cleanup_cmd])
    a.wait()
    b = runCommand([cleanup_cmd], (REMOTE_ADDR, REMOTE_SSH_PORT), outputStream=remote_output_log)
    b.wait()
    time.sleep(10)


  clean_up()

  for bsz in tqdm.tqdm([32]):
    filename = f"0403-v3bench-bsz-{bsz}.txt"

    common_args = [
      "--max-running-requests", f"{bsz}",
    ]

    # setup sglang servers.
    sglang_local_args = SGLANG_COMMON_ARGS.copy()  + common_args.copy()

    env = []

    localServer = runCommand([f"{' '.join(env)} python3 -m sglang.launch_server"] + sglang_local_args, outputStream=local_output_log)
    # remoteServer = runCommand([f"{' '.join(env)} python3 -m sglang.launch_server"] + sglang_remote_args, (REMOTE_ADDR, REMOTE_SSH_PORT), remote_output_log)

    wait_server(MASTER_ADDR, 8080)

    logger.info("The server is ready! Wait some seconds to let the server warm up.")
    time.sleep(10)
    logger.info("Start benchmarking...")

    BENCHMARK_ARGS = [
      "--model", "default",
      "--host", f"{MASTER_ADDR}",
      "--port", "8080", # sglang server port. Not dist init port..
      "--endpoint", "/v1/chat/completions",
      "--dataset-name", "jsonl",
      "--num-prompts", f"{ 280 }", 
      "--dataset-path", "/sgl-workspace/upload/dataset/qa_out_0216_r1_300_max_25k_formatted.jsonl",
      "--max-concurrency", f"{ bsz }",
      "--backend", f"openai-chat",
      "--tokenizer", f"{MODEL}",
      "--jsonl-output-len", "4096",
      "--save-result",
      "--result-filename", filename
    ]
    benchmarkClient = runCommand(["python3 -m openai_benchmark.benchmark_serving"] + BENCHMARK_ARGS, outputStream=bench_output_log)
    benchmarkClient.wait()

    # shutdown sglang servers.
    clean_up()





if __name__ == '__main__':
  do_bs_exp()

