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


PREFILL_ADDR = "127.0.0.1"
# PREFILL_INIT_PORT = "20000"
PREFILL_SSH_PORT = "2222"
PREFILL_SERVE_PORT = "8080"

DECODE_ADDR = "127.0.0.1"
# DECODE_INIT_PORT = "20000"
DECODE_SSH_PORT = "2222"
DECODE_SERVE_PORT = "8090"

LB_ADDR = "127.0.0.1"
LB_SERVE_PORT = "8100"

EXTRA_SSH_ARGS = "" # "-i ~/ytwu/.ssh/id_ed25519"

NETDEVICE = "eth0"

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL="/home/qspace/upload/luban_cache/model/luban-llm_deepseek_v3-model_path/DeepSeek-V3/"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# SGLANG_PATH = "/root/ytwu/save/sglang"
# SGLANG_PATH = ""

SGLANG_COMMON_ARGS = [
  f"--model-path", MODEL,
  "--trust-remote-code",
  "--disable-radix-cache",
  "--schedule-policy", "fcfs",
  "--host", "0.0.0.0",
  "--mem-fraction-static", "0.70",
  "--disable-overlap-schedule",
  "--chunked-prefill-size 32768",
  "--allow-auto-truncate" if MODEL == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' else ''
]

# in future we will have multiple nodes in P or D instances.
# "--tp", "1",
# "--dist-init-addr", f"{MASTER_ADDR}:{MASTER_PORT}",
# "--nnodes", "2",

PREFILL_ARGS = [
  "--tp 1",
  "--nnodes 1",
  "--disaggregation-mode prefill",
  "--port", f"{PREFILL_SERVE_PORT}",
]

DECODE_ARGS = [
  "--tp 1",
  "--nnodes 1",
  "--disaggregation-mode decode",
  "--port", f"{DECODE_SERVE_PORT}",
]

PREFILL_OUTPUT_LOG='/tmp/sgl-prefill.log'
DECODE_OUTPUT_LOG='/tmp/sgl-decode.log'
BENCH_OUTPUT_LOG='/tmp/sgl-bench.log'
LB_OUTPUT_LOG='/tmp/sgl-lb.log'
os.system("rm -rf " + PREFILL_OUTPUT_LOG + " " + DECODE_OUTPUT_LOG + " " + BENCH_OUTPUT_LOG + " " + LB_OUTPUT_LOG)

prefill_output_log = open(PREFILL_OUTPUT_LOG, 'w')
decode_output_log = open(DECODE_OUTPUT_LOG, 'w')
bench_output_log = open(BENCH_OUTPUT_LOG, 'w')
lb_output_log = open(LB_OUTPUT_LOG, 'w')

remotes = [
  (PREFILL_ADDR, PREFILL_SSH_PORT, prefill_output_log),
  (DECODE_ADDR, DECODE_SSH_PORT, decode_output_log),
]


def runCommand(cmd: list[str], remoteAddr: tuple[str, str] = None, outputStream = subprocess.DEVNULL) -> subprocess.Popen:
    if remoteAddr is not None:
      # source ~/.bashrc fails to alias proxy_on. Dirty hack to fix.
      # PROXY_ON = "&& export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export http_proxy=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export https_proxy=http://hk-mmhttpproxy.woa.com:11113"
      
      PROXY_ON = ""

      # PS1=[] dirtyhack to bypass ~/.bashrc checking
      remote_cmd = ' '.join(['ssh', EXTRA_SSH_ARGS, '-p', remoteAddr[1], f'root@{remoteAddr[0]}', f'"PS1=[] source ~/.bashrc {PROXY_ON} && env && (', *cmd, ')"'])
      logger.info(f"runCommand remotely: {remote_cmd}")
      proc = subprocess.Popen(remote_cmd, shell=True, stdout=outputStream, stderr=outputStream)
      return proc
    else:
      # run_func in a new process
      logger.info(f"runCommand locally: {' '.join(cmd)}")
      proc = subprocess.Popen(' '.join(cmd), shell=True, stdout=outputStream, stderr=outputStream)
      return proc


def wait_server(addr, port):
  port=int(port)
  import socket
  # poll the server until it is ready.
  logger.info(f"wait_server: {addr}:{port}")
  while True:
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # set s timeout 1s
        s.settimeout(1)
        s.connect((addr, port))
        http_request = b"GET / HTTP/1.1\r\nHost: %s:%d\r\n\r\n" % (addr.encode(), port)
        s.sendall(http_request)
        response = s.recv(4096)
        print(f"Received response: {response.decode('utf-8', errors='ignore')}")
        return
    except socket.error as e:
      time.sleep(1)
      continue
    

def do_exp():

  def clean_up():
    logger.info("clean up...")
    cleanup_cmd = "ps -ef | grep 'sglang' | grep -v grep | grep -v defunct | awk '{print \$2}' | xargs -r kill -SIGKILL; ps aux | grep 'sglang' | grep -v defunct; pkill -f sglang"
    for addr, ssh_port, output_log in remotes:
      b = runCommand([cleanup_cmd], (addr, ssh_port), output_log)
      b.wait()
    time.sleep(10)

  clean_up()

  for bsz in tqdm.tqdm([1]):
    filename = f"0408-pd-{bsz}.txt"

    exp_varying_args = [
      "--max-running-requests", f"{bsz}",
    ]

    # setup sglang servers.
    sglang_prefill_args = SGLANG_COMMON_ARGS.copy() + PREFILL_ARGS.copy() + exp_varying_args.copy()
    sglang_decode_args = SGLANG_COMMON_ARGS.copy() + DECODE_ARGS.copy() + exp_varying_args.copy()

    prefill_env = [
      "CUDA_VISIBLE_DEVICES=0",
      "UCX_TLS=tcp,cuda",
      f"UCX_NET_DEVICES={NETDEVICE}",
      "UCX_LOG_LEVEL=info"
    ]

    decode_env = [
      "CUDA_VISIBLE_DEVICES=1",
      "UCX_TLS=tcp,cuda",
      f"UCX_NET_DEVICES={NETDEVICE}",
      "UCX_LOG_LEVEL=info"
    ]

    prefillServer = runCommand([f"{' '.join(prefill_env)} python3 -m sglang.launch_server"] + sglang_prefill_args, (PREFILL_ADDR, PREFILL_SSH_PORT), prefill_output_log)
    decodeServer = runCommand([f"{' '.join(decode_env)} python3 -m sglang.launch_server"] + sglang_decode_args, (DECODE_ADDR, DECODE_SSH_PORT), decode_output_log)

    wait_server(PREFILL_ADDR, PREFILL_SERVE_PORT)
    wait_server(DECODE_ADDR, DECODE_SERVE_PORT)

    logger.info("All PD servers are ready! Wait some seconds to let the server warm up.")
    time.sleep(10)

    lb = runCommand([
      "python3 -m sglang.srt.disaggregation.mini_lb",
      "--prefill", f"http://{PREFILL_ADDR}:{PREFILL_SERVE_PORT}",
      "--decode", f"http://{DECODE_ADDR}:{DECODE_SERVE_PORT}",
      "--host 0.0.0.0", 
      "--port", f"{LB_SERVE_PORT}",
    ], outputStream=lb_output_log)

    time.sleep(1)
    logger.info("Start benchmarking...")

    BENCHMARK_ARGS = [
      "--model", "default",
      "--host", f"{LB_ADDR}",
      "--port", f"{LB_SERVE_PORT}",
      "--endpoint", "/v1/chat/completions",
      "--dataset-name", "jsonl",
      "--num-prompts", f"{ 10 }", 
      # "--dataset-path", "/sgl-workspace/upload/dataset/qa_out_0216_r1_300_max_25k_formatted.jsonl",
      "--dataset-path", "/sgl-workspace/upload/dataset/easy.jsonl",
      "--max-concurrency", f"{ bsz }",
      "--backend", f"openai-chat",
      "--tokenizer", f"{MODEL}",
      "--jsonl-output-len", "128",
      "--save-result",
      "--result-filename", filename
    ]
    benchmarkClient = runCommand(["python3 -m openai_benchmark.benchmark_serving"] + BENCHMARK_ARGS, outputStream=bench_output_log)
    benchmarkClient.wait()

    # shutdown sglang servers.
    clean_up()





if __name__ == '__main__':
  do_exp()

