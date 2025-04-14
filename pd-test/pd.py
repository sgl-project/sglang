import os
import sys
import time
import subprocess
import openai_benchmark
import logging
import random
import numpy as np
import tqdm

from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# (HOST, SSH_PORT, SERVE_PORT, BOOTSTRAP_PORT, extra_env)
PREFILLS = (
  ("29.226.64.219", "2222", "8080", "9500", ["CUDA_VISIBLE_DEVICES=0,1"]),
  ("29.226.64.219", "2222", "8081", "9520", ["CUDA_VISIBLE_DEVICES=2,3"]),
)

DECODES = (
  ("29.226.64.239", "2222", "8090", "10000", ["CUDA_VISIBLE_DEVICES=4,5"]),
  ("29.226.64.239", "2222", "8091", "10010", ["CUDA_VISIBLE_DEVICES=6,7"]),
)

LB_HOST = "127.0.0.1"
LB_SSH_PORT = "2222"
LB_SERVE_PORT = "15000"

EXTRA_SSH_ARGS = "" # "-i ~/ytwu/.ssh/id_ed25519"

# NETDEVICE = "eth0"
# NETDEVICE = "lo"

# NETDEVICE="mlx5_bond_1:1"
# UCX_TLS="rc,gdr_copy,rc_x,cuda_copy,cuda_ipc"

NETDEVICE="bond1"
UCX_TLS="tcp,gdr_copy,cuda_copy,cuda_ipc"

TP=2

# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL="/home/qspace/upload/luban_cache/model/luban-llm_deepseek_v3-model_path/DeepSeek-V3/"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL='/home/qspace/upload/luban_cache/model/dpsk-1-5B/'

SGLANG_COMMON_ARGS = [
  f"--model-path", MODEL,
  "--trust-remote-code",
  "--disable-radix-cache",
  "--schedule-policy", "fcfs",
  "--host", "0.0.0.0",
  "--mem-fraction-static", "0.70",
  "--disable-overlap-schedule",
  "--chunked-prefill-size 32768",
  "--allow-auto-truncate" if MODEL == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' else '',

  "--disable-cuda-graph" # For PD.
]

# in future we will have multiple nodes in P or D instances.
# "--dist-init-addr", f"{MASTER_ADDR}:{MASTER_PORT}",
# "--nnodes", "2",

BENCH_OUTPUT_LOG='/tmp/sgl-bench.log'
LB_OUTPUT_LOG='/tmp/sgl-lb.log'
os.system("rm -rf " + BENCH_OUTPUT_LOG + " " + LB_OUTPUT_LOG)

bench_output_log = open(BENCH_OUTPUT_LOG, 'w')
lb_output_log = open(LB_OUTPUT_LOG, 'w')

all_machines: list[tuple[str, str]] = []
for addr, ssh_port, _, _, _ in PREFILLS + DECODES:
  all_machines.append((str(addr), str(ssh_port)))

# add lb
all_machines.append((LB_HOST, LB_SSH_PORT))

# remove repeating
all_machines = list(set(all_machines))

def runCommand(cmd: list[str], remoteAddr: Optional[tuple[str, str]] = None, outputStream = subprocess.DEVNULL) -> subprocess.Popen:
    if remoteAddr is not None:
      # source ~/.bashrc fails to alias proxy_on. Dirty hack to fix.
      # PROXY_ON = "&& export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export http_proxy=http://hk-mmhttpproxy.woa.com:11113 && \
      #   export https_proxy=http://hk-mmhttpproxy.woa.com:11113"
      
      PROXY_ON = ""

      # PS1=[] dirtyhack to bypass ~/.bashrc checking
      remote_cmd = ' '.join(['ssh -o StrictHostKeyChecking=no', EXTRA_SSH_ARGS, '-p', remoteAddr[1], f'root@{remoteAddr[0]}', f'"PS1=[] source ~/.bashrc {PROXY_ON} && env && (', *cmd, ')"'])
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
        # print(f"Received response: {response.decode('utf-8', errors='ignore')}")
        return
    except socket.error as e:
      time.sleep(1)
      continue
    

def do_exp():

  def clean_up():
    logger.info("clean up...")
    cleanup_cmd = "ps -ef | grep 'sglang' | grep -v grep | grep -v defunct | awk '{print \$2}' | xargs -r kill -SIGKILL; ps aux | grep 'sglang' | grep -v defunct; pkill -f sglang" # type: ignore
    for addr, ssh_port in all_machines:
      b = runCommand([cleanup_cmd], (addr, ssh_port))
      b.wait()
    time.sleep(5)

  clean_up()

  prefill_logs = [] 
  decode_logs = []
  for i in range(len(PREFILLS)):
    path=f"/tmp/sgl-prefill-{i}.log"
    os.system("rm -rf " + path)
    prefill_logs.append(open(path, 'w'))
  for i in range(len(DECODES)):
    path=f"/tmp/sgl-decode-{i}.log"
    os.system("rm -rf " + path)
    decode_logs.append(open(path, 'w'))

  prefill_list_boostrap_port = ""
  for i in range(len(PREFILLS)):
    # host:bootstrap_port
    prefill_list_boostrap_port += f"{PREFILLS[i][0]}:{PREFILLS[i][3]},"

  for bsz in tqdm.tqdm([32]):
    filename = f"0411-{len(PREFILLS)}P{len(DECODES)}D-TP{TP}-bsz-{bsz}.txt"

    prefill_servers = []
    decode_servers = []

    for i, (HOST, SSH_PORT, SERVE_PORT, BOOTSTRAP_PORT, extra_env) in enumerate(PREFILLS):
      # setup sglang servers.
      sglang_prefill_args = SGLANG_COMMON_ARGS.copy() + [
        "--disaggregation-mode prefill",
        f"--tp {TP}",
        "--nnodes 1",
        "--max-running-requests", f"{bsz}",
        "--disaggregation-mode prefill",
        f"--disaggregation-bootstrap-port {BOOTSTRAP_PORT}",
        f"--disaggregation-decode-instance-num {len(DECODES)}",
        "--port", f"{SERVE_PORT}",
      ]
      prefill_env = [
        f"UCX_TLS={UCX_TLS}",
        f"UCX_NET_DEVICES={NETDEVICE}",
        "UCX_LOG_LEVEL=info",
        # "NCCL_DEBUG=INFO",
      ] + extra_env

      server = runCommand([f"{' '.join(prefill_env)} python3 -m sglang.launch_server"] + sglang_prefill_args, (HOST, SSH_PORT), prefill_logs[i])
      prefill_servers.append(server)
    


    for i, (HOST, SSH_PORT, SERVE_PORT, BOOTSTRAP_PORT, extra_env) in enumerate(DECODES):
      sglang_decode_args = SGLANG_COMMON_ARGS.copy() + [
        f"--tp {TP}",
        "--nnodes 1",
        "--disaggregation-mode decode",
        "--max-running-requests", f"{bsz}",
        f"--disaggregation-prefill-bootstrap-addr {prefill_list_boostrap_port}",
        "--port", f"{SERVE_PORT}",
      ]

      decode_env = [
        f"UCX_TLS={UCX_TLS}",
        f"UCX_NET_DEVICES={NETDEVICE}",
        "UCX_LOG_LEVEL=info",
        # "NCCL_DEBUG=INFO",
      ] + extra_env

      server = runCommand([f"{' '.join(decode_env)} python3 -m sglang.launch_server"] + sglang_decode_args, (HOST, SSH_PORT), decode_logs[i])
      decode_servers.append(server)



    # wait_server(PREFILL_HOST, PREFILL_SERVE_PORT)
    # wait_server(DECODE_HOST, DECODE_SERVE_PORT)

    for i, (HOST, SSH_PORT, SERVE_PORT, BOOTSTRAP_PORT, _) in enumerate(PREFILLS):
      wait_server(HOST, SERVE_PORT)
      logger.info(f"Prefill server {HOST}:{SERVE_PORT} is ready!")

    for i, (HOST, SSH_PORT, SERVE_PORT, BOOTSTRAP_PORT, _) in enumerate(DECODES):
      wait_server(HOST, SERVE_PORT)
      logger.info(f"Decode server {HOST}:{SERVE_PORT} is ready!")

    logger.info("All PD servers are ready! Wait some seconds to let the server warm up.")
    time.sleep(10)


    prefill_list_sport_bport = ""
    for i in range(len(PREFILLS)):
      # host:port:bootstrap_port
      prefill_list_sport_bport += f"http://{PREFILLS[i][0]}:{PREFILLS[i][2]}:{PREFILLS[i][3]},"
      
    decode_list_sport = ""
    for i in range(len(DECODES)):
      # host:port
      decode_list_sport += f"http://{DECODES[i][0]}:{DECODES[i][2]},"


    # remove lasting ,
    prefill_list_sport_bport = prefill_list_sport_bport[:-1]
    decode_list_sport = decode_list_sport[:-1]

    lb = runCommand([
      "python3 -m sglang.srt.disaggregation.mini_lb",
      "--prefill", prefill_list_sport_bport,
      "--decode", decode_list_sport,
      "--host 0.0.0.0", 
      "--port", f"{LB_SERVE_PORT}", 
    ], remoteAddr=(LB_HOST, LB_SSH_PORT), outputStream=lb_output_log) # type: ignore

    time.sleep(1)
    logger.info("Start benchmarking...")

    BENCHMARK_ARGS = [
      "--model", "default",
      "--host", f"{LB_HOST}",
      "--port", f"{LB_SERVE_PORT}",
      "--endpoint", "/v1/chat/completions",
      "--dataset-name", "jsonl",
      "--num-prompts", f"{ bsz * 3 }", 
      "--dataset-path", "/sgl-workspace/upload/dataset/qa_out_0216_r1_300_max_25k_formatted.jsonl",
      # "--dataset-path", "/sgl-workspace/upload/dataset/easy.jsonl",
      # "--dataset-path", "/sgl-workspace/upload/dataset/long-easy.jsonl",
      "--max-concurrency", f"{ bsz }",
      "--backend", f"openai-chat",
      "--tokenizer", f"{MODEL}",
      "--jsonl-output-len", "4096",
      "--save-result",
      "--result-filename", filename
    ]
    benchmarkClient = runCommand(["python3 -m openai_benchmark.benchmark_serving"] + BENCHMARK_ARGS, outputStream=bench_output_log) # type: ignore
    benchmarkClient.wait()

    # shutdown sglang servers.
    clean_up()





if __name__ == '__main__':
  do_exp()

