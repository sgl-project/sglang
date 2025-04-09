


# sglang small 小模型单机测试，用于调试sglang改动
在8*L20开发机上：
```bash
conda activate /data/home/yongtongwu/conda-sglang
python3 -m sglang.launch_server \
--model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--enable-torch-compile \
--torch-compile-max-bs 1 \
--tp 4 \
--trust-remote-code \
--host 0.0.0.0 \
--port 8080 # > /tmp/sglang.log 2>&1 &
# tail -f /tmp/sglang.log
```

在8*H20部署机上：
```bash

SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log \
PYTHONPATH=/sgl-workspace/:/sgl-workspace/sglang/:/sgl-workspace/sglang/sgl-kernel:$PYTHONPATH python3 -m sglang.launch_server \
--model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1_distill_qwen_1_5b-model_path/DeepSeek-R1-Distill-Qwen-1.5B \
--enable-torch-compile \
--torch-compile-max-bs 1 \
--tp 4 \
--trust-remote-code \
--host 0.0.0.0 \
--port 8080 > /tmp/sglang.log 2>&1 &
tail -f /tmp/sglang.log

# --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \


# SEND REQUEST TO PROFILE
python3 -m sglang.bench_serving --backend sglang \
--tokenizer /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1_distill_qwen_1_5b-model_path/DeepSeek-R1-Distill-Qwen-1.5B \
--num-prompts 10 \
--dataset-name random \
--random-input 8192 \
--random-output-len 1 \
--profile \
--host 127.0.0.1 \
--port 8080

```


# sglang serving

```bash

# Master node


mkdir -p /root/sglang/profile_log

SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log \
python3 -m sglang.launch_server \
--model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
--max-running-requests 1 \
--tp 16 \
--dist-init-addr 29.226.51.165:20002 \
--nnodes 2 \
--node-rank 0 \
--trust-remote-code \
--disable-radix-cache \
--schedule-policy fcfs \
--host 0.0.0.0 \
--disable-overlap-schedule \
--port 8080 > /tmp/sglang.log 2>&1 &

tail -f /tmp/sglang.log


# side node
mkdir -p /root/sglang/profile_log
SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log \
python3 -m sglang.launch_server \
--model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
--max-running-requests 1 \
--tp 16 \
--dist-init-addr 29.226.51.165:20002 \
--nnodes 2 \
--node-rank 1 \
--trust-remote-code \
--disable-radix-cache \
--schedule-policy fcfs \
--host 0.0.0.0 \
--disable-overlap-schedule \
--port 8080 > /tmp/sglang.log 2>&1 &

tail -f /tmp/sglang.log




# SEND REQUEST TO PROFILE
python3 -m sglang.bench_serving --backend sglang \
--tokenizer /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
--num-prompts 1 \
--dataset-name random \
--random-input-len 15000 \
--random-output-len 16 \
--host 127.0.0.1 \
--profile \
--port 8080

```


kill sglang process:
```bash
# #!/bin/bash
ps -ef | grep 'sglang' | grep -v grep | awk '{print $2}' | xargs kill -SIGKILL; ps aux | grep 'sglang' | grep -v defunct

# # 检查是否还有残留进程
# remaining_processes=$(ps -ef | grep 'sglang' | grep -v grep)
# if [ -z "$remaining_processes" ]; then
#     echo "所有包含 sglang 的进程已成功终止。"
# else
#     echo "仍有包含 sglang 的进程未终止，可考虑发送 SIGKILL 信号强制终止。"
# fi
```

# sglang benchmark

```bash
# on devcloud test machine
conda activate vllm66
cd ~/upload
python benchmarks_old/benchmark_serving_new.py \
    --model default \
    --host 29.123.192.10 \
    --port 8080 \
    --endpoint /v1/chat/completions \
    --dataset-name json \
    --dataset-path dataset/qa_out_0216_r1_300.json \
    --num-prompts 1 \
    --max-concurrency 8 \
    --backend openai-chat \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
    --json-output-len 4096 \
    --add-uuid \
    --save-result
    
# --tokenizer /home/qspace/data/mmsearchlibral208csvr/rdata_models/DeepSeek-R1 \

# on gemini machine
# pip3 install datasets --index-url=https://mirrors.tencent.com/pypi/simple --extra-index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

# mkdir -p /root/sglang/profile_log
# export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# python3 /root/save/benchmarks/benchmark_serving.py \
#     --model default \
#     --host 29.123.192.10 \
#     --port 8080 \
#     --endpoint /v1/chat/completions \
#     --dataset-name jsonl \
#     --dataset-path /root/save/dataset/qa_out_0216_r1_300.jsonl \
#     --num-prompts 1 \
#     --max-concurrency 4 \
#     --backend openai-chat \
#     --tokenizer /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1/ \
#     --add-uuid \
#     --profile \
#     --save-result




python benchmarks/benchmark_serving.py \
    --model default \
    --host 29.123.192.10 \
    --port 8080 \
    --endpoint /v1/chat/completions \
    --dataset-name jsonl \
    --dataset-path ~/upload/dataset/max2/qa_out_0216_r1_300_max_30k_formatted.jsonl \
    --max-concurrency 1 \
    --backend openai-chat \
    --tokenizer /home/qspace/data/mmsearchlibral208csvr/rdata_models/DeepSeek-R1 \
    --jsonl-output-len 4096 \
    --num-prompts 4 \
    --save-result
    # 删掉了--num-samples 代表使用整个数据集（总条数281）

```






# Nsys Profile


```bash

# master node
nsys launch --trace-fork-before-exec=true --cuda-graph-trace=node \
python3 -m sglang.launch_server \
--model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
--enable-torch-compile \
--torch-compile-max-bs 8 \
--tp 16 \
--dist-init-addr 29.123.192.10:20000 \
--nnodes 2 \
--node-rank 0 \
--trust-remote-code \
--host 0.0.0.0 \
--port 8080 > /tmp/sglang.log 2>&1 &
tail -f /tmp/sglang.log


# side node
python3 -m sglang.launch_server \
--model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
--enable-torch-compile \
--torch-compile-max-bs 8 \
--tp 16 \
--dist-init-addr 29.123.192.10:20000 \
--nnodes 2 \
--node-rank 1 \
--trust-remote-code \
--host 0.0.0.0 \
--port 8080 > /tmp/sglang.log 2>&1 &
tail -f /tmp/sglang.log


# master node
nsys start --output profile.sglang.out

# benchmark client node. do benchmarking 
# See [Benchmark Client](#benchmark-client)

nsys stop 



```


<!-- 
# Nsys Profile (discarded)


```bash


# Possible --trace values are one or more of 'cuda', 'nvtx', 'cublas', 'cublas-verbose', 'cusolver', 'cusolver-verbose', 'cusparse', 
#            'cusparse-verbose', 'mpi', 'oshmem', 'ucx', 'nvvideo', 'osrt', 'cudnn', 'opengl', 'opengl-annotations', 'openacc', 'openmp', 'vulkan', 'vulkan-annotations' or 'none'

# master node

# pip3 install nvtx --index-url=https://mirrors.tencent.com/pypi/simple --extra-index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

nsys launch --trace-fork-before-exec=true --cuda-graph-trace=node --delay 600 --duration 360000 --output profile.sglang.out \
  --force-overwrite true \
  -t nvtx,cuda,cublas,oshmem,osrt,cudnn,cusparse \
  python3 -m sglang.bench_one_batch_server \
  --input-len 15000 \
  --output-len 800 \
  --batch-size 1 \
  --model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --tp 16 \
  --dist-init-addr 29.123.192.10:20000 \
  --nnodes 2 \
  --node-rank 0 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080 > /tmp/sglang.log 2>&1 &
tail -f /tmp/sglang.log

# slave node
# nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --delay 600 --duration 360000 \
  python3 -m sglang.bench_one_batch_server \
  --input-len 15000 \
  --output-len 800 \
  --batch-size 1 \
  --model-path /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1 \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --tp 16 \
  --dist-init-addr 29.123.192.10:20000 \
  --nnodes 2 \
  --node-rank 1 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080 > /tmp/sglang.log 2>&1 &
tail -f /tmp/sglang.log
``` -->



