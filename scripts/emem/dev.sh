####################################################################################################
# A10 dev

sudo docker run -d -it \
    --ulimit memlock=-1  --ulimit stack=67108864  --ulimit core=-1 \
    --ipc=host --network=host --privileged \
    --shm-size=64g --gpus all \
    -v /mnt/vdb1:/data-mnt \
    --name sglang-elasticmem \
    mirrors-ssl.aliyuncs.com/lmsysorg/sglang:v0.5.5.post1 bash

# sglang
pip install vllm==0.11.0

# elasticmem
git clone https://github.com/sglang-bot/elasticmem && cd elasticmem
pip install -e . --no-build-isolation
python3 test/test_elastic.py

##########
# gpt-oss
for _ in {1..2}; do
  ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
  ps aux | grep "sglang::" | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 1
done

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
export SGLANG_ELASTIC_MEM_POOL=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
rm -rf nohup.out
nohup python3 -m sglang.launch_server \
  --log-level debug \
  --model /data-mnt/gpt-oss-20b-bf16/ \
  --tp 2 \
  --attention-backend triton \
  --cuda-graph-max-bs 1024 \
  --mem-fraction-static 0.7 \
  --hybrid-kvcache-ratio 1.0 \
  --context-length 65536 &

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 2048 --random-input 8192 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 128

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 2048 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 256

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 8192 --random-output 8192 --random-range-ratio 0.5 \
  --max-concurrency 24

##############################
# tiny-random-llama-4-8E
##############################
# diff --git a/config.json b/config.json
# index ce581b5..42d45c9 100644
# --- a/config.json
# +++ b/config.json
# @@ -52,13 +52,13 @@
#      "rope_theta": 500000.0,
#      "router_aux_loss_coef": 0.001,
#      "router_jitter_noise": 0.0,
# -    "torch_dtype": "float32",
# +    "torch_dtype": "bfloat16",
#      "use_cache": true,
#      "use_qk_norm": true,
#      "vocab_size": 202048
#    },
#    "tie_word_embeddings": false,
# -  "torch_dtype": "float32",
# +  "torch_dtype": "bfloat16",
#    "transformers_version": "4.51.3",
#    "vision_config": {
#      "_attn_implementation_autoset": true,

for _ in {1..2}; do
  ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
  ps aux | grep "sglang::" | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 1
done

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
export SGLANG_ELASTIC_MEM_POOL=true
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
rm -rf nohup.out
nohup python3 -m sglang.launch_server \
  --log-level debug \
  --model /data-mnt/tiny-random-llama-4-8E/ \
  --attention-backend fa3 \
  --cuda-graph-max-bs 1024 \
  --mem-fraction-static 0.3 \
  --hybrid-kvcache-ratio 1.0 \
  --context-length 65536 &

sleep 3

tail -f nohup.out

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 2048 --random-input 24576 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 800

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 8192 --random-output 8192 --random-range-ratio 0.5 \
  --max-concurrency 1024

nohup python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 8192 --random-output 8192 --random-range-ratio 0.5 \
  --max-concurrency 24 > bench.out &

####################################################################################################
# qwen3-next

for _ in {1..2}; do
  ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
  ps aux | grep "sglang::" | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 1
done

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
export SGLANG_ELASTIC_MEM_POOL=true
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
python3 -m sglang.launch_server \
  --log-level debug \
  --model /data-mnt/Qwen3-Next-80B-A3B-Instruct/ \
  --tp 2 \
  --attention-backend triton \
  --cuda-graph-max-bs 1024 \

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 512 --random-input 24576 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 128

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 8192 --random-output 8192 --random-range-ratio 0.5 \
  --max-concurrency 1024

####################################################################################################

curl -L -X POST 'http://127.0.0.1:30000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
--data-raw '{
  "messages": [
    {
      "content": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
      "role": "user"
    }
  ],
  "model": "xxx",
  "max_tokens": 64,
  "stream": false
}'

curl -X POST 127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "activities": ["CPU", "GPU"]
  }'

curl -X POST 127.0.0.1:30000/stop_profile

####################################################################################################
# h20 dev

for _ in {1..2}; do
  ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
  ps aux | grep "sglang::" | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 1
done

export SGLANG_ELASTIC_MEM_POOL=true
export SGLANG_RATIO=0.5
export PORT=30000
CUDA_VISIBLE_DEVICES=0,1,2,3 \

export SGLANG_ELASTIC_MEM_POOL=false
export SGLANG_RATIO=0.5
export PORT=30001
CUDA_VISIBLE_DEVICES=4,5,6,7 \

nohup python3 -m sglang.launch_server \
  --log-level debug \
  --model /home/t4/models/lvm-data/Llama-4-Scout-17B-16E-Instruct \
  --tp 4 \
  --schedule-conservativeness 100 \
  --attention-backend fa3 \
  --hybrid-kvcache-ratio ${SGLANG_RATIO} \
  --context-length 200000 \
  --port ${PORT} \
  > nohup.emem.${SGLANG_ELASTIC_MEM_POOL}.ratio.${SGLANG_RATIO}.port.${PORT}.out 2>&1 \
  &

echo "" > nohup.bench.${SGLANG_ELASTIC_MEM_POOL}.ratio.${SGLANG_RATIO}.port.${PORT}.out

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /home/t4/models/lvm-data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1024 --random-input 1024 --random-output 7168 --random-range-ratio 1 \
  --max-concurrency 384 --seed 0 \
  --port ${PORT} \
  >> nohup.bench.${SGLANG_ELASTIC_MEM_POOL}.ratio.${SGLANG_RATIO}.port.${PORT}.out 2>&1

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /home/t4/models/lvm-data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 192 --random-input 8192 --random-output 65536 --random-range-ratio 1 \
  --max-concurrency 64 --seed 1 \
  --port ${PORT} \
  >> nohup.bench.${SGLANG_ELASTIC_MEM_POOL}.ratio.${SGLANG_RATIO}.port.${PORT}.out 2>&1

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /home/t4/models/lvm-data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1024 --random-input 1024 --random-output 7168 --random-range-ratio 1 \
  --max-concurrency 384 --seed 2 \
  --port ${PORT} \
  >> nohup.bench.${SGLANG_ELASTIC_MEM_POOL}.ratio.${SGLANG_RATIO}.port.${PORT}.out 2>&1
