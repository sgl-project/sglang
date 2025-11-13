sudo docker run -d -it \
    --ulimit memlock=-1  --ulimit stack=67108864  --ulimit core=-1 \
    --ipc=host --network=host --privileged \
    --shm-size=64g --gpus all \
    -v /home/zhikuan.psc/data:/data-home \
    -v /mnt/vdb1:/data-mnt \
    --name sglang-elasticmem \
    mirrors-ssl.aliyuncs.com/lmsysorg/sglang:v0.5.5.post1 bash

pip config set global.index-url https://mirrors.aliyun.com/pypi/web/simple
pip install -e . --no-build-isolation
python3 test/test_elastic.py

pip install vllm==0.11.0
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# outlines 0.1.11 requires outlines_core==0.1.26, but you have outlines-core 0.2.11 which is incompatible.
# sglang 0.5.5.post1 requires nvidia-cutlass-dsl==4.2.1, but you have nvidia-cutlass-dsl 4.3.0.dev0 which is incompatible.

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
rm -rf nohup.out
nohup python3 -m sglang.launch_server \
  --log-level debug \
  --model /data-mnt/tiny-random-llama-4-8E/ \
  --attention-backend fa3 \
  --mem-fraction-static 0.3 \
  --hybrid-kvcache-ratio 1.0 \
  --context-length 32767 &

sleep 3

tail -f nohup.out

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 24576 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 2048

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
    "num_steps": 10000,
    "activities": ["MEM"],
    "merge_profiles": true
  }'
