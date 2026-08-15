# Create dummy weights:
# 1. Create a folder `~/llama-3.1-405b-fp8-dummy` and create `config.json` and tokenizer under this folder.
# 2. Get `config.json`` from ./config.md
# 3. Download the tokenizer
#   wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer.json
#   wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer_config.json

# Launch vllm
# python3 -m vllm.entrypoints.openai.api_server --model ~/llama-3.1-405b-fp8-dummy/ --load-format dummy --disable-log-requests --tensor-parallel-size 8 --max-model-len 10000

# offline
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 3000 --random-input 1024 --random-output 1024 > vllm_log11
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 4000 --random-input 1024 --random-output 512 > vllm_log12
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 800 --random-input 4096 --random-output 2048 > vllm_log13
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 1500 --random-input 4096 --random-output 1024 > vllm_log14
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 6000 --random-input 256 --random-output 512 > vllm_log15
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 2000 > vllm_log21

# online
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 > vllm_log31
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 > vllm_log32
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 > vllm_log33
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 > vllm_log34
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 1024 --random-output 1024 > vllm_log35
