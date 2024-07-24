# create ~/llama-3.1-405b-fp8-dummy and create config.json and tokenizer:
# config.json from https://gist.github.com/zhyncs/748597c44d47b45fa15866a4ae2c2b29?permalink_comment_id=5128893
# wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer.json?download=true
# wget wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer_config.json?download=true

# Launch vllm
# python3 -m vllm.entrypoints.openai.api_server --model ~/llama-3.1-405b-fp8-dummy/ --load-format dummy --disable-log-requests --tensor-parallel-size 8 --max-model-len 10000

# offline
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 > vllm/log11
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 > vllm/log12
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 512 --random-range-ratio 0.5 > vllm/log13
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 512 --random-range-ratio 0.5 > vllm/log14
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 2500 > vllm/log21

# online
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 300 --request-rate 1 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log31
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 600 --request-rate 2 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log32
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log33
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log34
python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log35
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name random --num-prompt 1000 --request-rate 32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > vllm/log36
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 1 > vllm/log41
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 2 > vllm/log42
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 4 > vllm/log43
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 8 > vllm/log44
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 16 > vllm/log45
# python3 ../../python/sglang/bench_serving.py --backend vllm --dataset-name sharegpt --num-prompt 1000 --request-rate 32 > vllm/log46
