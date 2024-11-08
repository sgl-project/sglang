# Create dummy weights:
# 1. Create a folder `~/llama-3.1-405b-fp8-dummy` and create `config.json` and tokenizer under this folder.
# 2. Get `config.json`` from ./config.md
# 3. Download the tokenizer
#   wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer.json
#   wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer_config.json

# Launch sglang
# python -m sglang.launch_server --model-path ~/llama-3.1-405b-fp8-dummy/ --load-format dummy --tp 8 --quant fp8 --disable-radix --mem-frac 0.87

# offline
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 3000 --random-input 1024 --random-output 1024 > sglang_log11
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 4000 --random-input 1024 --random-output 512 > sglang_log12
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 800 --random-input 4096 --random-output 2048 > sglang_log13
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 1500 --random-input 4096 --random-output 1024 > sglang_log14
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 6000 --random-input 256 --random-output 512 > sglang_log15
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 2000 > sglang_log21

# online
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 > sglang_log31
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 > sglang_log32
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 > sglang_log33
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 > sglang_log34
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 1024 --random-output 1024 > sglang_log35
