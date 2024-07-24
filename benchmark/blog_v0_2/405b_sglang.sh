# create ~/llama-3.1-405b-fp8-dummy and create config.json and tokenizer:
# config.json from https://gist.github.com/zhyncs/748597c44d47b45fa15866a4ae2c2b29?permalink_comment_id=5128893
# wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer.json?download=true
# wget wget https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/resolve/main/tokenizer_config.json?download=true 

# Launch sglang
# python -m sglang.launch_server --model ~/llama-3.1-405b-fp8-dummy/ --load-format dummy --tp 8 --quant fp8 --disable-radix --mem-frac 0.88

# offline
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 > sglang/log11
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 > sglang/log12
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 512 --random-range-ratio 0.5 > sglang/log13
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 512 --random-range-ratio 0.5 > sglang/log14
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 2500 > sglang/log21

# online
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 300 --request-rate 1 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log31
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 600 --request-rate 2 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log32
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log33
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log34
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log35
# python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 1000 --request-rate 32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 > sglang/log36
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 1 > sglang/log41
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 2 > sglang/log42
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 4 > sglang/log43
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 8 > sglang/log44
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 16 > sglang/log45
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompt 1000 --request-rate 32 > sglang/log46
