# Launch trtllm
# https://gist.github.com/zhyncs/748597c44d47b45fa15866a4ae2c2b29?permalink_comment_id=5129302

# offline
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log11
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log12
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2500 --random-input 1024 --random-output 512 --random-range-ratio 0.5 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log13
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2500 --random-input 4096 --random-output 512 --random-range-ratio 0.5 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log14
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 2500 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log21

# online
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 300 --request-rate 1 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log31
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 600 --request-rate 2 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log32
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log33
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log34
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log35
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 1000 --request-rate 32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log36
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 1 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log41
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 2 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log42
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 4 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log43
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 8 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log44
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 16 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log45
# python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 1000 --request-rate 32 --model meta-llama/Meta-Llama-3-8B-Instruct > trtllm/log46
