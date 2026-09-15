# Launch trtllm
# https://github.com/sgl-project/tensorrt-demo

# offline
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 3000 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log11
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 4000 --random-input 1024 --random-output 512 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log12
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 800 --random-input 4096 --random-output 2048 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log13
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 1500 --random-input 4096 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log14
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 6000 --random-input 256 --random-output 512 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log15
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name sharegpt --num-prompt 2000 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log21

# online
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log31
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log32
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log33
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log34
python3 ../../python/sglang/bench_serving.py --backend trt --dataset-name random --num-prompt 3200 --request-rate 16 --random-input 1024 --random-output 1024 --model /root/Meta-Llama-3-8B-Instruct > trtllm_log35
