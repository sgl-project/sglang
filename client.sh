curl http://127.0.0.1:30000/flush_cache           
python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.98 --num-prompts 5 --max-concurrency 1 --output-file res.jsonl     
curl http://127.0.0.1:30000/flush_cache   
python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.98 --num-prompts 20 --max-concurrency 4 --output-file res.jsonl 
curl http://127.0.0.1:30000/flush_cache
python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.98 --num-prompts 80 --max-concurrency 16 --output-file res.jsonl
curl http://127.0.0.1:30000/flush_cache
python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.98 --num-prompts 160 --max-concurrency 32 --output-file res.jsonl
curl http://127.0.0.1:30000/flush_cache
