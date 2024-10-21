# SGLang Data Parallel Router (Experimental)

The router can route requests to the sglang instances on multi-node with easy-to-extend routing policy.

## Usage

1. Launch workers

```python
export CUDA_VISIBLE_DEVICES=0; python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 127.0.0.1 --port 9000
export CUDA_VISIBLE_DEVICES=1; python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 127.0.0.1 --port 9002
```

2. Launch a router

> Note: for multi-node, replace `worker-urls` with accessible http endpoint

```python
python -m sglang.srt.router.launch_router --host 127.0.0.1 --port 8080 --policy round_robin --worker-urls http://127.0.0.1:9000 http://127.0.0.1:9002
```

3. Send a curl request to the router

```bash
curl -X POST http://127.0.0.1:8080/generate  -H "Content-Type: application/json" -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

