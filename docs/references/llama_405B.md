# Run Llama 3.1 405B

## Run 405B (fp8) on a Single Node

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

## Run 405B (fp16) on Two Nodes

```bash
# on the first node, replace 172.16.4.52:20000 with your own node ip address and port

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0

# on the second node, replace 172.18.45.52:20000 with your own node ip address and port

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.18.45.52:20000 --nnodes 2 --node-rank 1
```
