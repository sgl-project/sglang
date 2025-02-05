# Run Multi-Node Inference

## Llama 3.1 405B

**Run 405B (fp16) on Two Nodes**

```bash
# on the first node, replace 172.16.4.52:20000 with your own node ip address and port

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0

# on the second node, replace 172.18.45.52:20000 with your own node ip address and port

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.18.45.52:20000 --nnodes 2 --node-rank 1
```
Note that Llama 3.1 405B could also
### Run 405B (fp8) on a Single Node

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

## DeepSeek_v3_r1 671B
<!-- deep seek multi-node link -->
To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run `tp=16`, let `10.0.0.1` be the hostname of the first node and `5000` be an available port. You can use the following commands. If you encounter a deadlock, try adding `--disable-cuda-graph`.
```bash
# Node 0
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1
```
