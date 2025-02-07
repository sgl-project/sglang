# Run Multi-Node Inference

## Llama 3.1 405B

**Run 405B (fp16) on Two Nodes**

```bash
# replace 172.16.4.52:20000 with your own node ip address and port of the first node

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0

# replace 172.18.45.52:20000 with your own node ip address and port of the second node

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.18.45.52:20000 --nnodes 2 --node-rank 1
```

Note that LLama 405B (fp8) can also be launched on a single node.

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

## DeepSeek V3/R1

Please refer to [DeepSeek documents for reference.](https://docs.sglang.ai/references/deepseek.html#running-examples-on-multi-node).
