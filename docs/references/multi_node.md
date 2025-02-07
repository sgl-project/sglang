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

## Multi-Node Inference on SLURM

This example showcases how to serve SGLang server across multiple nodes by SLURM. Submit the following job to the SLURM cluster.

```
#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2

# Total tasks across all nodes

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB

# Enter Partition Name to Request for a Certain Partition

#SBATCH --partition="lmsys.org"

# Requesting for 8 GPUs (change as per your needs)

#SBATCH --gres=gpu:8

#SBATCH --time=12:00:00

# Load required modules or set environment variables if needed

echo "[INFO] Activating environment on node $SLURM_PROCID"
if ! source ENV_FOLDER/bin/activate; then
    echo "[ERROR] Failed to activate environment" >&2
    exit 1
fi

# Define parameters
model=MODEL_PATH
tp_size=16

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# Define the NCCL init address using the hostname of the head node

HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Set different OUTLINES_CACHE_DIR before starting process on each node to prevent weird sqlite errors that arise on SLURM nodes when running multinode

export OUTLINES_CACHE_DIR="/tmp/node_0_cache"

# Launch processes with srun

srun --ntasks=1 --nodes=1 --exclusive --output="SLURM_Logs/%x_%j_node0.out" \
    --error="SLURM_Logs/%x_%j_node0.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --tp "$tp_size" \
    --nccl-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank 0 &

export OUTLINES_CACHE_DIR="/tmp/node_1_cache"

srun --ntasks=1 --nodes=1 --exclusive --output="SLURM_Logs/%x_%j_node1.out" \
    --error="SLURM_Logs/%x_%j_node1.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --tp "$tp_size" \
    --nccl-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank 1 &

# Wait for localhost:30000 to accept connections

while ! nc -z localhost 30000; do
    sleep 1
    echo "[INFO] Waiting for localhost:30000 to accept connections"
done

echo "[INFO] localhost:30000 is ready to accept connections"

# Keeps waiting and doesn't let server die until end of requested timeframe. You can optionally wait for a fixed duration

wait
```

Then, you can test the server by sending requests following other [documents](https://docs.sglang.ai/backend/openai_api_completions.html).

Thanks for [aflah02](https://github.com/aflah02) for providing the example, based on his [blog post](https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang).
