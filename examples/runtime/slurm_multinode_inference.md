# Multi-Node LLM Inference with SGLang on SLURM Enabled Clusters

SLURM Submit Script - 

```
#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2
#SBATCH --ntasks=2  # Total tasks across all nodes
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB

#SBATCH --partition="" # Enter Partition Name to Request for a Certain Partition 
#SBATCH --gres=gpu:8 # Requesting for 8 GPUs (change as per your needs)

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

# Test server via sending a CURL request

response=$(curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
-H "Authorization: Bearer None" \
-H "Content-Type: application/json" \
-d '{
  "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "List 3 countries and their capitals."
    }
  ],
  "temperature": 0,
  "max_tokens": 64
}')

echo "[INFO] Response from server:"
echo "$response"

# Run inference via a Python script

python sglang_requester.py --model "meta-llama/Meta-Llama-3.1-405B-Instruct"

wait # Keeps waiting and doesn't let server die until end of requested timeframe. You can optionally wait for a fixed duration
```

Requesting via Python -

```
import openai
import argparse
import time
import os

def perform_request(client, messages, model):
    s_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Lower temperature for more focused responses
        max_tokens=20000,  # Reasonable length for a concise response
        top_p=1,  # Slightly higher for better fluency
        n=1,  # Single response is usually more stable
        seed=20242,  # Keep for reproducibility
    )
    e_time = time.time()
    print("Time taken for request: ", e_time - s_time)

    return response.choices[0].message.content

def main(args):
    print("Arguments: ", args)
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    message = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hey"},
    ]
    response = perform_request(client, message, args.model)
    print(response)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    args = argparser.parse_args()
    main(args)
```

[This blog post](https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang) contains more details about why you would use SLURM and who is this useful for