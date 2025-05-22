# Router for Data Parallelism

Given multiple GPUs running multiple SGLang Runtimes, SGLang Router distributes the requests to different Runtimes with its unique cache-aware load-balancing algorithm.

The router is an independent Python package, and it can be used as a drop-in replacement for the OpenAI API.

## Installation

```bash
pip install sglang-router
```

Detailed usage of the router can be found in [launch_router](https://github.com/sgl-project/sglang/blob/main/sgl-router/py_src/sglang_router/launch_router.py) and [launch_server](https://github.com/sgl-project/sglang/blob/main/sgl-router/py_src/sglang_router/launch_server.py). Also, you can directly run the following command to see the usage of the router.

```bash
python -m sglang_router.launch_server --help
python -m sglang_router.launch_router --help
```

The router supports two working modes:

1. Co-launch Router and Runtimes
2. Launch Runtimes and Router separately

## Co-launch Router and Runtimes

This will be a drop-in replacement for the existing `--dp-size` argument of SGLang Runtime. Under the hood, it uses multi-processes to launch multiple workers, wait for them to be ready, then connect the router to all workers.

```bash
python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dp-size 4 --host 0.0.0.0
```

After the server is ready, you can directly send requests to the router as the same way as sending requests to each single worker.

Please adjust the batchsize accordingly to achieve maximum throughput.

```python
import requests

url = "http://localhost:30000/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print(response.json())
```

## Launch Runtimes and Router Separately

This is useful for multi-node DP. First, launch workers on multiple nodes, then launch a router on the main node, and connect the router to all workers.

```bash
python -m sglang_router.launch_router --worker-urls http://worker_url_1 http://worker_url_2
```

## Dynamic Scaling APIs

We offer `/add_worker` and `/remove_worker` APIs to dynamically add or remove workers from the router.

- `/add_worker`

Usage:

```bash
curl -X POST http://localhost:30000/add_worker?url=http://worker_url_1
```

Example:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30001

curl -X POST http://localhost:30000/add_worker?url=http://127.0.0.1:30001

# Successfully added worker: http://127.0.0.1:30001
```

- `/remove_worker`

Usage:

```bash
curl -X POST http://localhost:30000/remove_worker?url=http://worker_url_1
```

Example:

```bash
curl -X POST http://localhost:30000/remove_worker?url=http://127.0.0.1:30001

# Successfully removed worker: http://127.0.0.1:30001
```

Note:

- For cache-aware router, the worker will be removed from the tree and the queues.

## Fault Tolerance

We provide retries based for failure tolerance.

1. If the request to a worker fails for `max_worker_retries` times, the router will remove the worker from the router and move on to the next worker.
2. If the total number of retries exceeds `max_total_retries`, the router will return an error.

Note:

- `max_worker_retries` is 3 and `max_total_retries` is 6 by default.

## Routing Strategies

### Cache-Aware Load-Balancing Router

The native router combines two strategies to optimize both cache utilization and request distribution:

1. Cache-Aware Routing (Approximate Tree)
2. Load-Balancing Routing (Shortest Queue with Balance Thresholds)

The router dynamically switches between these strategies based on load conditions:

- Uses load balancing when the system is imbalanced
- Uses cache-aware routing when the system is balanced

A system is considered imbalanced if both conditions are met:

1. (max_load - min_load) > balance_abs_threshold
2. max_load > balance_rel_threshold * min_load

***Cache-Aware Routing (Approximate Tree)***

When the workers are considered to be balanced, the router maintains an approximate radix tree for each worker based on request history, eliminating the need for direct cache state queries on each worker. The tree stores raw text characters instead of token IDs to avoid tokenization overhead.

Process:

1. For each request, find the worker with the highest prefix match.

   - If match rate > cache_threshold, route the request to the worker with highest match (likely has relevant data cached)
   - If match rate ≤ cache_threshold, route the request to the worker with smallest tree size (most available cache capacity)

2. Background maintenance: Periodically evict least recently used leaf nodes on the approximate tree to prevent memory overflow.

***Load-Balancing (Shortest Queue)***

For unbalanced systems, this strategy tracks pending request counts per worker and routes new requests to the least busy worker. This helps maintain optimal load distribution across workers.

## Configuration Parameters

1. `cache_threshold`: (float, 0.0 to 1.0, default: 0.5)
   - Minimum prefix match ratio to use highest-match routing.
   - Below this threshold, the request will be routed to the worker with most available cache space.

2. `balance_abs_threshold`: (integer, default: 32)
   - Absolute difference threshold for load imbalance detection.
   - The system is potentially imbalanced if (max_load - min_load) > abs_threshold.

3. `balance_rel_threshold`: (float, default: 1.0001)
   - Relative ratio threshold for load imbalance detection.
   - The system is potentially imbalanced if max_load > min_load * rel_threshold.
   - Used in conjunction with `balance_abs_threshold` to determine the final imbalance state.

4. `eviction_interval`: (integer, default: 60)
   - Interval in seconds between LRU eviction cycles for the approximate trees.
   - Background thread periodically evicts least recently used nodes to maintain tree size.

5. `max_tree_size`: (integer, default: 16777216)
   - Maximum nodes on the approximate tree.
   - When exceeded, LRU leaf nodes are evicted during the next eviction cycle.
