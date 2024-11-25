# SGLang Router

SGLang router is a standalone module implemented in Rust to achieve data parallelism across SGLang instances.

## Installation

```bash
pip install sglang-router
```

## Usage
The router offers two modes:

### 1. Co-launch workers and router
This will be a drop-in replacement for the existing `--dp-size`. This part of code will be moved into sglang core.
Under the hood, it uses multi-processes to launch multiple sglang workers, wait for them to be healthy, then launch the router.

```bash
$ python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dp-size 8
```

### 2. Launch only router
This is useful for multi-node DP. You can launch workers on different nodes, then connect the router to them.

```bash
$ python -m sglang_router.launch_router --worker-urls http://worker1:8000 http://worker2:8000

$ python -m sglang_router.launch_router --help
usage: launch_router.py [-h] [--host HOST] [--port PORT] [--worker-urls WORKER_URLS [WORKER_URLS ...]]
                       [--policy {random,round_robin,cache_aware}] [--cache-threshold CACHE_THRESHOLD]
                       [--balance-abs-threshold BALANCE_ABS_THRESHOLD] [--balance-rel-threshold BALANCE_REL_THRESHOLD]
                       [--eviction-interval EVICTION_INTERVAL] [--max-tree-size MAX_TREE_SIZE]

options:
  -h, --help            show this help message and exit
  --host HOST          Host address to bind the router server (default: 127.0.0.1)
  --port PORT          Port number to bind the router server (default: 30000)
  --worker-urls WORKER_URLS [WORKER_URLS ...]
                       List of worker URLs (e.g., http://worker1:8000 http://worker2:8000) (default: None)
  --policy {random,round_robin,cache_aware}
                       Load balancing policy to use (default: cache_aware)
  --cache-threshold CACHE_THRESHOLD
                       Cache threshold (0.0-1.0) for cache-aware routing (default: 0.5)
  --balance-abs-threshold BALANCE_ABS_THRESHOLD
                       Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold (default: 32)
  --balance-rel-threshold BALANCE_REL_THRESHOLD
                       Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold (default: 1.0001)
  --eviction-interval EVICTION_INTERVAL
                       Interval in seconds between cache eviction operations (default: 60)
  --max-tree-size MAX_TREE_SIZE
                       Maximum size of the approximation tree for cache-aware routing (default: 16777216)
```

## Strategy

### Cache-Aware Load-Balancing Router

This router combines two strategies to optimize both cache utilization and request distribution:

1. Cache-Aware Routing (Approximate Tree)
2. Load-Balancing Routing (Shortest Queue with Balance Thresholds)

The router dynamically switches between these strategies based on load conditions:
- Uses load balancing when the system is imbalanced
- Uses cache-aware routing when the system is balanced

A system is considered imbalanced if both conditions are met:
1. (max_load - min_load) > balance_abs_threshold
2. max_load > balance_rel_threshold * min_load

#### 1. Cache-Aware Routing (Approximate Tree)
This strategy maintains an approximate radix tree for each worker based on request history,
eliminating the need for direct cache state queries. The tree stores raw text characters
instead of token IDs to avoid tokenization overhead.

Process:
- For each request, find the worker with the highest prefix match
- If match rate > cache_threshold:
  - Route to the worker with highest match (likely has relevant data cached)
- If match rate ≤ cache_threshold:
  - Route to the worker with smallest tree size (most available cache capacity)
- Background maintenance:
  - Periodically evict least recently used leaf nodes to prevent memory overflow

#### 2. Load-Balancing (Shortest Queue)
This strategy tracks pending request counts per worker and routes new requests
to the least busy worker when the system is detected to be imbalanced. This helps
maintain optimal load distribution across workers.

### Configuration Parameters

1. `cache_threshold`: (float, 0.0 to 1.0, default: 0.5)
   - Minimum prefix match ratio to use highest-match routing
   - Below this threshold, routes to worker with most available cache space

2. `balance_abs_threshold`: (integer, default: 32)
   - Absolute difference threshold for load imbalance detection
   - System is potentially imbalanced if (max_load - min_load) > abs_threshold

3. `balance_rel_threshold`: (float, default: 1.0001)
   - Relative ratio threshold for load imbalance detection
   - System is potentially imbalanced if max_load > min_load * rel_threshold
   - Used in conjunction with abs_threshold to determine final imbalance state

4. `eviction_interval`: (integer, default: 60)
   - Interval in seconds between LRU eviction cycles for the approximate trees
   - Background thread periodically evicts least recently used nodes to maintain tree size

5. `max_tree_size`: (integer, default: 16777216)
   - Maximum nodes per tree
   - When exceeded, LRU leaf nodes are evicted during the next eviction cycle

## Development

- Rust and Cargo installed

```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

- Python with pip installed


### Build Process

#### 1. Build Rust Project

```bash
cargo build
```

#### 2. Build Python Binding

##### Option A: Build and Install Wheel
1. Build the wheel package:
```bash
pip install setuptools-rust wheel build
python -m build
```

2. Install the generated wheel:
```bash
pip install <path-to-wheel>
```

##### Option B: Development Mode

For development purposes, you can install the package in editable mode:

Warning: Using editable python binding can suffer from performance degradation!! Please build a fresh wheel for every update if you want to test performance.

```bash
pip install -e .
```

**Note:** When modifying Rust code, you must rebuild the wheel for changes to take effect.

### CI/CD Setup

The continuous integration pipeline consists of three main steps:

#### 1. Build Wheels
- Uses `cibuildwheel` to create manylinux x86_64 packages
- Compatible with major Linux distributions (Ubuntu, CentOS, etc.)
- Additional configurations can be added to support other OS/architectures
- Reference: [cibuildwheel documentation](https://cibuildwheel.pypa.io/en/stable/)

#### 2. Build Source Distribution
- Creates a source distribution containing the raw, unbuilt code
- Enables `pip` to build the package from source when prebuilt wheels are unavailable

#### 3. Publish to PyPI
- Uploads both wheels and source distribution to PyPI

The CI configuration is based on the [tiktoken workflow](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/.github/workflows/build_wheels.yml#L1).
