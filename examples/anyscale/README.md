# Multi-Node SGLang Deployment on Anyscale

This example demonstrates how to deploy a multi-node SGLang server on Anyscale using Ray.

## Overview

Two deployment modes are available:

1. **Multiprocessing Backend** (`job.yaml` + `driver.py`): Each node runs a separate `sglang.launch_server` process. Nodes coordinate via explicit `dist-init-addr`.

2. **Ray Actor Backend** (`job_ray_actor.yaml` + `driver_ray_actor.py`): Uses SGLang's built-in Ray actor scheduler with automatic multi-node discovery. The Engine automatically discovers GPU nodes, creates per-node placement groups, and infers `dist_init_addr`.

**Default Configuration:**
- Model: `Qwen/Qwen3-1.7B`
- Nodes: 2
- Total Tensor Parallelism: 8 (4 GPUs per node)
- Total GPUs: 8

## Files

- `job.yaml` - Anyscale job config for **multiprocessing backend**
- `driver.py` - Driver script for multiprocessing backend (launches separate processes per node)
- `job_ray_actor.yaml` - Anyscale job config for **Ray actor backend**
- `driver_ray_actor.py` - Driver script for Ray actor backend (uses `sglang.Engine` with `use_ray=True`)
- `Dockerfile` - Docker image definition based on `anyscale/ray:2.53.0-py312-cu129`

## Prerequisites

1. **Anyscale Account** - Access to Anyscale platform
2. **Docker Registry** - Push your custom image to a registry accessible by Anyscale
3. **GPU Instances** - Access to GPU instances (e.g., `g5.12xlarge` with A10G GPUs)

## Setup

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t sglang-ray:latest -f Dockerfile .

# Tag for your registry
docker tag sglang-ray:latest YOUR_REGISTRY/sglang-ray:latest

# Push to registry
docker push YOUR_REGISTRY/sglang-ray:latest
```

### 2. Update job.yaml

Edit `job.yaml` and update:
- `compute_config.cloud` - Your Anyscale cloud name
- `compute_config.region` - Your preferred region
- `image_uri` - Your Docker image URI

### 3. Submit the Job

**Multiprocessing Backend:**
```bash
# Using Anyscale CLI
anyscale job submit -f job.yaml
```

**Ray Actor Backend:**
```bash
# Using Anyscale CLI
anyscale job submit -f job_ray_actor.yaml
```

Or using the SDK:
```python
import anyscale
from anyscale.job.models import JobConfig

# For multiprocessing backend
config = JobConfig.from_yaml('job.yaml')
anyscale.job.submit(config)

# For Ray actor backend
config = JobConfig.from_yaml('job_ray_actor.yaml')
anyscale.job.submit(config)
```

## Driver Script Usage

### Multiprocessing Backend (driver.py)

The driver script can be customized with command-line arguments:

```bash
python driver.py \
  --model-path Qwen/Qwen3-1.7B \
  --tp 8 \
  --nnodes 2 \
  --port 30000 \
  --dist-init-port 20000
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `Qwen/Qwen3-1.7B` | HuggingFace model ID or local path |
| `--tp` | `4` | Total tensor parallelism size across all nodes |
| `--nnodes` | `2` | Number of nodes |
| `--port` | `30000` | Server port |
| `--dist-init-port` | `20000` | Distributed initialization port |

### Ray Actor Backend (driver_ray_actor.py)

This uses SGLang's built-in Ray actor backend with automatic multi-node discovery:

```bash
python driver_ray_actor.py \
  --model-path Qwen/Qwen3-1.7B \
  --tp 8 \
  --port 30000
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `Qwen/Qwen3-1.7B` | HuggingFace model ID or local path |
| `--tp` | `4` | Total tensor parallelism size (auto-distributed across nodes) |
| `--port` | `30000` | Server port |

**Key Differences from Multiprocessing Backend:**
- No need to specify `--nnodes` - automatically discovered from Ray cluster
- No need for `--dist-init-port` - automatically inferred
- Uses `sglang.Engine(use_ray=True)` instead of launching separate processes
- Automatic placement group creation with STRICT_PACK strategy per node

## How It Works

### Multiprocessing Backend

1. **Ray Initialization**: The driver connects to the Ray cluster managed by Anyscale.

2. **Placement Groups**: Creates STRICT_SPREAD placement groups to ensure each actor runs on a different node.

3. **Actor Creation**: Creates `SGLangServerActor` instances, one per node, each bound to specific GPUs.

4. **Server Launch**: Each actor launches `sglang.launch_server` with:
   - Same model path
   - Same TP size
   - Shared `dist-init-addr` pointing to Node 0
   - Unique `node-rank`

5. **Health Check**: Waits for the master server (Node 0) to respond to health checks.

6. **Testing**: Sends a test generation request to verify the server works.

### Ray Actor Backend

1. **Ray Initialization**: Connects to the Ray cluster.

2. **GPU Node Discovery**: Uses `ray._private.state.available_resources_per_node()` to discover all GPU nodes (not `ray.nodes()` which is for debugging only).

3. **Cluster Topology**: Creates a `RayClusterTopology` based on the required world_size (tp_size * pp_size), validating homogeneous GPU counts across nodes.

4. **Placement Groups**: Creates one STRICT_PACK placement group per node, each with `gpus_per_node` bundles of 1 GPU.

5. **Rank Assignment**: Computes (pp_rank, tp_rank) → (node_idx, local_gpu_idx) mapping.

6. **Actor Launch**: Creates `SchedulerActor` instances with:
   - Proper placement group and bundle index for node/GPU assignment
   - Runtime GPU discovery via `ray.get_runtime_context().get_accelerator_ids()`
   - Shared `dist_init_addr` inferred from primary worker IP

7. **Event Loop**: Each actor runs its scheduler event loop for request handling.

## Architecture

### Multiprocessing Backend

```
┌─────────────────────────────────────────────────────────────────┐
│                        Anyscale Cluster                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │      Node 0         │       │      Node 1         │         │
│  │  (Head Node)        │       │  (Worker Node)      │         │
│  │                     │       │                     │         │
│  │  ┌───────────────┐  │       │  ┌───────────────┐  │         │
│  │  │ SGLang Server │  │ NCCL  │  │ SGLang Server │  │         │
│  │  │   node_rank=0 │◄─┼───────┼─►│   node_rank=1 │  │         │
│  │  │   tp=2        │  │       │  │   tp=2        │  │         │
│  │  │   port=30000  │  │       │  │   port=30000  │  │         │
│  │  └───────────────┘  │       │  └───────────────┘  │         │
│  │        │            │       │                     │         │
│  │    [GPU0] [GPU1]    │       │    [GPU0] [GPU1]    │         │
│  └────────┼────────────┘       └─────────────────────┘         │
│           │                                                     │
│           ▼                                                     │
│      API Endpoint                                               │
│   http://NODE0_IP:30000                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ray Actor Backend

```
┌─────────────────────────────────────────────────────────────────┐
│                        Anyscale Cluster                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     sglang.Engine                        │   │
│  │                    (use_ray=True)                        │   │
│  │  ┌─────────────────────────────────────────────────────┐│   │
│  │  │           ray_cluster_utils                         ││   │
│  │  │  • discover_gpu_nodes()                             ││   │
│  │  │  • create_cluster_topology()                        ││   │
│  │  │  • compute_rank_to_node_assignment()                ││   │
│  │  └─────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│           ┌──────────────────┼──────────────────┐              │
│           ▼                  ▼                  ▼              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ PlacementGroup 0│ │ PlacementGroup 1│ │ PlacementGroup N│   │
│  │  (STRICT_PACK)  │ │  (STRICT_PACK)  │ │  (STRICT_PACK)  │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           │                   │                   │            │
│  ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐   │
│  │     Node 0      │ │     Node 1      │ │     Node N      │   │
│  │ ┌─────┐ ┌─────┐ │ │ ┌─────┐ ┌─────┐ │ │ ┌─────┐ ┌─────┐ │   │
│  │ │Sched│ │Sched│ │ │ │Sched│ │Sched│ │ │ │Sched│ │Sched│ │   │
│  │ │Actor│ │Actor│ │ │ │Actor│ │Actor│ │ │ │Actor│ │Actor│ │   │
│  │ │TP=0 │ │TP=1 │◄┼─┼►│TP=2 │ │TP=3 │◄┼─┼►│TP=N │ │TP=M │ │   │
│  │ └──┬──┘ └──┬──┘ │ │ └──┬──┘ └──┬──┘ │ │ └──┬──┘ └──┬──┘ │   │
│  │   GPU0   GPU1   │ │   GPU0   GPU1   │ │   GPU0   GPU1   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│           │                   │                   │            │
│           └───────────────────┴───────────────────┘            │
│                          NCCL                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Server fails to start

1. Check NCCL connectivity between nodes:
   ```bash
   # Set NCCL_DEBUG=INFO for detailed logs
   ```

2. Verify GPU availability:
   ```bash
   ray status  # Check available resources
   ```

3. Check firewall rules allow communication on:
   - Port 20000 (dist-init)
   - Port 30000 (API server)
   - NCCL ports (dynamic, typically high ports)

### Out of Memory

- Reduce model size or increase GPU memory
- Check if other processes are using GPU memory

### Connection timeouts

- Increase timeout in `wait_for_server_ready()`
- Check network connectivity between nodes
- Verify security groups allow inter-node traffic

## Scaling

To use more GPUs or nodes, adjust:

1. **job.yaml**: Update `instance_type` and `min_nodes`/`max_nodes`
2. **driver.py arguments**: Update `--tp` and `--nnodes`

Example for 4 nodes with TP=4 each (16 GPUs total):
```bash
python driver.py --model-path meta-llama/Llama-2-70b-hf --tp 4 --nnodes 4
```
