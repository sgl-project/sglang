# Slurm Examples

Slurm batch scripts for running SGLang on HPC clusters.

## Prerequisites

- A Slurm cluster with GPU nodes.
- SGLang installed on all compute nodes, or a container image with SGLang.

## Examples

### Single-Node Serving

Launch a single-node server for a model like Llama:

```bash
sbatch examples/slurm/single-node.slurm
```

Customize with environment variables:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct GPUS=2 sbatch examples/slurm/single-node.slurm

# Use a container image (Apptainer/Singularity)
MODEL=meta-llama/Llama-3.1-8B-Instruct CONTAINER_IMAGE=docker://lmsysorg/sglang:latest sbatch examples/slurm/single-node.slurm
```

### Multi-Node DeepSeek Serving

Launch a multi-node server for DeepSeek models (e.g., DeepSeek-V3 on 2x8 GPUs):

```bash
sbatch examples/slurm/multi-node-deepseek.slurm
```

Customize with environment variables:

```bash
MODEL=deepseek-ai/DeepSeek-V3 GPUS_PER_NODE=8 sbatch examples/slurm/multi-node-deepseek.slurm

# With a container
MODEL=deepseek-ai/DeepSeek-V3 CONTAINER_IMAGE=docker://lmsysorg/sglang:latest sbatch examples/slurm/multi-node-deepseek.slurm
```

## Slurm Directives

Edit the `#SBATCH` directives at the top of each script to match your cluster:

| Directive | Description |
|---|---|
| `--partition` | Slurm partition name (e.g., `gpu`, `compute`) |
| `--account` | Slurm account for billing |
| `--time` | Maximum job runtime |
| `--gpus-per-node` | GPUs to request per node |

## Container Support

The scripts auto-detect the container runtime. Set `CONTAINER_IMAGE` to use one:

```bash
CONTAINER_IMAGE=docker://lmsysorg/sglang:latest          # Docker Hub image (Apptainer/Singularity)
CONTAINER_IMAGE=/path/to/sglang.sqsh                     # Local squashfs image
```

Supported runtimes: Apptainer, Singularity, Enroot.

## See Also

- DeepSeek V3 multi-node instructions: [benchmark/deepseek_v3/README.md](../benchmark/deepseek_v3/README.md)
- K8s distributed serving: [docker/k8s-sglang-distributed-sts.yaml](../docker/k8s-sglang-distributed-sts.yaml)
- Docker Compose example: [docker/compose.yaml](../docker/compose.yaml)
