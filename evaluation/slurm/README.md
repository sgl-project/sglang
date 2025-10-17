# Introduction

SLURM (Simple Linux Utility for Resource Management) is an open-source resource manager and job scheduler widely used for managing large-scale compute clusters.

more details follow [USERS: Get started with Slurm on OCI cluster - IT - AI Platform - Confluence](https://amd.atlassian.net/wiki/spaces/ITAI/pages/757629783/USERS+Get+started+with+Slurm+on+OCI+cluster) and [SLURM MAIN PAGE - IT - AI Platform - Confluence](https://amd.atlassian.net/wiki/spaces/ITAI/pages/757629927/SLURM+MAIN+PAGE)

# Ues slurm to lanuch 1P1D

Launch a Terminal and execute the following commands, nd a job ID will be output once the launch is successful.

```
sbatch auto_pd_distributed.sbatch
```

Check SLURM job status

```
squeue -u $USER
```

Cancel SLURMSLURM job

```
scancel <job_id>
```

There are still some issues at present. The sbatch script cannot automatically perform task cleanup operations. You need to manually stop the docker on each node or execute it using the following script.

```
#!/bin/bash

# get node list
NODES=("useocpm2m-401-013" "useocpm2m-401-016")  # Replace with the actual node name

echo "Starting DeepSeek cleanup..."

for node in "${NODES[@]}"; do
    echo "=== Cleaning up on $node ==="

    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $node "
        docker rm -f deepseek-prefill 2>/dev/null || echo 'No prefill container found'
        docker rm -f deepseek-router 2>/dev/null || echo 'No router container found'
        docker rm -f deepseek-decode 2>/dev/null || echo 'No decode container found'
        echo 'Checking for related processes:'
        ps aux | grep -i deepseek | grep -v grep || echo 'No related processes found'
    "

    echo "Cleanup completed on $node"
    echo
done

echo "All cleanup operations completed!"
```