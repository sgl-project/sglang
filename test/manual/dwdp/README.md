# ROCm DWDP development tests

Submit every GPU test through Slurm:

```bash
NODE_LIST=smci355-ccs-aus-n08-25,smci355-ccs-aus-n08-29,smci355-ccs-aus-n08-33,smci355-ccs-aus-n09-21,smci355-ccs-aus-n09-25 \
SLURM_EXCLUDE_NODES=smci355-ccs-aus-n08-21 \
bash scripts/ci/amd/submit_dwdp_rocm_slurm.sh vmm-poc
```

Available arms are printed by the submit script. The submitter removes
`smci355-ccs-aus-n08-21` from the candidate pool, adds it to Slurm's exclusion
list, and verifies the resulting job description.

Before calling `sbatch`, the submitter requires Slurm state `idle` and reads
each candidate's per-GPU VRAM usage. By default a node is rejected if any GPU
uses more than 4 GiB, if the eight GPUs differ by more than 2 GiB, or if all
eight GPUs cannot be queried. The allocated job repeats the same check before
compilation or model startup to close the scheduling race. Thresholds can be
changed with `MAX_USED_VRAM_GIB`, `MAX_VRAM_SKEW_GIB`, and
`EXPECTED_GPU_COUNT`. If no node passes, no test job is submitted. The check
reads memory counters only; it never resets a GPU or inspects or terminates
processes.

After the container exits, the job gives VRAM up to 60 seconds to return below
the same limits. Persistent allocations mark the job failed and the node
unsafe for later submissions, even if the functional test itself passed.

Do not run ROCm GPU tests directly on the login host, reset GPUs, or terminate
unrelated processes. CPU-only lint and static tests may run locally.

Model smoke tests accept environment overrides:

```bash
NODE_LIST=smci355-ccs-aus-n08-25,smci355-ccs-aus-n08-33,smci355-ccs-aus-n09-21 \
SLURM_EXCLUDE_NODES=smci355-ccs-aus-n08-21 \
BACKENDS="vmm ipc" \
BASE_PORT=39000 \
MODEL_PATH=/models/DeepSeek-R1-0528-MXFP4-th \
bash scripts/ci/amd/submit_dwdp_rocm_slurm.sh model-smoke
```
