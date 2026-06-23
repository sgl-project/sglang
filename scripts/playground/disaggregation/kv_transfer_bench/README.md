# KV Transfer Latency Benchmark

This benchmark measures two-node GPU-buffer transfer latency through SGLang's
Mooncake transfer engine. It does not launch a model server. Use it before the
full PD flip experiment to understand the relation between transfer size and
communication latency.

## Files

- `kv_transfer_latency.py`: Python benchmark with `target` and `initiator` roles.
- `run_target.sh`: Docker wrapper for the target node.
- `run_initiator.sh`: Docker wrapper for the initiator node.

## Node Plan

Start with the cleanest pair:

```text
target:    lingjun-102
initiator: lingjun-099
```

Then sweep more pairs:

```text
099 -> 100
099 -> 101
099 -> 102
100 -> 101
100 -> 102
101 -> 102
```

Each node needs:

```bash
docker images | grep sglang-pd-switch
docker run --rm --gpus all sglang-pd-switch:tianciJ nvidia-smi
nvidia-smi
```

Accept a node only when the image exists, Docker can see all 8 GPUs, and
`nvidia-smi` shows no unrelated process occupying the GPU selected by `GPU_ID`.

## Common Environment

Best case: run from the repository checkout on each server:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/kv_transfer_bench
export SGLANG_REPO=/home/tiancij/sglang
export SGLANG_IMAGE=sglang-pd-switch:tianciJ
export OUTPUT_DIR=/tmp/kv-transfer-bench
export GPU_ID=0
export IB_DEVICE=mlx5_0
export PROTOCOL=rdma
```

If the server does not have the full repo, copy only this directory to the
server and run from that copied directory. In that case, do not set
`SGLANG_REPO`; the wrappers mount the current benchmark directory and use the
SGLang package installed inside `sglang-pd-switch:tianciJ`.

Example:

```bash
mkdir -p /root/kv_transfer_bench
# copy kv_transfer_latency.py, run_target.sh, run_initiator.sh, README.md here
cd /root/kv_transfer_bench
unset SGLANG_REPO
export SGLANG_IMAGE=sglang-pd-switch:tianciJ
```

If RDMA is not ready and you only want a TCP sanity check:

```bash
export PROTOCOL=tcp
export MC_FORCE_TCP=1
```

## First Pair: 099 -> 102

### 1. Start Target On 102

On `lingjun-102`:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/kv_transfer_bench
export SGLANG_REPO=/home/tiancij/sglang
export SGLANG_IMAGE=sglang-pd-switch:tianciJ
export OUTPUT_DIR=/tmp/kv-transfer-bench
export HOST_IP=$(hostname -I | awk '{print $1}')
export GPU_ID=0
export IB_DEVICE=mlx5_0
export PROTOCOL=rdma
export MAX_BYTES=2GB
./run_target.sh
```

If 102 only has `/root/kv_transfer_bench`, replace the first two lines with:

```bash
cd /root/kv_transfer_bench
unset SGLANG_REPO
```

Keep this process running. It prints a line like:

```text
TARGET_INFO_JSON={"bytes":2147483648,...,"session_id":"192.168.0.42:12345"}
```

### 2. Pass Target Info To 099

Copy the full `TARGET_INFO_JSON=...` value from the target output. On
`lingjun-099`, paste it as an environment variable:

```bash
export TARGET_INFO_JSON='{"bytes":2147483648,"gpu_id":0,"host":"192.168.0.42","ib_device":"mlx5_0","protocol":"rdma","ptr":123456,"session_id":"192.168.0.42:12345"}'
```

Use the actual JSON from 102, not the example above.

### 3. Run Initiator On 099

Start with a small sweep:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/kv_transfer_bench
export SGLANG_REPO=/home/tiancij/sglang
export SGLANG_IMAGE=sglang-pd-switch:tianciJ
export OUTPUT_DIR=/tmp/kv-transfer-bench
export HOST_IP=$(hostname -I | awk '{print $1}')
export GPU_ID=0
export IB_DEVICE=mlx5_0
export PROTOCOL=rdma
export SIZES=1MB:64MB:x2
export WARMUP=2
export REPEAT=5
./run_initiator.sh
```

If 099 only has `/root/kv_transfer_bench`, replace the first two lines with:

```bash
cd /root/kv_transfer_bench
unset SGLANG_REPO
```

If this succeeds, run the full sweep:

```bash
export SIZES=1MB:2GB:x2
export WARMUP=3
export REPEAT=20
./run_initiator.sh
```

Results are written on the initiator node:

```text
/tmp/kv-transfer-bench/summary.csv
/tmp/kv-transfer-bench/samples.jsonl
```

## Acceptance Criteria

For each pair:

```text
target prints target_ready=true
initiator completes without warmup transfer failure
summary.csv has one row per data size
samples.jsonl has repeat_count rows per data size
error_count is 0 for every summary row
nvidia-smi shows no benchmark process left after target is stopped
```

## Stop Target

Press `Ctrl-C` in the target terminal. Then verify:

```bash
nvidia-smi
```
