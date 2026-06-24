# KV Transfer Auto Experiment Runbook

Date: 2026-06-24

## Goal

Run the remaining capped Mooncake-background experiments from node 099 without manual step-by-step orchestration. Node 099 starts foreground/background initiators; node 102 starts target containers and RDMA receive counter monitors over SSH.

## Script

Use:

```bash
python3 /root/kv_transfer_bench/kv_auto_experiment.py
```

Required on both nodes:

- `/root/kv_transfer_bench/kv_transfer_latency.py`
- `/root/kv_transfer_bench/kv_auto_experiment.py` on node 099
- Docker image `sglang-pd-switch:tianciJ`
- Passwordless SSH from 099 to 102, normally `root@192.168.0.41`

## Fixed Missing Matrix

Run suite:

```bash
python3 kv_auto_experiment.py --suite fixed-missing
```

Included runs:

- `800_4x200_bg{1,10,50,90}_cap200_moonbg_fixed`
- `200_1x200_tail_single_nic_bg{1,10,50,90}_cap200_moonbg_fixed`
- `200_1x200_head_rdma_bg{1,10,50,90}_cap200_moonbg_fixed`
- `200_1x200_head_tcp_bg{1,10,50,90}_cap200_moonbg_fixed`

Rate policy:

- Per-lane cap is 200 Gbps for these missing runs.
- Background rate is `cap * bg_percent`.
- Foreground rate is `cap - background_rate`.

Outputs:

- Raw logs: `/tmp/kv-transfer-bench/auto/<run>/raw/`
- Aggregated matrix: `/tmp/kv-transfer-bench/auto/<run>/aggregated-summary.csv`
- RDMA receive counter: `/tmp/kv-transfer-bench/auto/<run>/raw/rdma-rcv-monitor.csv`

## Competition Matrix

Run suite:

```bash
python3 kv_auto_experiment.py --suite competition
```

Included two-flow runs:

- `comp_2flows_2GB_vs_{256MB,512MB,1GB,2GB,4GB}_fair`
- `comp_2flows_2GB_vs_{256MB,512MB,1GB,2GB,4GB}_uncapped`

Included multi-flow runs:

- `comp_{1,2,4,8}flows_2GB_equal_fair`
- `comp_{1,2,4,8}flows_2GB_equal_uncapped`

Raw per-flow timing is kept in:

- `/tmp/kv-transfer-bench/auto/competition/<run>/raw/competition-flow*.log`
- `/tmp/kv-transfer-bench/auto/competition/<run>/raw/competition-flow*-samples.jsonl`
- `/tmp/kv-transfer-bench/auto/competition/<run>/competition-events.jsonl`
- `/tmp/kv-transfer-bench/auto/competition/<run>/competition-summary.csv`

The `fair` cases explicitly split the 200 Gbps lane cap evenly across flows. The `uncapped` cases omit per-flow rate limits to compare Mooncake's natural sharing behavior.

## Common Commands

List all planned runs:

```bash
python3 kv_auto_experiment.py --suite list
```

Dry-run one run:

```bash
python3 kv_auto_experiment.py --suite fixed-missing --only 800_4x200_bg1_cap200_moonbg_fixed --dry-run
```

Run one fixed-missing experiment:

```bash
python3 kv_auto_experiment.py --suite fixed-missing --only 800_4x200_bg1_cap200_moonbg_fixed
```

Run one competition experiment:

```bash
python3 kv_auto_experiment.py --suite competition --only comp_2flows_2GB_vs_512MB_fair
```

Run in background:

```bash
mkdir -p /tmp/kv-transfer-bench/auto
nohup python3 kv_auto_experiment.py --suite fixed-missing \
  > /tmp/kv-transfer-bench/auto/fixed-missing.log 2>&1 &
tail -f /tmp/kv-transfer-bench/auto/fixed-missing.log
```

## Host Overrides

Defaults assume:

- Node 099 TCP source host: `192.168.0.39`
- Node 102 TCP target host: `192.168.0.41`
- Node 102 SSH host: `root@192.168.0.41`

Override if needed:

```bash
export TARGET_SSH_HOST=root@192.168.0.41
export TARGET_SSH_KEY=~/.ssh/kvbench_102
export KV_HEAD_TCP_SRC_HOST=192.168.0.39
export KV_HEAD_TCP_TGT_HOST=192.168.0.41
export KV_SINGLE_NIC_IB_DEVICE=mlx5_0
```

