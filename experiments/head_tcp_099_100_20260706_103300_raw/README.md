# Head TCP sweep, cloud-099 to cloud-100

Run time: 2026-07-06 10:33 Asia/Shanghai

Nodes:
- Source: cloud-099, 192.168.0.42
- Target: cloud-100, 192.168.0.40

Command:

```bash
python3 scripts/playground/disaggregation/kv_transfer_bench/kv_auto_experiment.py \
  --suite head-tcp-sweep \
  --target-ssh-host root@192.168.0.40 \
  --head-tcp-src-host 192.168.0.42 \
  --head-tcp-tgt-host 192.168.0.40 \
  --out-root /root/kv_transfer_bench/results/head_tcp_099_100_20260706_103300
```

Runs:

| run | rows | total errors | avg bandwidth all sizes (GB/s) | avg bandwidth >=1GiB (GB/s) | 2GiB bandwidth mean (GB/s) | 2GiB latency mean (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 100_1x100_head_tcp_bg0_fine_dense | 46 | 0 | 1.645 | 1.699 | 1.724 | 1159.947 |
| 200_1x200_head_tcp_bg0_fine_dense | 46 | 0 | 1.871 | 1.971 | 1.967 | 1016.958 |

Size coverage:
- 1MiB through 32MiB in 1MiB steps
- 48MiB, 64MiB, 96MiB, 128MiB, 192MiB, 256MiB, 384MiB, 512MiB, 768MiB, 1GiB, 1.25GiB, 1.5GiB, 1.75GiB, 2GiB

Local data layout:
- `source/`: source-side logs, per-size sample JSONL, shard summary CSV, aggregated summary CSV, and run log copied from cloud-099
- `target/`: target-side `target-bond0.log` and `target-bond0.json` copied from cloud-100
