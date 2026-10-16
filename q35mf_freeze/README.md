# Frozen tree for five Qwen3.5-397B-A17B NVFP4 AgentX operating points (GB300)

This directory documents how to reproduce five measured AgentX operating points with this tree.
The code is the merge commit `4d7b8aba94` of two lines that both fork from `a6a8c56015`:

- `99859a6dba`: stacked decode-path optimizations (used by points 2, 3, 4 and 5);
- `72cbd15b54`: PD prefill / decode package (used by point 1).

Both parents are ancestors of this branch. The merged tree equals the base plus both diffs line for line.
Every non-stock behaviour sits behind a default-off environment switch, so each point turns on only the
switches listed for it below. With no switch set, the tree runs the base code paths.

## Common setup (all points)

| Item | Value |
|---|---|
| Container image | `lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560`; this tree's `python/sglang` is mounted over the image's editable install (`/sgl-workspace/sglang/python/sglang`, read-only). Keep the image's compiled artifacts (`_version.py`, Rust `*.so`) in the mounted tree. |
| Checkpoint | `Qwen3.5-397B-A17B-NVFP4-V2` mounted at `/model`; served name `nvidia/Qwen3.5-397B-A17B-NVFP4`. Experts NVFP4, attention / shared expert FP8 per-tensor, FP8 KV cache. The MTP draft runs as shipped (BF16): no `--speculative-draft-model-quantization`, no `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE`. |
| Acceptance | Forced acceptance length 4.8 for 6 MTP steps (`SGLANG_SIMULATE_ACC_LEN=4.8`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`), as the benchmark harness injects for this model. |
| Client | InferenceX `9330deaf` mounted at `/infmax-workspace`, `bash /infmax-workspace/benchmarks/multi_node/agentic_srt.sh` with the point's `client.env` (aiperf `754356e9`, dataset `semianalysis_cc_traces_weka_062126_256k`, scenario `agentic-coding`, warmup 10 per lane). Duration 3600 s. Set `RESULT_FILENAME` and the `<..._IP>` placeholders. |
| PD runtime (points 1-3) | dynamo `61c37bc3` plus the upstream `e18e53fa` attention-DP-size fix (installed by the worker scripts from a wheel cache under `/configs/dynamo-wheels`); NATS + etcd on the head node (`infra.sh`); `dynamo.frontend` behind nginx 1.27.4 (`nginx.conf`, consistent hash on `X-Dynamo-Session-ID`); KV transfer with Mooncake over NVLink, so P and D must sit in one NVL72 rack. |
| Prefill weights (PD) | Load the prefill workers from a shared-filesystem copy, not a `/dev/shm` copy: the HiCache host pools need that host memory. |

Placeholders in the scripts: `<P{i}_HOST>` / `<P{i}_IP>` = prefill worker i (worker 0 also runs infra, nginx and the client),
`<D0_N{k}_HOST>` / `<D0_N{k}_IP>` = node k of the decode worker, `<NODE{k}_HOST>` / `<NODE{k}_IP>` = node k of an aggregated server.
The worker scripts are the exact rendered launch commands of the reference runs; only hosts and IPs were replaced.

## The five points

Each row lists the topology, the launch files, the switches the point turns on and the client concurrency.

| Point | Topology (GPUs) | Files | Switches turned on (all others off) | Conc. |
|---|---|---|---|---:|
| 1 C704 PD | 3 prefill workers (TP1, PP4, 4 GPUs each) + 1 decode worker (DEP8 over 2 nodes) = 20 | `point1_c704_pd/` | prefill: `SGLANG_Q35MF_GDN_PREFILL_SPLIT_CONV`; decode: `SGLANG_Q35MF_PDDEC_FENCE`, `SGLANG_Q35MF_PDDEC_DEFER_PREALLOC`, `SGLANG_Q35MF_PDDEC_NONBLOCKING_ADMIT`, `SGLANG_Q35MF_PDDEC_SPLIT_HOIST` | 704 |
| 2 PD mini-unit, decode cap 10 | 1 prefill (TP1, HiCache ratio 12) + 1 decode (TP2, max running 10) = 3 | `point2_f6_pd_mini_cap10/` | decode only: the 24 switches set to 1 in `decode_w0.sh` (GDN verify set, attention front / gate quant, host-sync and post-verify graph set, fused routing, draft-path set); `SGLANG_ENABLE_METADATA_GLUE_GRAPH=0` | 56 |
| 3 PD mini-unit, decode cap 6 | 1 prefill (TP1) + 1 decode (TP2, max running 6) = 3 | `point3_f3_pd_mini_cap6/` | decode only: same switch set as point 2 (see `decode_w0.sh`) | 38 |
| 4 Aggregated TP2 b1 | 1 server, TP2, max running 1 = 2 | `point4_tp2_b1_c10/` | the 25 switches set to 1 in `server_node0.sh` (the point 2 set plus `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1`) | 10 |
| 5 Aggregated TP8 b1 (C7) | 1 server, TP8 over 2 nodes, max running 1 = 8 | `point5_c7_tp8/` | the 32 switches set to 1 in `server_node*.sh`, plus `--moe-runner-backend flashinfer_trtllm --flashinfer-allreduce-fusion-backend cutedsl` | 7 |

Notes:
- Point 2 and 3 decode workers run on GPUs 1-2 of the node (`CUDA_VISIBLE_DEVICES=1,2` in `decode_w0.sh`); the prefill worker uses GPU 0.
- Point 4 ran with a fixed autotune tactic snapshot (FlashInfer autotune cache seeded before start); points 1, 2, 3 and 5 run the autotuner at start (reusing its on-disk cache when one is present).
- `SGLANG_Q35MF_GDN_DIRECT_SOLVE=0` and `SGLANG_Q35MF_GDN_EMPTY_HISTORY=0` in points 2-4 restate the defaults.
- Measured results (the reference runs and the reproductions on this tree) are kept with the benchmark notes, not in this directory.

## Switch reference

Each switch is defined in `python/sglang/srt/environ.py` with a one-line description; the default of every switch is off.
Point 1 sets none of the switches of points 2-5, and points 2-5 set none of point 1's switches.
