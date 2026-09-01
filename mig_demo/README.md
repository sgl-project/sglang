# Emulating multi-GPU SGLang serving on a single GPU

Demo of [stas00/ml-engineering: Emulating multiple GPUs with a single GPU](
https://github.com/stas00/ml-engineering/blob/master/training/emulate-multi-node.md#emulating-multiple-gpus-with-a-single-gpu)
applied to **SGLang serving** (the doc targets training): several independent
SGLang server processes treat one physical H200 as a multi-GPU / multi-node
deployment, for development and testing of distributed code paths without a
multi-GPU box.

## Environment

- 1x NVIDIA H200 141GB (rx devbox `mig-demo`, h200-sci-k8s), driver 580.105.08, CUDA 13.0
- torch 2.13.0+cu130 (bundles NCCL 2.29.7) — **NCCL 2.31.2 preloaded** via
  `LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2`
  (`pip install -U --target /root/nccl-new 'nvidia-nccl-cu13>=2.31'`; the pip
  pin-conflict warning is expected — that is why the preload copy exists)
- sglang 0.5.18 (`lmsysorg/sglang:latest` image), model `Qwen/Qwen3-0.6B`

## MIG caveat (platform-specific)

The doc's recipe carves the GPU into MIG instances. On this platform the
devbox pod lacks `CAP_SYS_ADMIN`, so `nvidia-smi -i 0 -mig 1` fails with
`Insufficient Permissions` (exit 4) — MIG must be enabled host-side by the
cluster. All cases below therefore run the emulated ranks on the **whole GPU**,
which NCCL 2.31's `NCCL_MULTI_RANK_GPU_ENABLE=1` permits. Trade-off vs MIG:
no memory/compute isolation between ranks (emulated with
`--mem-fraction-static` caps instead), and no P2P/CUMEM path between
instances — the transport falls back to host shared memory.

## The three required ingredients

```bash
export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2  # NCCL >= 2.31
export NCCL_MULTI_RANK_GPU_ENABLE=1  # allow >1 rank per physical GPU
export NCCL_NVLS_ENABLE=0            # no NVLink SHARP between emulated ranks
```

Without `NCCL_MULTI_RANK_GPU_ENABLE=1` the emulated launch dies at
distributed init: `RuntimeError: NCCL error: invalid usage` (negative control
verified).

## Case 1 — TP=2 across two emulated nodes on one GPU

`--tp` is GLOBAL; `--nnodes 2` puts one scheduler rank in each process. Each
process binds `cuda:0` of the same H200 (`CUDA_VISIBLE_DEVICES=0` for both).

```bash
# node 0 (serves HTTP) — start first
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --nnodes 2 --node-rank 0 --tp 2 --host 127.0.0.1 --port 30000 \
  --dist-init-addr 127.0.0.1:25000 \
  --disable-custom-all-reduce --mem-fraction-static 0.35
# node 1 (scheduler + dummy health server on its own port)
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --nnodes 2 --node-rank 1 --tp 2 --host 127.0.0.1 --port 30001 \
  --dist-init-addr 127.0.0.1:25000 \
  --disable-custom-all-reduce --mem-fraction-static 0.35
```

SGLang pitfalls handled (from source): `--dist-init-addr` is mandatory (else
each node draws its own random `nccl_port` and rendezvous hangs); node 1 needs
a distinct `--port` (its dummy health server would clash with node 0's HTTP);
`--disable-custom-all-reduce` skips the custom-AR init that cannot parse a
UUID/shared CUDA_VISIBLE_DEVICES (auto-falls back to NCCL anyway).

Verified: `nvidia-smi` shows both server processes resident on GPU 0
(~54 GiB each), `/health` up on both ports, greedy generations through the
TP=2 engine produce **byte-identical output to the TP=1 baseline**.

| case | decode throughput |
|---|---:|
| TP=1 baseline (single process) | 576.3 tok/s |
| TP=2 emulated, 2 ranks on 1 GPU | 3.8 tok/s |
| TP=2 emulated + `NCCL_CUMEM_ENABLE=1` | 3.8 tok/s (no change) |

Why so slow: 28 layers x 2 all-reduces per token = 56 tiny collectives per
decode step, and with both ranks on one device NCCL cannot use P2P — measured
interconnect between the emulated ranks:

```
all_reduce 512MiB payload, 2 ranks on one H200: busbw = 59.0 GB/s
```

vs the doc's 308 GB/s for 2 real H200s and 413 GB/s for 2x `3g.71gb` MIG
instances of one H200. This emulates the **topology**, not the performance —
a dev/testing tool, exactly as the doc warns.

## Case 2 — PP=2 across two emulated nodes on one GPU

Same launch shape with `--pp-size 2 --tp 1`. Pipeline stages exchange one p2p
send/recv per microbatch instead of per-layer all-reduces, so the slow
transport hurts far less:

| case | decode throughput |
|---|---:|
| PP=2 emulated, 2 stages on 1 GPU | 252.7 tok/s (44% of TP=1 baseline) |

Greedy output again byte-identical to baseline.

## Case 3 — two independent TP=1 servers (DP-like) on one GPU

No collectives at all: two `launch_server` processes, `--mem-fraction-static`
caps emulating MIG-style memory partitioning, benched concurrently.

| case | KV tokens | c=1 per engine | c=8 per engine | c=16 per engine |
|---|---:|---:|---:|---:|
| instance A (:30000), `--mem-fraction-static 0.42` | 517,529 | 319.7 | 4,732.6 | 4,945.9 |
| instance B (:30001), `--mem-fraction-static 0.84` | 540,798 | 325.2 | 4,687.0 | 4,973.8 |
| **aggregate** | **1,058,327** | **645.0** | **9,419.6** | **9,919.7** |

### vs a single instance on the whole GPU (`bench_single.sh`)

Equal-total-concurrency comparison, tok/s:

| total concurrency | single instance | DP-like aggregate | winner |
|---:|---:|---:|---|
| 2 (1/engine) | 582.7 | 645.0 | DP +10.7% |
| 16 (8/engine) | 8,955.3 | 9,419.6 | DP +5.2% |
| 32 (16/engine) | **15,499.2** | 9,919.7 | **single +56%** |
| KV capacity | 1,164,732 | 1,058,327 | single +9% |

Verdict: co-location wins only at low concurrency (two schedulers overlap
CPU-side overheads). At high load a single engine batches all 32 requests
into one efficient decode step while the two co-located engines run two
batch-16 decodes on contending SMs — single instance wins by 56% and also
holds 9% more KV. The real value of the DP-like shape is isolation
(separate KV pools, independent restart/failure domains, per-tenant QoS),
not peak throughput.

(All requests share one prompt, so RadixCache prefix caching amortizes
prefill; these are decode-throughput measurements.)

Mem-fraction pitfall: the second server computes its minimum viable fraction
*after* the first server's pool exists, so it must be strictly greater than
the first's (~+0.01). Symmetric 0.35 and 0.40 both failed with
`Loaded weights leave no GPU memory for the KV cache ... minimum viable = ...`;
asymmetric 0.44/0.47 starts.

But the fractions do NOT partition memory proportionally. sglang sizes each
pool as `fraction x total_gpu_mem - memory_already_used`, so the second
server only gets the delta between the two fractions:

| server | fraction | KV pool (K+V) | max_total_num_tokens | context_len |
|---|---:|---:|---:|---:|
| A (:30000, first) | 0.44 | 58.4 GiB | 546,661 | 40,960 |
| B (:30001) | 0.47 | 3.4 GiB | 31,877 | 40,960 |

B ended up with 1/17 of A's KV capacity from a 3-point fraction gap — below
one full 40,960-token context. For a roughly even split, the second server's
fraction must be ~2x the first's (pool_B = fraction_B x total - used_A);
verified: 0.42/0.84 lands 517,529 / 540,798 tokens (within 5%). Hardware
MIG partitions would not have this coupling; these software caps only
emulate them coarsely.

## Files

- `launch_emulated_tp2.sh` — canonical TP=2 emulated launch + health wait
- `run_case.sh` — parameterized case runner (`PP=2 ./run_case.sh ...` for PP)
- `dp_like.sh` — Case 3 runner
- `bench_single.sh` — single-instance baseline at the same loads (c=1/8/16/32)
- `profile_dp.sh` — Case 3 torch-profiler capture (`/start_profile` on both
  instances under concurrent load; traces in `traces/`, open in Perfetto)
- `traces/dp-instance-{A,B}.trace.json.gz` — Chrome traces of both instances
  profiled simultaneously (60 forward steps, CPU+GPU activities)
- `verify_and_bench.py` — correctness + tok/s probe against `/generate`
- `nccl_same_gpu_test.py` — minimal proof NCCL 2.31 runs 2 ranks on one GPU
- `all_reduce_bench_2rank.py` — emulated-interconnect busbw measurement
