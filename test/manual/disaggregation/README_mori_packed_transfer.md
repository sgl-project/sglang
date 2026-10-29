# MoRI heterogeneous-TP page packing

`SGLANG_MORI_KV_PACKING=1` enables sender-side page packing for schema-proven
contiguous NHD GQA caches with prefill TP smaller than decode TP. Decode TP
must divide the KV head count and be a multiple of prefill TP. Full attention
and one optional ordinary SWA pool are supported. The destination receives
its final local-head page layout directly; it does not unpack a staging pool.

Both peers must advertise matching versioned byte-layout metadata. Legacy
peers and unsupported layouts retain the existing path; heterogeneous SWA
still raises its existing error unless its direct head-slice layout is proven.
The unpacked direct SWA path is also available with packing disabled, providing
a correctness reference for the newly supported topology.

Scope excludes PP/CP/DCP, replicated KV heads, draft KV, recurrent state,
quantization scale sidecars, HND, and unified page-major storage. Deferred
decode KV release must be enabled. Storage proofs use `view`, `wire_entries`
and `regions`, including actual registered pointers, sizes and layer order.

`SGLANG_MORI_KV_PACK_BUFFER_SIZE_MB` defaults to 64. Each transfer worker
lazily owns one registered buffer shared across peers/components. Larger
payloads use bounded batches; insufficient capacity for one page falls back.
Every batch drains before reuse. Partial submissions retain all known
statuses; failed or ambiguous submissions permanently quarantine the sender
registration and keep the remote release hold. This PR has no producer fusion or
overlapped multi-buffer pipeline.

## Validation

CPU contract/lifetime checks:

```bash
python3 test/registered/unit/disaggregation/test_mori_packed_contract.py
python3 test/registered/unit/disaggregation/test_mori_packed_sender_lifetime.py
```

Real GPU gather, randomized pages and untouched-page guards, fake transport:

```bash
pytest -q test/registered/kernels/ops/memory/test_mori_packed_gather.py
```

The manual harness requires exactly two processes, actual GPU memory
registration and MoRI RDMA. It checks all destination bytes independently
after every iteration. Its baseline submits one `batch_write` per K/V tensor,
matching the production direct path. Same-node example:

```bash
MORI_DISABLE_AUTO_XGMI=1 torchrun --standalone --nproc-per-node=2 \
  test/manual/disaggregation/bench_mori_packed_transfer.py \
  --prefill-tp 4 --decode-tp 8 --decode-rank 1 --heads 16 \
  --page-size 16 --tokens 1024 --swa-window 512 --mode all \
  --warmup 3 --iterations 10
```

Repeat with `--order packed,baseline` and `--packed-capacity-bytes 32768`
to exercise reversed order and multiple batches. Two-node execution uses
`torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0` (or `1`) with a shared
master address/port, and `MORI_BENCH_BIND_HOST` set to each node's routable IP.

Reported timings include submission, drain and/or a Gloo barrier; they are
not network-only latency or model-serving performance. Submitted descriptor
counts are API offset/size entries, not posted WQEs or verbs calls. Old
prototype results do not validate this extraction onto current upstream;
model serving and cross-node RDMA need independent validation on rebased sources.

## Rebased Muse experiment

Source base `1776e2dcf8ec663054901e0e371eedd546cd00f3`, packing code
`58ee16666202d4f19f65618e1b77ba59a7333308`. Serving additionally used the
experimental ROCm runtime from [#44667](https://github.com/sgl-project/sglang/pull/44667),
pinned at `37197b72708c0bc6f89c5c54a1719df22455613c`, with routing call-arity
and optional lm-head output-buffer compatibility fixes. Those runtime changes
are outside this PR.

Checkpoint: `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`, revision
`c7bc74fd64371315a091bc15f9723882eead0dc1`. MI355X, text only, TP=EP,
P4/D8, BF16 KV, page16, Triton attention/MoE, graphs/radix/overlap off,
context4096, max-total-tokens16384, concurrency1, seed20261028.
Decode stayed in the same process; prefill restarted per arm. Direct HTTP
submitted decode before prefill. Every arm had 16 shard-covering warmups and
50 measured requests, ISL1024/OSL1. These times include client orchestration.

| Arm, in execution order | TTFT median ms | Mean ms | p95 ms |
| --- | ---: | ---: | ---: |
| Direct A | 137.796 | 137.908 | 139.491 |
| Packed64 A | 119.651 | 123.667 | 130.883 |
| Packed32 A | 118.747 | 147.203 | 120.510 |
| Packed64 B | 119.001 | 146.824 | 121.399 |
| Packed32 B | 118.724 | 149.490 | 120.919 |
| Direct B | 136.881 | 137.034 | 139.786 |

Retain the ~1.5 s forward stalls in all statistics. Their cause remains under
investigation; median gains do not establish a mean/tail improvement. Both
32 and 64 MiB fit each 1024-token full/SWA component in one batch. Capacity 1 MiB
regressed in transport screening. Capacity defaults remain unchanged.

OSL64 confirmation, 8 warmups +10 measured per arm: direct/packed32/packed64
TTFT medians 141.482/121.894/121.954 ms. All 450 sequences across both workloads,
including warmups and 54 sequences of 64 tokens, matched their direct references.
This is one fixed prompt and does not measure task accuracy.

A separate post-timing trace recorded two full gather kernels with grid
`[32,256,1]` and two SWA gathers with grid `[92,256,1]` on each prefill rank,
corresponding to both destination peers. Trace timings are excluded above.
