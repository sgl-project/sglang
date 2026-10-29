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
statuses; ambiguous submissions quarantine the source registration and remote
release hold until transport recovery. This PR has no producer fusion or
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
model serving and cross-node RDMA must be revalidated on the rebased source.
