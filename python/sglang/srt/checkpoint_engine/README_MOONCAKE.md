# Loading SGLang weights from Mooncake Store (first load)

Load model weights into a running SGLang server from a
[Mooncake](https://github.com/kvcache-ai/Mooncake) distributed DRAM store,
instead of reading safetensors from disk at startup.

This historical delivery branch is based on SGLang commit
`9b0c470a6a13bedc860c6836f5d07b6c2f62e61a` (`JD-v0.5.10.rc0`). It is not a
port to the current `main` branch. Runtime code is the original standalone
first-load patch; this guide includes subsequent operational clarifications.

Why: the store holds one copy of the weights in (remote) host DRAM, shared by
any number of server instances and nodes, and restore speed does not depend
on the local pagecache. On the originating fork (8xH200 / TP8 / 554 GiB FP8
checkpoint): ~34 s injection, identical before and after `drop_caches`, vs
44-134 s for disk reads depending on cache state. The acceptance run of this
patch (8xH20, single store instance, same checkpoint) measured 80.1 s
end-to-end injection at 6.92 GiB/s.

**Scope: first load (cold start) only.** This directory does not implement
foreground/background model switching or any lifecycle management; a server
loads once at startup. Multi-node loaders (`--parallel-size` < world size)
are inherited from upstream but not exercised by this flow.

## How it works

```
                    one-time upload                every cold start
/models/<ckpt>  --dump_to_mooncake-->  mooncake   --mooncake_loader-->  SGLang
(safetensors)     ~2GiB buckets         store       fetch (RDMA)         (started with
                  + JSON manifest     (host DRAM)   -> NCCL bcast         --load-format dummy)
                                                    -> ZMQ/CUDA-IPC apply
```

1. `dump_to_mooncake.py` packs tensors into buckets (target size
   `max(256 MiB, largest tensor)` — ≈2.2 GiB for Kimi-K2.5, whose largest
   tensor dominates) plus a JSON manifest under keys
   `ckpt:<name>:{manifest,bucket:*}`. Multi-shard checkpoints are verified
   against `model.safetensors.index.json` so a missing shard cannot be
   uploaded as a "complete" checkpoint.
2. SGLang starts with `--load-format dummy --checkpoint-engine-wait-weights-before-ready`:
   the process boots without touching the checkpoint; `/ping` answers
  immediately. The weight-wait gate is time-limited: after its timeout the
  baseline can become ready without valid weights (see Step 3).
3. `mooncake_loader.py` (one process per GPU, via torchrun) fetches each
   bucket from the store (owner rank only), NCCL-broadcasts it to all ranks,
   and hands tensor metadata to the server's `/update_weights_from_ipc`
   workers, which copy views straight out of a shared CUDA-IPC GPU buffer.
   Data never moves through host memory or the HTTP layer.
4. When the update succeeds the server flips ready and `/health` returns 200.

Detected failures fail closed: missing buckets abort before the server is
involved, manifests are validated against the packing rule before any GPU
work, a per-bucket load error reported by the worker fails the whole load, a
mid-load error fails the HTTP update (and thus readiness) on both sides, and
a failed quantization post-hook fails the load instead of serving corrupted
weights. The known limits: bucket existence checks are not content
checksums, and an outright killed process (loader SIGKILL, node loss) can
leave the server stuck in the update and needing a restart — see Pitfalls.

## Requirements

- From this branch's repository root, install
  `pip install -e './python[checkpoint-engine]'` and
  `pip install mooncake-transfer-engine==0.3.9` (tested store client).
  Do not replace the patched source with a separately installed public
  SGLang release. Verify `pip show checkpoint-engine` reports 0.4.0. The loader
  requires checkpoint-engine 0.4.0's two-stage finalize; this patch updates
  the extra's pin accordingly (the previous `checkpoint-engine==0.1.2` pin
  speaks a different, incompatible protocol).
- The dependency pin also affects other checkpoint-engine callers in the
  environment, including existing IPC/RL weight updates. Regress those paths
  before integrating this historical branch into a shared deployment.
- RDMA NIC reachable from all parties (or `--protocol tcp` for functional
  tests).
- GPU headroom on each rank for the staging buffer:
  `max_bucket_size x --pipeline-depth` (~2 GiB x 4 by default), freed when
  the loader exits.
- Host DRAM on store nodes: the segment size is locked at store startup
  (aggregate across store instances) and must fit every checkpoint you keep
  in it. Size it deliberately — an oversized segment can OOM the node.

## Step 0 — start master + store service (once per cluster)

Both ship inside the `mooncake` pip package. On the master node:

```bash
mooncake_master --port=50051 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8081 \
    --default_kv_lease_ttl=300000 \
    --eviction_high_watermark_ratio=0.95
```

On each store node (contributes `MOONCAKE_GLOBAL_SEGMENT_SIZE` of its DRAM):

```bash
export MOONCAKE_LOCAL_HOSTNAME=<this-node-ip>
export MOONCAKE_MASTER=<master-ip>:50051
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_GLOBAL_SEGMENT_SIZE=32gb        # size for your checkpoints
export MOONCAKE_LOCAL_BUFFER_SIZE=0
export MOONCAKE_PROTOCOL=rdma                   # or tcp
export MOONCAKE_DEVICE=<rdma-nic>               # e.g. mlx5_bond_0
python3 -m mooncake.mooncake_store_service
```

Watch the store log for `Using specified RDMA devices` and a successful GID
selection — that is the difference between real RDMA and a silent TCP
fallback. Then wait for `Store service started successfully`: large pinned
segments register tens of seconds after the mount line, and puts fail with
rc=-200 until then (the dump tool also probes for this itself).

## Step 1 — upload a checkpoint (once per checkpoint)

```bash
python3 -m sglang.srt.checkpoint_engine.dump_to_mooncake \
    --checkpoint-path /models/Qwen3-0.6B --checkpoint-name Qwen3-0.6B \
    --master-addr <master-ip>:50051 --metadata-server P2PHANDSHAKE \
    --protocol rdma --rdma-devices <rdma-nic>
```

`--checkpoint-name` is the contract: dump, health-check and loader must all
use the same name. Re-uploading a name requires removing it first
(`remove_from_mooncake.py`).

## Step 2 — verify (before every use)

Store contents are **volatile**: a store service restart loses everything.
Check before relying on it:

```bash
python3 -m sglang.srt.checkpoint_engine.mooncake_store_health_check \
    --checkpoint-name Qwen3-0.6B \
    --master-addr <master-ip>:50051 --metadata-server P2PHANDSHAKE \
    --protocol rdma --rdma-devices <rdma-nic>
# exit 0 = complete; 1 = not uploaded; 2 = incomplete; treat ANY non-zero
# (including a 139 segfault when the store is unreachable) as unhealthy.
```

## Step 3 — start SGLang with dummy weights

```bash
python3 -m sglang.launch_server \
    --model-path /models/Qwen3-0.6B --tp 1 --port 30000 \
    --load-format dummy --checkpoint-engine-wait-weights-before-ready
```

- `--model-path` still points at the checkpoint directory: config and
  tokenizer are read from it; tensor data is not.
- Do **not** set `model_loader_extra_config` (e.g. multithreaded disk read
  options) together with `--load-format dummy` — the dummy loader rejects it
  at startup.
- `/ping` answers 200 as soon as HTTP is up and does not prove model readiness.
  `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` (seconds, default 120) bounds how long
  the server waits for weights. Set it in the server environment well above
  the worst-case total time until injection completes, including any delay
  before starting the loader. **The baseline logs an error on timeout but
  can still become ready with random weights; this patch does not fix that
  path.** Do not admit traffic until the loader exits 0, `/health` is 200,
  and a minimal inference has passed. Increasing the timeout does not make
  the timeout path fail closed. The first `/health` probe
  right after injection may still return 503 while the initial generation
  warms up; it settles within seconds.
- If the weight wait times out or the load outcome is uncertain, keep the
  target out of routing and restart it before retrying. A later `/health`
  200 does not turn that attempt into a successful load.
- Caveat: with `--tokenizer-worker-num > 1` only the worker that handles the
  update request flips ready; keep it at 1 (default) for this flow.
- **Do not rely on `/health` alone.** The OpenAI-compatible endpoints do not
  refuse requests before weights arrive — a request sent to a dummy-weight
  server returns garbage tokens. Route traffic only through something that
  enforces the combined readiness checks above, never straight at a booting
  server.

## Step 4 — inject the weights

One loader process per GPU, `--parallel-size` = TP size:

```bash
torchrun --nproc-per-node 1 \
    -m sglang.srt.checkpoint_engine.mooncake_loader \
    --checkpoint-name Qwen3-0.6B \
    --endpoint http://localhost:30000 --parallel-size 1 \
    --master-addr <master-ip>:50051 --metadata-server P2PHANDSHAKE \
    --protocol rdma --rdma-devices <rdma-nic> --flush-cache --pipeline-depth 2
```

If your torch install has no `torchrun` entry point, use the equivalent
`python3 -m torch.distributed.run`.

The loader validates the manifest layout and bucket existence (not content
checksums), waits for `/ping`, POSTs
`/update_weights_from_ipc`, streams all buckets through the
fetch/bcast/apply pipeline, then finalizes (the second finalize round waits
up to `SGLANG_IPC_POST_HOOK_TIMEOUT_MS`, default 120 s, for the server-side
quantization post-hook). Require loader exit 0, `/health` 200, and a successful
minimal inference. For correctness acceptance, compare deterministic output
against the same checkpoint loaded from disk; HTTP success alone does not
establish correct weights.

Use a fixed prompt and compare its deterministic token IDs or logprobs to a
recorded disk-load baseline. The short chat request below is only a smoke
check and does not replace that correctness check.

```bash
curl -sf http://localhost:30000/health
curl -sf http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}'
```

## Tuning and environment knobs

| Knob | Default | Meaning |
|---|---|---|
| `--pipeline-depth` | 4 | staging buffer slots per rank (GPU mem = max_bucket x depth); explicitly use 2 for this first-load recipe |
| `--update-timeout` / `SGLANG_IPC_UPDATE_TIMEOUT` | 1800 s | separate bound applied to the `/ping` wait, the initial IPC handshake, and the HTTP update request; not one global loader deadline |
| `SGLANG_IPC_POST_HOOK_TIMEOUT_MS` | 120000 | wait for the server-side post-hook ack (quant repack; ~3 s on 554 GiB FP8) |
| `SGLANG_IPC_APPLY_ACK_TIMEOUT_MS` | 300000 | per-bucket apply ack bound; converts a dead worker into a failed load |
| `SGLANG_CE_IPC_LINGER_MS` | 5000 | server-side ZMQ teardown bound (never set to unbounded) |
| `MOONCAKE_MASTER` / `MOONCAKE_DEVICE` / `MOONCAKE_PROTOCOL` ... | — | env fallbacks for the corresponding CLI flags |

## Pitfalls (all learned the hard way)

- **Always pass `--metadata-server P2PHANDSHAKE` explicitly** (or leave the
  harmonized defaults alone). Historic tool defaults disagreed and produced
  `setup rc=-200` style failures that look like network problems.
- **Store data is volatile** — health-check before every use; re-dump after
  a store restart (554 GiB re-uploads in ~2-3 min per 100 GiB on RDMA).
- **Restart master and store together.** A store service restarted under a
  running master leaves stale segment metadata behind; the next upload fails
  with `put_from ... code -200`. Bounce both, then re-dump.
- **One dump at a time.** Keep uploads serialized; concurrent dumps against
  one store are unvalidated territory. (Historic -200 failures blamed on
  concurrency were actually the store warm-up window above.)
- **The client buffer must hold the manifest in one read.** Large-model
  manifests (per-tensor metadata for hundreds of buckets) run to tens of MB;
  an undersized `--buffer-size` surfaces as "manifest not found" even though
  the health-check passes. The loader defaults to 256 MB, which covers
  ~TB-scale checkpoints; don't lower it without measuring your manifest.
- **The health-check can segfault** (exit 139) when the master is down;
  script wrappers must treat any non-zero exit as unhealthy.
- **Not every loader failure can abort the server side.** Store/fetch
  errors abort cleanly (the server gets the failure and the update returns
  400). But if the loader was killed outright (SIGKILL, node loss), or it
  timed out waiting for a server ack (the server side is stuck rather than
  dead), the server may be left waiting inside the update — restart the
  server process in that case.
- **A rank dying mid-NCCL-broadcast** — or an owner rank failing its store
  fetch — surfaces on the other ranks only after the process-group timeout
  (300 s): they are already blocked in the collective for that bucket. The
  load still fails closed, just slowly; be patient before diagnosing a hang.
- Dumping very large checkpoints right after `drop_caches` has occasionally
  stalled on cold reads; if a dump wedges, kill and rerun it.

## Files

| File | Role |
|---|---|
| `mooncake_loader.py` | one-shot injector (torchrun, one process per GPU) |
| `dump_to_mooncake.py` | safetensors -> store uploader |
| `mooncake_store_health_check.py` | manifest + bucket existence probe, not content checksums |
| `remove_from_mooncake.py` | delete a checkpoint / prefix from the store |
| `checkpoint_engine_worker.py` | server-side IPC worker (part of sglang) |

Provenance: loader and tools evolved from the
[kvcache-ai/checkpoint_engine](https://github.com/kvcache-ai/checkpoint_engine)
examples in the JD-AI-Infra fast-startup project, hardened by its production
debugging (two-stage finalize handling, ZMQ teardown deadlock defenses,
fail-closed error paths).
