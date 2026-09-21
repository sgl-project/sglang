# Experimental LZ4 KV transfer and compressed HiCache L2

This opt-in prototype shares KV encoding between Mooncake P/D transfer and a
compressed Prefill HiCache Host pool. Attention continues to use its existing
KV tensors. HiCache owns prefix matching, backup, restoration, eviction and ACKs;
P/D owns transfer ranges and remote completion. The shared compression executor
owns representation conversion and bounded execution resources.

## Revision and validation boundary

The original r4 snapshot is commit `d88215a2e6`, based on
`8ae4a39b50ffcdb44175cd2fc5594032740ba888`. It passed a reported six-group,
180-request normal-path run with Qwen3-8B on two H20 GPUs. Ordinary LZ4 fell back
to raw data throughout that workload; forced LZ4 increased encoded payload by
about 0.348%, excluding descriptors. These results establish neither a speedup
nor a storage-capacity benefit. The full online fault matrix remains pending.

This branch integrates upstream main at
`1da8ac10e1` and requires a new image and new GPU/RDMA acceptance run.
Build from the complete branch checkout. The old r4 overlay, file hashes and
online results describe the old snapshot and must not be used to certify this
revision. Source delivery must identify both the base revision and all overlay hashes.

The LZ4 payload and chunk descriptor remain v2, with nvCOMP `5.3.0.16`.
Upstream added per-entry KV lengths to registration frame 19; compression now
uses frames 20 and 21. The advertised capability includes `registration-v2` and the `verify-0`/`verify-1` requirement so
older experimental peers are rejected during bootstrap. Upgrade both workers
together. With compression disabled, no compression fields are appended.

## Supported configuration

The initial scope is Qwen3-8B, BF16, FlashInfer, 1P1D, TP/PP/DP/CP/DCP=1,
page size 1 and prefill chunk size 1024. The reference settings use 512 MiB
compression workspace, 8 GB Prefill L2 and no Decode L2. Compressed L2 requires
write-through and the Python UnifiedRadixCache. Overlap scheduling, CUDA graphs,
HiSparse, speculative decoding, LoRA, sessions, external cache linkers and L3
are outside this prototype's supported scope. Runtime P/D role switching is
not supported with the required staging buffers.

A compression object contains one token's K/V across all layers. The 4 KiB
Host storage block size does not change KV pages or the attention layout.
Ordinary mode retains raw fallback; FORCE is a verification mode that permits
LZ4 expansion and requires VERIFY=1. This is not a reproduction of KVServe's
algorithms or controller.

## Data flow and ownership

```text
Prefill GPU KV ---- shared encoder ---- P/D wire buffer ---- RDMA ---- Decode restore
                          |                    ^
                          v                    | read lease; release after copy
                 HiCache fixed-block pool ----+
                          |
                          +---- restore + verify ---- HiCache admission ticket
```

The Host pool uses logical handles with generations and non-contiguous 4 KiB
block chains. `alloc(n)` atomically reserves handles and output-upper-bound
blocks before HiCache submits a backup. For example, a 197632-byte bound needs
49 blocks; an actual 148224-byte payload needs 37. `prepare_write()` freezes the
write mapping and returns the unused blocks for reuse. Allocation does not
require a contiguous extent or move existing objects.

```text
FREE -> RESERVED -> WRITING -> READY -> RETIRED -> FREE
```

HiCache selects eviction victims in logical-page units. `available_size()`
reports pages that can actually be reserved. Logical eviction removes the
reuse index, but blocks remain allocated until outstanding leases drain.
Uncertain drain failures retain resources and quarantine the affected path.

Each write window contains at most 64 objects. The executor copies encoded GPU
bytes into bounded pinned Host staging, confirms D2H completion, scatters them
into blocks and publishes the objects. HiCache acknowledges the complete node
only after all required objects succeed. A restore ticket becomes admissible
only after decompression, GPU writeback and required byte verification finish.

Two pinned Host staging channels separate writes from reads. Restore and P/D
share the read channel. `acquire()` takes a read lease without copying data;
`materialize_pages()` gathers at most 64 objects in an execution thread.
P/D copies them into an independent wire buffer. Host leases protect that copy;
the wire buffer remains live until RDMA finishes. The scheduler does not wait
on a Future, staging channel or GPU event. With no GPU batch, pending compressed
L2 work receives a 1 ms scheduling yield; ordinary upstream storage/P/D yields
remain unchanged.

## Budget and diagnostics

The 8 GB budget includes the data arena, fixed object/block metadata, hashes,
indexes, two pinned staging channels and bounded scratch. Python objects,
temporary indexing allocations, logging and other serving allocations are
outside this pool budget; it is not a process RSS limit. GPU workspace and
Decode receive staging are also separate budgets.

`KV_COMPRESSION_STATS.l2` exposes free/reserved/live/retired blocks, actual
encoded bytes, internal padding, staging use, readers, writers and quarantined
objects. Allocation, lock wait, maximum lock hold, gather/scatter and staging
wait are recorded separately. Timers overlap and must not be summed as request
latency. `SGLANG_KV_COMPRESSION_TRACE_STORE=1` adds node, handle, generation and
page-identity logs; shared cache operations do not invent request ownership.

## Local checks

Run the control-flow and storage suite from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SGLANG_COMPRESSION_STANDALONE_TEST=1 PYTHONPATH=python \
python -m pytest -q \
  test/registered/unit/mem_cache/test_compressed_hicache.py \
  test/registered/unit/mem_cache/test_compressed_block_pool.py \
  test/registered/unit/disaggregation/test_pd_compression_protocol.py \
  test/registered/unit/disaggregation/test_pd_compression_validation.py
```

The standalone switch bypasses package initialization on a CPU workstation.
Storage code runs directly; selected serving methods run with controlled
collaborators. These checks do not import or exercise the complete GPU server.
The registered tests also provide the executable entry points used by CPU CI.
The admission-timing fixture covers identity tracing both enabled and disabled.

For a capacity replay, supply the frozen workload and request order:

```bash
python test/manual/kv_transfer/replay_block_capacity.py \
  --workload workload.json --order request-order.jsonl \
  --node-pages 1024 8192 --output capacity.json
```

The replay uses the production pool geometry and a simplified split-aware LRU.
Default encoded lengths are synthetic, not captured KV. `--lengths` accepts
measured lengths keyed by prefix SHA256 and fails on missing entries. It does
not execute actual HiCache scheduling, CUDA or RDMA.

## Image and online validation

Build the complete updated checkout with the existing dependencies. Generate a
new source-hash manifest for that checkout. On each GPU worker, run:

```bash
python test/manual/kv_transfer/check_pd_compression_image.py --manifest /path/to/new-manifest.json
python -m pytest -q test/manual/kv_transfer/test_pd_compression_gpu.py --junitxml=/tmp/gpu.xml
python test/manual/kv_transfer/check_pd_compression_image.py --manifest /path/to/new-manifest.json --gpu-report /tmp/gpu.xml
```

Compare collected test node IDs with the new manifest; failures, missing cases
and skips do not count as GPU acceptance. The GPU tests cover nvCOMP, block
storage, actual-byte restoration and write-failure drain, not cross-node RDMA.

The launcher accepts `MODEL_PATH`, `IB_DEVICE`, `ROLE` and GPU binding settings.
Inspect its command with:

```bash
DRY_RUN=1 bash test/manual/kv_transfer/launch_pd_compression.sh
```

| Group | COMPRESSION_MODE | Prefill ENABLE_HICACHE | HICACHE_COMPRESSION | FORCE |
|---|---|---:|---|---:|
| native | off | 1 | off | 0 |
| passthrough | passthrough | 1 | passthrough | 0 |
| force-l2 | lz4 | 1 | lz4 | 1 |
| off | off | 0 | off | 0 |
| force | lz4 | 0 | off | 1 |
| lz4 | lz4 | 1 | lz4 | 0 |

Decode uses ENABLE_HICACHE=0 and HICACHE_COMPRESSION=off. Its P/D mode, FORCE
and VERIFY must match Prefill. Re-establish native/off baselines with the new
image. Run `native -> passthrough -> force-l2 -> off -> force -> lz4`, then fault
validation. Use fresh workers for each group, with no resets within a group.

```bash
python test/manual/kv_transfer/validate_pd_compression.py run \
  --workload workload.json --output results/force-l2 --phase force-l2 \
  --router "$ROUTER" --execution-contract execution-contract.json \
  --prefill-log-command "$PREFILL_LOG_COMMAND" \
  --decode-log-command "$DECODE_LOG_COMMAND"
python test/manual/kv_transfer/validate_pd_compression.py audit results/force-l2
python test/manual/kv_transfer/validate_pd_compression.py compare results/native results/force-l2
```

The frozen acceptance workload requires 38 requests per cached group and 14
per uncached group, complete same-path output-token equality, four concurrent
requests and three cycles restoring/verifying/adopting 8191 Host pages. Each
cycle must drain naturally within 180 seconds with three consecutive fresh
zero-work snapshots, without request-time health failures or Router 503s.
Missing/null cache-source details fail the Host-hit requirement.

Forced LZ4 must have per-object evidence of encoding, transfer, L2 publication,
old-object reuse and restore after GPU eviction. Save image/source hashes,
configuration, workload fingerprints, raw responses, diagnostics and metrics.
Stop on a failed gate. Old-image results cannot substitute for this run.

## Fault validation

Fault injection is off by default. A test must target one explicit operation,
for example:

```text
SGLANG_KV_COMPRESSION_TEST_FAULT={"operation":"write:4294967296","point":"after_scatter","kind":"error"}
```

Obtain the exact `write:<first handle>` or `restore:<first handle>` identity
from the current run. Supported points are after_scatter, before_publish,
before_restore and after_restore_copy; kinds are error, checksum and undrained.
A test is invalid unless its specified injection actually executes.

Cover cancellation and late completion, partial writes/publication failures,
bad descriptors or hashes, restoration failure, RDMA disconnect and peer exit.
Never publish failed targets. Safe failures must drain before reclamation and
allow a subsequent request; uncertain CUDA/RDMA drain must retain ownership
and quarantine the worker. Local injection is not a real RDMA failure test.
The complete online fault matrix and performance assessment remain required.

## Review repair: admission, ownership and evidence

A completed compressed restore already owns destination GPU pages. HiCache now
exposes only the matching request's ready, generation-checked reservation to
PrefillAdder. Its initial budget check subtracts those pages once; global free
capacity and other requests receive no credit. Admission still checks the new
tail and output reservation. Stale or cancelled tickets are reconciled without
publishing their target pages.

`decode_pages()` now returns a `DecodedPages` owner. Borrow `owner.raw` and
`owner.to_staging_order()` only while open, discard borrowed views, then call
`owner.close(consumer_stream)` after writeback and verification. The reservation
covers output, a possible layout copy, Host-input upload and declared consumer
scratch until GPU work drains. An impossible reservation fails before allocation;
external restore threads may wait for temporary pressure without holding the
executor lock. The sole executor thread must never wait for its own workspace.
Cancellation interrupts receive-side reservation waits. Uncertain drain retains
both allocations and charges. Backend-private nvCOMP allocations, registered
wire buffers, receive rings and model KV are separate; this is not a total CUDA
memory cap. Record process/device peaks separately in image testing.

Compression quarantine now blocks incoming requests, Prefill admission and
Decode preallocation/prebuilt admission. It retains uncertain resources and sends an explicit unhealthy signal to the
tokenizer. `/ready`, `/health` and `/health_generate` remain unavailable until
worker replacement; successful old responses do not clear quarantine. GPU byte
corruption that drains safely remains a request failure and can be followed by
a healthy request. These are different fault classes.

Before `run`, create `execution-contract.json` from the actual deployment. It
must contain source_manifest_sha256, image_digest, model, model_revision, dtype,
kv_cache_dtype, topology, page_size and chunk_tokens. Use the same contract for
all six groups of the same image; group switches are captured by phase and
startup diagnostics. Retain the actual expanded launcher configuration beside
it. Do not use the original r4 digest. A template is provided in
`execution_contract.example.json`; replace every placeholder before running.
Comparisons reject self-comparison, incompatible phase pairs and contract or
sampling differences. Historical evidence without a contract may be audited,
but does not certify a new deployment.

The frozen 8192-token L2 replay now requires exactly 8191 Host-adopted and verified
pages and device=0, host=8191, storage=0 in the response. Missing/null source
details, partial restore, duplicate restore events, changed output tokens or
missing requests fail. Natural drain requires three independently observed
snapshots within 180 seconds; a single record declaring consecutive=3 is invalid.
New runs retain snapshot identities and reject stale duplicates.

The GPU test inventory includes decoded-owner lifetime at 1/64/65/1024 full
Qwen3-8B-shaped pages, too-small budgets, and actual payload/target corruption.
Use the manifest's collected node IDs, never an old hard-coded case count.
Local CPU success and GPU collection/skips are not GPU or RDMA acceptance.
