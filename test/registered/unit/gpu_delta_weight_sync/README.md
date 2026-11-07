# GPU delta receiver

The experimental receiver applies canonical XOR deltas directly to existing
execution-layout buffers. Small complete-value targets use overwrite semantics.
It does not requantize weights, change checkpoint loading, replace the ordinary
allocator, or alter the independent disk update API.

See [GPU-accelerated Weight Teleportation](https://humansand.ai/blog/gpu-accelerated-weight-teleportation)
for the end-to-end design and performance motivation.

Implementation ownership is explicit: `layout.py` admits immutable canonical
bindings and consumer identities; `receiver.py` prepares publications and runs
the fixed decode/apply pipeline. `host.py` owns publication control and rank-local
CUDA mappings; `host_worker.py` builds the shared encoded, prepared and metadata
images in one persistent CPU-only process per engine-host. `tokenizer.py`
coordinates the public update and
correlates internal replies; `session.py` owns scheduler integration and each
rank's state. HTTP routes and IPC schemas stay in `http.py` and `io.py`.

## Admission and supported layouts

Canonical names, shapes and dtypes come from the immutable local safetensors
checkpoint headers at first admission. The standard loader's selected target and
bundled-MTP files are supported; custom, secondary, encrypted or transformed
sources are not. Packed live buffers cannot reconstruct this source inventory.
Names and shapes do not prove that the live engine has the correct base bytes.

`models.py` maps the DeepSeek MLA/DSA family, including GLM's shared runtime
implementation: canonical name mutation, unequal Q/KV-A fusion, indexer fusion
and BF16-to-FP32 norm replacement, MLA consumer views and static draft exclusions.
Bundled MTP layer weights stay static; its shared embedding/head buffers follow
the target. A new model family supplies startup bindings and derived views through
the same mapping interface. GDN/KDA and other model mappings are not implemented.

`bindings.py` owns dense TP/vocabulary slicing and consumer identity snapshots.
`flashinfer.py` factors native expert layouts into affine byte views. Segments
exclude padding and preserve the canonical byte order; there is no per-update
weight or mask permutation. Aliased images receive one XOR; independent consumer
images each receive it. The admitted FlashInfer matrix is:

| Native weight format | TRTLLM / routed | CuTe DSL | MegaMoE |
| --- | --- | --- | --- |
| BF16 | Yes | No native path admitted | No native path admitted |
| FP8, 128×128 blocks, UE8M0 scales | Yes | No | No |
| MXFP8 | Yes | No | Yes |
| MXFP4 with MXFP8 activation | Yes | No | Yes |
| NVFP4 | W4A4 | W4A4 or W4A16 | W4A4 |

These are byte-layout contracts for existing inference paths, not new inference
kernels. Load-time weight quantization and conversion fallbacks are rejected.
Block128 FP8 accepts exact UE8M0 scale values in compact U8 wire storage and expands
them losslessly into the consumer's FP32 API image; arbitrary FP32 scale factors
are unsupported. NVFP4 derived calibration reuses the ordinary inference helpers.
MegaMoE rejects changes to its captured input-normalization scalar and incompatible
gate/up alphas before any live writes. CuTe wrappers and derived buffers must
already be initialized. Layout/native tests cover the matrix; full serving
qualification remains NVFP4 W4A16.

Attention, routed-expert and shared-expert TP follow their existing loader
partition groups, including padded expert slices. Each rank decodes complete
canonical tensors and applies only its admitted byte views. Attention DCP is
supported within an attention-TP group: its width must divide attention TP.
DeepSeek/GLM MLA's unquantized replicated Q/K consumers receive the same XOR mask
through prepared affine views; local K/V caches refresh from the local KV-B shard.
This adds no apply-time all-gather or kernel specialization. Quantized replicated
Q projections remain unsupported.

Prefill CP reuses those module-owned partitions: CP partners may repeat attention
views while routed and shared experts retain their own TP groups. The optional
`enable_cp_decode_attn_tp` optimization is rejected because it caches independent
weight slices and temporarily swaps live parameter storage. Prefill CP has unit
coverage; a dedicated serving end-to-end test is not included.

Fixed topology is required when routed experts are bound: no presharded canonical
checkpoint, EPLB, redundant experts or elastic joiners. The selected FlashInfer
runner must match its dispatcher. Dense-only mappings do not need a MoE
configuration. Ordinary globally ordered control broadcast is required;
local broadcast, LoRA, disaggregated serving or pipeline widths above one,
shared IPC weight storage and HPC-Ops derived-weight caches are rejected at
admission. Control replies include every original TP/CP/DP participant.

## Ownership and lifecycle

Miles exclusively owns model updates, pause/resume, memory residency and topology
for a stream's lifetime. There is no server-wide ownership lock, XOR retry,
rollback or automatic replay of an uncertain update.

Routes, IPC schemas, tokenizer coordination and scheduler handlers live under
`srt/weight_sync/gpu_delta/`; shared files only register them. The four HTTP POST
endpoints are an internal interface to Miles:

| Endpoint | Purpose |
| --- | --- |
| `get_gpu_delta_info` | Admit the receiver and return its canonical plan, original participants and version. |
| `update_weights_from_gpu_delta` | Prepare, apply and resume one publication, returning compact APPLIED acknowledgements, full RESUMED receipts and server timings. |
| `get_gpu_delta_status` | Inspect the identified session without repeating its update. |
| `clear_gpu_delta_state` | Release retained delta resources while preserving weights and committed version. |

The update request carries the negotiated participants, caller-generated session
ID and publication descriptor: manifest path/SHA256, stream, base/target versions
and plan digest, plus `validate_payload_hash=true` by default. Rolling updates and
HF-base recovery use the same request shape;
the tokenizer does not reread the manifest or repeat admission. Each engine's
tokenizer owns the following internal phases:

1. **Prepare:** the CPU process authenticates the manifest and builds shared
   payloads and numeric metadata. Rank-local background work qualifies the canonical
   plan and prepares small GPU metadata, raw targets and derived calibration.
   It takes no model-update writer
   lock and neither writes model/KV storage nor pauses generation.
2. **Apply:** close generation admission, pause scheduling, fence readers, retract
   requests and optionally flush caches. Allocate two decoded-mask slots, bind
   pointers, decode and apply. `APPLIED` requires GPU completion and successful
   native size/status checks. A failed fence cannot reclaim KV/cache.
3. **Resume:** after every original rank of that engine applies, the tokenizer
   sends the internal resume message. SGLang records the model version and reopens
   generation. Miles advances its baseline only after all engines finish.
   Independent engines need not pause together.

Preparation failure aborts that engine's preparation while old-version serving
continues. Failed or aborted delta sessions require restart before another update.
Internal apply/resume failures leave admission closed. A lost HTTP
response may instead mean the server has already completed and resumed; Miles
does not abort or replay the update on transport failure. The session ID permits
inspection of that outcome. There is no separate global quiesce/commit round.

Updates accept `flush_cache=true` and `abort_all_requests=false`. Miles may disable
flushing only after invalidating KV and multimodal caches. Reader fencing and retraction still run;
optional abort occurs after new admission closes. Updates preserve the pause
present at the apply boundary. An internally retained pause commits the version
and releases per-update resources without reopening generation; its
`RESUMED` receipt has `generation_paused=true` and null `resumed_ns`/`blocked_s`.
The optional `weight_version` labels admitted generations independently of the
transport's target sequence; omission uses the transport target version.

`clear_gpu_delta_state` is valid when idle or successfully resumed. It drains each
rank's FIFO release work, closes workers/host allocations and drops
planning, decoder and stream objects. After every rank acknowledges, one process
per engine-host deletes the shared encoded and prepared caches. Weights,
publications, committed
version, process identity, small provenance and pause state remain unchanged.
Describe can recreate the backend at that version. Framework allocator/JIT caches
are not cleared.

HF-base recovery requires a fresh version-0 engine using the matching immutable
checkpoint. Miles negotiates its new participants and submits the base-relative
publication through the same update endpoint. The publication path must be visible
to all ranks. Delta resources are retained by default (`release_state=false`);
`release_state=true` clears them after successful activation while retaining the
APPLIED/RESUMED receipts in the response. Explicit `clear_gpu_delta_state` remains
available independently. Cleanup uses the same all-rank close/cache-release barriers
and is excluded from the returned activation phase timings.

## Payload and cache contracts

Publications select `snappy-zstd`, `lz4-zstd`, `lz4`, `bitcomp-zstd` or `bitcomp`,
with a positive integer `frame_bytes` up to 4 MiB (default 1 MiB). Miles defaults
both initial synchronization and subsequent updates to `bitcomp-zstd`. SGLang
selects the codec from each publication's manifest; it has no codec environment
variable or independent default. Bitcomp uses sparse, lossless U8 encoding; it
never requantizes weights. GPU-delta does not require Blackwell: the decoder selects
DE for Snappy/LZ4 on SM100/SM103 and CUDA kernels otherwise. Each inference backend
retains its own hardware and weight-format requirements. DE frames must fit the
device's compressed and decoded limits. All paths validate geometry, alignment and
capacity. Streams can change codec without changing their canonical plan; native
decoders are cached by inner codec.

Wrapped matrix frames use independent GPU-compressed outer Zstd chunks with at
most 1 MiB output. CPU workers unwrap them into shared host storage. Plain
LZ4/Bitcomp has an empty outer `frames` list and equal encoded/decoded extents;
workers copy the inner compressed bytes. DE reads its host arena directly;
CUDA decoders receive batch-sized H2D copies through two GPU slots.
There is no full-publication HBM copy.
Unchanged frames are omitted. Scalars/vectors use `raw_bytes`: complete target
values, no XOR/compression, and no payload when unchanged.

One engine-host creator reads and optionally SHA256-checks owner files in parallel
into retained tmpfs slices. Source identity/extent is checked across each read;
all tasks drain before READY or failure. Namespace/publication records bind the
original cohort, stream, path, digest, session, versions and hash policy. Mapped
encoded bytes remain owned until every submitted worker drains.

For every codec, one fixed original rank per engine-host owns a persistent CPU
child that parses the manifest and prepares the complete canonical image. The
child owns its reusable payload pool, publication lock, current index and buffer
growth. It uses a separate Python interpreter/GIL, imports no Torch and performs
no CUDA work. Small IPC records describe the shared files; payloads and numeric
metadata do not cross the IPC channel. The child publishes READY after building
the image and accepts REUSABLE only for the exact released publication.
Cancellation terminates and reaps the child, ending its write lease. A late abort
may retain a complete READY image but never makes it reusable. Clear drains and
closes the child; parent death also terminates it. Wrapped
codecs use one native Zstd `BufferWithSegments` per bounded worker group over the
encoded mapping, without per-frame input views or payload copies. Segment spans
remain bounded by their logical files. `multi_decompress_to_buffer(threads=0)`
checks declared decoded sizes, then workers scatter temporary ordinary CPU output
into the persistent image. Native decoding creates no extra workers.
Every rank maps that same prepared tmpfs inode.
CUDA decoders register their mappings for asynchronous H2D; rank-local plans
select the required tensors without another host copy.
Followers create no payload pool; abort cancels their owner wait. Global manifest
and encoded mappings are dropped after host preparation.

DE ranks own private CUDA 13 `HOST_NUMA`/`PINNED` allocations with
`CU_MEM_CREATE_USAGE_HW_DECOMPRESS`; these are never exported/imported. Each DE rank
maps the shared prepared image without registration and copies only its required
tensor spans into its private arena, without another file read, hash or
outer decode. All arenas and caches fit the initial extent, rounded to allocation
granularity. Growth reserves twice the required extent; fitting
updates reuse storage and CUDA registrations. Shared prepared capacity is counted
once per engine-host, while DE rank-private capacity is counted per rank.

Successful all-rank resume authorizes encoded and prepared-cache reuse. The session
queues release and view cleanup ahead of its next prepare on the same FIFO executor.
BUILDING precedes overwrite; failed/aborted publications remain nonreusable, and
late old releases cannot release newer bytes. Explicit clear removes retained
encoded and prepared files only after the rank-close barrier.

## Fixed decode/apply pipeline

Compressed tensors are grouped by active model layer, with separate embedding,
head and remaining standalone groups first. Empty groups are omitted. Each batch
must fit HBM; there is no intra-layer streaming. Active membership, normalized
views and canonical offsets are cached, while frame lengths, arena offsets and
omitted ranges are rebuilt from each publication.

Preparation builds one numeric frame table, validates workspace/output geometry,
uploads small metadata/raw targets and prepares derived values. It loads the three
fixed Triton families and caches occupancy per module/device, then prebinds launch
closures. No model geometry enters compilation-cache keys. The families cover
packed contiguous/two-axis word access, contiguous-source word-affine targets and
general byte/affine fallback. The policy uses a full word-affine tile grid and at
most four resident waves for packed/general; custom grids are test-only controls.
There is no online autotuning or pause-time scratch fitting.

After the reader fence, exactly two largest-batch decoded HBM slots are allocated
when compressed work exists. One DE stream shares temporary workspace; each slot
has its own status/size row. One apply stream owns the sticky error gate. DE slot
`i % 2` is reused only after its prior apply completes. DE0 starts first, overlapping
small raw-target copies. Apply i is queued before host submission of DE i+1 because
nvCOMP's Async call can wait for prior work on its calling stream.

The CUDA path uses one compute stream for decode, status checks and Triton apply.
A separate copy stream prefetches compressed data into two batch-sized slots.
The next H2D is queued before the current decode; a slot may be overwritten only
after its previous decode completes. Contiguous host spans, including alignment
padding, are coalesced during prepare. Bitcomp's batched API can synchronize its
calling stream to inspect headers; the prequeued copy can still overlap GPU work.
Large compressed and decoded slots are allocated only after the reader fence.

Only omitted frame gaps/tails are zeroed. A fully covered batch skips zeroing; one
status kernel checks native sizes/statuses before each grouped XOR. Aligned source
binding patches the prepared pointer slab and uploads it once. Unexpected source
misalignment rebuilds conservative runtime descriptors; later aligned binding
restores the template. Derived consumer copies are gated by the sticky error.
Completion joins transfer, decode and apply before scratch release; failed cleanup
attempts to drain both streams. Foreign tensor aliases retain allocations and registered
producer/consumer streams are fenced before free in their original CUDA context.
The ordinary Torch allocator and its settings remain unchanged.

Raw targets are packed once into an aligned pinned arena and uploaded during
prepare, including batched lossless UE8M0 expansion and BF16 norm conversion where
admitted. Dtype-grouped copies update existing buffers during pause. The canonical
plan digest is qualified once, but fresh frames and native decode status are
validated every update. Before apply, identity checks cover every live parameter
slot and consumer object, with tensor geometry deduplicated at admission. Pointer
identity cannot detect unrelated same-buffer writes by another updater.

## Configuration and observations

Feature environment variables are development controls, not a stable public API.
Keep them fixed for the stream's lifetime. Batch size is captured at backend
creation, worker count when its pool is first needed, and sorting by each decoder.

| Setting | Default | Meaning |
| --- | --- | --- |
| `GPU_DELTA_LAYERS_PER_BATCH` | `1` | Positive count of active adjacent model layers per batch |
| `GPU_DELTA_CPU_WORKERS` | `32` | Reusable workers on one owner per engine-host, for every codec; range 1–64 |
| `GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS` | `0` | `0`/`1`, native sorting on the paused DE call |
| `GPU_DELTA_HOST_CACHE_DIR` | `/dev/shm/sglang-gpu-delta-<uid>` | Private user-owned tmpfs visible to the engine's host-local cohort |
| `SGLANG_ENABLE_NVTX_OPERATIONS` | `0` | Standard SGLang detailed operation annotations, including GPU delta |

There is no decode-depth or legacy timing/NVTX knob. Wrapped workers have independent
Zstd contexts and at most four tasks per worker. Two host-local EP4 engines use two
preparation pools on either decoder path. Engine IDs and
`host_cache_id`, not hostname alone, define cache
sharing. Independent engines share no cache locks or release lifecycle.

Profiling the CPU child requires the optional NVIDIA `nvtx` Python package:
`python -m pip install --no-deps nvtx==0.2.17`. It is imported only when
`SGLANG_ENABLE_NVTX_OPERATIONS=1`; ordinary execution does not require it.

Payload verification is selected per update by `validate_payload_hash=True` on
`update_weights_from_gpu_delta`. Miles passes its configured policy explicitly;
SGLang does not read a hash-policy environment variable. Every rank sharing a
publication must agree. After successful all-rank apply/resume releases the cache,
the next publication may choose another policy; unchecked bytes cannot satisfy a
checked publication. A receiver requiring hashes rejects
`payload_checksum_format="none"`; a skipping receiver may accept hashed
publications. Skipping trusts contents without payload SHA authentication.
Manifest SHA, exact file extent/identity, path containment, Zstd output length and
native decode checks still apply.

CPU timings, bytes and counters remain available for W&B independently of NVTX.
The paused APPLIED acknowledgement retains identity, publication provenance, state,
scheduler timing and `result.applied`/`result.target_version`. Full metrics stay in
the session and are returned by RESUMED and status, avoiding their serialization
before the all-rank apply barrier. Receipts contain no diagnostic CUDA timing
events. Host spans are nested and worker sums overlap; do not add them or infer
device bandwidth from host durations.

| Receipt fields | Interpretation |
| --- | --- |
| `host_prepare_s` | Full background preparation |
| `host_manifest_read_parse_s`, `host_plan_validate_s`, `host_tensor_prepare_s` | Manifest, canonical-plan qualification, local tensor planning |
| `host_plan_cache_reused`, `host_batch_plan_reused` | Admitted digest / current active-geometry cache reused; no old frames retained |
| `host_rank_prepare_s` | Encoded admission, global metadata release, local allocation/fill and final encoded-view release |
| `host_rank_metadata_release_s` | CPU child drops the parsed manifest and encoded view after shared preparation |
| `host_cpu_worker_spawn_s`, `host_cpu_worker_wait_s` | Owner's process-launch call / complete preparation IPC wait; wait includes child work and is not additive with its phases |
| `host_cpu_worker_pid`, `host_cpu_worker_start_ticks` | Shared child identity, repeated on the engine's ranks; followers do not own another child |
| `host_rank_prepare_body_s`, `host_rank_prepare_call_s` | Through local snapshot return / local fill and job cleanup; caller-minus-body includes view release/bookkeeping |
| `host_rank_layout_s`, `host_rank_allocation_s` | Local ordering/offset planning / arena growth |
| `host_raw_pack_s`, `raw_bytes`, `raw_h2d_bytes` | CPU raw packing, direct bytes, aligned arena upload bytes |
| `host_metadata_prepare_s`, `host_metadata_wait_s` | Small GPU input/status setup including its own stream waits / those waits alone |
| `host_apply_kernel_prepare_s`, `apply_kernel_cache_misses`, `apply_kernel_cache_reused` | Record construction and fixed-module cache lookup; misses are Python cache misses, not necessarily compilation |
| `paused_setup_host_s`, `paused_apply_host_wall_s` | Slot allocation/binding/uploads / complete apply through completion and scratch release |
| `h2d_bytes`, `apply_metadata_h2d_bytes`, `apply_descriptor_h2d_bytes`, `decoder_metadata_h2d_bytes` | Explicit raw, metadata and CUDA compressed-input uploads; descriptor bytes are a subset of apply metadata bytes |
| `compressed_input_buffers`, `compressed_input_capacity_bytes`, `compressed_input_h2d_bytes` | CUDA input-slot count, total input-slot capacity and copied bytes including alignment padding; zero on DE |
| `de_host_input_bytes` | Compressed bytes read directly by DE, not an explicit H2D transfer |
| `decode_stages`, `decoded_buffers`, `decoded_scratch_bytes`, `decoder_workspace_bytes` | Fixed 2; actual slots (0 for raw-only); total slot bytes; temporary workspace, not peak HBM |
| `decoded_zero_ranges`, `decoded_zero_bytes` | Omitted canonical ranges cleared before decode |
| Batch/tensor/group/contract/grid counters | Actual runtime work; kernel cache reuse is a per-rank boolean |

CPU slab construction and launch prebinding are outside
`host_apply_kernel_prepare_s`. Shared-cache fields have distinct scopes:

- `host_encoded_cache_created`/`reused` identify creator and followers;
  `host_encoded_cache_wait_s` is the mutex wait and `access_s` includes
  wait/build/attachment through lock release. `build_s` repeats the cached creator
  duration on followers and must not be summed across ranks.
- Creator-only `host_encoded_cache_read_hash_s` encloses submission and joins;
  `read_worker_sum_s` and `sha256_worker_sum_s` sum overlapping tasks. `hash_files`
  and `hash_bytes` count work once. `skip_payload_hash` is reported on every rank;
  skipped SHA time/counts are zero.
- `host_rank_outer_zstd_decode_s` encloses task submission/join, including raw
  copies; `host_rank_outer_zstd_worker_decode_sum_s` sums each worker's native
  batch decode and scatter into the shared image.
  `host_rank_outer_zstd_{encoded_bytes,decoded_bytes,tensors,frames}` counts
  full-image wrapped work only on the owner and zero on followers.
  Plain LZ4/Bitcomp uses `host_rank_encoded_copy_s` and
  `host_rank_encoded_copy_{worker_sum_s,bytes,tensors}` instead; copy counters
  exclude raw targets. Inactive codec counters are zero.
- `host_rank_{arena_bytes,capacity_bytes,capacity_generation,mapping_reused}` and
  `host_rank_allocation_{s,calls,bytes}` describe local mappings and required bytes.
  CUDA ranks share physical capacity; its allocation calls count registrations.
  `host_rank_private_capacity_bytes` counts only private DE input storage.
  `host_rank_staging_copy_{s,bytes}` measures DE's local copy from the shared image;
  it is zero for CUDA decode.
  `host_encoded_cache_capacity_{bytes,generation}` and creator-only
  `host_encoded_cache_allocation_{s,calls,bytes}` describe shared tmpfs.
  `host_prepared_cache_*` describes the shared prepared image; `build_s` repeats
  its owner duration on followers and must not be summed. `host_rank_cpu_workers`
  records pool size, zero for followers on either path. Capacity counters exclude
  temporary native Zstd output slabs and do not measure total host RAM usage.

`scheduler_timing.blocked_s` is the pause metric: process-local `monotonic_ns` from
setting the pause flag before reader fencing through resume clearing it. It includes
retraction, flush, apply and the wait for that engine's ranks, excluding background
prepare and post-resume cleanup. Open, failed or caller-retained pauses have null
`resumed_ns`/`blocked_s`. Resume returns the completed receipts without an extra RPC;
this is neither GPU idle time nor HTTP latency or first-token recovery.

The paired Miles `tests/manual/gpu_delta/bench_gpu_delta.py` reuses persistent
checkpoints/publications for one EP8 or two EP4 engines. `--profile-version` selects
an NVTX capture gate around one update, with correctness comparisons outside timed
updates. Standard operation annotations also cover non-delta SGLang work. Async
`gpu_delta.pause` ranges intentionally remain open for poisoned/failed/retained
pauses; synchronous worker/phase ranges balance on exceptions. NVTX and CUDA work
can overlap, and Async enqueue durations can include GPU backpressure.

## Validation

The CPU suite is grouped by contract: `bindings` covers canonical slicing and
FlashInfer byte maps; `layout` covers model mapping and consumer admission;
`receiver` covers preparation, plan caches and the decode/apply lifecycle;
`apply` and `derived` cover execution planning and consumer updates.
`codec_admission` covers ABI/frame geometry and outer decoding. The API suite
groups HTTP routing, tokenizer coordination and reply correlation. Checkpoint,
host-cache, memory and session/scheduler tests retain separate fixtures. The independent CPU model mapping exercises extensibility,
not a newly qualified serving model. CPU mocks do not prove CUDA/native behavior.

```bash
python -m pytest -q test/registered/unit/gpu_delta_weight_sync \
  --ignore=test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/manual/gpu_delta/test_codec.py
python -m pytest -q test/manual/gpu_delta/test_host_cuda.py
python -m pytest -q test/manual/gpu_delta/test_apply_cuda.py
python -m pytest -q test/manual/gpu_delta/test_device_allocator_cuda.py
```

CPU cases register in `base-a-test-cpu`; layout CUDA cases register in
`base-b-test-4-gpu-b200` and use serving-image FlashInfer/CuTe dependencies without
nvCOMP. They compare against ordinary loader/preprocessing helpers and check
padding, independent/aliased images, exact bytes, failure gates and graph addresses.

Manual tests use the paired CUDA 13 Miles image, `nvidia-libnvcomp-cu13==6.0.0.12`,
`zstandard`, `python-snappy` and `lz4`; missing prerequisites fail. The Snappy/LZ4
DE cases and the DE allocator test require SM100/SM103. CUDA Bitcomp/Snappy/LZ4
cases do not require DE. Together they exercise sorting off/on, fixed-slot reuse,
known byte output and ownership without sending malformed compressed streams into
nvCOMP. The host tests additionally check one-owner preparation, shared bytes and
capacity reuse/growth, cancellation and failure lifetimes. The host CUDA test
checks shared registered mappings and asynchronous H2D after a peer unregisters,
plus private DE rank allocations and once-per-engine file verification. Its
two-engine DE case
and the allocator context-restoration test each require two DE-capable devices.
Use separate fresh processes for each ambient allocator mode:

```bash
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
  python -m pytest -q test/manual/gpu_delta/test_device_allocator_cuda.py test/manual/gpu_delta/test_codec.py
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False \
  python -m pytest -q test/manual/gpu_delta/test_device_allocator_cuda.py test/manual/gpu_delta/test_codec.py
```

See [NVIDIA's DE requirements](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html).
