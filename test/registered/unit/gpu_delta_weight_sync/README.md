# GPU delta receiver

The experimental receiver applies canonical XOR deltas directly to existing
execution-layout buffers. Small complete-value targets use overwrite semantics.
It does not requantize weights, change checkpoint loading, replace the ordinary
allocator, or alter the independent disk update API.

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

Fixed topology is required when routed experts are bound: expert TP=1, no
presharded canonical checkpoint, EPLB, redundant experts or elastic joiners. The
selected FlashInfer runner must match its dispatcher. Dense-only mappings do not
need a MoE configuration. Ordinary globally ordered control broadcast is required;
local broadcast, LoRA, disaggregated serving, unsupported attention/pipeline
parallelism, shared IPC weight storage and HPC-Ops derived-weight caches are
rejected at admission. Delta application adds no distributed collectives.

## Ownership and lifecycle

One coordinator exclusively owns model updates, pause/resume, memory residency
and topology for a stream's lifetime. Mixing another updater or administrative
mutation into those engines is unsupported. There is no server-wide ownership
lock, XOR retry, rollback or automatic replay of an uncertain update.

Routes, IPC schemas, tokenizer coordination and scheduler handlers live under
`srt/weight_sync/gpu_delta/`; shared files only register them. Each engine follows:

1. **Prepare:** background work authenticates the manifest, qualifies the canonical
   plan, admits the encoded cache and prepares local host payloads, small GPU
   metadata, raw targets and derived calibration. It takes no model-update writer
   lock and neither writes model/KV storage nor pauses generation.
2. **Apply:** close generation admission, pause scheduling, fence readers, retract
   requests and optionally flush caches. Allocate two decoded-mask slots, bind
   pointers, decode and apply. `APPLIED` requires GPU completion and successful
   native size/status checks. A failed fence cannot reclaim KV/cache.
3. **Resume:** after every original rank of that engine applies, Miles sends
   resume. SGLang records the model version and reopens generation. The trainer
   advances its baseline only after all engines finish. Independent engines need
   not pause together.

Partial mutation or uncertain apply/resume keeps admission closed and requires
restart. Preparation may be aborted before apply dispatch, while old-version
serving continues. Arbitrary call ordering, retries and concurrent administration
are outside the exclusive-coordinator contract; there is no separate global
quiesce/commit round or resume certificate.

Staged apply and one-shot updates accept `flush_cache=true` and
`abort_all_requests=false`. A trusted caller may disable flushing only after
invalidating KV and multimodal caches. Reader fencing and retraction still run;
optional abort occurs after new admission closes. One-shot updates preserve the
pause present at the apply boundary. `resume_gpu_delta(keep_pause=true)` commits
the version and releases delta resources without reopening generation; its
`RESUMED` receipt has `generation_paused=true` and null `resumed_ns`/`blocked_s`.
The optional `weight_version` labels admitted generations independently of the
transport's target sequence; omission uses the transport target version.

`clear_gpu_delta_state` is valid when idle or successfully resumed. It drains each
rank's FIFO release work, closes workers/private host allocations and drops
planning, decoder and stream objects. After every rank acknowledges, one process
per engine-host deletes the shared encoded cache. Weights, publications, committed
version, process identity, small provenance and pause state remain unchanged.
Describe can recreate the backend at that version. Framework allocator/JIT caches
are not cleared.

A standalone load requires a fresh version-0 engine using the matching immutable
HF base. Keep it out of routing until success; the publication path must be visible
to all ranks. The loader discovers fresh participants and composes a 0→V update:

```bash
curl -f -X POST http://localhost:30000/update_weights_from_gpu_delta \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"/checkpoints/step-100/gpu-delta/manifest.json"}'
curl -f -X POST http://localhost:30000/clear_gpu_delta_state \
  -H 'Content-Type: application/json' -d '{}'
```

`release_state=true` clears delta resources after a successful one-shot load.
Miles recovery uses `false` to retain the backend for later staged updates.

## Payload and cache contracts

Publications select `snappy-zstd`, `lz4-zstd` or plain `lz4`, with a positive integer
`frame_bytes` up to 4 MiB (default 1 MiB). The decoder checks actual compressed and
decoded lengths against its device limit, plus geometry, alignment and capacity.
An expanding compressed frame beyond that limit is rejected; there is no software
inner decoder, frame splitting or raw-matrix fallback. Streams can change codec
without changing their canonical plan; native decoders are cached by inner codec.

Wrapped matrix frames use independent outer Zstd chunks with at most 1 MiB output.
The CPU worker pool unwraps them directly into each rank's original host arena;
DE reads the packed inner Snappy/LZ4 bytes from there. Plain LZ4 has an empty outer
`frames` list and equal encoded/decoded extents; workers copy each local tensor's
inner arena once. There is no encoded HBM ring or full-publication HBM copy.
Unchanged frames are omitted. Scalars/vectors use `raw_bytes`: complete target
values, no XOR/compression, and no payload when unchanged.

One engine-host creator reads and optionally SHA256-checks owner files in parallel
into retained tmpfs slices. Source identity/extent is checked across each read;
all tasks drain before READY or failure. Namespace/publication records bind the
original cohort, stream, path, digest, session, versions and hash policy. Ranks then
release the cache lock and independently prepare local payloads. Global manifest
and foreign entries are dropped after READY, before local preparation. Mapped
encoded-file views remain owned until every submitted worker drains.

Each rank owns a private CUDA 13 `HOST_NUMA`/`PINNED` allocation with
`CU_MEM_CREATE_USAGE_HW_DECOMPRESS`; it is never exported/imported or shared through
an FD broker. Ordinary pinned Torch memory does not establish DE capability.
Both the rank arena and encoded cache fit the initial extent, rounded to their
allocation granularity. Growth reserves twice the required extent; fitting updates
reuse storage. Host RAM and tmpfs capacity are separate.

Successful all-rank resume authorizes encoded-cache reuse. The session queues
release and view cleanup ahead of its next prepare on the same FIFO executor.
BUILDING precedes overwrite; failed/aborted publications remain nonreusable, and
late old releases cannot release newer bytes. Explicit clear removes retained
encoded files only after the rank-close barrier.

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
has its own status/size row. One apply stream owns the sticky error gate. Slot
`i % 2` is reused only after its prior apply completes. DE0 starts first, overlapping
small raw-target copies. Apply i is queued before host submission of DE i+1 because
nvCOMP's Async call can wait for prior work on its calling stream.

Only omitted frame gaps/tails are zeroed. A fully covered batch skips zeroing; one
status kernel checks native sizes/statuses before each grouped XOR. Aligned source
binding patches the prepared pointer slab and uploads it once. Unexpected source
misalignment rebuilds conservative runtime descriptors; later aligned binding
restores the template. Derived consumer copies are gated by the sticky error.
Completion joins DE and apply before scratch release; failed cleanup attempts to
drain both streams. Foreign tensor aliases retain allocations and registered
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
Keep them fixed for the stream's lifetime. Batch size, worker count and hash policy
are captured when the backend is created; sorting is captured by each decoder.

| Setting | Default | Meaning |
| --- | --- | --- |
| `GPU_DELTA_LAYERS_PER_BATCH` | `1` | Positive count of active adjacent model layers per batch |
| `GPU_DELTA_CPU_WORKERS` | `32` | Reusable workers per rank, allowed range 1–32 |
| `GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS` | `0` | `0`/`1`, native sorting on the paused DE call |
| `GPU_DELTA_HOST_CACHE_DIR` | `/dev/shm/sglang-gpu-delta-<uid>` | Private user-owned tmpfs visible to the engine's host-local cohort |
| `GPU_DELTA_SKIP_PAYLOAD_HASH` | `0` | `1` skips payload SHA256; manifest authentication and size/path/native checks remain |
| `SGLANG_ENABLE_NVTX_OPERATIONS` | `0` | Standard SGLang detailed operation annotations, including GPU delta |

There is no decode-depth or legacy timing/NVTX knob. Wrapped workers have independent
Zstd contexts and at most four tasks per worker. Two EP4 engines can use eight pools
of 32 workers. Engine IDs and `host_cache_id`, not hostname alone, define cache
sharing. Independent engines share no cache locks or release lifecycle.

Set the same payload hash policy in the Miles/Ray environment. A receiver requiring
hashes rejects a publication with `payload_checksum_format="none"`; a skipping
receiver may accept hashed publications. Skipping trusts contents without payload
SHA authentication. Exact file extent/identity, path containment, Zstd output
length and native decode checks still apply.

CPU timings, bytes and counters remain available for W&B independently of NVTX.
Receipts contain no diagnostic CUDA timing events. Host spans are nested and worker
sums overlap; do not add them or infer device bandwidth from host durations.

| Receipt fields | Interpretation |
| --- | --- |
| `host_prepare_s` | Full background preparation |
| `host_manifest_read_parse_s`, `host_plan_validate_s`, `host_tensor_prepare_s` | Manifest, canonical-plan qualification, local tensor planning |
| `host_plan_cache_reused`, `host_batch_plan_reused` | Admitted digest / current active-geometry cache reused; no old frames retained |
| `host_rank_prepare_s` | Encoded admission, global metadata release, local allocation/fill and final encoded-view release |
| `host_rank_metadata_release_s` | Narrow entries and drop global manifest/content, before tensor planning |
| `host_rank_prepare_body_s`, `host_rank_prepare_call_s` | Through local snapshot return / local fill and job cleanup; caller-minus-body includes view release/bookkeeping |
| `host_rank_layout_s`, `host_rank_allocation_s` | Local ordering/offset planning / arena growth |
| `host_raw_pack_s`, `raw_bytes`, `raw_h2d_bytes` | CPU raw packing, direct bytes, aligned arena upload bytes |
| `host_metadata_prepare_s`, `host_metadata_wait_s` | Small GPU input/status setup including its own stream waits / those waits alone |
| `host_apply_kernel_prepare_s`, `apply_kernel_cache_misses`, `apply_kernel_cache_reused` | Record construction and fixed-module cache lookup; misses are Python cache misses, not necessarily compilation |
| `paused_setup_host_s`, `paused_apply_host_wall_s` | Slot allocation/binding/uploads / complete apply through completion and scratch release |
| `h2d_bytes`, `apply_metadata_h2d_bytes`, `apply_descriptor_h2d_bytes`, `decoder_metadata_h2d_bytes` | Explicit raw/metadata uploads; descriptor bytes are a subset of apply metadata bytes |
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
  copies; `host_rank_outer_zstd_worker_decode_sum_s` sums workers.
  `host_rank_outer_zstd_{encoded_bytes,decoded_bytes,tensors,frames}` counts local
  wrapped work. Plain LZ4 uses `host_rank_encoded_copy_s` and
  `host_rank_encoded_copy_{worker_sum_s,bytes,tensors}` instead; copy counters
  exclude raw targets. Inactive codec counters are zero.
- `host_rank_{arena_bytes,capacity_bytes,capacity_generation,mapping_reused}` and
  `host_rank_allocation_{s,calls,bytes}` describe private arenas.
  `host_encoded_cache_capacity_{bytes,generation}` and creator-only
  `host_encoded_cache_allocation_{s,calls,bytes}` describe shared tmpfs.
  `host_rank_cpu_workers` records pool size.

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
FlashInfer byte maps; `layout`, `apply` and `derived` cover planning/caches and
consumer updates; `codec_admission` covers ABI/frame geometry and outer decoding.
Checkpoint, host-cache, memory, session/scheduler, tokenizer and HTTP tests retain
separate fixtures. The independent CPU model mapping exercises extensibility,
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

Manual DE tests require Blackwell, CUDA 13, `nvidia-libnvcomp-cu13==5.3.0.16`,
`zstandard`, `python-snappy` and `lz4`; missing prerequisites fail. They exercise
Snappy/LZ4 sorting off/on, fixed-slot reuse, known byte output and ownership without
sending malformed compressed streams into nvCOMP. The host CUDA test additionally
checks private rank allocations, once-per-engine file verification and warm
reuse/growth. The allocator test requires two CUDA devices for context restoration.
Use separate fresh processes for each ambient allocator mode:

```bash
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
  python -m pytest -q test/manual/gpu_delta/test_device_allocator_cuda.py test/manual/gpu_delta/test_codec.py
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False \
  python -m pytest -q test/manual/gpu_delta/test_device_allocator_cuda.py test/manual/gpu_delta/test_codec.py
```

See [NVIDIA's DE requirements](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html).
