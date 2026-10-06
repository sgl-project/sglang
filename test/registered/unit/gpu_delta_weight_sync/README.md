# GPU delta receiver tests

The experimental receiver applies canonical XOR deltas directly to existing
execution-layout buffers. Static bundled MTP layer weights remain unchanged;
its shared embedding/head buffers continue to follow the target. The disk update
API remains independent. Pure Torch byte transforms cover CuTe DSL NVFP4 expert
layouts and BF16 dense storage;
the standalone MegaMoE transform helper does not admit an integrated MegaMoE
runtime on this branch.

Canonical names, shapes and dtypes come from the original immutable local
safetensors checkpoint headers at first delta admission. The feature supports the
standard checkpoint loader and its selected target/bundled-MTP files; custom,
secondary or transformed sources are outside this contract. It does not wrap
model methods or change ordinary checkpoint loading. Packed live parameters alone
cannot recover the canonical source inventory.

One Miles coordinator exclusively owns these engines' model updates, pause/resume,
memory residency and topology for the stream's lifetime. Mixing another weight
updater or administrative mutation into the same engine is unsupported. Ordinary
APIs retain their existing behavior; this feature does not intercept them to
implement a server-wide ownership lock. A stream starts with fresh engines and
never automatically replays an uncertain XOR update.

HTTP routes, tokenizer coordination and scheduler control handlers live under
`srt/weight_sync/gpu_delta/`. Shared integration consists of route, IPC schema,
communicator and scheduler-handler registration. Preparation takes no model-update
writer lock. Partial mutation poisons the delta session instead of attempting
recovery or falling back to a different update path.

```bash
python -m pytest -q test/registered/unit/gpu_delta_weight_sync \
  --ignore=test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/manual/gpu_delta/test_codec.py
python -m pytest -q test/manual/gpu_delta/test_host_cuda.py
```

Layout algebra, allocator admission, protocol and session tests run in the
existing `base-a-test-cpu` suite. The CUDA layout test runs in
`base-b-test-4-gpu-b200` with the serving package's FlashInfer/CuTe DSL dependencies;
it does not require nvCOMP.

The codec suite is manual-only because registered CI does not provision its
prebuilt nvCOMP dependency. It requires Blackwell, `nvidia-libnvcomp-cu13==5.3.0.16`,
`zstandard`, `python-snappy` and `lz4` (the last two are CPU test oracles). It
checks hardware Snappy and LZ4 decoding with sorting off/on against known input
bytes from a DE-capable host allocation, reusing decoded HBM slots. No malformed compressed
streams are sent to nvCOMP. Missing hardware or dependencies fail the manual
suite. The registered `test_gpu_delta_layout_cuda.py` compares layouts and derived
scale buffers with the existing SGLang/FlashInfer loader helpers and checks MLA
source views, failure gating and destination addresses across CUDA graph replay.

Preparation reads, validates and hashes encoded publication files once per engine-host.
Each rank unwraps only its local tensors directly into its own retained, original
DE-capable host allocation for CPU and GPU access.
Preparation constructs CPU descriptors without reading weights or stopping serving;
it prepares small GPU metadata/workspace and raw-target inputs, but never
allocates large decoded-mask slots or runs DE/model application.
There is no staging selector or full-publication HBM copy. Every retained changed
matrix frame uses the selected inner codec, including inputs whose
compressed representation expands; there is no raw-frame fallback. Once every
original rank reports `PREPARED`, Miles fans out
`update_weights_from_delta`. Each local handler closes generation admission,
pauses scheduling, fences existing readers, retracts requests, flushes caches and
applies the delta. It returns `APPLIED` only after GPU completion and decoder
checks. Miles waits for each engine's original ranks to apply, then sends that
engine `resume_weights_from_delta(session_id)`. Independent
engines prepare, apply and resume separately; the trainer waits for all engines
before advancing its update baseline. Resume records the new version and reopens
generation after successful local acknowledgments.

Layout admission caches physical scale destinations and BF16 MLA/FP32 scale source
views. Aligned NVFP4 blockscales apply canonical bytes through strided destination
views; padded geometries retain the explicit zero-padded mask transform. Primary
and MMA images that alias receive one XOR, while independent consumer images each
receive it. Derived refresh uses a device predicate and writes into the existing
destination without allocating a transformed source or replacement buffer.
Before apply, identity checks cover live parameter roots and independent consumer
objects/storage. Feature-owned immutable views do not need separate scans, and
the in-place apply does not repeat the walk afterward.

Compressed tensors form batches of adjacent active target model layers. Separate
non-layer groups hold embeddings, the language-model head, and remaining standalone
weights, in that order; empty groups are omitted.
`GPU_DELTA_LAYERS_PER_BATCH` defaults to 1; positive values group that many active
layers in model order, even when unchanged layers are absent. Each complete batch
must fit HBM; there is no intra-layer streaming. DE reads the compressed frames directly from the rank-owned host arena into HBM;
there is no encoded HBM ring, compressed H2D copy or transfer-stage selector.
Decoded slots each fit the largest batch, including non-layer groups. A dedicated DE stream
decodes the next batch while the apply stream validates and applies the current
batch. One nvCOMP call handles all retained frames in a batch. Slot reuse waits
for its previous apply event, including consumption of that slot's size/status
rows. Temporary DE workspace is shared because DE submissions are ordered.

The large decoded-mask slots are allocated only after scheduler pause and
its reader fence. Preparation already allocates the small nvCOMP temporary
workspace, per-slot status/size rows and descriptor slabs, and uploads immutable
input metadata and raw targets on feature-owned streams. Relative output offsets,
slot bounds, metadata views and status-validation callbacks are also prepared here,
without launching decode or apply. Temporary CPU frame and raw-packing plans are
not retained after preparation; the decode plan keeps the host/GPU buffer leases
and metadata needed for execution. After pause, actual output pointers are checked
and uploaded; scratch-dependent apply setup and first-use tuning remain paused.
Slots are allocated once per update, reused across layers, and
released after completion before resume. The
PyTorch native caching allocator may reuse their storage; the feature retains no
large HBM lease during normal rollout. Host arena capacity and static CPU plans
remain persistent.
Preparation records the omitted frame gaps and tails. Only those byte ranges are
zeroed before decode; a fully covered batch skips zeroing. One device kernel checks
all decoded sizes/statuses and ORs failures into the sticky apply gate.

Layer membership and normalized affine source/destination views are cached for
the current active tensor set and layer count. Fresh compressed lengths, frame
lists and arena offsets are prepared for each update. Singleton and adjacent
contiguous axes are collapsed without changing byte order. All affine images in
a streaming batch share one XOR launch. Its exact uniform contracts, counts and
tile intervals are compile-time parameters, without model-name classification or
a runtime geometry table. Only fresh source/destination pointers are uploaded.
Proven four-byte alignment and
row/tail geometry select uint32 XOR; other contracts use byte XOR in the same
kernel. Contiguous inner tiles need only scalar base-address arithmetic. Irregular
padded scales retain their explicit transform.

Paused setup loads the chosen module and queries CUDA's actual residency once per
compiled kernel/device. A small first-use search compares one
CTA per 2048-byte tile with a static grid of at most four resident waves. It uses
the complete batch's geometry/counts and disjoint synthetic source/target regions
in unused decoded scratch, never live weights or the decoder. One warmup and three
timed trials per candidate use events on the paused apply stream. The sum of
borrowed footprints is capped at 4 GiB per device; no additional weight-size
allocation is made. Nonfitting batches or an exhausted budget use the measured
naive policy. Fitting depends on the selected grouping and scratch capacity.
Both measured and fixed
choices are cached in-process by complete geometry, counts, proven alignment and
device. Warm plan reuse performs no fitting check, occupancy query or tuning.
Cold tuning is included in the pause; warm geometry reuses its cached choice.
No configuration selection, counter reset or dynamic work stealing occurs in the
layer loop.

DE0 is submitted first; the small raw-target apply can overlap it. Apply i is
queued before submission of DE i+1 because nvCOMP's Async call may wait for earlier
work on its calling stream. Ready/free events protect decoded-slot reuse. All
status checks and consumers of the sticky error flag run on the apply stream;
the DE stream does not race that flag. Completion joins every DE operation and
mask application before scratch release; failed cleanup drains both streams.

There is no separate global quiesce or commit round. A participant may apply
before another fails to pause; failed or uncertain activation never authorizes
resume, rollback or automatic replay. Miles owns the ordered API sequence and
sends resume only after every rank of that engine reports successful apply.
Concurrent administration, retries and arbitrary call ordering are unsupported;
SGLang does not revalidate caller identity/session echoes or accept a separate
resume certificate. A failed
fence must not reclaim KV/cache. Preparation can be aborted before update dispatch,
while serving continues on the old version.

The admitted topology uses ordinary globally ordered control broadcast. Local
control broadcast and elastic EP joiners are unsupported. Delta application itself
has no distributed collectives, and shared IPC weight storage is excluded.

The `GPU_DELTA_*` environment variables below are development/debug knobs,
not a stable user-facing configuration API.

`GPU_DELTA_CODEC` accepts `snappy-zstd` (default) and `lz4-zstd`. The
receiver freezes it at backend admission, advertises it in its participant plan,
and rejects any unsupported value. The sender and receiver must agree before a
publication; manifest fields cannot override the launch constraint. Legacy codec
values and removed encoder/outer selectors have no migration layer.

Protocol 4 carries the selected `codec` and explicit `frame_bytes` (64 KiB, 1 MiB or
4 MiB; default 1 MiB). Matrix frames contain only input/output offsets and lengths, without
redundant codec/file fields. Each natural tensor's outer descriptor names one
immutable owner file and independent Zstd chunks of at most 1 MiB output, exactly
covering its aligned inner-codec arena. LZ4 uses raw byte blocks with bitshuffle
disabled. The sender computes both the inner codec and outer Zstd on GPU; the receiver unwraps Zstd on CPU directly into each rank's original host arena,
then decodes model-layer batches
directly from host for in-place apply. Natural tensor boundaries remain unchanged
in the publication format.
The decoder caches the device's hardware operation limit and rejects a frame if
its actual compressed or decoded length exceeds that limit. A 4 MiB frame whose
compressed representation expands beyond a 4 MiB device limit is rejected;
there is no frame splitting or software fallback. Outer Zstd stays at 1 MiB.
There is no GPU outer decoder, legacy protocol or automatic fallback.

`GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS=0` (default) or `1` selects nvCOMP
hardware chunk sorting once when the decoder is constructed, for either codec.
Sorting executes inside the paused decode call; preparation does not sort or
reorder the frames. The codec extension and sorting options await native GPU
qualification and matched benchmarks; existing Snappy results do not measure LZ4.

`GPU_DELTA_DECODE_STAGES` accepts 2 (default), 3 or 4 and is frozen at backend
initialization. Each extra stage adds one largest-batch decoded HBM buffer and
small status/size rows, allocated at the same preparation/pause boundaries.
Batch i reuses slot i modulo the depth only after that slot's prior apply finishes.
DE submissions still share one ordered stream/workspace, and apply i is queued
before the host submits DE i+1. More slots can defer a reuse wait but do not remove
nvCOMP's calling-stream wait or guarantee lower pause latency.

`GPU_DELTA_CPU_WORKERS` defaults to 32 (bounded to 1–32) per rank.
Each rank uses reusable workers with independent Zstd contexts. Local tensors
are grouped into at most four times as many tasks as workers, preserving strict
per-frame checks and direct writes into the rank arena. Two EP4 engines therefore
have up to eight pools of 32 decode workers. Workers touch only CPU buffers;
CUDA setup remains on each rank's preparation thread.

`GPU_DELTA_HOST_CACHE_DIR` defaults to `/dev/shm/sglang-gpu-delta-<uid>` and
must be a private, user-owned tmpfs directory large enough for encoded publication
files. The decoded inner-codec/raw arenas are CUDA-owned host RAM, outside tmpfs.
Ranks of one engine on the same physical host must see the same cache directory;
across containers, explicitly mount the same host tmpfs there. Engine IDs select
separate subdirectories and advertised `host_cache_id` values. Independent engines
share no cache locks or release lifecycle. Container hostname does not infer sharing.
Miles still sends the negotiated `host_tensor_names` union. Each receiver checks
that it covers its local bindings, then decodes only those local tensor names.

Each rank qualifies its local views while admitting the canonical tensor/view
plan and retains only detached static definitions. Later publications compare
every static field directly, without repeating local view matching; reordered
views use the same canonical normalization. Immutable binding and derived-image
storage keys are cached, while each publication remaps its fresh frame offsets
and omitted-byte ranges in one pass. Frames and payloads remain
publication-specific. Private arena index/state records use `orjson`; atomic
replacement and canonical namespace/publication digests are unchanged.

One creator per engine-host validates all publication frame metadata, including
foreign experts, then reads and SHA-256 checks owner files in parallel using its
existing CPU pool. Each task copies into its disjoint retained tmpfs slice and
checks source identity/extent across the read; all tasks drain before READY or failure. The
namespace and build/release mutex bind the original engine participants and delta
stream; publication metadata binds the manifest path, digest, session and versions.
READY certifies the encoded cache only after verification succeeds. Ranks then
release the cache lock and independently decode local tensors into their own
arenas; raw targets are copied beside the inner-codec frames. There is no second
source-file read, full decoded temporary or intermediate decoded host copy.
Each rank drains all submitted decode tasks before returning or raising.

Each rank's arena uses CUDA 13 `cuMemCreate` with `HOST_NUMA`, `PINNED` and
`CU_MEM_CREATE_USAGE_HW_DECOMPRESS`, without export handles. The original mapping
is admitted with `CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE`. There is no
CUDA memory import, FD broker, compiled extension or software inner-decoder fallback.
Ordinary `cudaHostAlloc`/`cudaHostRegister` memory does not establish this contract.
See [NVIDIA's DE requirements](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html).

Both the rank arena and shared encoded cache reserve the required extent for the
initial capacity, rounded to their allocation granularity. Later growth reserves
twice the required extent; fitting updates reuse the allocation. Each rank owns
its DE allocation and frees it after its streams drain. Encoded tmpfs capacity is
owned by the engine-host cache and has its own generation.

Miles sends resume only after all original ranks of that engine have applied.
Successful resume authorizes encoded-cache reuse. The session queues release and
view cleanup on its FIFO executor ahead of the next prepare. Apply, abort and
failure cannot release a publication. Late old releases cannot release newer
bytes. BUILDING is recorded before overwrite; failed or aborted publications
remain nonreusable. Retained encoded files require explicit cleanup.

Canonical rank-0/rank-1 tensors instead negotiate `raw_bytes`: complete target
values with no XOR, frames or compression envelope. Unchanged values omit their
payload. Preparation packs changed local scalars/vectors into one aligned pinned
arena, uploads it and performs any BF16-to-FP32 norm conversion during
preparation. After pause, dtype-grouped `torch._foreach_copy_` updates existing buffers before matrix
application; derived NVFP4 scales refresh afterward. Late decoder failure retains
the same poisoned-session behavior. This small bypass remains one whole-model
packed path rather than being split into layer batches.

`raw_bytes` counts direct payload bytes; `raw_h2d_bytes` includes arena alignment.
`host_raw_pack_s` is preparation CPU packing; raw H2D also occurs in preparation.

Preparation reports manifest loading/parsing (`host_manifest_read_parse_s`), plan
validation (`host_plan_validate_s`), frame validation (`host_encoded_cache_frames_validate_s`),
local tensor planning (`host_tensor_prepare_s`) and full preparation
(`host_prepare_s`). `host_metadata_prepare_s` covers small GPU input setup and
status-callback preparation, including its own stream waits (`host_metadata_wait_s`).
`paused_setup_host_s` includes decoded-slot allocation, output-pointer binding and
uploads, and scratch-dependent apply setup; `paused_apply_tune_s` isolates cold tuning.
`paused_apply_host_wall_s` includes setup, the decode/apply pipeline, completion
and scratch release, but the scheduler's full `blocked_s` remains the pause metric.

`de_host_input_bytes` counts compressed bytes read directly by DE. `h2d_bytes`
counts explicit raw-target and metadata transfers; it no longer counts an encoded
Snappy copy. These are different traffic categories, not a throughput estimate.
`decode_stages` reports the configured depth; `decoded_buffers` counts actual slots.
`decoded_scratch_bytes` is their total allocation;
`decoder_workspace_bytes` is temporary DE workspace. Neither is peak HBM usage.
`decoded_zero_ranges`/`decoded_zero_bytes` count omitted canonical bytes cleared
before decode. Batch/group/contract/grid and cold tuning counters retain their
ordinary meanings. `host_batch_plan_reused` reports cached active tensor geometry.

With `GPU_DELTA_TIMING=1`, events report decode on the DE stream and layout apply,
raw apply and derived refresh on the apply stream. `paused_gpu_pipeline` excludes
setup. These spans overlap and must not be summed. nvCOMP can wait inside an Async
call, so host enqueue durations can contain GPU backpressure.

`host_encoded_cache_created`/`host_encoded_cache_reused` identify the creator
and followers. Creator-only `host_encoded_cache_read_hash_s` measures parallel
file verification, including path checks, submission and all joins.
`host_encoded_cache_read_worker_sum_s` and `host_encoded_cache_sha256_worker_sum_s`
sum overlapping worker intervals; neither is additive with the wall span.
`host_encoded_cache_hash_files`, `hash_bytes` and `frames_validations` (with the
same prefix) count shared work once. Verification finishes before local decode starts. `host_encoded_cache_wait_s` isolates the
cache mutex wait. `host_encoded_cache_build_s` repeats the cached build duration
on followers and must not be summed across ranks. `host_rank_prepare_s` includes
cache access, rank allocation and local outer decode.

`host_plan_cache_reused` reports whether the canonical plan's static definitions
were already qualified. Every publication still authenticates its manifest and
checks names, shapes, dtypes, encodings, byte counts and rank views against the
admitted plan, then validates all changing payload/frame extents. The cache holds
only detached static definitions, not an old manifest or payload.

Each rank reports `host_rank_outer_zstd_decode_s` for CPU task submission/join
wall time, including raw copies. `host_rank_outer_zstd_validate_s` and
`host_rank_outer_zstd_worker_decode_sum_s` sum worker durations, not critical-path
time. The same prefix's `encoded_bytes`, `decoded_bytes`, `tensors` and `frames`
count that rank's local outer decode work. Rank arenas report
`host_rank_{arena_bytes,capacity_bytes,capacity_generation,mapping_reused}` and
`host_rank_allocation_{s,calls,bytes}`. Shared encoded storage separately reports
`host_encoded_cache_capacity_{bytes,generation}` and creator-only
`host_encoded_cache_allocation_{s,calls,bytes}`. `host_rank_cpu_workers` records
each rank's pool size.

Preparation can contend for host bandwidth and performs small GPU input work
while rollout continues. It never reserves large decoded-mask slots or runs
DE/application/cold scratch tuning before pause. The manual two-engine, two-rank
CUDA test checks private original allocations, local exact-byte DE, once-per-engine
encoded verification, warm reuse and growth. CPU mocks establish control and byte
semantics only; hardware capability, DE and overlap require native validation.

Runtime updates do not hash weights or build a custom compiled extension. Exact
weight-content comparisons are confined to tests. State/version checks prevent
replay but do not prove weight equality. Unsupported layouts are rejected. The
exclusive-controller contract requires a fresh baseline; pointer checks do not
detect arbitrary same-buffer writes by another updater.

Admission also reuses the ordinary updater's shared CUDA IPC weight-cache and
HPC-Ops derived-weight-cache exclusions before creating a delta plan or session.

The paired Miles feature contains `tests/manual/gpu_delta/bench_gpu_delta.py`, which launches
one EP8 or two EP4 engines and measures the snappy-zstd contract using persistent
altered checkpoint and publications. `GPU_DELTA_TIMING=1` enables
phase events without synchronizing every tensor; correctness comparisons stay
outside timed updates.

Each original-rank receipt includes `scheduler_timing` on that process's
`monotonic_ns` clock. `blocked_s` runs from the scheduler's pause flag, before
the existing reader fence, until its resume clears that flag. It includes
retraction, cache flush, apply and the wait for its own engine ranks to acknowledge
apply; it excludes
background preparation and post-resume cleanup. Open or failed intervals retain
null `resumed_ns` and `blocked_s`. Resume responses carry the completed receipts,
so measurement needs no extra synchronization or status RPC. This measures
scheduler blocking, not GPU idle time, HTTP latency or first-token recovery.
