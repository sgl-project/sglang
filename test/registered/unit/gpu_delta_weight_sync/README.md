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
`srt/weight_sync/gpu_delta_*`. Shared integration consists of route, IPC schema,
communicator and scheduler-handler registration. Preparation takes no model-update
writer lock. Partial mutation poisons the delta session instead of attempting
recovery or falling back to a different update path.

```bash
python -m pytest -q test/registered/unit/gpu_delta_weight_sync \
  --ignore=test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_layout_cuda.py
python -m pytest -q test/manual/weight_sync/test_gpu_delta_codec.py
python -m pytest -q test/manual/weight_sync/test_gpu_delta_host_cuda.py
```

Layout algebra, allocator admission, protocol and session tests run in the
existing `base-a-test-cpu` suite. The CUDA layout test runs in
`base-b-test-4-gpu-b200` with the serving package's FlashInfer/CuTe DSL dependencies;
it does not require nvCOMP.

The codec suite is manual-only because registered CI does not provision its
prebuilt nvCOMP dependency. It requires Blackwell, `nvidia-libnvcomp-cu13==5.3.0.16`,
`zstandard` and `python-snappy`. It checks hardware Snappy decoding against
known input bytes, reusing a bounded encoded tensor buffer. No malformed compressed
streams are sent to nvCOMP. Missing hardware or dependencies fail the manual
suite. The registered `test_gpu_delta_layout_cuda.py` compares layouts and derived
scale buffers with the existing SGLang/FlashInfer loader helpers and checks MLA
source views, failure gating and destination addresses across CUDA graph replay.

Preparation checks compressed artifact SHA-256 and unwraps outer Zstd once per
host sharing domain. Each rank registers the retained shared Snappy/raw arena for
CUDA and constructs descriptors without reading weights or stopping serving.
There is no staging selector or full-publication HBM copy. Every retained changed
matrix frame is Snappy, including inputs whose
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

Compressed tensors form one batch per target model layer, plus one standalone
batch for embedding/head and other non-layer weights. Each batch must fit HBM;
there is no intra-layer streaming. Preparation coalesces adjacent rank-owned
ranges of the shared pinned Snappy arena into bulk transfers, without copying
foreign experts or making a second per-rank host arena. GPU scratch fits the
largest batch, and one nvCOMP call decodes all retained frames in that batch.
Two encoded slots let a copy stream prefetch the next batch while the apply
stream decodes and updates the current batch. Decoded scratch and decoder
workspace remain single-buffered.
Preparation records the omitted frame gaps and tails. Only those byte ranges are
zeroed before decode; a fully covered batch skips zeroing. One device kernel checks
all decoded sizes/statuses and ORs failures into the sticky apply gate.

Layer membership, layout groups and destination geometry are reusable. Fresh
compressed lengths, frame lists and arena offsets are prepared for each update.
Singleton and adjacent contiguous axes are collapsed without changing byte order.
Expert layout metadata comes from one representative per canonical
`.experts.<id>.<suffix>` family, retaining the layer and complete projection/suffix;
each expert still contributes its own destination pointers. Uniform experts and
strided layouts keep the uniform batched XOR kernel. Contiguous non-expert tensors
with unequal lengths share one per-layer launch: a compact cumulative tile lookup
schedules exactly the useful byte tiles, with no largest-tensor padding or expanded
per-tile table. Equal-length groups use the uniform kernel. Irregular padded scales
retain their explicit transform.

The current active tensor set caches geometry and CPU prefix metadata. Fresh apply
pointers and the small prefix/length tails share one metadata upload before pause.
Bulk H2D starts only during paused apply. Ready/free events protect each
encoded slot; decoded scratch, status checks and updates remain ordered on the
apply stream. Completion joins the final copy, and cancellation drains both
streams before releasing shared host views. The existing reader fence and
device-side decoder failure gate remain in place.

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

`GPU_DELTA_CODEC=snappy-zstd` is the sole contract and the default. The
receiver freezes it at backend admission, advertises it in its participant plan,
and rejects any unsupported value. The sender and receiver must agree before a
publication; manifest fields cannot override the launch constraint. Legacy codec
values and removed encoder/outer selectors have no migration layer.

Protocol 4 carries `codec="snappy-zstd"` and explicit `frame_bytes` (64 KiB or
1 MiB). Matrix frames contain only input/output offsets and lengths, without
redundant codec/file fields. Each natural tensor's outer descriptor names one
immutable owner file and independent Zstd chunks of at most 1 MiB output, exactly
covering its aligned Snappy arena. The sender computes both Snappy and outer Zstd
on GPU; the receiver always unwraps Zstd on CPU directly into a host-shared arena,
registers each process's mapping for CUDA, then transfers model-layer batches for
hardware decoding and in-place apply. Natural tensor boundaries remain unchanged
in the publication format.
There is no GPU outer decoder, legacy protocol or automatic fallback.

`GPU_DELTA_CPU_WORKERS` defaults to 32 (bounded to 1–32) per engine-host.
One creator uses that many reusable CPU workers, each with its own Zstd context;
its other ranks attach to the completed arena. Natural tensors are grouped into
at most four times as many tasks as workers, preserving strict per-frame checks
and direct writes into the shared arena. Two EP4 engines therefore use two
independent pools: up to 64 decode workers plus two SHA workers on that host.
Workers touch only CPU buffers; CUDA setup remains on each rank's original
preparation thread.

`GPU_DELTA_HOST_CACHE_DIR` defaults to `/dev/shm/sglang-gpu-delta-<uid>` and
must be a private, user-owned directory on tmpfs with enough space for the wrapped
payloads and expanded arenas during construction. Ranks of one engine on the
same physical host must see the same directory and IPC/mount namespace; across
containers, explicitly mount the same host tmpfs there. Engine IDs select separate
subdirectories and advertised `host_cache_id` values. Independent engines
deliberately duplicate CPU buffers and work; they share no build locks or release
lifecycle. Container hostname is not used to infer sharing.
Miles negotiates the canonical tensor-name union per cache ID and sends it in
`host_tensor_names`. This negotiated union covers each receiver's local names; foreign
experts outside that union are not decoded.

Each rank qualifies the canonical tensor/view plan once and retains only detached
static definitions. Later publications compare every static field directly;
reordered views use the same canonical normalization. Frames and payloads remain
publication-specific. Private arena index/state records use `orjson`; atomic
replacement and canonical namespace/publication digests are unchanged.

One creator per engine-host validates all publication frame metadata, including
foreign experts, then copies owner files into retained tmpfs mappings and checks source
identity/extent across the read. One dedicated worker SHA-256 checks those exact
retained bytes while CPU workers decode independent canonical tensors directly
into one shared Snappy arena; raw targets are copied beside them. No second
payload read, full decoded temporary or per-rank Snappy copy is needed.
Aliased bindings and ranks within that engine reuse the same physical bytes. The
namespace and its build/release mutex bind the original engine participants, delta
stream and host tensor union; publication metadata
binds the canonical manifest path, digest, session and versions. READY is published
only after SHA verification, every decode task and exact chunk/window/output
check passes. Hash and decode are both joined on failure before ownership is
dropped; unverified bytes never become available for GPU use.

Each backend retains its MAP_SHARED mapping and CUDA registration across updates.
The first decoded/encoded allocations reserve the required extent, rounded to
64 MiB. A later capacity increase reserves twice the newly required extent with
the same rounding. A fitting later publication reuses both allocations and each
rank's existing registration. Growth allocates a new inode;
registered inodes are never resized. Each rank maps/registers the same shared
physical pages through its own VA; there is no full per-rank Snappy copy. CUDA
registration/unregistration runs outside the host build mutex. Torch's pinned
allocator does not own this external memory.

Miles sends resume only after the engine's ranks have finished H2D and their
update-stream fences. Successful local resume authorizes shared-arena reuse. Another engine's state does not authorize or block this release. The
session queues generation-specific release and view cleanup on its existing FIFO
executor, off the scheduler thread and ahead of the next local prepare. Local
apply, abort, failure and ordinary close cannot release a shared generation. A
late old release cannot release a newer publication. BUILDING is recorded before
overwriting bytes, so a decode failure cannot expose stale READY metadata.

Capacity and registration remain resident for the backend lifetime. Cold/growth
registration is measured separately; warm reuse does not register again. Failed
or aborted generations remain nonreusable, and retained tmpfs files require
explicit cleanup after consumers exit. There is no automatic eviction, pageable
fallback, per-rank region registration or full Snappy HBM residency.

Canonical rank-0/rank-1 tensors instead negotiate `raw_bytes`: complete target
values with no XOR, frames or compression envelope. Unchanged values omit their
payload. Preparation packs changed local scalars/vectors into one aligned pinned
arena, uploads it once and performs any BF16-to-FP32 norm conversion. During the
pause, dtype-grouped `torch._foreach_copy_` updates existing buffers before matrix
application; derived NVFP4 scales refresh afterward. Late decoder failure retains
the same poisoned-session behavior. This small bypass remains one whole-model
packed path rather than being split into layer batches.

`raw_bytes` counts direct payload bytes; `raw_h2d_bytes` includes arena alignment.
`host_raw_pack_s` is preparation CPU packing; direct H2D also occurs before pause.

Preparation reports manifest loading/parsing (`host_manifest_read_parse_s`), plan
validation (`host_plan_validate_s`), frame validation (`host_frames_validate_s`),
local tensor preparation (`host_tensor_prepare_s`), arena/decoder setup
(`host_decoder_prepare_s`) and its final GPU wait (`host_ready_wait_s`).
`host_prepare_s` covers the complete preparation. Shared construction, registration
and local tensor metadata are separate phases; nested timings and concurrent
worker durations must not be summed as wall time.
`decoder_metadata_uploads` counts metadata slabs and
`decoder_metadata_h2d_bytes` counts their uploaded bytes; neither changes the
matrix/raw payload byte counts.
`compressed_batches`, `compressed_h2d_spans` and `apply_groups` count layer
decodes, bulk copies and grouped XOR launches. `apply_variable_groups` counts the
mixed-length linear groups; `apply_prefix_h2d_bytes` counts their compact lookup
rows within the apply metadata slab. `decoded_zero_ranges`/`decoded_zero_bytes`
describe only omitted canonical bytes cleared before decode. `host_batch_plan_reused`
reports reuse of the active tensor plan. Encoded/decoded scratch and decoder
workspace byte counts describe reserved working buffers, not peak HBM usage.
With debug timing enabled, `paused_layer_h2d` measures copy-stream work and
`paused_copy_wait` measures the apply stream waiting for ready data; these overlap
with decode/apply and must not be added together.

`host_payload_cache_created`/`host_payload_cache_reused` distinguish the one
creator from followers. Creator-only `host_payload_read_s`, `host_payload_sha256_s`,
`host_payload_hash_files` and `host_payload_hash_bytes` expose once-host read/hash;
`host_payload_read_sha256_s` is their work sum, not a sequential critical path.
SHA overlaps decode: `host_payload_decode_hash_s` measures the combined wall span,
and `host_payload_hash_wait_s` is the hash tail waited after decoding. Do not add
SHA duration to decode wall time. Followers report zero work for these counters.
`host_shared_prepare_s` includes cache wait/attachment or construction;
`host_payload_cache_wait_s` isolates the build/attachment mutex wait.

`host_plan_cache_reused` reports whether the canonical plan's static definitions
were already qualified. Every publication still authenticates its manifest and
checks names, shapes, dtypes, encodings, byte counts and rank views against the
admitted plan, then validates all changing payload/frame extents. The cache holds
only detached static definitions, not an old manifest or payload.

Creator-only `host_outer_zstd_decode_s` is CPU task submission/join wall time
(including raw copies). `host_outer_zstd_validate_s` and
`host_outer_zstd_worker_decode_sum_s` sum worker durations, not critical-path time.
The `host_outer_zstd_encoded_bytes`, `decoded_bytes`, `tensors` and `frames`
counters count each reconstructed host tensor/chunk once.
`host_shared_build_s` is the same cached build duration for all consumers and must
not be summed across ranks. `host_shared_arena_bytes` is the publication's used
extent; `host_shared_capacity_bytes`, `host_shared_capacity_generation` and
`host_shared_capacity_inode` identify the retained decoded allocation.
`host_shared_mapping_reused` and `host_shared_registration_reused` describe each
rank's reuse. `host_shared_registered_bytes`/`host_shared_register_calls` count only
new registration (zero on warm reuse), while
`host_shared_registration_capacity_bytes` reports current registered capacity.
`host_shared_register_s` measures registration or its reuse check.
Creator-only `host_shared_allocation_{s,calls,bytes}` and
`host_encoded_allocation_{s,calls,bytes}` distinguish cold/growth from warm builds;
`host_encoded_capacity_bytes`/`host_encoded_capacity_generation` identify staging
capacity. `host_outer_zstd_cpu_workers` records the creator pool size.

These costs occur outside explicit scheduler pause but can contend with serving.
Release/view cleanup runs on the session executor before later preparation;
unregistration occurs only on capacity replacement or backend teardown. The manual
two-process oracle exercises cold, fitting and growing updates for both 4/8-worker
pools: exact asynchronous H2D bytes, unchanged warm VA with zero register calls,
new inode/registration on growth and final disposal. Its smaller capacity alignment
keeps the oracle bounded; full-model measurements use production's 64 MiB alignment.
CPU tests cover corrupt data, creator deduplication, host-union binding, retained
bytes, worker draining, nonreusable aborts, stale release and scheduler/FIFO ordering.

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
