# GPU delta receiver tests

The experimental receiver applies canonical XOR deltas directly to existing
execution-layout buffers. Static bundled MTP layer weights remain unchanged; its existing shared embedding/head buffers continue to follow the target. It retains the disk update API independently. Pure
Torch byte transforms cover CuTe DSL NVFP4 expert layouts and BF16 dense storage;
the standalone MegaMoE transform helper does not admit an integrated MegaMoE
runtime on this branch.

```bash
python -m pytest -q \
  test/registered/unit/weight_sync/test_gpu_delta_layout.py \
  test/registered/unit/weight_sync/test_gpu_delta_payload.py \
  test/registered/unit/weight_sync/test_gpu_delta_session.py
python -m pytest -q test/manual/weight_sync/test_gpu_delta_codec.py
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
scale buffers with the existing SGLang/FlashInfer loader helpers.

Preparation checks compressed artifact SHA-256 on CPU, pins immutable encoded
buffers and constructs descriptors without reading weights or stopping serving.
Preparation reserves the largest required encoded/decoded tensor arenas. During
apply, each required matrix tensor is uploaded from pinned host memory, decoded and
applied before reusing those arenas. There is no staging selector or full-publication
HBM copy. Every retained changed matrix frame is Snappy, including inputs whose
compressed representation expands; there is no raw-frame fallback. After every original rank is prepared, the coordinator retracts generation and waits for actual
`QUIESCED` receipts. Only then does `update_weights_from_delta` mutate weights.
All original engines must commit before any resumes. A failure after possible
mutation poisons the session and requires a fresh engine; no rollback is promised.

`WEIGHT_DELTA_CODEC=snappy-zstd` is the sole contract and the default. The
receiver freezes it at backend admission, advertises it in its participant plan,
and rejects any unsupported value. The sender and receiver must agree before a
publication; manifest fields cannot override the launch constraint. Legacy codec
values and removed encoder/outer selectors have no migration layer.

Protocol 4 carries `codec="snappy-zstd"` and explicit `frame_bytes` (64 KiB or
1 MiB). Matrix frames contain only input/output offsets and lengths, without
redundant codec/file fields. Each natural tensor's outer descriptor names one
immutable owner file and independent Zstd chunks of at most 1 MiB output, exactly
covering its aligned Snappy arena. The sender computes both Snappy and outer Zstd
on GPU; the receiver always unwraps Zstd on CPU directly into final pinned tensor
buffers, then streams tensor Snappy bytes for hardware decoding and in-place apply.
There is no GPU outer decoder, legacy protocol or automatic fallback.

Owner files are SHA-256 checked once into pageable memory. The preparation worker
unwraps only local tensors using one reused Zstd context and caches tensors shared
by multiple local bindings. It validates chunk/frame extents, alignment, bounded
window/output and exact decoded length before reporting `PREPARED`. GPU Zstd may
omit content size; the bounded decoder output remains mandatory. Invalid envelopes
fail before any model mutation. Full Snappy HBM residency is not supported.

Canonical rank-0/rank-1 tensors instead negotiate `raw_bytes`: complete target
values with no XOR, frames or compression envelope. Unchanged values omit their
payload. Preparation packs changed local scalars/vectors into one aligned pinned
arena, uploads it once and performs any BF16-to-FP32 norm conversion. During the
pause, dtype-grouped `torch._foreach_copy_` updates existing buffers before matrix
application; derived NVFP4 scales refresh afterward. Late decoder failure retains
the same poisoned-session behavior. The matrix streaming path is unchanged.

`raw_bytes` counts direct payload bytes; `raw_h2d_bytes` includes arena alignment.
`host_raw_pack_s` is preparation CPU packing; direct H2D also occurs before pause.

`host_payload_read_sha256_s` reports wrapped owner-file loading/verification;
`host_outer_zstd_validate_s`, `host_outer_zstd_pin_allocate_s` and
`host_outer_zstd_decode_s` separate envelope validation, pinned allocation and CPU
decode. Encoded/decoded byte and tensor counts use the same `host_outer_zstd_`
prefix and count each reconstructed local tensor once. `host_outer_zstd_frames`
counts outer chunks. These preparation costs
are outside the explicit scheduler pause. Background work and pre-pause status
handlers can still contend with serving; the pause metric does not measure that
interference. Each rank still reads the small outer files;
there is no cross-process shared pinned-memory cache. The CPU tests include
exact reconstruction, every truncation point, trailing/concatenated frames,
checksum corruption, protocol mismatch, local-only preparation and buffer reuse.

Runtime updates do not hash weights or build a custom compiled extension. Exact
weight-content comparisons are confined to tests. State/version checks prevent
replay but do not prove weight equality. Unsupported layouts and previously
mutated baselines are rejected, so a new stream starts with a freshly loaded
engine.

Admission also reuses the ordinary updater's shared CUDA IPC weight-cache and
HPC-Ops derived-weight-cache exclusions before creating a delta plan or session.

The paired Miles feature contains `tests/manual/bench_gpu_delta.py`, which launches
one EP8 engine and measures the snappy-zstd contract using persistent
altered checkpoint and publications. `WEIGHT_DELTA_TIMING=1` enables
phase events without synchronizing every tensor; correctness comparisons stay
outside timed updates.

Each original-rank receipt includes `scheduler_timing` on that process's
`monotonic_ns` clock. `blocked_s` runs from the scheduler's pause flag, before
the existing reader fence, until its resume clears that flag. It includes
retraction, cache flush, cohort waits, apply and commit coordination; it excludes
background preparation and post-resume cleanup. Open or failed intervals retain
null `resumed_ns` and `blocked_s`. Resume responses carry the completed receipts,
so measurement needs no extra synchronization or status RPC. This measures
scheduler blocking, not GPU idle time, HTTP latency or first-token recovery.
