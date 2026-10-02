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
`zstandard` and `python-snappy`. It compares both GPU decoders with identical
known input bytes, reusing a bounded encoded tensor buffer. No malformed compressed
streams are sent to nvCOMP. Missing hardware or dependencies fail the manual
suite. The registered `test_gpu_delta_layout_cuda.py` compares layouts and derived
scale buffers with the existing SGLang/FlashInfer loader helpers.

Preparation checks compressed artifact SHA-256 on CPU, pins immutable encoded
buffers and constructs descriptors without reading weights or stopping serving.
Preparation reserves the largest required encoded/decoded tensor arenas. During
apply, each required matrix tensor is uploaded from pinned host memory, decoded and
applied before reusing those arenas. There is no staging selector or full-publication
HBM copy. Snappy and Zstd profiles may contain raw frames when compression expands
individual inputs. After every original rank is prepared, the coordinator retracts generation and waits for actual
`QUIESCED` receipts. Only then does `update_weights_from_delta` mutate weights.
All original engines must commit before any resumes. A failure after possible
mutation poisons the session and requires a fresh engine; no rollback is promised.

Snappy matrix deltas use protocol 3 with a CPU Zstd envelope around each tensor's
Snappy bytes (`snappy-independent-*-zstd-v1`). CPU and GPU producers emit the same
format; plain Snappy publications are rejected. Protocol 2 retains the ordinary
Zstd transport (`zstd-independent-*-v1`). The immutable manifest selects the
decoder, with no envelope setting or automatic fallback. Snappy owner files are
SHA-256 checked once into pageable memory. The background preparation worker
unwraps only local tensors directly into their final pinned buffers, using one
reused Zstd context and caching a tensor shared by multiple local bindings.
It validates exact frame extent, content size, inner offsets and output length
before reporting `PREPARED`; malformed envelopes fail before any model mutation.
Protocol 4 additionally admits GPU-produced outer Zstd chunks
(`snappy-independent-*-gpu-zstd-v1`). Each natural tensor remains one descriptor;
its independent chunks have at most 1 MiB output and exactly cover its aligned
inner Snappy arena. The receiver always unwraps these on CPU directly into that
same final pinned tensor buffer, with one reused Zstd context and no intermediate
Snappy copy. It then uses the unchanged per-tensor H2D/hardware-Snappy apply path.
There is no GPU outer decoder or full-Snappy HBM residency option on the receiver.
Only the sender chooses the opt-in `WEIGHT_DELTA_SNAPPY_OUTER=gpu` path.
GPU chunks may omit content size, but bounded window/output, exact encoded frame
extent, and exact reconstructed size are checked before accepting the result.

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
counts outer chunks (one per natural tensor for protocol 3). These preparation costs
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
one EP8 engine and measures streaming Zstd and wrapped Snappy with the same persistent
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
