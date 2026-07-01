# FlexKV ↔ sglang integration

A `RadixCache` subclass that routes sglang's host-tier KV cache through a
FlexKV [`KVManager`](https://github.com/taco-project/FlexKV) (CPU / SSD /
Remote offload). Same integration pattern as
[`LMCRadixCache`](../lmcache/README.md): `FlexKVRadixCache` overrides
`match_prefix` / `init_load_back` / `cache_finished_req` / `evict`; a
`FlexKVConnector` façade talks to `KVManager`, `KVTPClient`, and a
3-axis (PP × CP × TP) sync context.

---

## Quick start (single H20, single GPU, Qwen3-8B)

This walks through everything the verification on H20-GPU-11 actually
exercised. Adjust paths / model / GPU as needed.

### 1. Prereqs

* `lmsysorg/sglang:dev` (or any sglang container with CUDA 12.x + torch 2.10+).
* This sglang fork (branch `feat/flexkv-main-connector`) and FlexKV
  (branch `main`) checked out somewhere reachable from the container
  — e.g. `/raid/fly/sglang-connector-dir/{sglang,FlexKV}`. Verified
  against FlexKV main at `aa74e39` (PR #184); older commits down to
  the layerwise integration also work.

### 2. Start a container with both repos mounted

```bash
docker run -d --name flexkv-sglang \
  --gpus all --ipc=host --network host \
  --shm-size=32g --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v /raid/fly:/raid/fly \
  --workdir /raid/fly/sglang-connector-dir \
  --entrypoint "" \
  lmsysorg/sglang:dev sleep infinity

docker exec flexkv-sglang bash -c "
  apt-get update -qq &&
  apt-get install -y numactl libnuma-dev libxxhash-dev liburing-dev cmake ninja-build
"
```

### 3. Install sglang fork (editable) + FlexKV

```bash
docker exec flexkv-sglang bash -c '
  set -e
  git config --global --add safe.directory "*"

  # sglang fork: install in editable mode, replacing the prebuilt sglang
  cd /raid/fly/sglang-connector-dir/sglang
  pip install --no-deps -e python

  # FlexKV: pin to main, init the xxHash submodule, debug C++ build.
  cd /raid/fly/sglang-connector-dir/FlexKV
  git checkout main && git pull --ff-only
  git submodule update --init third_party/xxHash
  pip install -q cython ninja pybind11
  FLEXKV_ENABLE_METRICS=0 bash build.sh --debug

  # Smoke check
  python3 -c "
import sglang, flexkv
from flexkv.kvmanager import KVManager
from sglang.srt.mem_cache.storage.flexkv import flexkv_comm
from sglang.srt.mem_cache.registry import registered_radix_cache_backends
import sglang.srt.mem_cache.storage.flexkv  # registers
print(\"flexkv ok\", flexkv.__file__)
print(\"sglang ok\", sglang.__file__)
print(\"registered backends:\", registered_radix_cache_backends())
"
'
```

If the build hangs on `pip install sglang-kernel`, see
[Troubleshooting](#troubleshooting).

### 4. Minimal FlexKV YAML

```yaml
# /raid/fly/sglang-connector-dir/flexkv_min.yaml
cpu_cache_gb: 16
```

That's enough to enable a 16 GiB CPU offload pool. See
[`example_config_mp.yaml`](example_config_mp.yaml) for SSD / remote /
distributed knobs.

### 5. Launch the server (MP / synchronous mode)

```bash
docker exec -d flexkv-sglang bash -c '
  cd /raid/fly/sglang-connector-dir
  CUDA_VISIBLE_DEVICES=0 \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  python3 -m sglang.launch_server \
    --model-path /raid/fly/model/Qwen3-8B \
    --port 30000 --tp-size 1 \
    --enable-flexkv \
    --flexkv-config-file /raid/fly/sglang-connector-dir/flexkv_min.yaml \
    --mem-fraction-static 0.45 --max-running-requests 8 \
    > /tmp/sglang.log 2>&1
'
```

`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` bypasses the prebuilt
`sglang-kernel` version assertion (the `lmsysorg/sglang:dev` image ships
0.4.2.post2; main expects ≥ 0.4.3). Not a FlexKV-specific issue;
remove when the container image is refreshed.

Wait ~2 min for the model load + CUDA graph capture. Confirm with:

```bash
docker exec flexkv-sglang bash -c '
  grep -E "fired up|Connector ready" /tmp/sglang.log | tail -2
'
```

Expected (key lines):

```
[FlexKV] Connector ready ...: layerwise=False, prefetch=False
The server is fired up and ready to roll!
```

### 6. Send a request and observe a cache hit

```bash
# First call: priming — fresh prefill, FlexKV stores the prefix.
docker exec flexkv-sglang bash -c '
  curl -s http://127.0.0.1:30000/generate -X POST \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"The capital of France is\",
         \"sampling_params\": {\"max_new_tokens\": 5, \"temperature\": 0}}"
'

# Flush the GPU radix (FlexKV CPU pool keeps the data) and re-send.
docker exec flexkv-sglang bash -c '
  curl -s http://127.0.0.1:30000/flush_cache -X POST
  curl -s http://127.0.0.1:30000/generate -X POST \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"The capital of France is\",
         \"sampling_params\": {\"max_new_tokens\": 5, \"temperature\": 0}}"
'
```

Look at the second response's `meta_info`:

```json
"cached_tokens": 4,
"cached_tokens_details": { "device": 0, "host": 4 },
```

`host: 4` confirms the bytes came back from FlexKV's CPU pool. The
server log should also show a matching D2H/H2D bandwidth line:

```
[FLEXKV] ... H2D transfer request: N finished transfer data size: 0.0xx GB ... 30+ GB/s
```

### 7. Layerwise mode

Add `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1` before `python3 -m
sglang.launch_server`. Everything else is identical. On the second
request you'll see `cached_tokens_details: {"device": N, "host": 0}`
(in IP mode the load happens inside `match_prefix` so sglang accounts
for it as device-side) and a log line `LAYERWISE transfer request: N
finished ...`. The startup log will also include
`[FlexKV] Eventfd handshake complete ... counters=3 layers=<N>`.

---

## Correctness verification

Numerical match against a no-FlexKV baseline (greedy decoding,
deterministic). Scripts are in this repo's testing notes; the canonical
two are reproduced below.

```bash
# Phase 1: capture the no-FlexKV baseline.
docker exec -d flexkv-sglang bash -c '
  CUDA_VISIBLE_DEVICES=0 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  python3 -m sglang.launch_server \
    --model-path /raid/fly/model/Qwen3-8B --port 30000 --tp-size 1 \
    --mem-fraction-static 0.45 > /tmp/sglang.log 2>&1
'
# ... wait until ready ...
docker exec flexkv-sglang python3 /raid/fly/sglang-connector-dir/sglang/python/sglang/srt/mem_cache/storage/flexkv/verify_outputs.py --phase baseline
docker exec flexkv-sglang bash -c "pkill -9 -f launch_server; sleep 3"

# Phase 2: relaunch with --enable-flexkv and compare.
docker exec -d flexkv-sglang bash -c '
  CUDA_VISIBLE_DEVICES=0 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  python3 -m sglang.launch_server \
    --model-path /raid/fly/model/Qwen3-8B --port 30000 --tp-size 1 \
    --enable-flexkv --flexkv-config-file /raid/fly/sglang-connector-dir/flexkv_min.yaml \
    --mem-fraction-static 0.45 > /tmp/sglang.log 2>&1
'
# ... wait until ready ...
docker exec flexkv-sglang python3 /raid/fly/sglang-connector-dir/sglang/python/sglang/srt/mem_cache/storage/flexkv/verify_outputs.py --phase test
```

Expected last line: `Total mismatches: 0`. Each prompt is run twice
(R1 fresh / R2 after `flush_cache`); both R1 and R2 outputs must
byte-equal the baseline.

Repeat the Phase-2 launch with `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1`
to validate the layerwise path.

---

## Selecting the backend

Two equivalent CLI flags:

```bash
# Auto-selection chain (matches --enable-lmcache style)
python3 -m sglang.launch_server --enable-flexkv \
  --flexkv-config-file /path/to/flexkv_config.yaml ...

# Explicit registry path
python3 -m sglang.launch_server --radix-cache-backend flexkv \
  --flexkv-config-file /path/to/flexkv_config.yaml ...
```

Either flag also sets `FLEXKV_CONFIG_PATH` so you can omit
`--flexkv-config-file` and configure FlexKV purely through env vars.

---

## Modes

### MP (synchronous, default)

* `match_prefix` calls `FlexKVConnector.lookup_kv` only.
* When `host_hit_length > 0`, the scheduler later calls
  `init_load_back`, which allocates the uncached slots and fires
  `retrieve_kv` (FlexKV `launch` + `wait`).
* `cache_finished_req` runs `put_match` + `launch` and stashes the
  in-flight FlexKV task id. Source-node lock is held until
  `check_completed_stores` (called from `check_hicache_events` /
  `evict`) signals completion.

This is the path you'll use under any non-trivial deployment topology
(DP > 1, multi-instance, multi-node, ...).

### IP / layerwise (`FLEXKV_ENABLE_LAYERWISE_TRANSFER=1`)

* `match_prefix` allocates the uncached slots and fires
  `start_load_kv_layerwise` immediately.
* A `FlexKVLayerDoneCounter` is registered onto sglang's KV pool via
  `register_layer_transfer_counter`; the per-layer hook blocks each
  forward layer on its own eventfd until the FlexKV transfer worker
  signals the layer is staged.
* Layerwise mode requires the FlexKV transfer worker's UDS socket
  (`/tmp/flexkv_layerwise_eventfd.sock` by default) to be reachable —
  the connector handshakes with it at startup. The socket path is
  computed by FlexKV's `build_layerwise_eventfd_socket_path` from the
  same dp/pp/instance settings, so configuration is taken care of as
  long as you launch FlexKV consistently.

---

## Files

* `flexkv_radix_cache.py` — `FlexKVRadixCache(RadixCache)`. Overrides
  `match_prefix`, `init_load_back`, `cache_finished_req`, `evict`,
  `check_hicache_events`, `reset`.
* `flexkv_connector.py` — `FlexKVConnector`. Owns the `KVManager`,
  `KVTPClient`, and the cross-rank sync context. Public methods:
  `lookup_kv`, `retrieve_kv`, `start_load_kv_layerwise`, `store_kv`,
  `check_completed_stores`, `prefetch_async`, …
* `flexkv_comm.py` — `FlexKVComm` (3-axis PP × CP × TP sync built on
  torch.distributed) + the eventfd / `SCM_RIGHTS` shims used by the
  layerwise transfer UDS handshake. **`FlexKVLayerLoadingEvent` here
  carries the layerwise correctness fix** (drain stale eventfd
  signals on reset, switch `wait` to `select.select` to keep blocking
  semantics on a NONBLOCK fd).
* `__init__.py` — registers the `"flexkv"` factory with
  `sglang.srt.mem_cache.registry`.

---

## TP / PP / CP / DP

FlexKV runs one `KVManager` per DP route (=
`instance_id * dp_size + dp_rank`). Every other rank in the same
fan-out is the "sync follower" — `FlexKVComm` broadcasts the
leader's lookup / store decisions via gloo CPU groups so non-leader
ranks know which task ids and slot mappings to use.

Supported:

* **TP** (any size) — typical sglang topology.
* **DP** (`dp_size > 1`) and multi-instance — FlexKV automatically
  switches its `KVManager` to server-client mode.
* **PP** (`pp_size > 1`) — including cross-node PP. The PP receiver
  forwards its slot mappings back to FlexKV's
  `TransferManagerOnRemote` via the same ZMQ channel used for GPU
  registration.
* **CP** (`attn_cp_size > 1`) — sync handled symmetrically with TP.
* **DP attention** (`enable_dp_attention=True`) — the inner
  `attn_tp_size` is what FlexKV uses for register-side routing.

---

## Environment variables

* `FLEXKV_CONFIG_PATH` — full FlexKV YAML / JSON config (also set
  automatically by `--flexkv-config-file`).
* `FLEXKV_ENABLE_LAYERWISE_TRANSFER` — `1` to enable layerwise mode.
* `FLEXKV_LAYERWISE_EVENTFD_SOCKET` — UDS socket path (default
  `/tmp/flexkv_layerwise_eventfd.sock`); auto-suffixed per
  `(pp_rank, dp_client_id)` when those dims are > 1.
* `FLEXKV_MASTER_HOST` / `FLEXKV_MASTER_PORTS` — multi-node master
  endpoint for `TransferManagerOnRemote`. Default
  `localhost:5556,5557,5558`. With `nnodes > 1` we also fall back to
  `server_args.dist_init_addr`'s host.
* `FLEXKV_KV_CACHE_DTYPE` — override KV dtype when sglang uses
  `--kv-cache-dtype auto`.
* `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` — bypass the prebuilt
  `sglang-kernel` version assertion (not FlexKV-specific).

---

## Troubleshooting

* **`fatal: not a git repository ... third_party/xxHash`** — FlexKV's
  build.sh needs an actual git checkout for the submodule. If you
  rsync'd FlexKV without `.git/`, sync it: `rsync -az
  /path/to/FlexKV/.git/ <remote>:<dir>/FlexKV/.git/` then
  `git config --global --add safe.directory "*"`.
* **`fatal: detected dubious ownership`** — same fix:
  `git config --global --add safe.directory "*"`.
* **`xxhash.h: No such file or directory`** — submodule not init'd.
  `cd FlexKV && git submodule update --init third_party/xxHash`.
* **`dist/lease_meta_mempool.h: No such file or directory`** — your
  rsync excluded `csrc/dist/`. The directory `FlexKV/csrc/dist/` is
  source, not a build artifact; re-sync without `--exclude='dist'`.
* **`No module named 'Cython'`** — install: `pip install cython ninja pybind11`.
* **`sglang-kernel is installed with version 0.4.2.post2, which is
  less than the minimum required version 0.4.3`** — either run with
  `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` or refresh
  `pip install -U sglang-kernel`. The download is ~600 MB and can
  take a long time on slow links.
* **`cudaHostRegister failed with error code 100` (cudaErrorNoDevice)**
  — happens when the FlexKV transfer subprocess can't init CUDA on
  the assigned device. Usually a stuck previous session; restart
  the container.
* **`[FlexKV] Waiting for FlexKV ready` loops > 60 s** — the
  KVManager subprocess crashed at boot. Check `/tmp/sglang.log` for
  the actual stack (usually a CUDA-init or torch-mp issue).
* **Layerwise mode: server hangs at "Eventfd connected attempts=..."**
  — the `LayerwiseTransferWorker` hasn't started yet. Wait — it can
  take 20-30 s after `Eventfd server created`. If it never advances,
  check the FlexKV-side log lines beginning with `[LayerwiseWorker]`.

---

## Status

* MP (synchronous) path — verified end-to-end on Qwen3-8B (H20-3e):
  output byte-equal to no-FlexKV baseline across short / medium / long
  prompts. ~30–46 GB/s observed for D2H stores and ~37 GB/s for H2D
  loads.
* IP (layerwise) path — verified end-to-end with the fix in
  `flexkv_comm.py`. ~7–12 GB/s per-layer (smaller per-call payload).
* PP / CP / DP / multi-node — code paths driven by `FlexKVComm`,
  carried over from the production-validated `BaseKVConnector`
  integration. Not exercised in single-GPU smoke tests; needs a
  multi-node run before shipping.

### Known limitations

* Hybrid models (Mamba / SWA / DSV4 indexer auxiliary pools) are not
  supported through this connector — only the primary KV pool is
  hooked up. HiCache's multi-pool `batch_*_v2` interface would map
  here but requires `PoolTransfer` + `PoolHitPolicy` plumbing in
  `FlexKVConnector`.
* Write-back acks are per-request (one `dec_lock_ref` per
  `cache_finished_req`), not per-page like HiCache's
  `flush_write_through_acks`.
* `--radix-cache-backend=flexkv` and `--enable-flexkv` are
  mutually equivalent today; we don't yet emit a deprecation
  warning if both are set.

## Benchmarks

Reproducible SWE-bench harness lives under [`benchmarks/`](./benchmarks/).
It launches sglang three times against the same prompt set (baseline,
`--enable-hierarchical-cache`, `--enable-flexkv`) and prints a side-by-side
summary of TTFT / E2E / throughput.

### Setup

Model: Qwen3-8B on 1× H20. Server flags:

    --attention-backend triton --mem-fraction-static 0.32
    --max-running-requests 32 --chunked-prefill-size 16384
    --context-length 32000

Workload: 120 prompts sampled from
[`princeton-nlp/SWE-bench_Lite_oracle`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_oracle)
with input length ≤ 28k tokens (p50 = 7088, max = 27961). Two passes —
pass 1 populates the host cache, pass 2 is the measured run. `qps=2.0`,
`concurrency=24`, `max_new_tokens=32`, `temperature=0`.

### Warm-pass results

| Config | TTFT avg / p50 / p90 / p99 | E2E p50 | Throughput | Output tok/s | H2D / D2H |
| --- | --- | --- | --- | --- | --- |
| baseline | 6.86 / 8.04 / 9.88 / 10.89 s | 8.15 s | 1.86 req/s | 37.7 | — |
| `--enable-hierarchical-cache` | **0.04 / 0.04 / 0.06 / 0.06 s** | 0.23 s | 2.02 req/s | 40.8 | — |
| `--enable-flexkv`             | **0.05 / 0.05 / 0.07 / 0.08 s** | 0.24 s | 2.02 req/s | 40.8 | 86 / 155 |

Server-side (via `ReqTimeStats` in the sglang log): 76 / 76 non-EOS-immediate
warm-pass requests have `cached_input_len == input_len` for both `hicache`
and `flexkv` (100 % prefix recovery); baseline stays at ~59 tokens
(system-prompt header only). The 86 `H2D transfer` log lines under `flexkv`
confirm the CPU-tier loadbacks actually fired.

### Output correctness

Byte-level diff of generated text across 32 prompts, `temperature=0`:

* baseline: cold pass == warm pass (32 / 32; fully deterministic without cache)
* `hicache`: warm vs baseline warm — 29 / 32 identical, 3 diverge
* `flexkv`:  warm vs baseline warm — 29 / 32 identical, 3 diverge (mostly the same 3 as `hicache`)

The ~10 % divergence at `temperature=0` is the well-known KV-cache-reuse
artifact caused by floating-point non-associativity between "prefill in
place" and "load pre-computed KV" paths; it affects the mainline
`--enable-hierarchical-cache` at the same rate and is not FlexKV-specific.

### Reproducing

    cd python/sglang/srt/mem_cache/storage/flexkv/benchmarks
    export MODEL=/path/to/Qwen3-8B
    bash run_swebench.sh                    # runs baseline, hicache, flexkv
    bash run_swebench.sh flexkv             # runs one config

Outputs land in `./bench_out_swebench/<config>.{json,log,server.log}`.
The `.log` file contains the per-config summary printed above; the
`_server.log` file has `ReqTimeStats` lines suitable for the cache-hit
sanity check.
