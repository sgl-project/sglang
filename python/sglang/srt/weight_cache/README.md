# Experimental DeepSeek V4.1 recovery

This extension extends the existing Weight Cache Daemon feature introduced in
[#27139](https://github.com/sgl-project/sglang/pull/27139) and described in
[Fast Engine Recovery: Sub-Second Engine Restart for SGLang via Weight Cache Daemon](https://www.lmsys.org/blog/2026-08-21-sglang-fast-recovery)
to DeepSeek V4.1, with Engram in either GPU or host memory.

The article addresses the repeated checkpoint I/O and weight processing paid on
every engine restart, which prolong service outages during worker failures and
rolling updates. It retains processed weights in a persistent daemon and lets
replacement engines reuse them through CUDA IPC. This extension builds on that feature
and addresses the gaps that prevent the same recovery path from working for V4.1.

**Why this extension is needed:**

- **V4.1 needs more than tensor import.** Its FP8 dense / FP4 expert path has
  post-load layouts, quantization metadata and cached tensor references that a
  replacement engine must restore consistently. Related failures are documented
  in [#39794](https://github.com/sgl-project/sglang/pull/39794). This extension restores
  the processed state, rebinds mHC and TopK references, and checks backend/layout
  compatibility instead of merely relaxing the IPC admission check or repeating
  transforms on shared weights.
- **Host Engram needs its own resident recovery path.** GPU tensor IPC cannot
  share its host backing. This extension preserves daemon-owned host tables and imports
  them in replacement workers; optional HOST_NUMA VMM reuses allocation handles
  to avoid repeated `cudaHostRegister` on unchanged tables.
- **Faster checkpoint reads still pay checkpoint I/O on restart.** The
  InstantTensor and positioned-read PRs below accelerate loading from storage.
  This extension complements them by reusing already processed GPU weights and host
  Engram after a worker restart, while the daemon survives. Their cold-load
  optimizations can still benefit initial daemon startup, but are not included
  in this patch.

The result is a V4.1-specific implementation of the article's resident-weight
recovery path, plus an experimental prewarmed standby for its active-standby
scenario. Complete weight reload and service recovery results are reported below;
full long-context acceptance and other remaining gates are still pending.

### Related work

- [#27139 — Support fast engine recovery through weight cache](https://github.com/sgl-project/sglang/pull/27139): the original feature this extension extends.
- [#38306 — Support post-load lifecycle and MXFP4 IPC](https://github.com/sgl-project/sglang/pull/38306): related model-support work describing K3/DSV3 validation; this extension adds V4.1-specific state restoration and host Engram recovery.
- [#39794 — Reject routed MXFP4 experts for IPC](https://github.com/sgl-project/sglang/pull/39794): documents why accepting a base FP8 configuration is insufficient for routed-expert post-load layouts.
- [#40453 — Add InstantTensor loader support](https://github.com/sgl-project/sglang/pull/40453), and [#42919 — Load V4.1 with the Engram host table](https://github.com/sgl-project/sglang/pull/42919): accelerate checkpoint loading while keeping host-resident tables off the GPU stream.
- [#42920 — Parallel positioned reads for Engram host tables](https://github.com/sgl-project/sglang/pull/42920): accelerates storage-to-host loading with parallel `preadv`; this extension addresses resident recovery and mapping reuse instead.

## Host Engram configuration

Use Linux/NVIDIA CUDA with the repository's supported dependencies. The optional
`cuda_vmm` backend additionally requires `cuda.bindings.driver`, device support
for HOST_NUMA VMM and POSIX FD export/import. Unsupported devices fail explicitly.

Set the same environment, checkpoint, source, backend choices, and model overrides
in the daemon and every client. For example, for a local TP4/EP1 deployment:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank

common=(
  --model-path "$MODEL_PATH" --trust-remote-code
  --tp 4 --attn-dp-size 1 --ep-size 1
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4
  --attention-backend dsv4
  --json-model-override-args '{"vision_n_layers":0}'
  --model-loader-extra-config '{"dsv41_weight_cache":true,"dsv41_host_table_backend":"cuda_vmm"}'
  --context-length 4096 --max-total-tokens 8192 --max-running-requests 4
  --chunked-prefill-size 4096 --max-prefill-tokens 4096
  --mem-fraction-static 0.65 --random-seed 42
  --disable-cuda-graph --disable-prefill-cuda-graph --disable-overlap-schedule
)
# Terminal 1: load and retain the processed model.
python -m sglang.srt.weight_cache.daemon "${common[@]}"
# Terminal 2: import resident weights; wait for the daemon to be ready first.
python -m sglang.launch_server "${common[@]}" \
  --weight-cache-mode client --host 127.0.0.1 --port 22941 --nccl-port 22942
```

These commands describe the tested configuration; capacity and backend support
must be checked for your hardware. Omit the host-table environment settings for
GPU-resident Engram. Select `"dsv41_host_table_backend":"memfd"` for the comparison
backend. The opt-in `dsv41_weight_cache` flag is required for V4.1 IPC recovery.

VMM allocates NUMA host storage once in the daemon, exports its allocation FD,
and imports it in each worker. Clients expose a CUDA tensor alias to physical
host memory without copying the table or calling `cudaHostRegister` again. Each
process still pays for its own GPU access mapping. CPU checkpoint loading and
initial allocation remain cold-start costs.

The post-load manifest restores tensor aliases, cached tensor references,
quantization flags, mHC seams and TopK correction bias references. The contract
and layout fingerprint reject incompatible backend/configuration/source choices.
Host descriptors validate logical size, padded allocation size, NUMA placement,
and tensor bounds. Host-table objects own imported mappings, CUDA handles and
FDs; retire tensor/kernel users before closing these objects.

When upgrading this implementation, restart the weight daemon before importing
its cache with upgraded clients. The compatibility fingerprint includes the
daemon, IPC loader, model-load orchestration and weight-processing source files;
a client rejects a cache produced by different recovery code. This intentionally
requires a fresh daemon load after source changes.

## Prewarmed standby workers

`python -m sglang.srt.weight_cache.standby --config config.json` manages two
independent client process groups. A worker becomes ready only after HTTP health
and a generation warmup. It promotes an already ready worker after active-worker
failure and replenishes the failed worker in the background. Manage the weight
daemon separately.

A configuration has two `workers`, each with a distinct `name`, loopback HTTP
`url`, and a `command` argument list that launches its client server. Optional
worker `env` entries extend the inherited environment. Use distinct HTTP/NCCL
ports and budget both workers' private KV pools. Other settings include `host`,
`port`, `log_dir`, `warmup`, `health_path`, `health_interval`, `health_timeout`,
`poll_interval`, `startup_timeout`, `warmup_timeout`, and `restart_backoff`.
Inspect `/recovery/status` for readiness and promotion events.

Only stateless generation/OpenAI inference routes and model metadata are exposed.
Requests containing `session_params` are rejected. Failed requests are never
replayed; truncated streams are closed. HTTP health does not prove GPU forward
progress. Two workers sharing the same GPU do not isolate device failures, and
only one worker remains available while the standby is rebuilding.

## Validation and limitations

The experiment used four NVIDIA L20D GPUs, TP4/EP1/attention-DP1,
DeepSeek-V4.1-Flash native FP8 dense/FP4 expert weights, context 4096 and a maximum
of four requests. CUDA Graph and overlap were disabled.

### Engram in GPU memory

Disk reload compared with resident CUDA IPC recovery:

| Metric | Before | After | Speedup |
| --- | --- | --- | --- |
| Complete main-model weight reload | **50.56 s** | **1.02 s** | **49.6×** |
| Complete service recovery: process launch → healthy service | **95.09 s** | **36.53 s** | **2.60×** |

Values are medians: two final matched disk runs and three recovery runs.
Compilation/autotuning caches were warm. The weight reload covers the entire
main model, including Engram and post-load state; service recovery also includes
communication, KV and runtime initialization.

### Engram in shared host memory

| Host layout and comparison | Metric | Before | After | Speedup |
| --- | --- | --- | --- | --- |
| Shared table: disk reload → resident IPC + memfd | Complete main-model weight reload | **85.45 s** | **14.44 s** | **5.9×** |
| Per-rank shards: existing memfd recovery → HOST_NUMA VMM | Complete main-model weight reload | **4.77 s** | **3.36 s** | **1.42×** |
| Per-rank shards: existing memfd recovery → HOST_NUMA VMM | Launch-helper return → polled service readiness | **45.54 s** | **40.53 s** | **1.12×** |

Host recovery values are medians of three runs; the shared-table disk baseline
has one sample. Shared tables and per-rank shards are separate experiments:
85.45 s is not a matched baseline for 3.36 s. Exact process-launch to healthy
service time was not measured for the host VMM comparison; its readiness timer
starts after the launch helper returns.

Both placements require an already loaded, surviving daemon for optimized
recovery; initial daemon startup is excluded.

Across 30 prompt comparisons, all text and input/output token IDs matched.
24/30 also matched logprobs exactly. The 1401-token case differed even in a memfd
same-service repeat; strict long-context numerical acceptance remains unresolved.
8/51/203 GB allocation probes matched an independent CPU gather oracle. Four-rank
shared-table gather and CUDA Graph replay passed at component scope only; full
model shared layout and full model CUDA Graphs were not validated.

Earlier memfd standby fault injection measured 205–422 ms from launcher SIGKILL
to the first response token. This is takeover by an already warmed worker, not
VMM worker reconstruction. VMM plus standby has not had a combined full-model
fault-injection run.

The experiment used driver 580.105.08, CUDA driver API 13.0, Torch 2.13.0+cu130,
FlashInfer 0.6.18 and sglang-kernel 0.4.7. The frozen source expected kernel 0.4.9;
its version check was explicitly bypassed. This dependency mismatch is an open
validation gap, not a recommended deployment configuration.

Full-capacity throughput, speculative/draft execution, other topologies,
multi-node operation, daemon loss and in-flight state recovery are unvalidated.
This extension does not implement the parallel checkpoint-read optimization in
SGLang #42920. Keep this feature experimental until the remaining gates pass.

The CPU contract tests can be run individually:

```bash
for test in test_host_vmm_weight_cache test_host_engram_weight_cache \
            test_dsv41_weight_cache_state test_weight_cache_protocol test_warm_standby; do
  python "test/registered/unit/model_loader/${test}.py"
done
```
