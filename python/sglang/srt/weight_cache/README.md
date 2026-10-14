# Experimental DeepSeek V4.1 recovery

This extension lets a replacement worker import the daemon's processed DeepSeek
V4.1 weights and derived model state. GPU weights use Torch CUDA IPC. Host Engram
storage uses shared memfds by default, or explicitly selected CUDA `HOST_NUMA`
VMM allocations. The daemon must remain alive; this does not recover a failed GPU,
a failed daemon, or in-flight request/KV state.

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

### Complete main-model weight reload time

Each row compares the same Engram placement; all numbers cover the entire main
model, including Engram and post-load state, rather than an isolated mapping call.

| Configuration | Before | After | Speedup |
| --- | --- | --- | --- |
| GPU-resident Engram: disk reload → resident CUDA IPC | 50.56 s | 1.02 s | 49.6× |
| Shared host Engram: disk reload → resident IPC + memfd | 85.45 s | 14.44 s | 5.9× |
| Per-rank host Engram: existing memfd recovery → HOST_NUMA VMM | 4.77 s | 3.36 s | 1.42× |

GPU disk reload uses the median of two final matched runs; other recovery values
are medians of three runs. Shared host disk reload has one sample. These are
separate experiments and comparisons; the shared-host disk result is not a
baseline for the per-rank VMM result. Optimized workers import an already loaded
daemon; initial daemon loading is excluded.

### Time until the service is usable

With GPU-resident Engram, complete process-launch to healthy-service time changed
from **95.09 s to 36.53 s** (median), a **2.60×** speedup. Compilation/autotuning
caches were warm for both sides. Weight recovery is approximately one second,
but communication, KV and runtime initialization still keep reconstruction in
the tens of seconds.

With per-rank host Engram, VMM clients took approximately **40.53 s** to polled
readiness, versus **45.54 s** for existing memfd clients (medians). This timer
starts after the launch helper returns; exact process-launch to ready time was
not measured for this comparison.

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
