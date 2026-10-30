# LMCache Connector for SGLang

This document describes how to use LMCache as KV Cache Management Backend for SGLang engine.
For more details about LMCache, please refer to: https://lmcache.ai

## Install LMCache

### Method 1: with pip

```bash
pip install lmcache
```

### Method 2: from source

Clone LMCache project:

```bash
git clone https://github.com/LMCache/LMCache
```

Install:

```bash
cd LMCache
pip install -e . --no-build-isolation
```


## Use LMCache

SGLang uses `LMCacheUnifiedRadixCache` with LMCache's multiprocess connector.
The standalone LMCache daemon owns the external cache and can survive SGLang
process restarts. Daemon host and port come from the LMCache YAML config.

Terminal 1 â€” start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

Use the bundled `example_config_mp.yaml` (or any YAML setting `mp_host` / `mp_port`):

Terminal 2 â€” start SGLang:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_mp.yaml
```

For full LMCache config options see https://docs.lmcache.ai/api_reference/configurations.html.

## Engine-driven shared-memory transfers

Engine-driven transfers keep GPU copies in the SGLang worker. With a shared
host IPC namespace, a named SHM pool lets the worker copy KV directly to/from
the daemon's CPU buffers, avoiding bulk inline message payloads. The daemon
and SGLang processes must see the same `/dev/shm`, even when they run in
different containers. The pool must use non-lazy allocation.

Start the daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 16 --no-l1-use-lazy \
  --chunk-size 256 --eviction-policy LRU \
  --supported-transfer-mode engine_driven \
  --shm-name sglang_kv_pool
```

Connect SGLang using `example_config_mp_engine_driven.yaml` from this directory:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --enable-lmcache \
  --lmcache-config-file example_config_mp_engine_driven.yaml \
  --enable-deterministic-inference --attention-backend triton \
  --mem-fraction-static 0.1 --max-total-tokens 65536 --context-length 4096 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
```

Confirm the SGLang startup log reports `Creating EngineDrivenContextShm`.
`Creating EngineDrivenContextInline` means the shared-memory path was not
selected. Omitting the daemon's `--shm-name` selects inline engine-driven
transfers with the same client configuration.

Validation used MI355X, ROCm 10, PyTorch 2.11 and LMCache
[a005bb1](https://github.com/LMCache/LMCache/tree/a005bb19715d1e7a32372e6c4c850b73366cbcc5).
Engine-driven SHM preserved the restored 1024-token prefix byte-for-byte
across 56 KV tensors on each of two TP ranks. Rerun the checks when changing
the engine or driver versions.

### Measure external reuse

From the repository root, benchmark a running isolated server:

```bash
python test/manual/mem_cache/benchmark_lmcache_reuse.py \
  --base-url http://127.0.0.1:30000 --model Qwen/Qwen3-0.6B \
  --input-len 3073 --output-len 64 --concurrency 8 \
  --warmup 4 --waves 16 --output reuse.json
```

The benchmark uses separate cache salts, primes external KV, and flushes GPU
KV between waves. It checks output equality and full external-only hits. The
server's GPU token pool must fit the concurrent requests; the example above
uses 65536 tokens. Its wave throughput includes prefill and decode but excludes
priming and cache flushes, so it is not steady-state production throughput.
Use `--prefill-url` with a PD router to flush the prefill server. For a separate
recomputation run, use `--expected-storage-hit 0` against a server that bypasses
external loading, keeping the model and other serving settings fixed.

On the validated stack, two runs per transport with the workload above gave
85.61 output tokens/s for inline transfers and 592.27 output tokens/s for SHM,
aggregated over measured wave durations. Median TTFT was 5229.15 ms and
128.54 ms respectively. All 512 measured requests returned identical tokens
across transports, with 3072 external hit tokens and zero device hits each.
Both paths used the same non-lazy 16 GiB L1 pool, ordered inline/SHM/SHM/inline.
These results compare engine-driven transports on this workload; they do not
establish gains over IPC transfers or another serving engine.

## Disaggregated prefill

Enable LMCache on the prefill server with `--disaggregation-mode prefill`
alongside the usual PD transfer configuration. The prefill server retrieves
reusable KV from LMCache and sends it to decode through the configured PD
backend. Run the decode server without `--enable-lmcache`.

The standalone LMCache service can retain KV across prefill server restarts.
Keep the model, KV dtype, tensor-parallel layout, and LMCache configuration
consistent between runs. Early prefix sends wait for asynchronous LMCache
loads on the forward stream, including with `--disable-overlap-schedule`.

PD decode, DP attention, DCP, and speculative decoding remain unsupported
with `--enable-lmcache`.

For strict output comparisons, enable `--enable-deterministic-inference` with
a supported attention backend on both PD servers. Prefix reuse changes the
prefill computation shape; non-deterministic kernels can produce different
tokens even when the restored prefix KV is byte-identical.

### Validate PD reuse

With SGLang, LMCache, and the PD transfer backend installed in the same active
Python environment, run the manual end-to-end test from the repository root:

```bash
LMCACHE_MP_TRANSFER_MODE=engine_driven \
  python test/manual/mem_cache/test_disaggregation_lmcache.py -v
```

The test starts its own daemon and prefill/decode servers. It compares greedy
tokens against a no-LMCache baseline, verifies external hits after a local
cache flush and both engine restarts, and checks cache-salt isolation.
It requires two GPUs, or four with `SGLANG_TEST_LMCACHE_TP_SIZE=2`.
Set `LMCACHE_MP_TRANSFER_MODE=lmcache_driven` to exercise IPC transfers and
`SGLANG_TEST_LMCACHE_DISABLE_OVERLAP=1` to disable overlap scheduling.
Set `SGLANG_TEST_LMCACHE_SHM=1` with engine-driven mode to validate the named
SHM pool. The test creates its own unique pool name.
The default model is `Qwen/Qwen3-0.6B`; override it with
`SGLANG_TEST_LMCACHE_MODEL`. `SGLANG_TEST_LMCACHE_BASE_PORT` (default `31500`)
reserves the next seven ports for this test. This is a functional test, not a
latency or throughput benchmark.

On the tested MI355X / ROCm 10 stack, TP=2 passed this test with engine-driven
LMCache transfers and `MC_FORCE_TCP=1` for Mooncake. Named SHM also passed
with Qwen3-0.6B and Qwen3-8B, including both engine restarts and strict
16-token output parity. These are functional smoke tests rather than a model
accuracy benchmark. The default same-host
Mooncake HIP path failed cold-request token parity, and TP=2 LMCache-driven
IPC retrieval faulted even in colocated SGLang. These paths need separate
validation; the TCP result does not establish their correctness or RDMA
performance.
