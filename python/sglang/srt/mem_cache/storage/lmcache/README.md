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

Terminal 1 — start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

Use the bundled `example_config_mp.yaml` (or any YAML setting `mp_host` / `mp_port`):

Terminal 2 — start SGLang:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_mp.yaml
```

For full LMCache config options see https://docs.lmcache.ai/api_reference/configurations.html.

## Avoid loading small external hits

Use `--lmcache-min-retrieve-tokens N` to recompute external hits shorter than
`N` tokens. The default is `0`, preserving existing loading behavior. This is
an operator-calibrated threshold, not an online cost estimator: compare loading
and recomputation on the deployed model, hardware, KV dtype, storage tier and
workload before choosing it.

SGLang skips the lookup when even a fully cached prompt cannot reach the
threshold. Otherwise it looks up normally and applies the threshold to the
external tokens beyond the common rank-local GPU prefix. A rejected hit
releases all lookup locks before GPU allocation or transfer. The local GPU
prefix remains reusable and storing computed KV remains enabled.

For example, with `--lmcache-min-retrieve-tokens 512`, a 256-token external hit
is recomputed; a 512-token external hit can be loaded. A 768-token total hit
with 512 tokens already resident in the engine counts as only 256 external
tokens. Long prompts with short partial hits may still incur storage prefetch
cost before SGLang can decide. Recalibrate after deployment changes and check
held-out mixed-hit latency and throughput; fewer transfers alone do not prove
a performance improvement.

Run the manual GPU test to check lookup bypass, rejection of a short partial
hit on a longer prompt, and successful reuse of KV stored after recomputation:

```bash
SGLANG_TEST_LMCACHE_TP_SIZE=2 \
  python test/manual/mem_cache/test_lmcache_load_admission.py -v
```

The default is TP=1. The test owns its daemon and engine, uses engine-driven
transfers, and requires LMCache in the active Python environment. Override
`SGLANG_TEST_LMCACHE_MODEL` or `SGLANG_TEST_LMCACHE_BASE_PORT` (default `31700`)
to select the model or the four consecutive ports used by the test.

Set `SGLANG_TEST_LMCACHE_SHM=1` to run the same checks with a named, non-lazy
SHM pool. The daemon and workers must share `/dev/shm`. Calibrate the threshold
using the actual transfer path; inline and SHM can have different break-even
points.

## Decide from measured costs

Pass `--lmcache-load-cost-profile costs.json` to compare measured loading and
recomputation costs after lookup. The default is unset. The minimum-token
gate still applies independently. Profile decisions use the actual external
hit beyond the common rank-local prefix, before GPU allocation or transfer.
Declined hits follow the same lock-release and store lifecycle as the minimum
gate. No request-time file reads or additional collectives are needed.

For example, these **illustrative costs are not deployment measurements**:

```json
{
  "version": 1,
  "metric": "request_elapsed_ms",
  "min_savings": 0.05,
  "calibration": "Replace with measurements on the deployed configuration",
  "profiles": [{
    "model_name": "my-model",
    "world_size": 1,
    "chunk_size": 256,
    "lookup_tokens": 1024,
    "engine_computed_tokens": 256,
    "external_tokens": 512,
    "load_ms": 10.0,
    "recompute_ms": 8.0
  }]
}
```

Only an exact match of all six geometry fields uses a measurement. The model
name must match SGLang's model path, and world size is TP times PP. Lookup
tokens are the page-aligned prefix queried by SGLang, excluding the protected
tail; they can differ from the full input length. External tokens count the
actual hit beyond the common engine prefix, including for partial hits.
Unmeasured shapes keep normal loading behavior. There is no interpolation.

The default saving margin is 5%: recomputation must be cheaper by at least
that fraction before declining. Costs must be positive and finite. Invalid
schemas, duplicate shapes and invalid margins fail at startup. Supported
metric labels are `ttft_ms`, `request_elapsed_ms` and `wave_elapsed_ms`;
the chosen label describes the supplied costs, not an online measurement or
learning mechanism. A TTFT improvement can reduce throughput.

Calibrate both branches with the same external keys and already-issued lookup
path, then validate on separate requests. A run that bypasses lookup using a
large minimum-token gate does not measure post-lookup recomputation cost.
Keep hardware, KV dtype, transfer mode, storage, concurrency, output length and
engine settings fixed. These deployment properties are not inferred from the
file; choose a matching profile and recalibrate when they change.

Set `SGLANG_TEST_LMCACHE_COST_PROFILE=1` in the manual admission test above to
exercise synthetic measured accept/decline decisions and an unmeasured shape,
including the partial-hit recompute/store/reuse path. This checks correctness;
the synthetic costs are not suitable for performance tuning.
