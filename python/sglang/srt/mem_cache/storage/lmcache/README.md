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
