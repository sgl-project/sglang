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

LMCache supports two transport modes. Pick one based on whether the cache should outlive the SGLang process and be shared across instances.

### Mode A: in-process (default)

Uses `LMCacheLayerwiseConnector`. KV transfer happens per layer inside the SGLang process; the cache lives and dies with the server.

Firstly, setup LMCache config. An example config is set at `example_config.yaml`. For more settings please refer to https://docs.lmcache.ai/api_reference/configurations.html.

Secondly, setup SGLang serving engine with lmcache:

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=example_config.yaml

python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache
```

### Mode B: multi-process daemon

Uses `LMCacheMPConnector`. SGLang issues a single blocking retrieve over a ZMQ socket and skips the per-layer KV transfer hook entirely; the daemon owns the KV store, so it survives SGLang restarts and can be shared across SGLang instances.

Terminal 1 — start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --chunk-size 256 --l1-size-gb 4 \
  --eviction-policy LRU --disable-observability
```

Terminal 2 — start SGLang pointing at the daemon:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-mp-host 127.0.0.1 --lmcache-mp-port 5556
```

Setting `--lmcache-mp-host` is the trigger that switches connectors. When unset, SGLang uses Mode A.
