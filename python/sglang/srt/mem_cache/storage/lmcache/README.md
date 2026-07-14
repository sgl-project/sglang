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

LMCache supports two transport modes. **MP (multi-process, default)** issues a single blocking retrieve over ZMQ to a standalone daemon that owns the KV store and survives SGLang restarts. **IP (in-process)** uses an embedded layerwise connector — the cache lives and dies with the SGLang process. Mode selection is currently a code-level setting on `LMCRadixCache._mode`; only MP is reachable by default.

### MP mode (default): multi-process daemon

Uses `LMCacheMPConnector`. Daemon host/port come from the LMCache YAML config (`mp_host`, `mp_port`).

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

### IP mode: in-process

Uses `LMCacheLayerwiseConnector`. KV transfer happens per layer inside the SGLang process; the cache lives and dies with the server. To enable, set `LMCRadixCache._mode = LMCacheMode.IP` in the source.

IP mode currently requires the actual token allocator page size to be 1. Servers that use larger allocator pages must use MP mode. The constructor rejects unsupported IP configurations before reading LMCache configuration, accessing KV buffers, creating CUDA streams, or constructing a connector. Page-size-one IP mode keeps its existing layerwise transfer lifecycle.

The LMCache config still controls chunk_size and storage; `mp_host` / `mp_port` are ignored on this path. Use the bundled `example_config_ip.yaml`:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_ip.yaml
```

## Page-aligned load-back ownership

MP load-back allocates complete pages using the token allocator's actual page size. Only logical uncached destinations are exposed to LMCache; temporary padding slots are never part of the connector mapping. A trailing cache hit that does not fill its final allocator page is discarded after the blocking retrieve relinquishes the mapping.

The radix node key, device indices returned to the scheduler, evictable-size accounting, and eventual node eviction all use the same complete-page prefix. The remaining allocated pages are returned to the allocator exactly once. This keeps every allocator page owned entirely by either the radix node or the allocator, including when the configured storage page size differs from the actual allocator page size.
