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

Firstly, setup LMCache config. An example config is set at `example_config.yaml`. For more settings please refer to https://docs.lmcache.ai/api_reference/configurations.html.

Secondly, setup SGLang serving engine with lmcache:

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=example_config.yaml

python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache
```

If `--enable-lmcache` is set without LMCache MP arguments, SGLang uses the
existing in-process LMCache connector.

To connect SGLang to an LMCache multi-process daemon, start the LMCache MP
service separately according to the LMCache documentation, then launch SGLang
with the LMCache MP host and port:

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=example_config.yaml

python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-mp-host 127.0.0.1 \
  --lmcache-mp-port 5555
```
