# Ascend MemCache as L3 KV Cache

This document explains how to use **Ascend MemCache** as the L3 KV Cache backend for **SGLang HiCache**.

Related documentation:

- [Ascend MemCache Build Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md)
- [Ascend MemCache Config Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md)
- [Ascend MemCache Python API](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)
- [SGLang HiCache Design](https://docs.sglang.io/advanced_features/hicache_design.html)
- [Ascend MemFabric](https://gitcode.com/Ascend/memfabric_hybrid)
- [Ascend MemCache](https://gitcode.com/Ascend/memcache)

## About MemCache

MemCache is a distributed cache system from Ascend, built on MemFabric underneath, and can provide a high-performance distributed memory pool.
In SGLang HiCache, MemCache can be used as the L3 KV Cache backend to store and reuse KV cache.


## Install Ascend Memcache

[Memcache Official Document](https://gitcode.com/Ascend/memcache/blob/master/doc/install_run.md)

```bash
pip install memcache_hybrid
```

## Deploy MemCache
### Metaservice
add `metaservice_config.json`
```json
{
    // Meta service start-up url; in K8s meta service master-standby HA, auto-set to Pod IP at startup
    "meta_service_url": "tcp://127.0.0.1:5000",

    // Config store url; in K8s, auto-set to Pod IP at startup
    "config_store_url": "tcp://127.0.0.1:6000",

    // HTTP metrics service url
    "metrics_url": "http://127.0.0.1:8000",

    // Log level: debug, info, warn, error
    "log_level": "info"
}
```

Pass MetaService options via `metaservice_config.json` (see above). Keys below match `memcache_hybrid.MetaConfig` field names.

| Key | Type | Required | Default | Valid range | Description |
| --- | --- | --- | --- | --- | --- |
| `meta_service_url` | string | optional | `tcp://127.0.0.1:5000` | `tcp://<ip>:<port>` | Meta service listen address. Port in [1025, 65535]. |
| `config_store_url` | string | optional | `tcp://127.0.0.1:6000` | `tcp://<ip>:<port>` | Config store address. Port in [1025, 65535]. |
| `metrics_url` | string | optional | `http://127.0.0.1:8000` | `http://<ip>:<port>` | HTTP metrics endpoint. Port in [1025, 65535]. |
| `ha_enable` | boolean | optional | `false` | `true` / `false` | Enable MetaService master/backup HA in a K8s cluster. |
| `log_level` | string | optional | `info` | `debug` / `info` / `warn` / `error` | Log level. |
| `log_path` | string | optional | `/var/log/memcache_hybrid` | relative or absolute path | Log directory. Absolute paths start with `/`. |
| `log_rotation_file_size` | integer | optional | `20` | [1, 500] | Log rotation file size in MB. |
| `log_rotation_file_count` | integer | optional | `50` | [1, 50] | Number of rotated log files to keep. |
| `evict_threshold_high` | integer | optional | `90` | [1, 99] | Eviction high-water mark (%). Max is 99. Eviction is skipped when a single put exceeds 1% of capacity. |
| `evict_threshold_low` | integer | optional | `80` | [0, 98] | Eviction low-water mark (%) after eviction completes. |

For more options, see [MemCache Configuration Guide — MetaService Config](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md#metaservice-config).


## Quick Start Ascend_memcache as L3 backend

### Shell 1: Start Meta service

```bash
python -m sglang.srt.mem_cache.storage.ascend_memcache.start_meta_service --config_path "${metaservice_config_path}"
```

### Shell 2: Start SGLang Server


```bash
python -m sglang.launch_server \
  --model-path ${model_path} \
  --hicache-io-backend kernel_ascend \
  --attention-backend ascend \
  --enable-hierarchical-cache \
  --hicache-storage-backend ascend_memcache \
  --hicache-mem-layout page_first_kv_split \
  --hicache-storage-backend-extra-config '{"meta_service_url":"tcp://127.0.0.1:5000", "config_store_url":"tcp://127.0.0.1:6000", "log_level":"info", "world_size":256, "protocol": "device_sdma", "dram_size": "1GB"}'
```

Pass LocalService options via `--hicache-storage-backend-extra-config` (JSON). Keys below match `memcache_hybrid.LocalConfig` field names.

| Key | Type | Required | Default | Valid range | Description |
| --- | --- | --- | --- | --- | --- |
| `meta_service_url` | string | optional | `tcp://127.0.0.1:5000` | `tcp://<ip>:<port>` | Meta service address. Port in [1025, 65535]. In HA, `<ip>` is the cluster IP. |
| `config_store_url` | string | optional | `tcp://127.0.0.1:6000` | `tcp://<ip>:<port>` | Config store address. Port in [1025, 65535]. |
| `log_level` | string | optional | `info` | `debug` / `info` / `warn` / `error` | Log level. |
| `world_size` | integer | optional | `256` | [1, 1024] | Max rank count. Cannot change after ranks connect; restart Meta to update. |
| `protocol` | string | **required** | `host_rdma` | `host_rdma`, `host_urma`, `host_tcp`, `host_shm`, `device_sdma`, `device_rdma` | Transport protocol. `host_shm` requires `dram_size` > 0, `hbm_size` = 0, and no hcom. |
| `hcom_url` | string | optional | `tcp://127.0.0.1:7000` | `tcp://<ip>:<port>` | HCOM address for the DRAM pool. Port in [1024, 65535]. |
| `dram_size` | string / integer | **required** | `1GB` | [0, 1TB] | DRAM pool size. Accepts `134217728`, `2048KB`, `200mb`, `2.5G`, `1TB`, etc. Auto-aligned to 2MB (`host_rdma` / `host_tcp` / `host_shm`) or 1GB (`device_sdma` / `device_rdma`). |
| `hbm_size` | string / integer | optional | `0` | [0, 1TB] | HBM pool size (same format as `dram_size`). Must be `0` when using `host_shm`. |
| `max_dram_size` | string / integer | optional | `64GB` | [0, 1TB] | Max `dram_size` across all local processes. |
| `max_hbm_size` | string / integer | optional | `0` | [0, 1TB] | Max `hbm_size` across all local processes. |


For more options, see [MemCache Configuration Guide — LocalService Config](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md#localservice-config).
