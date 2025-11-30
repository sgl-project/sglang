# MemCache Memory Pool as L3 Distributed KV Cache on ascend environment

This document describes how to use MemCache as the L3 Distributed KVCache Pool for SGLang.
For more details about MemCache, please refer to: [MemCache project](https://gitcode.com/Ascend/memcache)


Related documentation:
* [MemCache project](https://gitcode.com/Ascend/memcache)
* [MemFabric project](https://gitcode.com/Ascend/memfabric_hybrid)


## Build and Install MemCache in Container

### 1. Download the open source codes
```
git clone -b develop https://gitcode.com/Ascend/memcache.git memcache
```

### 2. Build install package
```
cd memcache
bash script/build_and_pack_run.sh
```

### 3. Install
```
cd output
bash memcache_hybrid-1.0.0_linux_aarch64.run
```
The default installation path is `/usr/local/memfabric_hybrid`.
And the **.whl will be installed into python site-packages

## Launch SGLang with MemCache

### 1. Launch MetaService
```
# 1. set the environment variables
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/memfabric_hybrid/latest/config/mmc-meta.conf

# 2. launch meta-service
/usr/local/memfabric_hybrid/latest/aarch64-linux/bin/mmc_meta_service
```
1). MetaService configuration can be provided via MMC_META_CONFIG_PATH environment variable.<br>
2). MetaService supports certificate-based secure communication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.<br>
3). meta-service can also be launched via python method, For details, refer to
https://gitcode.com/Ascend/memcache/blob/develop/README.md


### 2. Start the SGLang server with memcache(local-service).

```
# 1. set local service config
export MMC_LOCAL_CONFIG_PATH=/usr/local/memfabric_hybrid/latest/config/mmc-local.conf

# 2. launch sglang with --enable-hierarchical-cache and --hicache-storage-backend=memcache
python3 -m sglang.launch_server \
    --model-path /data/Qwen3-32B \
    --host 127.0.0.1 \
    --port 28002 \
    --trust-remote-code \
    --tp-size 2 \
    --mem-fraction-static 0.85 \
    --base-gpu-id 14 \
    --attention-backend ascend \
    --device npu \
    --disable-overlap-schedule \
    --log-level info \
    --disable-cuda-graph \
    --max-running-requests 8 \
    --context-length 3800 \
    --chunked-prefill-size 57344 \
    --max-prefill-tokens 30400 \
    --enable-hierarchical-cache \
    --hicache-storage-backend memcache &
```
1). Local-service configuration can be provided via MMC_LOCAL_CONFIG_PATH environment variables.<br>
2). The local-service is integrated into the tp worker process.
Local-service communication supports certificate-based secure authentication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.
