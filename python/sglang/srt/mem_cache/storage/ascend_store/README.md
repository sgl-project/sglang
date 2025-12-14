# MemCache Memory Pool as L3 Distributed KV Cache on ascend environment

This document describes how to use MemCache as the L3 Distributed KVCache Pool for SGLang.
For more details about MemCache, please refer to: [MemCache project](https://gitcode.com/Ascend/memcache)


Related documentation:
* [MemCache project](https://gitcode.com/Ascend/memcache)
* [MemFabric project](https://gitcode.com/Ascend/memfabric_hybrid)


## Build and Install MemCache in Container

### 1. Download open source codes and install MemCache
```
# download codes of MemCache
git clone https://gitcode.com/Ascend/memcache.git

# build install package
cd memcache
bash script/build_and_pack_run.sh

# install MetaService
cd output
bash memcache_hybrid-1.0.0_linux_aarch64.run

# install LocalService
pip3 install memcache/wheel/memcache_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
```

### 2. Download open source codes and install MemFabric
```
# download codes of MemFabric
git clone https://gitcode.com/Ascend/memfabric_hybrid.git

# build install package
cd memfabric_hybrid
bash script/build_and_pack_run.sh

# install MemFabric
cd output
bash memfabric_hybrid-1.0.0_linux_aarch64.run
pip3 install memfabric_hybrid/wheel/memfabric_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
pip3 install mk_transfer_adapter/wheel/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl
```

The default installation path is `/usr/local/`, and you can modify the installation path by `--install-path=${your path}`.
And the **.whl will be installed into python site-packages.

For more details about memcache build, please refer to: <br>
* [MemCache Build](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md)
* [MemFabric Build](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/doc/build.md)

## Launch SGLang with MemCache
### 1. Launch MetaService
```
# 1. set the environment variables
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf

# 2. launch meta-service
/usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service
```
1). The MMC_META_CONFIG_PATH environment variable specifies the configuration file of MetaService.
You need modify the configuration file content.<br>
For more detailed introduction of configuration items, please refer to [MemCache Configs](https://gitcode.com/Ascend/memcache/blob/master/doc/memcached_config.md).<br>
2). MetaService supports certificate-based secure communication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.<br>
3). MetaService can also be launched via python method, For details, refer to
https://gitcode.com/Ascend/memcache/blob/develop/README.md


### 2. Start the SGLang server with memcache(local-service).

```
# 1. set local service config
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

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
    --log-level info \
    --max-running-requests 8 \
    --context-length 3800 \
    --chunked-prefill-size 57344 \
    --max-prefill-tokens 30400 \
    --enable-hierarchical-cache-direct \
    --hicache-storage-backend memcache &
```
1). The MMC_LOCAL_CONFIG_PATH environment variable specifies the configuration file of LocaService.
You need modify the configuration file content.<br>
For more detailed introduction of configuration items, please refer to [MemCache Configs](https://gitcode.com/Ascend/memcache/blob/master/doc/memcached_config.md).<br>
<br>
2). LocalService is integrated into the tp worker process.
LocalService communication supports certificate-based secure authentication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.
