# Ascend MemCache Memory Pool as L3 KV Cache

This document describes how to use MemCache as the L3 KV cache for SGLang.
For more details about MemCache, please refer to: https://gitee.com/ascend/memfabric_hybrid/

## Build and Install MemCache in Container

### 1. Download the open source codes and its dependent submodule

```
#1. download source code
git clone -b br_A3_shm_bm_630_develop https://gitee.com/ascend/memfabric_hybrid.git memfabric_hybrid
```

### 2. Build install package

Use the one key build&package script to build:

```
cd memfabric_hybrid

bash script/build_and_pack_run.sh
```

### 3. Install
* **MetaService**
```
cd memfabric_hybrid/output

# install command
bash mxc-memfabric_hybrid-1.0.0_linux_aarch64.run --install
```
The default installation root path is `/usr/local/`.
We can add the `--install-path` parameter to modify the installation path.

* **LocalService**
```
1. install whl package
pip3 install memcache-1.0.0-cp311-cp311-linux_aarch64.whl

2. set env of LocalService
export MMC_LOCAL_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-local.conf
```
The LocalService will be integrated into and run within the inference engine's TP Worker processes.

## Launch SGLang with MemCache

### 1. Launch MetaService
```
# 1. set the environment variables
source /usr/local/mxc/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-meta.conf

# 2. launch through the binary
/usr/local/mxc/memfabric_hybrid/latest/aarch64-linux/bin/mmc_meta_service
```
1). MetaService configuration can be provided via MMC_META_CONFIG_PATH environment variable.<br>
2). MetaService supports certificate-based secure communication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.<br>
3). can also be started via python method, For details, refer to
https://gitee.com/ascend/memfabric_hybrid/blob/br_A3_shm_bm_630_develop/doc/zh/memcache.md


### 2. Start the SGLang server with LocalService.

```
# 1. set local service config
export MMC_LOCAL_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-local.conf

# 2. launch sglang with hierarchical-cache and memcache
python -m sglang.launch_server \
    --host 127.0.0.1 \
    --port 8011 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu
    --cuda-graph-max-bs 32 \
    --max-running-requests 64 \
    --enable-hierarchical-cache \
    --hicache-storage-backend memcache \
    --model-path [model_path]
```
1). MemCache client configuration can be provided via MMC_LOCAL_CONFIG_PATH environment variables.<br>
2). Inter-process communication supports certificate-based secure authentication.
However, you can disable certificate verification by setting xxx.xxx.tls.enable to false to achieve better performance, which may be accompanied by security risks.
