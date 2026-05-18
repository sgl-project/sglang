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


### SGLang installation with NPUs support

[SGLang NPU installation guide](https://github.com/nbbb24/sglang/blob/main/docs/platforms/ascend/ascend_npu.md)


### Install MemCache
[Official Document](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md)

#### Method 1: with pip

whl [pypi](https://pypi.org/project/memcache-hybrid/#files)
```bash
pip install memcache_hybrid
```

#### Method 2: from source

##### Clone MemCache

```bash
git clone https://gitcode.com/Ascend/memcache
cd memcache
git clean -xdf
git reset --hard
```

Fetch third-party libraries
```bash
# Initialize and update only the required submodules (excluding test dependencies)
git submodule update --init 3rdparty/

# Update memfabric_hybrid to the latest version of a specific branch (e.g., master)
# Replace 'master' with your target branch name
git -c submodule.3rdparty/memfabric_hybrid.branch=master submodule update --remote 3rdparty/memfabric_hybrid
```

Notes:
1. Use the `-c submodule.3rdparty/memfabric_hybrid.branch=<branch_name>` option to specify the target branch to fetch.

2. To fetch all submodules (including test dependencies), you can use `git submodule update --recursive --init`.

##### Install Memfabric

[Memfabric Build Official Document](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/doc/installation.md)
```bash
cd 3rdparty/memfabric_hybrid
git clean -xdf
git reset --hard
```

For build parameters, refer to [this document](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/doc/installation.md#%E4%BA%8C%E3%80%81-%E4%BD%BF%E7%94%A8-c-api)
```bash
bash script/build_and_pack_run.sh
```

The package will be saved at `output/memfabric-hybrid-${version}_${os}_${arch}.run`.  

Run:
```bash
cd output
bash memfabric-hybrid-${version}_${os}_${arch}.run
source /usr/local/memfabric_hybrid/set_env.sh
```



##### Build MemCache

Change directory to the memcache folder and build:

```bash
bash script/build_and_pack_run.sh --build_mode RELEASE
```

Build and run unit tests

```bash
bash script/run_ut.sh
```

The package will be saved at `output/memcache_hybrid-${version}_${os}_${arch}.run`

Run:
```bash
cd output
bash memcache_hybrid-${version}_${os}_${arch}.run
source /usr/local/memcache_hybrid/set_env.sh
```

## Deploy MemCache
### Metaservice

(1) Environment variables plus configuration file
(2) Configure directly in Python [refer to this document](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md#metaservice)

Recommended approach: add `metaservice_config.json`
```json
{
    "meta_service_url": " ",
    ...
}
```

### Localservice

(1) Environment variables plus configuration file
(2) Configure directly in Python [refer to this document](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md#localservice)


Recommended approach: add `localservice_config.json`
```json
{
    "protocol": " ",
    ...
}
```

```bash
export SGLANG_HICACHE_MEMCACHE_CONFIG_PATH=${localservice_config_path}
```

## Quick Start Ascend_memcache as L3 backend

### Shell 1: Start Meta service

```bash
python python/sglang/srt/mem_cache/storage/ascend_memcache/start_meta_service.py --config_path ${metaservice_config_path}
```

### Shell 2: Start SGLang Server


```bash
python -m sglang.launch_server \
  --model-path ${model_path} \
  --hicache-io-backend kernel_ascend \
  --attention-backend ascend \
  --enable-hierarchical-cache \
  --hicache-storage-backend ascend_memcache \
  --hicache-mem-layout page_first_kv_split 
```
