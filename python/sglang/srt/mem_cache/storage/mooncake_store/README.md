# Mooncake as L3 KV Cache

This document describes how to use Mooncake as the L3 KV cache for SGLang.

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

## Install Mooncake

### Method 1: with pip

```bash
pip install mooncake-transfer-engine
```

### Method 2: from source

Clone Mooncake project:

```bash
git clone https://github.com/kvcache-ai/Mooncake --recursive
```

Install dependencies:

```bash
cd Mooncake
bash dependencies.sh
```

Build the project. For additional build options, please refer to [the official guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html).

```bash
mkdir build
cd build
cmake ..
make -j
```

Install Mooncake:

```bash
sudo make install
```

## Deploy Mooncake

**Mooncake** is a distributed system that efficiently aggregates memory resources across multiple servers. It can also be deployed on a single server for simpler setups.

When integrated with **SGLang**, the system conceptually consists of four key components: `the master service`, `metadata service`, `store service`, and the `SGLang server`. Among them, the `master service` and `metadata service` are responsible for object and metadata maintenance. The `store service` manages a contiguous memory segment that contributes to the distributed KV cache, making its memory accessible to both local and remote `SGLang servers`. Data transfer occurs directly between the `store service` and `SGLang servers`, bypassing the `master service`.

### Single Server Deployment

There are four components for deploying Mooncake: metadata service, master service, store service and sglang instance.
Note: *Only **master service** is mandatory for single server deployment.*

**Launch Mooncake `metadata service`(Optional):**

```bash
python -m mooncake.http_metadata_server
```

**Launch Mooncake `master service`:**

```bash
mooncake_master --eviction_high_watermark_ratio=0.95
```

To start both the metadata and master services together:
```bash
mooncake_master --enable_http_metadata_server=true --eviction_high_watermark_ratio=0.95
```

**Understanding `eviction_high_watermark_ratio`:**

When a `PutStart` request fails due to insufficient memory, or when the eviction thread detects that space usage has reached the configured high watermark ratio, an eviction task is triggered to free up space by evicting a portion of objects.

Due to memory fragmentation, allocation failures may occur even when memory usage has not yet reached 100%. The actual threshold depends on the workload. This [benchmark document](https://kvcache-ai.github.io/Mooncake/performance/allocator_benchmark_result.html)
 provides memory allocation efficiency results under different scenarios. if excessive allocation failures are observed, consider lowering this parameter accordingly.

**Launch Mooncake `store service` (Optional):**

First, create and save a configuration file in JSON format. For example:

```json
{
    "local_hostname": "localhost",
    "metadata_server": "http://localhost:8080/metadata",
    "master_server_address": "localhost:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb",
    "local_buffer_size": 0
}
```

Parameter Explanation:

* `local_hostname`: The hostname of the `store service`.
* `metadata_server`: The network address of the `metadata service`. The default port is 8080.
* `master_server_address`: The network address of the `master service`. The default port is 50051.
* `protocol`: The protocol used by Mooncake. Supported values are `"rdma"` or `"tcp"`. For optimal performance, `"rdma"` is recommended.
* `device_name`: For `"rdma"`, you can leave this empty in most cases. Mooncake auto-discovers RDMA NICs by default. If you want to pin specific NICs (e.g., `mlx5_0,mlx5_1`), just set `device_name` accordingly. To list available devices, use `ibv_devices`.
* `global_segment_size`: The amount of memory contributed to the global memory pool. Accepts either bytes (integer) or a string with the `gb` suffix, e.g., `"16gb"`. A larger value allows Mooncake to cache more KV tensors.
* `local_buffer_size`: Local buffer is used to do request operations such as `Get` or `Put`. In this case, it is set to 0 because the instance functions solely as a storage server, contributing memory to the global pool without issuing any request operations.

Then start the `store service`:

```bash
python -m mooncake.mooncake_store_service --config=[config_path]
```

Note: If `MOONCAKE_GLOBAL_SEGMENT_SIZE` is set to a non-zero value when starting the `SGLang server`, launching the `store service` can be skipped. In this case, the `SGLang server` also takes on the role of the `store service`, which simplifies deployment but couples the two components together. Users can choose the deployment approach that best fits their needs.

**Start the `SGLang server` with Mooncake enabled:**

Mooncake configuration can be provided via environment variables. Note that, for optimal performance, the Mooncake backend currently supports only the `page_first` layout (which optimizes memory access patterns for KV cache operations).

There are three ways to prepare mooncakes:
1. Use environment variables;
2. Use json configuration files;
3. Additional configuration using the sglang parameter.

**Using env variables to configure Mooncake**

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
# Leave MOONCAKE_DEVICE empty for auto-discovery (default)
# To pin NICs, disable auto-discovery then set MOONCAKE_DEVICE, e.g.:
# export MC_MS_AUTO_DISC=0
# export MOONCAKE_DEVICE="mlx5_0,mlx5_1"
MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake\
    --model-path [model_path]
```

Parameter Explanation:

* `MOONCAKE_TE_META_DATA_SERVER`: The network address of the `metadata service`. The default port is 8080.
* `MOONCAKE_MASTER`: The network address of the `master service`. The default port is 50051.
* `MOONCAKE_PROTOCOL`: The protocol used by Mooncake. Supported values are `"rdma"` or `"tcp"`. For optimal performance, `"rdma"` is recommended.
* `MOONCAKE_DEVICE`: Optional for `"rdma"`. By default, Mooncake auto-discovers RDMA NICs. If you need to pin specific NICs, set `MOONCAKE_DEVICE` (comma-separated list, e.g., `mlx5_0,mlx5_1`).
* `MOONCAKE_GLOBAL_SEGMENT_SIZE`: The amount of memory contributed to the global memory pool. Accepts either bytes (integer) or a value with the `gb` suffix, e.g., `16gb`. If at least one `store service` is launched, this value can be set to `0`. In this case, the `SGLang server` will not contribute any memory to the system. Note that KV tensors cached in the contributed memory will be lost once this process terminates; however, this will not cause any system errors.

**Using JSON file to configure Mooncake**

```bash
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hicache/mooncake_config.json
echo '{
    "local_hostname": "localhost",
    "metadata_server": "http://localhost:8080/metadata",
    "master_server_address": "localhost:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb",
    "local_buffer_size": 0
}' > ${SGLANG_HICACHE_MOONCAKE_CONFIG_PATH}
```

**Using extra-config of sglang arguments to configure Mooncake**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost", "metadata_server": "http://127.0.0.1:8080/metadata", "global_segment_size": "4gb", "local_buffer_size": 16777216, "protocol": "rdma", "device_name": ""}'
```

**Important: Understanding Global Segment Size**

`global_segment_size` for `store service` and `MOONCAKE_GLOBAL_SEGMENT_SIZE` for `SGLang service`: This parameter specifies the amount of memory each instance contributes to the distributed memory pool. The total memory available for KV cache storage across the cluster is the sum of the memory contributed by all instances.

Adjust this value according to system’s available memory and expected cache requirements.

### Distributed Deployment

Distributed deployment of Mooncake is straightforward. Similar to the single-node setup, start one `metadata service` and one `master service` for this cluster. Then start a `store service` on each server.

Mooncake also supports high availability mode. This mode enhances fault tolerance by running the `master service` as a cluster of multiple master nodes coordinated through an `etcd` cluster. The master nodes use `etcd` to elect a leader, which is responsible for handling client requests. For more details about how to deploy in this mode, please refer to our [documents](https://kvcache-ai.github.io/Mooncake/) .

## Test Mooncake Store

This test is intended for developers to quickly verify that the MooncakeStore class interfaces are functioning correctly.

First, start the `metadata service` and `master service`. Then run the `test_mooncake_store.py`. 16MB global segments size is enough to run this test.

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
# Auto-discovery by default. To pin NICs:
# export MOONCAKE_DEVICE="mlx5_0,mlx5_1"
MOONCAKE_GLOBAL_SEGMENT_SIZE=16777216 \
python3 [path of test_mooncake_store.py]
```

If all tests pass, the message "✅ All tests passed" will be printed at the end.
