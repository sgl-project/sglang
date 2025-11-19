# Mooncake as L3 KV Cache

This document describes how to use Mooncake as the L3 KV cache for SGLang.

Related documentation:
* [Quick Start: SGLang HiCache with Mooncake Backend](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-quick-start.html)
* [Complete Guide: SGLang HiCache with Mooncake Backend](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html)
* [Mooncake x SGLang HiCache System Design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
* [HiCache System Design and Optimization](https://docs.sglang.ai/advanced_features/hicache_design.html)
* [SGLang HiCache with Mooncake Backend Benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html)

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

### Mooncake & SGLang HiCache

Mooncake serves as a high-performance L3 storage backend for SGLang HiCache, enabling distributed KV cache storage across multiple servers with RDMA-accelerated data transfer. This integration addresses the capacity limitations of traditional GPU-only or GPU+CPU caching by providing virtually unlimited cache storage through a distributed memory pool.

When a cache miss occurs in L1 and L2, HiCache automatically fetches the required KV cache from Mooncake's distributed memory pool. The system uses intelligent prefetching strategies to minimize latency, and utilize RDMA technology and zero-copy technique to ensure high-bandwidth, low-latency data transfer between SGLang instances and Mooncake storage nodes.

**Key Advantages:**

- **Scalable Capacity**: Aggregate memory across entire clusters into large distributed pools.
- **Cache Sharing**: KV caches can be shared by all SGLang instances in the cluster.
- **RDMA Acceleration**: Direct memory access eliminates CPU overhead and reduces latency.
- **Zero Copy**: Direct data transfer between L2 and Mooncake without intermediate copying, maximizing throughput.
- **Fault Tolerance**: Distributed architecture provides resilience against individual node failures.

This integration is particularly valuable for production deployments involving long-context models, multi-turn conversations, and high-throughput serving scenarios where traditional caching approaches become capacity-constrained.

## Install Mooncake

**Method 1: with pip**

```bash
pip install mooncake-transfer-engine
```

**Method 2: from source**

Clone Mooncake project:

```bash
git clone https://github.com/kvcache-ai/Mooncake --recursive
```

Install dependencies:

```bash
cd Mooncake
bash dependencies.sh
```

Build the project:

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

For more details, please refer to [Mooncake official installation guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html).

## Deployment

**Mooncake** is a distributed system that efficiently aggregates memory resources across multiple servers. It can also be deployed on a single server for simpler setups.

When integrated with **SGLang**, the system conceptually consists of four key components: `the master service`, `metadata service` (Optional), `store service`  (Optional), and the `SGLang server`. Among them, the `master service` and `metadata service` are responsible for object and metadata maintenance. The `store service` manages a contiguous memory segment that contributes to the distributed KV cache, making its memory accessible to both local and remote `SGLang servers`. Data transfer occurs directly between the `store service` and `SGLang servers`, bypassing the `master service`.

### Single Server Deployment

**Launch Mooncake `metadata service` (Optional):**

```bash
python -m mooncake.http_metadata_server
```

This service is responsible for centralized metadata management including internal connection status and related metadata.

Deployment of the `metadata service` can be skipped in the following cases:
* Mooncake supports non-centralized metadata management via a P2P handshake mechanism to exchange metadata. When using this mode, deployment of the `metadata service` can be skipped.
* Mooncake also supports embedding `mededata service` into `master service`. In this case, only the `master service` needs to be started.

**Launch Mooncake `master service`:**

The `master service` orchestrates the logical storage space pool across the entire cluster, managing KV cache space allocation and eviction.

To start `mooncake_master`:

```bash
mooncake_master --eviction_high_watermark_ratio=0.95
```

To start `mooncake_master` with embedded `metadata service` (so that a separate `metadata service` deployment can be skipped):

```bash
mooncake_master --enable_http_metadata_server=true --http_metadata_server_port=8080 --eviction_high_watermark_ratio=0.95
```

**Understanding `eviction_high_watermark_ratio`:**

When a `PutStart` request fails due to insufficient memory, or when the eviction thread detects that space usage has reached the configured high watermark ratio, an eviction task is triggered to free up space by evicting a portion of objects.

Due to memory fragmentation, allocation failures may occur even when memory usage has not yet reached 100%. The actual threshold depends on the workload. This [benchmark document](https://kvcache-ai.github.io/Mooncake/performance/allocator-benchmark-result.html) provides memory allocation efficiency results under different scenarios. if excessive allocation failures are observed, consider lowering this parameter accordingly.

**Launch Mooncake `store service` (Optional):**

First, create and save a configuration file in JSON format. For example:

```json
{
    "local_hostname": "localhost",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb",
    "local_buffer_size": 0
}
```

Note: If the `metadata service` is not deployed, set this field to:

```json
    "metadata_server": "P2PHANDSHAKE",
```

Then start the `store service`:

```bash
python -m mooncake.mooncake_store_service --config=[config_path] --port=8081
```

Mooncake `store service` configuration can also be provided via environment variables:

```bash
MOONCAKE_LOCAL_HOSTNAME="localhost" \
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
MOONCAKE_LOCAL_BUFFER_SIZE=0 \
python -m mooncake.mooncake_store_service --port=8081
```

**Parameter Explanation:**

* `local_hostname`, `MOONCAKE_LOCAL_HOSTNAME`: The hostname of the `store service`.
* `metadata_server`, `MOONCAKE_TE_META_DATA_SERVER` : The network address of the `metadata service`. The default port is 8080. If the `metadata service` is not deployed, set this field to: `"metadata_server": "P2PHANDSHAKE"`.
* `master_server_address`, `MOONCAKE_MASTER`: The network address of the `master service`. The default port is 50051.
* `protocol`, `MOONCAKE_PROTOCOL`: The protocol used by Mooncake. Supported values are `"rdma"` or `"tcp"`. For optimal performance, `"rdma"` is recommended.
* `device_name`, `MOONCAKE_DEVICE`: The RDMA devices used by Mooncake. This field can usually be left empty, as Mooncake automatically discovers available NICs by default. This parameter is required only when the protocol is set to `"rdma"` **and** a specific set of NICs needs to be used. Example: `"device_name": "mlx5_0,mlx5_1"`. To list available devices, run `ibv_devices`. **Note:** If the environment variable `MC_MS_AUTO_DISC` is set to `1`, any `device_name` or `MOONCAKE_DEVICE` configuration will be overridden, and Mooncake will switch to auto-discovery mode.
  - For tensor parallel deployments where different ranks should use different devices, you can specify device configurations using JSON format:
    ```json
    {
    "device_name": "{0: \"ib0,ib1\", 1: \"ib2,ib3\", 2: \"ib4,ib5\"}"
    }
    ```
  - Or in environment variables:
    ```bash
    MOONCAKE_DEVICE="{\"0\": \"ib0,ib1\", \"1\": \"ib2,ib3\", \"2\": \"ib4,ib5\"}"
    ```
* `global_segment_size`, `MOONCAKE_GLOBAL_SEGMENT_SIZE`: The amount of memory contributed to the global memory pool. Accepts either bytes (integer) or a string with the `gb` suffix, e.g., `"4294967296"` or `"4gb"`. A larger value allows Mooncake to cache more KV tensors.
* `local_buffer_size`, `MOONCAKE_LOCAL_BUFFER_SIZE`: Local buffer is used to do request operations such as `Get` or `Put`. In this case, it is set to 0 because the instance functions solely as a storage server, contributing memory to the global pool without issuing any request operations.

**Important: Understanding Global Segment Size**

`global_segment_size` and `MOONCAKE_GLOBAL_SEGMENT_SIZE`: This parameter specifies the amount of memory each instance contributes to the distributed memory pool. The total memory available for KV cache storage across the cluster is the sum of the memory contributed by all instances.

Adjust this value according to system’s available memory and expected cache requirements.

Note: If `MOONCAKE_GLOBAL_SEGMENT_SIZE` is set to a non-zero value when starting the `SGLang server`, launching the `store service` can be skipped. In this case, the `SGLang server` also takes on the role of the `store service`, which simplifies deployment but couples the two components together. Users can choose the deployment approach that best fits their needs.

**Start the `SGLang server` with Mooncake enabled:**

There are three ways to configure Mooncake:

1. Via extra configuration passed through sglang parameters
2. Using JSON configuration files
3. Using environment variables

Mooncake loads configuration in the following priority order:

1. If Mooncake-specific options are provided in `--hicache-storage-backend-extra-config`, they are used first.
2. If not, Mooncake checks whether the environment variable `DEFAULT_MOONCAKE_CONFIG_PATH_ENV` is set, and loads the JSON config file from that path.
3. If neither of the above is provided, Mooncake falls back to environment variables.

**Using extra-config of sglang arguments to configure Mooncake**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost", "metadata_server": "http://127.0.0.1:8080/metadata", "global_segment_size": "4gb", "protocol": "rdma", "device_name": ""}'
```

**Using JSON file to configure Mooncake**

SGLang server can load Mooncake config from `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`.

```bash
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hicache/mooncake_config.json

echo '{
    "local_hostname": "localhost",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb"
}' > ${SGLANG_HICACHE_MOONCAKE_CONFIG_PATH}

python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path]
```

**Using env variables to configure Mooncake**

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake\
    --model-path [model_path]
```

**Parameter Explanation:**

The Mooncake parameters used here are essentially the same as those configured for the `store service`.

In particular, for the `global segment size`, if at least one `store service` instance is running, this value can be set to `0`. In this case, the SGLang server will not contribute any memory to the system. Note that KV tensors stored in this contributed memory will be lost when the process exits; however, this will **not** cause any system errors.

**Important:** when `tp > 1`, each Tensor Parallel (TP) rank launches its own Mooncake backend instance and contributes `1/global_segment_size` memory. Therefore, the total memory consumption equals `global segment size`.

**HiCache Related Parameters for SGLang Server**

For a comprehensive overview of HiCache-related parameters, please refer to [this document](https://docs.sglang.ai/advanced_features/hicache_design.html#related-parameters).


Note that, for `--hicache-mem-layout {layer_first,page_first,page_first_direct}`, which specifies the memory layout for the host memory pool, `page_first` or `page_first_direct` are required if use Mooncake backend.

### Distributed Deployment

Distributed deployment of Mooncake is straightforward. Similar to the single-node setup, start one `metadata service` and one `master service` for this cluster. Then start a `store service` on each server.

Mooncake also supports high availability mode. This mode enhances fault tolerance by running the `master service` as a cluster of multiple master nodes coordinated through an `etcd` cluster. The master nodes use `etcd` to elect a leader, which is responsible for handling client requests. For more details about how to deploy in this mode, please refer to our [documents](https://kvcache-ai.github.io/Mooncake/).

### Prefill/Decode Disaggregation

In **PD disaggregation**, the configurations for the `metadata service`, `mooncake master`, and the optional `store service` remain the same as described above. The difference is that SGLang introduces three distinct roles: `prefill worker`, `decode worker`, and `router`.

Among these, the `prefill worker` supports enabling **HiCache**. To run with PD disaggregation, start from the [PD configuration](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration-v1.html), and add the HiCache-related parameters (as previously described for the `SGLang server`) to the `prefill worker`.

In the example below, one `prefill worker`, one `decode worker`, and one `router` are launched. HiCache is enabled on the `prefill worker` to optimize prefill performance.

**Prefill worker**:

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="mlx5_1" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-storage-prefetch-policy timeout \
    --hicache-storage-backend mooncake \
    --disaggregation-mode prefill \
    --disaggregation-ib-device "mlx5_1" \
    --base-gpu-id 0 \
    --port 30000
```

**Decode worker**:

```bash
python -m sglang.launch_server \
    --model-path [model_path] \
    --page-size 64 \
    --disaggregation-mode decode \
    --disaggregation-ib-device "mlx5_1" \
    --base-gpu-id 1 \
    --port 30001
```

**Router**:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://127.0.0.1:30000" \
    --decode "http://127.0.0.1:30001" \
    --host 0.0.0.0 \
    --port 8000
```

## Troubleshooting

**RDMA Registration Failure:**

* In some environments, RDMA registration may require root privileges. In this case, try running the program as root.
* In certain environments (e.g., eRDMA), there is an upper limit on the total amount of RDMA memory that can be registered. Once this limit is exceeded, registration will fail. To resolve this, you can lower the value of `MOONCAKE_GLOBAL_SEGMENT_SIZE`, or reduce the host memory allocated to HiCache in the `SGLang server` (since this memory is fully registered with RDMA to enable zero-copy).

**HiCache CPU Memory Usage:**

When using HiCache, the default L2 host DRAM (CPU memory) size for KV cache is **2 times** the size of the L1 device memory (GPU memory) for KV cache.

If the model is small but the GPU memory is large — especially in multi-TP (tensor parallel) setups — this may cause the L1 KV cache to become very large, which in turn can consume excessive CPU DRAM.

In such cases, you should manually configure an appropriate L2 cache size based on your hardware. This can be done by setting `--hicache-ratio` or `--hicache-size`.

**More Information:**

Additional troubleshooting information can be found [here](https://kvcache-ai.github.io/Mooncake/troubleshooting/troubleshooting.html).

## Test Mooncake Store

This test is intended for developers to quickly verify that the MooncakeStore class interfaces are functioning correctly.

First, start the `metadata service` and `master service`. Then run the `test_mooncake_store.py`. 16MB global segments size is enough to run this test.

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=16777216 \
python3 [path of test_mooncake_store.py]
```

If all tests pass, the message "✅ All tests passed" will be printed at the end.
