# Mooncake as L3 KV Cache

This document describes how to use Mooncake as the L3 KV cache for SGLang.

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly from the initiator’s DRAM/VRAM to the target’s DRAM/SSD in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/).

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

When integrated with **SGLang**, the system conceptually consists of four key components: `the master service`, `metadata service`, `store service`, and the `SGLang server`. Among them, the `master service` and `metadata service` are responsible for object and metadata maintenance. The `store service` manages a contiguous memory segment that contributes to the distributed KV cache, making its memory accessible to both local and remote `SGLang servers`. Data transfer occurs directly between the `store service` and `SGLang servers`, bypassing the master service.

Launch Mooncake master server:

```bash
mooncake_master
```

Launch Mooncake meta server:

```bash
python -m mooncake.http_metadata_server
```

Start the SGLang server with Mooncake enabled. Mooncake configuration can be provided via environment variables. Note that, for optimal performance, the Mooncake backend currently supports only the `page_first` layout.

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="erdma_0,erdma_1" \
MOONCAKE_MASTER=127.0.0.1:50051 \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake\
    --model-path [model_path]
```
