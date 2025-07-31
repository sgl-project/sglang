# Mooncake as L3 KV Cache

This document describes how to use Mooncake as the L3 KV cache for SGLang.
For more details about Mooncake, please refer to: https://kvcache-ai.github.io/

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

## Use Mooncake

Launch Mooncake master server:

```bash
mooncake_master
```

Launch Mooncake meta server:

```bash
python -m mooncake.http_metadata_server
```

Start the SGLang server with Mooncake enabled. Mooncake configuration can be provided via environment variables:

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
MOONCAKE_LOCAL_BUFFER_SIZE=134217728 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="erdma_0,erdma_1" \
MOONCAKE_MASTER=127.0.0.1:50051 \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake\
    --model-path [model_path]
```
