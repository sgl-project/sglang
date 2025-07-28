# Mooncake L3 store

### Directly install mooncake client

```bash
pip install mooncake
```

### Mooncake build steps
Building mooncake client and server from source code.
1. Clone mooncake source code 

```bash
git clone https://github.com/kvcache-ai/Mooncake
```

2. Install dependencies, stable Internet connection is required:

```bash
bash dependencies.sh
```

3. In the root directory of this project, run the following commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

4. Install Mooncake python package and mooncake_master executable

```bash
sudo make install
```

Please refer to https://kvcache-ai.github.io/Mooncake/ as more detailed guide.

### Launch Mooncake master and meta server

Launch mooncake master server:

```bash
./build/mooncake-store/src/mooncake_master
```

 Launch mooncake meta server:

```bash
python ./mooncake-transter-engine/example/http-metadata-server-python/bootstrap_server.py
```

### Mooncake config in SGLang

Mooncake config can be loaded from environment arguments, for example:

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_LOCAL_BUFFER_SIZE=268435456
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="erdma_0,erdma_1"
export MOONCAKE_MASTER=127.0.0.1:50051
```

Then launch the sglang server with argument --hicache-storage-backend mooncake

```bash
python -m sglang.launch_server --enable-hierarchical-cache --hicache-storage-backend mooncake
```

