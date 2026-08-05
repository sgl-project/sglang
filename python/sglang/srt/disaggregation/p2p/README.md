# P2P KV Cache Transfer Engine

This module implements a **peer-to-peer (P2P) CUDA-based KV cache transfer engine** for prefill-decode (PD) disaggregation in SGLang, designed specifically for **consumer GPUs (NVIDIA 4090 / 5090)**. It enables direct GPU-to-GPU KV cache transmission between prefill workers and decode workers without routing through host memory, reducing transfer latency and improving disaggregated serving throughput.

This backend targets **single-node** deployments only. For data-center GPUs (A/H/B series) or multi-node setups, use [Mooncake](../mooncake) or [NIXL](../nixl) instead.

## Table of Contents

- [Evaluation](#evaluation)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Driver Setup (NVIDIA 4090 / 5090)](#driver-setup-nvidia-4090--5090)
  - [Build & Install Transfer Engine](#build--install-transfer-engine)
- [Environment Variables](#environment-variables)
- [Launch Example](#launch-example)
- [Python API](#python-api)

## Evaluation

### Test Environment

- **Hardware**: NVIDIA 4090 × 4 (single node)
- **Model**: Qwen3-30B-A3B-FP8
- **TP size**: 2 (single instance uses GPU 0-1, PD uses GPU 0-1 for prefill and GPU 2-3 for decode)
- **KV cache**: fp8_e4m3, page-size 1
- **Max running requests**: 54 (calculated based on KvCache pool capacity with input 4096 / output 512)

### Accuracy

The following benchmarks confirm that switching to PD disaggregation with the P2P transfer backend does not degrade model accuracy compared to single-instance serving.

| Benchmark | Single Instance | PD Disaggregation (P2P) |
| --------- | --------------- | ----------------------- |
| GSM8K     | 0.9700          | 0.9900                  |
| C-Eval    | 0.8603          | 0.8670                  |
| MMLU      | 0.8800          | 0.8772                  |

### Performance

Test settings: input 4K tokens, output 512 tokens, `--random-range-ratio 1`, request rate 2 / 3 / 4.

| Request Rate | Mode            | Throughput (req/s) | Successful Requests | Input (tok/s) | Output (tok/s) | Mean E2E (ms) | Median E2E (ms) | P90 E2E (ms) | P99 E2E (ms) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) |
| ------------ | --------------- | ------------------ | ------------------- | ------------- | -------------- | ------------- | --------------- | ------------ | ------------ | -------------- | ---------------- | ------------- | -------------- | ---------------- | ------------- |
| 2            | Single Instance | 1.97               | 256                 | 8081.12       | 1010.14        | 16368.63      | 17078.59        | 18704.97     | 19154.06     | 328.88         | 306.55           | 674.36        | 31.39          | 32.83            | 36.74         |
| 2            | **PD (P2P)**    | **1.99**           | 256                 | **8132.93**   | **1016.62**    | **8239.38**   | **8290.15**     | **9186.63**  | **9459.92**  | **393.49**     | **305.75**       | **932.21**    | **15.35**      | **15.41**        | **17.51**     |
| 3            | Single Instance | 2.31               | 256                 | 9479.09       | 1184.89        | 29969.26      | 29969.26        | 37710.55     | 39852.57     | 10135.69       | 10788.92         | 22438.62      | 38.81          | 40.79            | 43.77         |
| 3            | **PD (P2P)**    | **2.86**           | 256                 | **11720.56**  | **1465.07**    | **10346.10**  | **10437.04**    | **11358.70** | **12025.76** | **872.31**     | **785.59**       | **2314.14**   | **18.54**      | **18.82**        | **19.84**     |
| 4            | **PD (P2P)**    | **3.54**           | 256                 | **14492.25**  | **1811.53**    | **13606.53**  | **13963.18**    | **15001.63** | **15376.39** | **1908.75**    | **1866.85**      | **4058.15**   | **22.89**      | **23.85**        | **24.48**     |

Key observations:

- At request rate 2, PD disaggregation reduces Mean E2E latency by ~**50%** (16368ms → 8239ms) and Mean TPOT by ~**51%** (31.39ms → 15.35ms)
- At request rate 3, single instance begins to saturate (TTFT spikes to 10s+), while PD maintains stable latency
- PD disaggregation sustains throughput up to request rate 4, achieving **3.54 req/s** where single instance cannot keep up

> To reproduce these results, refer to the commands in [Launch Example](#launch-example).

## Installation

### Prerequisites

- CUDA Toolkit (installed at `/usr/local/cuda`)
- Python 3.10+
- pybind11 (`pip install pybind11`)
- CMake >= 3.14

### Driver Setup (NVIDIA 4090 / 5090)

First, check whether your machine already supports P2P:

```bash
nvidia-smi topo -p2p rw
```

Each cell showing `OK` means P2P is supported between that GPU pair. **If all relevant GPU pairs already show `OK`, skip this section and proceed to [Build & Install Transfer Engine](#build--install-transfer-engine).**

The official NVIDIA driver does **not** enable P2P for 4090 by default. P2P can be enabled via a community-maintained open-source driver patch:

- Repository: https://github.com/tinygrad/open-gpu-kernel-modules
- Latest release (570.148.08): https://github.com/tinygrad/open-gpu-kernel-modules/releases/tag/570.148.08-p2p
- **Prerequisite**: The motherboard must support **Resizable BAR** (also known as Above 4G Decoding). This allows the PCIe BAR size to be adjusted to expose the full GPU memory to peers. If Resizable BAR cannot be enabled in the BIOS, the patched driver will not work.

After installing the patched driver, re-run `nvidia-smi topo -p2p rw` to confirm P2P is enabled before proceeding.

#### Step-by-step Installation Guide (Ubuntu 22.04, Driver 570.148.08)

> The latest release of [open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules/releases/tag/570.148.08-p2p) is `570.148.08-p2p`, so the example below installs NVIDIA driver `570.148.08` to match.

> Make sure kernel headers are installed before starting:
>
> ```bash
> sudo apt install -y linux-headers-$(uname -r)
> ```

**Step 1: Uninstall existing driver** (skip if no driver is currently installed)

```bash
# Stop display managers
sudo systemctl isolate multi-user.target
sudo systemctl stop gdm sddm lightdm nvidia-persistenced 2>/dev/null || true

# Kill processes using GPU
sudo fuser -k /dev/nvidia* 2>/dev/null || true

# Unload kernel modules
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true

# Run official uninstaller if available
if command -v /usr/bin/nvidia-uninstall >/dev/null 2>&1; then
  sudo /usr/bin/nvidia-uninstall
fi

# Clean up residual files
sudo rm -rf /usr/src/nvidia* /var/lib/dkms/nvidia* \
    /etc/modprobe.d/nvidia*.conf /etc/X11/xorg.conf 2>/dev/null || true

# Confirm no nvidia modules remain (no output is expected)
lsmod | grep -i nvidia || echo "nvidia modules not loaded."
```

**Step 2: Download required files**

Download the NVIDIA official driver runfile from https://www.nvidia.com/en-us/drivers/ — search for version `570.148.08`, select Linux x86_64, and download the `.run` file. The filename will be `NVIDIA-Linux-x86_64-570.148.08.run`. The official driver version must match the community kernel module version exactly; the latest community release is `570.148.08`.

Clone the P2P kernel module patch and check out the matching release tag:

```bash
git clone https://github.com/tinygrad/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
git checkout 570.148.08-p2p
cd ..
```

Download the CUDA 12.8 Toolkit installer (all versions available at https://developer.nvidia.com/cuda-toolkit-archive):

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
```

**Step 3: Install official driver without kernel modules**

```bash
sudo sh NVIDIA-Linux-x86_64-570.148.08.run --no-kernel-modules -s
```

**Step 4: Build and install P2P kernel modules**

Create the `x509.genkey` file for kernel module signing:

```bash
cat > x509.genkey << 'EOF'
[ req ]
default_bits = 4096
distinguished_name = req_distinguished_name
prompt = no
string_mask = utf8only
x509_extensions = myexts

[ req_distinguished_name ]
CN = Modules

[ myexts ]
basicConstraints=critical,CA:FALSE
keyUsage=digitalSignature
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid
EOF
```

Generate the signing certificate and install the modules:

```bash
cd open-gpu-kernel-modules

# Generate signing certificate
openssl req -new -nodes -utf8 -sha512 -days 36500 -batch -x509 \
  -config ../x509.genkey \
  -outform DER -out signing_key.x509 -keyout signing_key.pem

# Move certificate to kernel certs directory
sudo mv signing_key.pem signing_key.x509 $(find /usr/src/*-generic/certs -maxdepth 0)

# Fix System.map symlink
sudo ln -sf /boot/System.map-$(uname -r) /lib/modules/$(uname -r)/build/System.map

# Compile and install P2P modules (install.sh is in the repo root)
sh install.sh
cd ..
```

**Step 5: Verify P2P is enabled**

```bash
nvidia-smi topo -p2p rw
```

Each cell showing `OK` confirms P2P is active between the corresponding GPU pair.

**Step 6: Install CUDA Toolkit**

```bash
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent

# Configure environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Build & Install Transfer Engine

Configure the build with CMake:

```bash
cd python/sglang/srt/disaggregation/p2p
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir)
```

Compile the shared library:

```bash
make -j 8
```

This produces `cuda_p2p_transfer.cpython-310-x86_64-linux-gnu.so` in the `build/` directory.

Copy the compiled `.so` to the Python packages directory so it can be imported as `cuda_p2p_transfer`:

```bash
mv cuda_p2p_transfer.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages/
```

> **Note**: The `.so` filename encodes the Python version (e.g. `cpython-310` for Python 3.10). If you are using a different Python version, the filename will differ. Check the actual filename in the `build/` directory and update the path accordingly. For example, for Python 3.11 it would be `cuda_p2p_transfer.cpython-311-x86_64-linux-gnu.so` and the target directory would be `/usr/local/lib/python3.11/dist-packages/`.

Verify the installation:

```python
import cuda_p2p_transfer
```

## Environment Variables

| Variable                      | Description                                                                             | Default |
| ----------------------------- | --------------------------------------------------------------------------------------- | ------- |
| `MC_METRIC`                   | Enable transfer throughput logging                                                      | `0`     |
| `METRIC_TIME_STEP`            | Log print interval (seconds) for throughput stats. Only takes effect when `MC_METRIC=1` | `5`     |
| `EngineThreadNum`             | Number of transfer threads                                                              | `32`    |
| `SGLANG_P2P_BATCH_LIMIT`      | Maximum number of KV cache transfer requests batched in a single round                  | `512`   |
| `SGLANG_P2P_TRANSFER_TIMEOUT` | Timeout (seconds) for a single P2P transfer operation                                   | `60.0`  |
| `SGLANG_KVCACHE_LOG`          | Enable detailed KV cache transfer logging                                               | `0`     |

Set any variable to override its default before starting the server, for example:

```bash
export MC_METRIC=1
export EngineThreadNum=16
```

## Launch Example

The following commands reproduce the benchmark results above. Test environment: `Qwen3-30B-A3B-FP8` on a single node with 4 GPUs (prefill on GPU 0-1, decode on GPU 2-3).

> **Important: Do not set `CUDA_VISIBLE_DEVICES` to restrict GPUs for prefill or decode workers.**
> P2P transfer requires both prefill and decode workers to have visibility of all participating GPUs on the node. If `CUDA_VISIBLE_DEVICES` is set separately for each worker, each side can only see its own GPUs and will be unable to access the peer's memory, causing transfer failures.
>
> Use `--base-gpu-id` and `--gpu-id-step` to assign GPUs instead. With `--base-gpu-id <N> --tp-size <T> --gpu-id-step <S>`, the worker occupies GPUs `N, N+S, N+2S, ..., N+(T-1)*S` while remaining visible to all other workers on the node. The default step is `1`, so `--base-gpu-id 0 --tp-size 2` uses GPU 0,1 and `--base-gpu-id 2 --tp-size 2` uses GPU 2,3.

### 0. (Optional) Single Instance Server

> This is the single-instance baseline used for comparison in the benchmark. Skip if you only need PD disaggregation.

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=1 NCCL_NVLS_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_P2P_LEVEL=SYS \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen3-30B-A3B-FP8/ --trust-remote-code \
  --host 0.0.0.0 --tp 2 --base-gpu-id 0 --port 29999 \
  --max-running-request 54 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.89 --page-size 1 --cuda-graph-max-bs 64 \
  --disable-radix-cache --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-max-tokens 4096 --kv-cache-dtype fp8_e4m3
```

### 1. Start the Prefill Server

```bash
SGLANG_HACK_PD_DECODE_NUM_RESERVED_DECODE_TOKENS=1 MC_TE_METRIC=true \
  SGLANG_ENABLE_JIT_DEEPGEMM=1 NCCL_NVLS_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_P2P_LEVEL=SYS \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen3-30B-A3B-FP8/ --trust-remote-code \
  --host 0.0.0.0 --tp 2 --base-gpu-id 0 --port 29999 \
  --max-running-request 54 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.89 --page-size 1 --disable-radix-cache \
  --enable-nccl-nvls --enable-p2p-check \
  --disaggregation-mode prefill --disaggregation-transfer-backend p2p \
  --load-balance-method round_robin --tool-call-parser qwen \
  --enable-piecewise-cuda-graph --enforce-piecewise-cuda-graph \
  --piecewise-cuda-graph-max-tokens 4096 --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph
```

### 2. Start the Decode Server

```bash
SGLANG_HACK_PD_DECODE_NUM_RESERVED_DECODE_TOKENS=512 \
  SGLANG_ENABLE_JIT_DEEPGEMM=1 NCCL_NVLS_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_P2P_LEVEL=SYS \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen3-30B-A3B-FP8/ --trust-remote-code \
  --host 0.0.0.0 --tp 2 --base-gpu-id 2 --port 39999 \
  --max-running-request 54 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.89 --page-size 1 --cuda-graph-max-bs 64 \
  --disable-radix-cache --enable-nccl-nvls --enable-p2p-check \
  --disaggregation-mode decode --disaggregation-transfer-backend p2p \
  --tool-call-parser qwen --kv-cache-dtype fp8_e4m3
```

### 3. Start the Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:29999 9798 \
  --decode http://127.0.0.1:39999 \
  --host 0.0.0.0 --port 40001 \
  --policy random \
  --prometheus-host 0.0.0.0 --prometheus-port 9091 \
  --log-level info
```

### 4. Run Benchmark Client

```bash
python3 -m sglang.bench_serving \
  --backend sglang --num-prompts 256 \
  --random-input-len 4096 --random-output-len 512 \
  --dataset-name random \
  --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json \
  --seed 42 --host 0.0.0.0 --port 40001 \
  --random-range-ratio 1 --request-rate 4
```

### 5. Verify

```bash
curl -X POST http://127.0.0.1:40001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "How many letter e are in the word Deepseek?"
      }
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

## Python API

The `P2PTransferEngine` class is the main entry point. It is integrated into SGLang's disaggregation framework via `transfer_engine.py` and is selected when the P2P transfer backend is configured.

```python
from sglang.srt.disaggregation.p2p.transfer_engine import P2PTransferEngine

# hostname: IP or hostname of the current machine, used for session identification
engine = P2PTransferEngine(hostname="<host>", physical_gpu_id=0)

# Register a destination buffer and get its handle
handle = engine.register_buffer(dst_ptr)

# Transfer a single KV cache block
transfer_handle = engine.transfer(
    src_ptr, src_dev, dst_handle, dst_dev, dst_offset, length
)
transfer_handle.wait()

# Transfer multiple KV cache blocks in batch
transfer_handle = engine.transfer_many(
    src_ptrs, src_devs, dst_handles, dst_devs, dst_offsets, lengths
)
transfer_handle.wait()
```
