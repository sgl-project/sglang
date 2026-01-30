
# SGLang installation with NPUs support

You can install SGLang using any of the methods below. Please go through `System Settings` section to ensure the clusters are roaring at max performance. Feel free to leave an issue [here at sglang](https://github.com/sgl-project/sglang/issues) if you encounter any issues or have any problems.

## Component Version Mapping For SGLang
| Component         | Version                 | Obtain Way                                                                                                                                                                                                                   |
|-------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HDK               | 25.3.RC1                  | [link](https://hiascend.com/hardware/firmware-drivers/commercial?product=7&model=33) |
| CANN              | 8.5.0                     | [Obtain Images](#obtain-cann-image)                                                                                                                                                                                          |
| Pytorch Adapter   | 7.3.0                   | [link](https://gitcode.com/Ascend/pytorch/releases)                                                                                                                                                                          |
| MemFabric         | 1.0.5                   | `pip install memfabric-hybrid==1.0.5`                                                                                                                                                                 |
| Triton            | 3.2.0                   | `pip install triton-ascend`|
| SGLang NPU Kernel | NA                      | [link](https://github.com/sgl-project/sgl-kernel-npu/releases)                                                                                                                                                               |

<a id="obtain-cann-image"></a>
### Obtain CANN Image
You can obtain the dependency of a specified version of CANN through an image.
```shell
# for Atlas 800I A3 and Ubuntu OS
docker pull quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11
# for Atlas 800I A2 and Ubuntu OS
docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
```

## Preparing the Running Environment

### Method 1: Installing from source with prerequisites

#### Python Version

Only `python==3.11` is supported currently. If you don't want to break system pre-installed python, try installing with [conda](https://github.com/conda/conda).

```shell
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

#### CANN

Prior to start work with SGLang on Ascend you need to install CANN Toolkit, Kernels operator package and NNAL version 8.3.RC2 or higher, check the [installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

#### MemFabric-Hybrid

If you want to use PD disaggregation mode, you need to install MemFabric-Hybrid. MemFabric-Hybrid is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```shell
pip install memfabric-hybrid==1.0.5
```

#### Pytorch and Pytorch Framework Adaptor on Ascend

```shell
PYTORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
TORCH_NPU_VERSION=2.8.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$TORCH_NPU_VERSION
```

If you are using other versions of `torch` and install `torch_npu`, check [installation guide](https://github.com/Ascend/pytorch/blob/master/README.md)

#### Triton on Ascend

We provide our own implementation of Triton for Ascend.

```shell
pip install triton-ascend
```
For installation of Triton on Ascend nightly builds or from sources, follow [installation guide](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)

#### SGLang Kernels NPU
We provide SGL kernels for Ascend NPU, check [installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md).

#### DeepEP-compatible Library
We provide a DeepEP-compatible Library as a drop-in replacement of deepseek-ai's DeepEP library, check the [installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

#### CustomOps
_TODO: to be removed once merged into sgl-kernel-npu._
Additional package with custom operations. DEVICE_TYPE can be "a3" for Atlas A3 server or "910b" for Atlas A2 server.

```shell
DEVICE_TYPE="a3"
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run
./CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-2.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-2.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
```

#### Installing SGLang from source

```shell
# Use the last release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
```

### Method 2: Using Docker Image
#### Obtain Image
You can download the SGLang image or build an image based on Dockerfile to obtain the Ascend NPU image.
1. Download SGLang image
```angular2html
dockerhub: docker.io/lmsysorg/sglang:$tag
# Main-based tag, change main to specific version like v0.5.6,
# you can get image for specific version
Atlas 800I A3 : {main}-cann8.5.0-a3
Atlas 800I A2: {main}-cann8.5.0-910b
```
2. Build an image based on Dockerfile
```shell
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
# If there are network errors, please modify the Dockerfile to use offline dependencies or use a proxy
docker build -t <image_name> -f npu.Dockerfile .
```

#### Create Docker
__Notice:__ `--privileged` and `--network=host` are required by RDMA, which is typically needed by Ascend NPU clusters.

__Notice:__ The following docker command is based on Atlas 800I A3 machines. If you are using Atlas 800I A2, make sure only `davinci[0-7]` are mapped into container.

```shell

alias drun='docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/'

# Add HF_TOKEN env for download model by SGLang.
drun --env "HF_TOKEN=<secret>" \
    <image_name> \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

## System Settings

### CPU performance power scheme

The default power scheme on Ascend hardware is `ondemand` which could affect performance, changing it to `performance` is recommended.

```shell
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make sure changes are applied successfully
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor # shows performance
```

### Disable NUMA balancing

```shell
sudo sysctl -w kernel.numa_balancing=0
# Check
cat /proc/sys/kernel/numa_balancing # shows 0
```

### Prevent swapping out system memory

```shell
sudo sysctl -w vm.swappiness=10

# Check
cat /proc/sys/vm/swappiness # shows 10
```

## Running SGLang Service
### Running Service For Large Language Models
#### PD Mixed Scene
```shell
# Enabling CPU Affinity
export SGLANG_SET_CPU_AFFINITY=1
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

#### PD Separation Scene
1. Launch Prefill Server
```shell
# Enabling CPU Affinity
export SGLANG_SET_CPU_AFFINITY=1

# PIP: recommended to config first Prefill Server IP
# PORT: one free port
# all sglang servers need to be config the same PIP and PORT,
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8995 \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 0 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8000
```

2. Launch Decode Server
```shell
# PIP: recommended to config first Prefill Server IP
# PORT: one free port
# all sglang servers need to be config the same PIP and PORT,
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 1 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8001
```

3. Launch Router
```shell
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:8000 8995 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 \
    --port 6688
```

### Running Service For Multimodal Language Models
#### PD Mixed Scene
```shell
python3 -m sglang.launch_server \
    --model-path Qwen3-VL-30B-A3B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --tp 4 \
    --device npu \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --disable-radix-cache \
    --trust-remote-code \
    --enable-multimodal \
    --sampling-backend ascend
```
