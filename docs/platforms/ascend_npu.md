
# SGLang installation with NPUs support

You can install SGLang using any of the methods below. Please go through `System Settings` section to ensure the clusters are roaring at max performance. Feel free to leave an issue [here at sglang](https://github.com/sgl-project/sglang/issues) if you encounter any issues or have any problems.

## Installing SGLang

### Method 1: Installing from source with prerequisites

#### Python Version

Only `python==3.11` is supported currently. If you don't want to break system pre-installed python, try installing with [conda](https://github.com/conda/conda).

```shell
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

#### CANN

Prior to start work with SGLang on Ascend you need to install CANN Toolkit, Kernels operator package and NNAL version 8.3.RC2 or higher, check the [installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

#### MemFabric Adaptor

If you want to use PD disaggregation mode, you need to install MemFabric Adaptor. MemFabric Adaptor is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```shell
pip install mf-adapter==1.0.0
```

#### Pytorch and Pytorch Framework Adaptor on Ascend

At the moment NPUGraph optimizations are supported only in `torch_npu==2.6.0.post3` that requires 'torch==2.6.0'.
_TODO: NPUGraph optimizations will be supported in future releases of 'torch_npu' 2.7.1, 2.8.0 and 2.9.0_

```shell
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
TORCH_NPU_VERSION=2.6.0.post3
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$TORCH_NPU_VERSION
```

While there is no resleased versions of 'torch_npu' for 'torch==2.7.1' and 'torch==2.8.0' we provide custom builds of 'torch_npu'. PLATFORM can be 'aarch64' or 'x86_64'

```shell
PLATFORM="aarch64"
PYTORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/torch_npu/torch_npu-${PYTORCH_VERSION}.post2.dev20251120-cp311-cp311-manylinux_2_28_${PLATFORM}.whl
pip install torch_npu-${PYTORCH_VERSION}.post2.dev20251120-cp311-cp311-manylinux_2_28_${PLATFORM}.whl
```

If you are using other versions of 'torch' install 'torch_npu' from sources, check [installation guide](https://github.com/Ascend/pytorch/blob/master/README.md)

#### Triton on Ascend

We provide our own implementation of Triton for Ascend.

```shell
BISHENG_NAME="Ascend-BiSheng-toolkit_aarch64_20251121.run"
BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend/${BISHENG_NAME}"
wget -O "${BISHENG_NAME}" "${BISHENG_URL}" && chmod a+x "${BISHENG_NAME}" && "./${BISHENG_NAME}" --install && rm "${BISHENG_NAME}"
```
```shell
pip install triton-ascend==3.2.0rc4
```
For installation of Triton on Ascend nightly builds or from sources, follow [installation guide](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)

#### SGLang Kernels NPU
We provide our own set of SGL kernels, check [installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md).

#### DeepEP-compatible Library
We provide a DeepEP-compatible Library as a drop-in replacement of deepseek-ai's DeepEP library, check the [installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

#### CustomOps
_TODO: to be removed once merged into sgl-kernel-npu._
Additional package with custom operations. DEVICE_TYPE can be "a3" for Atlas A3 server or "910b" for Atlas A2 server.

```shell
DEVICE_TYPE="a3"
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
```

#### Installing SGLang from source

```shell
# Use the last release branch
git clone -b v0.5.6.post2 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_other.toml python/pyproject.toml
pip install -e python[srt_npu]
```

### Method 2: Using docker

__Notice:__ `--privileged` and `--network=host` are required by RDMA, which is typically needed by Ascend NPU clusters.

__Notice:__ The following docker command is based on Atlas 800I A3 machines. If you are using Atlas 800I A2, make sure only `davinci[0-7]` are mapped into container.

```shell
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t <image_name> -f npu.Dockerfile .

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

drun --env "HF_TOKEN=<secret>" \
    <image_name> \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend --host 0.0.0.0 --port 30000
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
