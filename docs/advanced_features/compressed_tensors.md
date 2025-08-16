# CompressedTensors

The [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) library extends the [safetensors](https://github.com/huggingface/safetensors) format, providing a versatile and efficient way to store and manage compressed tensor data. This library supports various quantization and sparsity schemes, making it a unified format for handling different model optimizations like GPTQ, AWQ, SmoothQuant, INT8, FP8, SparseGPT, and more. SGLang also supports running models that are quantized with the `compressed-tensors` format.

## CompressedTensors for MoE Models

To successfully run the Mixture of Experts (MoE) models that are quantized with the `compressed-tensors` format, please follow the steps below to ensure a smooth usage and development experience.

**1. Compilation Issue with `vllm`/`sgl-kernel` and CUDA 12.6**

`vllm` will install FlashAttention in the following steps. When compiling `vllm` and/or `sgl-kernel` with FlashAttention on a Hopper GPU using CUDA 12.6, you may encounter a segmentation fault:

```bash
kernel/build/_deps/repo-flash-attention-src/hopper/instantiations/flash_fwd_hdimall_bf16_paged_softcap_sm90.cu -o CMakeFiles/flash_ops.dir/_deps/repo-flash-attention-src/hopper/instantiations/flash_fwd_hdimall_bf16_paged_softcap_sm90.cu.o
Segmentation fault (core dumped)
```

⚠️ **Note**:  To ensure that FlashAttention compiles correctly on Hopper GPU Architecture (sm90), it is strongly [recommended](https://github.com/Dao-AILab/flash-attention/issues/1453) to use:
- nvcc version: 12.6
- ptxas version: 12.8

**(1) Check Current Versions**

Before proceeding, verify your current CUDA tool versions:
```bash
nvcc --version
ptxas --version
```
**(2) Update ptxas to 12.8 (if needed)**

- Save the following script to a file (e.g., `update_ptxas.sh`).
```bash
#!/usr/bin/env bash
# Source: https://github.com/Dao-AILab/flash-attention/blob/7ff1b621112ba8b538e2fc6a316f2a6b6f22e518/hopper/setup.py#L404
set -ex

if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_VERSION>"
    exit 1
fi

CUDA_VERSION=$1

if awk "BEGIN {exit !("$CUDA_VERSION" >= 12.6 && "$CUDA_VERSION" < 12.8)}"; then
    NVCC_ARCHIVE_VERSION="12.8.93"
    NVCC_ARCHIVE_NAME="cuda_nvcc-linux-x86_64-${NVCC_ARCHIVE_VERSION}-archive"
    NVCC_ARCHIVE_TAR="${NVCC_ARCHIVE_NAME}.tar.xz"
    NVCC_ARCHIVE_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/${NVCC_ARCHIVE_TAR}"

    wget "$NVCC_ARCHIVE_URL"
    tar -xf "$NVCC_ARCHIVE_TAR"

    mkdir -p /usr/local/cuda/bin
    cp "${NVCC_ARCHIVE_NAME}/bin/ptxas" /usr/local/cuda/bin/

    # Clean up temporary files
    rm -f "${NVCC_ARCHIVE_TAR}"
    rm -rf "${NVCC_ARCHIVE_NAME}"
fi
```

- Run the script with your CUDA version as the argument, using `sudo`:
```bash
sudo bash update_ptxas.sh 12.6
# Check the version
ptxas --version
```

**2. Install the Correct Version of `vllm`**

- If you are not using the latest version of SGLang, please install its corresponding vllm version accordingly.

- If you are using the latest version of SGLang, e.g., `lmsysorg/sglang:dev` from docker or building from source `main`, please refer to the following steps to install vllm without destroying the Python packages that you are using for SGLang.

    ```bash
    git clone https://github.com/vllm-project/vllm && cd vllm
    python use_existing_torch.py
    pip install -r requirements/build.txt
    pip install --no-build-isolation -e . -v
    ```

After completing the steps above, you should be able to run MoE models quantized with the `compressed-tensors` format.
