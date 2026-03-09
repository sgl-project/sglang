# sgl-kernel

Building and releasing sgl-kernel as a wheel is a part of the release workflow.  Check [release-whl-kernel.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/release-whl-kernel.yml) for details.

# sglang

`3rdparty/amd/wheel/sglang/pyproject.toml` is the AMD-specific pyproject for building the `amd-sglang` wheel.  It extends `python/pyproject_other.toml` with two ROCm-version extras (`rocm700`, `rocm720`) that pin the matching torch/triton/torchaudio/torchvision/sgl-kernel wheels, and renames the package to `amd-sglang`.

## Operation to build sglang wheel

```
$ git clone https://github.com/sgl-project/sglang.git && cd sglang
$ cp 3rdparty/amd/wheel/sglang/pyproject.toml python/pyproject.toml
$ cd python && python -m build
```

## Installation

### v0.5.9

ROCm 7.0.0:
```
pip uninstall sgl-kernel sglang amd-sglang
pip install "amd-sglang[all-hip,rocm700]" -i https://pypi.amd.com/rocm-7.0.0/simple --extra-index-url https://pypi.org/simple
```

ROCm 7.2.0:
```
pip uninstall sgl-kernel sglang amd-sglang
pip install "amd-sglang[all-hip,rocm720]" -i https://pypi.amd.com/rocm-7.2.0/simple --extra-index-url https://pypi.org/simple
```

Note: You must resolve the two dependencies, AITER and triton, below.  Others are optional depending on your applications.

## Manual Dependency Resolution

### Resolving AITER

[AITER](https://github.com/ROCm/aiter) is a fundamental dependency. Wheel-izing it is ongoing.
Until we can pin it reliably, install it manually (typically following the [ROCm docker recipe](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L106).

### Revolving triton

To avoid known issues in triton 3.5.1 installed by default, we recommend upgrading triton after installation.  In ROCm 7.0.0 environment,
```
pip install triton==3.6.0
```
or ROCm 7.2.0,
```
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.6.0%2Brocm7.2.0.gitba5c1517-cp310-cp310-linux_x86_64.whl
```

#### `torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`

After upgrading, you may hit this error during inference when PyTorch Inductor interacts with Triton metadata.

A pragmatic workaround is to guard the metadata access in Inductor's Triton heuristics so it only reads `cluster_dims` when the attribute exists:

```diff
--- a/opt/venv/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py
+++ b/opt/venv/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py
@@ -1759,6 +1759,8 @@
                 else (
                     (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                     if hasattr(binary, "metadata")
+                    and hasattr(binary.metadata, "num_ctas")
+                    and hasattr(binary.metadata, "cluster_dims")
                     else ()
                 )
             ),
```

### Resolving Dependencies for Distributed Inference

#### sgl-model-gateway

Install sgl-model-gateway as follows:

```
$ apt install openssl libssl-dev protobuf
$ export PATH="/$HOME/.cargo/bin:${PATH}" \
  && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && rustc --version && cargo --version # Prepare for a rust toolchain
$ python3 -m pip install --no-cache-dir setuptools-rust \
  && cd /sgl-workspace/sglang/sgl-model-gateway/bindings/python \
  && cargo build --release \
  && python3 -m pip install --no-cache-dir . \
  && rm -rf /root/.cache # Build and install sgl-model-gateway
```

#### [Mori](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L381)

### Resolving Dependencies for DeepSeek-V3.2

#### [TileLang](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L216)

#### [FHT (fast-hadamard-transform)](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L300)
