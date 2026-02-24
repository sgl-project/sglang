# sgl-kernel

Building and releasing sgl-kernel as a wheel is a part of the release workflow.  Check [release-whl-kernel.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/release-whl-kernel.yml) for details.

# sglang

`3rdparty/amd/wheel/sglang/pyproject.toml` is the AMD-specific pyproject for building the `amd-sglang` wheel.  It extends `python/pyproject_other.toml` with two ROCm-version extras (`rocm700`, `rocm720`) that pin the matching torch/triton/torchaudio/torchvision/sgl-kernel wheels, and renames the package to `amd-sglang`.

## Operation

```
$ git clone https://github.com/sgl-project/sglang.git && cd sglang
$ cp 3rdparty/amd/wheel/sglang/pyproject.toml python/pyproject.toml
$ cd python && python -m build
```

## Installation (Experimental)

ROCm 7.0.0:
```
pip install "amd-sglang[all-hip,rocm700]" -i https://pypi.amd.com/rocm-7.0.0/simple --extra-index-url https://pypi.org/simple
```

ROCm 7.2.0:
```
pip install "amd-sglang[all-hip,rocm720]" -i https://pypi.amd.com/rocm-7.2.0/simple --extra-index-url https://pypi.org/simple
```

## Resolving Triton

Triton 3.5.1 has a known issue in the ROCm 7.2.0 environment.  Replace the installation following the [ROCm docker recipe](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L472).

## Resolving AITER

[AITER](https://github.com/ROCm/aiter) is a fundamental dependency. Wheel-izing it is ongoing.
Until we can pin it reliably, install it manually (typically following the [ROCm docker recipe](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L106).

## Resolving Dependencies for PD Disaggregation

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

## Resolving Dependencies for DeepSeek-V3.2

### [TileLang](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L216)

### [FHT (fast-hadamard-transform)](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L300)
