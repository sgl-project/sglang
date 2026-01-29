# sgl-kernel

Building and releasing sgl-kernel as a wheel is a part of the release workflow.  Check [release-whl-kernel.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/release-whl-kernel.yml) for details.

# sglang

The patch, `wheel.patch`, describes the difference from the go-to practice `python/pyproject_other.toml` file, to one that enables a wheel build.  This section maintains the build process in short term, and is planned to be automated in the future.

## Operation

```
$ git clone https://github.com/sgl-project/sglang.git && cd sglang/3rdparty/amd/wheel/sglang
$ cp ../../../../python/pyproject_other.toml ./pyproject.toml
$ patch -p1 < wheel.patch
$ cp 3rdparty/amd/wheel/sglang/pyproject.toml python/ && cd ../../../../python
$ python -m build
```

## Installation (Experimental)

```
pip install "amd-sglang[all_hip,diffusion_hip]" -i https://pypi.amd.com/rocm7.1.1/simple
```

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
