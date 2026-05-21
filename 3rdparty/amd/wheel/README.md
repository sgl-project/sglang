# Limitations

* Wheels built for ROCm 7.0.0 have [a known issue](https://github.com/sgl-project/sglang/issues/20188).
* sglang-kernel wheels so far were built without multi-arch support and can only guarantee to work on gfx950.
* We suggest using [docker images](https://hub.docker.com/r/rocm/sgl-dev) for a better experience.

# sglang-kernel (formerly sgl-kernel)

> **Note:** We generally don't recommend direct installation of sglang-kernel, as it may not match arbitrary sglang versions in your target environment.

## Install Prebuilt Wheel

For example, to install [version 0.4.2.post2](https://sgl-project.github.io/whl/rocm720/sglang-kernel/):

```bash
pip install https://github.com/sgl-project/whl/releases/download/v0.4.2.post2/sglang_kernel-0.4.2.post2+rocm720-cp310-abi3-manylinux2014_x86_64.whl#sha256=86ce4335c7e12fa75f8ce24311ab5c3d52ec8c73a90b407868d0e7142c3d6421
```

## Reference

* [release-whl-kernel.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/release-whl-kernel.yml)

# sglang

Choose one of the following installation methods.

## Build from Source

```bash
cp python/pyproject_rocm.toml python/pyproject.toml
cd python && python -m build
```

## Install Prebuilt Wheel

For example, to install for [ROCm 7.2.0](https://pypi.amd.com/sglang/rocm720/simple/sglang/):

```bash
pip install "sglang[all-hip,rocm720]" -i https://pypi.amd.com/sglang/rocm-7.2.0/simple --extra-index-url https://pypi.org/simple
```

## Manual Dependency Resolution

### Resolving Triton

To avoid known issues in Triton 3.5.1 (installed by default), we recommend upgrading Triton after installation:

```bash
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.6.0%2Brocm7.2.0.gitba5c1517-cp310-cp310-linux_x86_64.whl
```

#### Known Issue: `torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`

After upgrading Triton, you may encounter this error during inference when PyTorch Inductor interacts with Triton metadata.

**Workaround:** Guard the metadata access in Inductor's Triton heuristics to only read `cluster_dims` when the attribute exists:

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

```bash
# Install system dependencies
apt install openssl libssl-dev protobuf-compiler

# Install Rust toolchain
export PATH="/$HOME/.cargo/bin:${PATH}" \
  && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && rustc --version && cargo --version

# Build and install sgl-model-gateway
python3 -m pip install --no-cache-dir setuptools-rust \
  && cd /sgl-workspace/sglang/sgl-model-gateway/bindings/python \
  && cargo build --release \
  && python3 -m pip install --no-cache-dir . \
  && rm -rf /root/.cache
```

#### Mori

See the [Mori installation steps](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L381) in the ROCm Dockerfile.

### Resolving Dependencies for DeepSeek-V3.2

#### [TileLang](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L216)

#### [FHT (fast-hadamard-transform)](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L300)
