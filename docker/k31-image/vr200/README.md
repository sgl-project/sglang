# Kimi K3.1 VR200: one serving image

`docker/k31-vr200-one.Dockerfile` builds one image for the Kimi K3.1 VR200 recipes in
`benchmark/kimi_k31_vr200/`: this SGLang tree, the sgl-nvidia-mega plugin, and the compiled parts the K3.1
paths use, on the SGLang nightly of 2026-11-07 for CUDA 13.4 (arm64, pinned by digest). The build steps are
`build_one.sh` in this directory; `addons/` holds the public FlashInfer #8186 Prims-TS diff and its SHA256SUMS.

## Build

Needs an arm64 host with Docker and BuildKit (`docker buildx`). No GPU is needed to build. The build fetches
from Docker Hub, GitHub, PyPI and crates.io. On a 64-core arm64 host it takes about 40 minutes (base image
pull included); the image is about 46 GB.

```bash
# the plugin code these recipes were checked with
git clone --branch kimi-k31-vr200-reference https://github.com/akhilg-nv/sgl-nvidia-mega.git ../sgl-nvidia-mega
# from the root of this SGLang checkout
docker buildx build --platform linux/arm64 -f docker/k31-vr200-one.Dockerfile \
  --build-context plugin=../sgl-nvidia-mega \
  --build-arg SGLANG_COMMIT=$(git rev-parse HEAD) \
  --build-arg PLUGIN_COMMIT=$(git -C ../sgl-nvidia-mega rev-parse HEAD) \
  -t k31-vr200-one --load .
```

`--build-arg K31_BUILD_JOBS=<n>` sets the compile parallelism (default 32). The plugin is copied from the
local checkout given as the `plugin` build context and installed editable at `/sgl-workspace/sgl-nvidia-mega`.

## Check

`build_one.sh` runs every step, logs each to `/opt/k31-image/one/build/<step>.log`, writes the step table to
`/opt/k31-image/one/steps.tsv`, and exits with the number of failed required steps, so the build fails if any
required step fails. Library versions stay the base image's (torch, FlashInfer, sgl-deep-gemm, sglang-kernel,
triton, nvidia-cutlass-dsl); the steps that install packages fail if they move any of them.

```bash
docker run --rm k31-vr200-one cat /opt/k31-image/one/steps.tsv
docker run --rm k31-vr200-one python3 /sgl-workspace/sglang/docker/k31-image/verify_image.py
```

`verify_image.py` exits 1 on this image with two expected notes: Prims-TS is FlashInfer main 26bb6dc9 plus
#8186, not the tree that `overlay_flashinfer_prims_ts.py` alone produces, and sgl-deep-gemm is the base image's
PyPI build (0.2.6), not the build of `build_deepgemm.py`. Any other note is a real failure.

## How it differs from the other K3.1 Dockerfiles

- `docker/k31-vr200.Dockerfile` starts from the SGLang nightly of 2026-10-27, copies this tree and the plugin
  over the base image's editable installs without reinstalling, and adds only the FlashInfer DynBlock SM107
  module; it builds no Rust TreeCore (its recipes run the Python TreeCore). Note that it copies this whole directory to `/opt/k31-image/vr200/`,
  so these files also land in that image (unused there).
- `docker/k31-vr200-one.Dockerfile` starts from the 2026-11-07 nightly and runs `build_one.sh`: it reinstalls
  this tree (`pip install --no-deps -e python`), builds the Rust TreeCore, FlashMLA `flashmla_ops` from this
  tree, the FlashInfer DynBlock (1024 experts) and SM107 fused-MoE modules, overlays FlashInfer Prims-TS
  (main 26bb6dc9 + #8186), installs flash-attn-4 4.0.0b36, the plugin (editable) and Dynamo 1.6.0 with the
  session-affinity runtime patches and the K3.1 `dynamo.sglang` edits, and checks the SM107 small-M GEMM tables.
- `docker/k31.Dockerfile` (the K3.1 recipe of #48414) starts from another nightly and does not install the plugin.
