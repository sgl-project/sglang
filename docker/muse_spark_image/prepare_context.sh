#!/bin/bash
# prepare_context.sh <commit> <out dir>: Docker build context for one commit of the Muse Spark refactor tree.
# The context holds python/sglang, aiter patch 05, the aiter CSV, the Dockerfile and muse_image_manifest.json;
# nothing from the working tree, so the image matches the commit byte for byte. The manifest also pins the two
# aiter files patch 05 produces: they are computed here from the base image's own copies (docker create + cp).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse "${1:?commit}^{commit}")
out=${2:?out dir}
base_image="lmsysorg/sglang-rocm:muse-spark-open@sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4"
# Upstream PR #44667 head (on main d33dd0566a): the tree's base.
base_commit=37197b72708c0bc6f89c5c54a1719df22455613c
patch_path=3rdparty/amd/muse_spark/patches/05-aiter-unified-attention-muse.patch
rm -rf "$out"
mkdir -p "$out"
git -C "$repo" archive --format=tar "$commit" python/sglang docker/muse_spark_image "$patch_path" | tar -x -C "$out"
mv "$out/docker/muse_spark_image/Dockerfile" "$out/docker/muse_spark_image/verify_manifest.py" "$out/"
mv "$out/docker/muse_spark_image/aiter_model_configs" "$out/"
mv "$out/$patch_path" "$out/"
rm -rf "$out/docker" "$out/3rdparty"

# The aiter files as patch 05 leaves them in the base image.
work=$(mktemp -d)
trap 'rm -rf "$work"; docker rm -f "zijiechen_ctx_aiter_$$" >/dev/null 2>&1 || true' EXIT
docker create --name "zijiechen_ctx_aiter_$$" "$base_image" >/dev/null
mkdir -p "$work/aiter/aiter/ops/triton/attention" "$work/aiter/aiter/ops/triton/_triton_kernels/attention"
for f in aiter/ops/triton/attention/unified_attention.py aiter/ops/triton/_triton_kernels/attention/unified_attention.py; do
  docker cp "zijiechen_ctx_aiter_$$:/sgl-workspace/aiter/$f" "$work/aiter/$f"
done
git -C "$work/aiter" init -q
git -C "$work/aiter" apply "$out/05-aiter-unified-attention-muse.patch"
mkdir -p "$out/.aiter_patched"
cp -r "$work/aiter/aiter" "$out/.aiter_patched/"

git -C "$repo" diff --name-status "$base_commit" "$commit" > "$out/.changes_vs_base"
python3 "$here/make_manifest.py" "$out" "$commit" "$out/.changes_vs_base" > "$out/muse_image_manifest.json"
rm -rf "$out/.changes_vs_base" "$out/.aiter_patched"
echo "context: $out (tree $commit)"
echo "build:   docker build --build-arg MUSE_TREE_COMMIT=$commit -t <tag> $out"
