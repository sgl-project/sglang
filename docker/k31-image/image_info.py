"""Read one platform of a K3.1 image from `docker buildx imagetools inspect --format '{{json .}}'`.

Usage: image_info.py <arch> < inspect.json

Prints shell assignments for the platform's manifest digest and the build labels
(PREV_DIGEST, PREV_SRC, PREV_NIGHTLY, PREV_LAYERS, PREV_FLASHMLA_TREE, PREV_FI_DYNBLOCK).
Exits 3 if the image has no linux/<arch> manifest. Handles a plain manifest, a
single-platform index (older builds carry an attestation next to the image) and a
multi-platform index.
"""

import json
import shlex
import sys

arch = sys.argv[1]
info = json.load(sys.stdin)
manifest = info["manifest"]
image = info.get("image") or {}

if "manifests" in manifest:
    digests = [
        entry["digest"]
        for entry in manifest["manifests"]
        if (entry.get("platform") or {}).get("os") == "linux"
        and (entry.get("platform") or {}).get("architecture") == arch
    ]
    if not digests:
        sys.exit(3)
    digest = digests[0]
else:
    digest = manifest["digest"]

if "config" in image:  # one platform: its config directly
    if image.get("architecture") != arch:
        sys.exit(3)
else:  # several platforms: one config per platform
    image = image.get(f"linux/{arch}") or {}
labels = (image.get("config") or {}).get("Labels") or {}

values = {
    "PREV_DIGEST": digest,
    "PREV_SRC": labels.get("ai.radixark.source.commit", ""),
    # Builds before the incremental scheme only had base.image (the nightly).
    "PREV_NIGHTLY": labels.get("ai.radixark.nightly.image") or labels.get("ai.radixark.base.image", ""),
    "PREV_LAYERS": labels.get("ai.radixark.overlay.layers", "1"),
    # Git tree of python/sglang/kernels/aot that its flashmla_ops was built from.
    "PREV_FLASHMLA_TREE": labels.get("ai.radixark.flashmla.tree", ""),
    # Hash of the flashinfer DynBlock scripts its jit-cache modules were rebuilt with.
    "PREV_FI_DYNBLOCK": labels.get("ai.radixark.flashinfer.dynblock", ""),
}
for name, value in values.items():
    print(f"{name}={shlex.quote(value)}")
