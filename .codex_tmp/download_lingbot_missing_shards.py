#!/usr/bin/env python3
from __future__ import annotations

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = "robbyant/lingbot-world-fast-diffusers"
CACHE_DIR = "/scratch/hf_cache/hub"
FILES = [
    "transformer/diffusion_pytorch_model-00014-of-00016.safetensors",
    "transformer/diffusion_pytorch_model-00015-of-00016.safetensors",
    "transformer/diffusion_pytorch_model-00016-of-00016.safetensors",
]


for filename in FILES:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        cache_dir=CACHE_DIR,
    )
    print(path)

print(
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["vae/*"],
        cache_dir=CACHE_DIR,
    )
)
