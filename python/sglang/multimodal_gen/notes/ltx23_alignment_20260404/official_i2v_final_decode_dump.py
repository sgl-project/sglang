from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEFAULT_LTX_REPO_ROOTS = ["/tmp/LTX-2-official", "/tmp/LTX-2"]
LTX_REPO_ROOT = Path(
    os.environ.get(
        "LTX_REPO_ROOT",
        next(
            (p for p in DEFAULT_LTX_REPO_ROOTS if Path(p).exists()),
            DEFAULT_LTX_REPO_ROOTS[0],
        ),
    )
)
CHECKPOINT_GLOB = (
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-22b-dev.safetensors"
)
FINAL_LATENTS_PATH = Path(
    os.environ.get("LTX23_OFFICIAL_FINAL_LATENTS", "/tmp/ltx23_official_i2v_final.pt")
)
DUMP_PATH = Path(
    os.environ.get("LTX23_OFFICIAL_DECODE_DUMP", "/tmp/ltx23_official_i2v_decode.pt")
)


def resolve_single_path(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern}, got {matches}")
    return matches[0]


@torch.inference_mode()
def main() -> None:
    checkpoint_path = resolve_single_path(CHECKPOINT_GLOB)

    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-core/src"))
    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-pipelines/src"))

    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.video_vae import (
        VAE_DECODER_COMFY_KEYS_FILTER,
        VideoDecoderConfigurator,
    )
    from ltx_pipelines.utils.blocks import Builder

    payload = torch.load(FINAL_LATENTS_PATH, map_location="cpu")
    latent = payload["video_latent_after"].to(device="cuda", dtype=torch.bfloat16)

    builder = Builder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )
    decoder = builder.build(device=torch.device("cuda"), dtype=torch.bfloat16).to(
        "cuda"
    )
    decoder.eval()

    raw_video = decoder(latent)
    uint8_chunks = list(decoder.decode_video(latent))
    if len(uint8_chunks) != 1:
        raise RuntimeError(f"Expected single decode chunk, got {len(uint8_chunks)}")

    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "raw_video": raw_video.detach().cpu().float(),
            "postprocessed_video": uint8_chunks[0].detach().cpu(),
        },
        DUMP_PATH,
    )
    print(DUMP_PATH)


if __name__ == "__main__":
    main()
