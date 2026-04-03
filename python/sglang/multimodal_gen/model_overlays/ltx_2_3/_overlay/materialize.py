import os

from huggingface_hub import snapshot_download

from sglang.multimodal_gen.runtime.utils.model_overlay import (
    _copytree_link_or_copy,
    _ensure_dir,
    _link_or_copy_file,
)

AUXILIARY_MODEL_ID = "Lightricks/LTX-2"
AUXILIARY_PATTERNS = [
    "audio_vae/**",
    "connectors/**",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "transformer/config.json",
    "vae/**",
    "vocoder/**",
]


def materialize(
    *,
    overlay_dir: str,
    source_dir: str,
    output_dir: str,
    manifest: dict,
) -> None:
    _ = overlay_dir, manifest

    auxiliary_dir = snapshot_download(
        repo_id=AUXILIARY_MODEL_ID,
        allow_patterns=AUXILIARY_PATTERNS,
        max_workers=8,
    )

    for component_name in (
        "audio_vae",
        "connectors",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "vae",
        "vocoder",
    ):
        _copytree_link_or_copy(
            os.path.join(auxiliary_dir, component_name),
            os.path.join(output_dir, component_name),
        )

    transformer_dir = os.path.join(output_dir, "transformer")
    _ensure_dir(transformer_dir)
    _link_or_copy_file(
        os.path.join(auxiliary_dir, "transformer", "config.json"),
        os.path.join(transformer_dir, "config.json"),
    )
    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-22b-dev.safetensors"),
        os.path.join(transformer_dir, "ltx-2.3-22b-dev.safetensors"),
    )

    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
        os.path.join(output_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"),
    )
    _link_or_copy_file(
        os.path.join(source_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        os.path.join(output_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    )
