"""Canonical metadata for the GPU-delta receiver's immutable startup checkpoint.

Read only at first delta admission. The exclusive controller must keep the
original local checkpoint unchanged for the engine's lifetime; this is metadata
discovery, not proof of the current model's weight values. Ordinary loaders and
model methods are never modified.
"""

from pathlib import Path

from safetensors import safe_open


def read_canonical_checkpoint_inventory(model_runner):
    """Read source names, shapes and dtypes without materializing any tensors.

    Only the standard local, serialized NVFP4 source contract is supported. Runtime
    layout admission separately checks the physical buffers and model mappings.
    """
    from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
    from sglang.srt.model_loader.weight_utils import (
        filter_duplicate_safetensors_files,
        maybe_add_mtp_safetensors,
    )

    model = model_runner.model
    config = model_runner.load_config
    model_config = model_runner.model_config
    loader = model_runner.loader
    # ModelOpt's already-quantized branch delegates to DefaultModelLoader.
    # Subclasses may override the iterator or transform its names and values.
    if type(loader) not in (DefaultModelLoader, ModelOptModelLoader):
        raise ValueError("GPU delta requires the standard local checkpoint loader")
    if type(loader) is ModelOptModelLoader and not model_config._is_already_quantized():
        raise ValueError("GPU delta does not support load-time ModelOpt conversion")
    if config.load_format not in {"auto", "safetensors", "fastsafetensors"}:
        raise ValueError("GPU delta requires a standard safetensors load format")
    if model_runner.is_draft_worker or config.draft_model_idx is not None:
        raise ValueError("GPU delta checkpoint admission requires the target model")
    if (
        getattr(model, "secondary_weights", ())
        or getattr(model, "allow_patterns_overrides", None) is not None
    ):
        raise ValueError("GPU delta does not support secondary or remapped sources")
    if config.decryption_key_file is not None:
        raise ValueError("GPU delta does not support encrypted checkpoint sources")
    quant_config = getattr(model, "quant_config", None)
    if not getattr(quant_config, "is_checkpoint_nvfp4_serialized", False) or getattr(
        quant_config, "is_nvfp4_online", False
    ):
        raise ValueError("GPU delta requires a serialized NVFP4 checkpoint")

    folder = Path(model_config.model_path)
    if not folder.is_dir():
        raise ValueError("GPU delta requires the original immutable local checkpoint")
    # Match the default loader's primary source (empty prefix, no draft remap).
    # Reuse its index filtering and bundled-MTP rules without invoking downloads,
    # checksum verification, tensor loading, or checkpoint page-cache prefetch.
    files = filter_duplicate_safetensors_files(
        [str(path) for path in folder.glob("*.safetensors")],
        str(folder),
        "model.safetensors.index.json",
    )
    files = maybe_add_mtp_safetensors(
        files, str(folder), "model.safetensors.index.json", model_config.hf_config
    )
    if not files:
        raise ValueError("GPU delta checkpoint has no selected safetensors shards")
    return _read_headers(files)


def _read_headers(files):
    inventory = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 - safe_open is not iterable.
                if name in inventory:
                    raise ValueError(f"ambiguous canonical checkpoint tensor: {name}")
                tensor_slice = source.get_slice(name)
                inventory[name] = {
                    "shape": tensor_slice.get_shape(),
                    "dtype": tensor_slice.get_dtype(),
                }
    if not inventory:
        raise ValueError("canonical checkpoint metadata is empty")
    return inventory
