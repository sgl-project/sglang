"""Read a checkpoint's quantization settings from its HF config files."""

import os


def has_hf_quant_config(model_path: str) -> bool:
    """Check if the model path contains hf_quant_config.json file.

    Args:
        model_path: Path to the model, can be local path or remote URL.

    Returns:
        True if hf_quant_config.json exists, False otherwise.
    """
    # Local paths are decided on the filesystem; the hub helpers below
    # reject them as invalid repo ids.
    if os.path.isdir(model_path):
        return os.path.isfile(os.path.join(model_path, "hf_quant_config.json"))

    from huggingface_hub import try_to_load_from_cache

    # Check if the model_path is a HuggingFace model ID and exists locally
    result = try_to_load_from_cache(model_path, "hf_quant_config.json")
    if isinstance(result, str):
        return True

    # Check if the model_path is a remote URL and exists on the HuggingFace Hub
    try:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        return hf_api.file_exists(model_path, "hf_quant_config.json")
    except Exception:
        return False


def get_quantization_config(hf_config) -> str | None:
    """Extract quantization method from HuggingFace config."""
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return quantization_config.get("quant_method")
    return None


def has_fp8_weights_in_checkpoint(model_path: str) -> bool:
    """Check if a model checkpoint actually contains FP8 (float8_e4m3fn) expert
    weight tensors by reading safetensors metadata headers.

    This is needed because some models (e.g. DeepSeek V3/R1) use native FP8 MoE
    experts without declaring it in quantization_config, while other models
    sharing the same architecture (e.g. Moonlight) are purely BF16.

    Accepts a local directory or a HuggingFace repo ID. For remote repos, only
    safetensors headers (a few KB) are fetched via byte-range reads; full
    shards are never downloaded.
    """
    import json
    import struct

    try:
        if os.path.isdir(model_path):

            def _open(name):
                return open(os.path.join(model_path, name), "rb")

            def _exists(name):
                return os.path.exists(os.path.join(model_path, name))

        else:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()

            def _open(name):
                return fs.open(f"{model_path}/{name}", "rb")

            def _exists(name):
                return fs.exists(f"{model_path}/{name}")

        if _exists("model.safetensors.index.json"):
            with _open("model.safetensors.index.json") as f:
                weight_map = json.loads(f.read()).get("weight_map", {})
            expert_files = sorted(
                {v for k, v in weight_map.items() if "experts" in k and "weight" in k}
            )
            shard_file = (
                expert_files[0]
                if expert_files
                else next(iter(sorted(set(weight_map.values()))), None)
            )
            if shard_file is None:
                return False
        elif _exists("model.safetensors"):
            shard_file = "model.safetensors"
        else:
            return False

        with _open(shard_file) as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if "experts" in key and "weight" in key:
                return meta.get("dtype") == "F8_E4M3"
        return False
    except Exception:
        return False
