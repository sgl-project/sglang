"""Engram table and gate-basis declarations in ModelSlim checkpoints."""

import logging
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.runtime_context import get_context, get_model

logger = logging.getLogger(__name__)


def table_dtype(
    storage, num_embeddings: int, dim: int, default: torch.dtype
) -> torch.dtype:
    if storage is None:
        return default
    if default == torch.uint8:
        raise ValueError("INT8 Engram storage cannot also use MXFP4 tables")
    expected = {
        "format": "int8_sym",
        "weight_dtype": "I8",
        "layout": "row_major",
        "group_size": FP8_BLOCK_SIZE,
        "axis": 1,
        "scale_suffix": "scale",
        "scale_dtype": "F8_E8M0",
        "weight_shape": [num_embeddings, dim],
        "scale_shape": [num_embeddings, dim // FP8_BLOCK_SIZE],
    }
    if not isinstance(storage, dict) or any(
        storage.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(f"Unsupported Engram embedding storage: {storage}")
    return torch.int8


def load_gate_basis(config, model: torch.nn.Module) -> None:
    """Load the optional QuaRot basis once for all Engram layers on this rank."""
    rotation = getattr(config, "engram_rotation_config", None)
    if not rotation:
        return
    expected = {
        "value_projection_rotated": True,
        "value_basis": "quarot_global",
        "key_and_gate_basis": "original",
        "runtime_delta_rotation": False,
    }
    if not isinstance(rotation, dict) or any(
        rotation.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"Unsupported ModelSlim Engram rotation: {rotation}")
    if getattr(model, "engram_gate_basis", None) is not None:
        return
    engram = next(
        (
            layer.engram
            for layer in model.layers
            if getattr(layer, "engram", None) is not None
        ),
        None,
    )
    if engram is None:
        return
    if engram.q_weight.is_meta:
        raise ValueError("Load Engram weights before loading the gate rotation")

    if get_context().is_config_namespace_published("model"):
        from sglang.srt.model_loader.loader import loaded_checkpoint_files

        # A checkpoint's config can still name a different, existing directory.
        source = get_model()
        loaded = loaded_checkpoint_files(source.model_path, source.revision)
        model_path = loaded[0] if loaded is not None else source.model_path
    else:
        model_path = getattr(config, "_name_or_path", None)
    if not model_path:
        raise ValueError("ModelSlim Engram rotation requires the checkpoint path")
    path = Path(model_path) / "optional" / "quarot.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"ModelSlim Engram rotation requires {path}")
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        if "global_rotation" not in tensors.keys():
            raise ValueError(f"Missing global_rotation in {path}")
        basis = tensors.get_tensor("global_rotation")
    dim = config.hidden_size
    if basis.shape != (dim, dim) or not basis.is_floating_point():
        raise ValueError(
            f"global_rotation must be a floating-point [{dim}, {dim}] tensor"
        )
    basis = basis.float()
    if not torch.isfinite(basis).all():
        raise ValueError("global_rotation contains non-finite values")
    model.register_buffer(
        "engram_gate_basis",
        basis.to(device=engram.q_weight.device),
        persistent=False,
    )
    logger.info("Loaded ModelSlim Engram gate rotation from %s", path)
