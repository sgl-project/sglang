# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from safetensors import safe_open


def inspect_ideogram_safetensors(paths: list[str]) -> Literal["fp8", "bf16"]:
    """Inspect the storage layout of one Ideogram component without loading tensors."""
    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate Ideogram tensor {name!r} in {path}")
                tensor = checkpoint.get_slice(name)
                tensors[name] = (tensor.get_dtype(), tensor.get_shape())

    if not tensors:
        raise ValueError("No tensors found in Ideogram checkpoint")
    if all(dtype == "BF16" for dtype, _ in tensors.values()):
        return "bf16"

    has_fp8 = False
    for name, (dtype, shape) in tensors.items():
        if dtype in {"BF16", "F16", "F32"}:
            continue
        if dtype != "F8_E4M3" or not name.endswith(".weight"):
            raise ValueError(f"Unsupported Ideogram tensor {name!r}: dtype {dtype}")
        has_fp8 = True
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        if scale_name not in tensors:
            raise ValueError(f"FP8 weight {name!r} is missing scale {scale_name!r}")
        scale_dtype, scale_shape = tensors[scale_name]
        if (
            len(shape) != 2
            or scale_shape != [shape[0]]
            or scale_dtype not in {"F32", "BF16", "F16"}
        ):
            raise ValueError(
                f"Invalid row-wise scale for FP8 weight {name!r}: "
                f"weight shape {shape}, scale shape {scale_shape}, dtype {scale_dtype}"
            )

    if not has_fp8:
        raise ValueError("Expected an FP8-with-scales or all-BF16 Ideogram checkpoint")
    return "fp8"
