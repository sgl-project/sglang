#!/usr/bin/env python3
"""检查 FP4 重量化前后的 tensor 结构和抽样反量化误差。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open


FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_fp4_expert_weight_name(name: str) -> bool:
    return (
        name.endswith(".weight")
        and (".ffn.experts." in name or ".ffn.shared_experts." in name)
    )


def choose_tensor_name(weight_map: dict[str, str]) -> str:
    routed = sorted(
        name for name in weight_map if name.endswith(".weight") and ".ffn.experts." in name
    )
    if routed:
        return routed[0]
    shared = sorted(
        name
        for name in weight_map
        if name.endswith(".weight") and ".ffn.shared_experts." in name
    )
    if shared:
        return shared[0]
    raise KeyError("no expert weight tensor found in index")


def unpack_fp4_values(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError(f"expected int8 2D packed tensor, got {packed.dtype}, {tuple(packed.shape)}")

    out_dim, packed_in_dim = packed.shape
    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    return torch.stack(
        (FP4_TABLE[low.long()], FP4_TABLE[high.long()]),
        dim=-1,
    ).reshape(out_dim, packed_in_dim * 2)


def dequant_fp4_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    fp4 = unpack_fp4_values(packed)
    out_dim, in_dim = fp4.shape
    expected_shape = (out_dim, in_dim // group_size)
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"scale shape mismatch: got {tuple(scale.shape)}, expected {expected_shape}"
        )
    full_scale = scale.float().repeat_interleave(group_size, dim=1)
    return fp4 * full_scale


def quantize_real_to_fp4(real: torch.Tensor, group_size: int, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    if real.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(real.shape)}")

    out_dim, in_dim = real.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by group_size={group_size}")

    num_groups = in_dim // group_size
    grouped = real.view(out_dim, num_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1)
    scale = torch.clamp(max_abs / 6.0, min=eps)

    normalized = grouped / scale.unsqueeze(-1)
    codebook = FP4_TABLE.view(1, 1, 1, 16)
    nibble = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)
    nibble = nibble.view(out_dim, in_dim)

    low = nibble[:, 0::2]
    high = nibble[:, 1::2]
    packed = (low | (high << 4)).view(torch.int8)
    return packed, scale.float()


def infer_group_size_from_shapes(weight: torch.Tensor, scale: torch.Tensor) -> int:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"expected 2D weight/scale, got {tuple(weight.shape)} and {tuple(scale.shape)}"
        )
    out_dim, packed_in_dim = weight.shape
    real_in_dim = packed_in_dim * 2
    if scale.shape[0] != out_dim:
        raise ValueError(
            f"scale first dim mismatch: weight out_dim={out_dim}, scale shape={tuple(scale.shape)}"
        )
    if scale.shape[1] == 0 or real_in_dim % scale.shape[1] != 0:
        raise ValueError(
            f"cannot infer group_size from weight {tuple(weight.shape)} and scale {tuple(scale.shape)}"
        )
    return real_in_dim // scale.shape[1]


def sample_abs_error(
    src_weight: torch.Tensor,
    src_scale: torch.Tensor,
    src_group_size: int,
    dst_weight: torch.Tensor,
    dst_scale: torch.Tensor,
    dst_group_size: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if tuple(src_weight.shape) != tuple(dst_weight.shape):
        raise ValueError(
            f"weight shape mismatch: src={tuple(src_weight.shape)} dst={tuple(dst_weight.shape)}"
        )

    out_dim, packed_in_dim = src_weight.shape
    in_dim = packed_in_dim * 2
    total = out_dim * in_dim
    actual_samples = min(samples, total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    flat_idx = torch.randint(total, (actual_samples,), generator=generator, device="cpu")
    rows = flat_idx // in_dim
    cols = flat_idx % in_dim

    src_dequant = dequant_fp4_grouped(src_weight, src_scale, src_group_size)
    dst_dequant = dequant_fp4_grouped(dst_weight, dst_scale, dst_group_size)

    src_values = src_dequant[rows, cols]
    dst_values = dst_dequant[rows, cols]
    abs_error = (src_values - dst_values).abs()
    worst_idx = int(abs_error.argmax().item()) if actual_samples > 0 else 0
    return {
        "samples": actual_samples,
        "max_abs_error": float(abs_error.max().item()) if actual_samples > 0 else 0.0,
        "mean_abs_error": float(abs_error.mean().item()) if actual_samples > 0 else 0.0,
        "worst": {
            "row": int(rows[worst_idx]) if actual_samples > 0 else None,
            "col": int(cols[worst_idx]) if actual_samples > 0 else None,
            "src": float(src_values[worst_idx]) if actual_samples > 0 else None,
            "dst": float(dst_values[worst_idx]) if actual_samples > 0 else None,
            "abs_error": float(abs_error[worst_idx]) if actual_samples > 0 else None,
        },
    }


def format_tensor_info(prefix: str, weight: torch.Tensor, scale: torch.Tensor, group_size: int) -> list[str]:
    real_in_dim = weight.shape[1] * 2
    return [
        f"{prefix} weight dtype: {weight.dtype}",
        f"{prefix} weight shape: {tuple(weight.shape)}",
        f"{prefix} real in_dim: {real_in_dim}",
        f"{prefix} scale dtype: {scale.dtype}",
        f"{prefix} scale shape: {tuple(scale.shape)}",
        f"{prefix} inferred group_size: {group_size}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 FP4 转换前后的 tensor 结构和抽样反量化误差。"
    )
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--tensor-name", type=str, default=None)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    src_index = load_json(args.src_dir / "model.safetensors.index.json")["weight_map"]
    dst_index = load_json(args.dst_dir / "model.safetensors.index.json")["weight_map"]

    tensor_name = args.tensor_name or choose_tensor_name(src_index)
    if not is_fp4_expert_weight_name(tensor_name):
        raise ValueError(f"tensor_name does not look like an expert FP4 weight: {tensor_name}")
    if tensor_name not in src_index:
        raise KeyError(f"{tensor_name} not found in src index")
    if tensor_name not in dst_index:
        raise KeyError(f"{tensor_name} not found in dst index")

    scale_name = tensor_name.removesuffix(".weight") + ".scale"
    if scale_name not in src_index:
        raise KeyError(f"{scale_name} not found in src index")
    if scale_name not in dst_index:
        raise KeyError(f"{scale_name} not found in dst index")

    src_shard = args.src_dir / src_index[tensor_name]
    dst_shard = args.dst_dir / dst_index[tensor_name]

    with safe_open(src_shard, framework="pt", device="cpu") as src_reader:
        src_weight = src_reader.get_tensor(tensor_name)
        src_scale = src_reader.get_tensor(scale_name)

    with safe_open(dst_shard, framework="pt", device="cpu") as dst_reader:
        dst_weight = dst_reader.get_tensor(tensor_name)
        dst_scale = dst_reader.get_tensor(scale_name)

    src_group_size = infer_group_size_from_shapes(src_weight, src_scale)
    dst_group_size = infer_group_size_from_shapes(dst_weight, dst_scale)

    print(f"tensor name: {tensor_name}")
    print(f"src shard: {src_shard.name}")
    print(f"dst shard: {dst_shard.name}")
    print()
    for line in format_tensor_info("src", src_weight, src_scale, src_group_size):
        print(line)
    print()
    for line in format_tensor_info("dst", dst_weight, dst_scale, dst_group_size):
        print(line)
    print()

    stats = sample_abs_error(
        src_weight=src_weight,
        src_scale=src_scale,
        src_group_size=src_group_size,
        dst_weight=dst_weight,
        dst_scale=dst_scale,
        dst_group_size=dst_group_size,
        samples=args.samples,
        seed=args.seed,
    )
    print(f"samples: {stats['samples']}")
    print(f"max_abs_error: {stats['max_abs_error']}")
    print(f"mean_abs_error: {stats['mean_abs_error']}")
    print(f"worst_sample: {stats['worst']}")


if __name__ == "__main__":
    main()
