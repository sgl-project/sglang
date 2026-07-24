#!/usr/bin/env python3
"""把 DeepSeek 风格的 packed FP4 权重重新量化为 group_size=128。

这个脚本对应的是：
1. 源权重仍然是 FP4（E2M1-like codebook，两个 4-bit 打包到一个 int8 字节）
2. 源 scale 是按较小 group（默认 32）提供
3. 目标仍然输出 FP4，只是把 scale 粒度改成更大的 group（默认 128）

脚本会处理：
- `layers.*.ffn.experts.*.weight`
- `layers.*.ffn.shared_experts.*.weight`

注意：
- 这不是 AWQ/GPTQ/Marlin 那类 `bits=4, group_size=128` 的线性 INT4 格式转换。
- 如果目标后端期待的是 `qweight/qzeros/scales/g_idx` 一类张量布局，本脚本不能直接满足，
  那种场景需要“反量化到浮点 -> 走目标量化器重新量化”。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def looks_like_fp8_config(src_dir: Path) -> bool:
    config_path = src_dir / "config.json"
    if not config_path.exists():
        return False
    config = load_json(config_path)
    quant_config = config.get("quantization_config", {})
    return (
        quant_config.get("quant_method") == "fp8"
        or quant_config.get("fmt") == "e4m3"
        or quant_config.get("weight_block_size") == [128, 128]
    )


def is_fp4_expert_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        (".ffn.experts." in name or ".ffn.shared_experts." in name)
        and name.endswith(".weight")
        and tensor.dtype == torch.int8
        and tensor.ndim == 2
    )


def unpack_fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """把 packed int8 中的两个 nibble 解成 FP4 码本值。"""
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError(f"expected int8 2D packed tensor, got {packed.dtype}, {packed.shape}")

    out_dim, packed_in_dim = packed.shape
    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    values = torch.stack(
        (FP4_TABLE[low.long()], FP4_TABLE[high.long()]),
        dim=-1,
    ).reshape(out_dim, packed_in_dim * 2)
    return values


def dequant_fp4_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    src_group_size: int,
) -> torch.Tensor:
    """按源 group_size 反量化得到实值张量。"""
    fp4 = unpack_fp4_values(packed)
    out_dim, in_dim = fp4.shape
    expected_scale_shape = (out_dim, in_dim // src_group_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape mismatch: got {tuple(scale.shape)}, expected {expected_scale_shape}"
        )

    full_scale = scale.float().repeat_interleave(src_group_size, dim=1)
    return fp4 * full_scale


def quantize_real_to_fp4(
    real: torch.Tensor,
    dst_group_size: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把实值重新量化到 FP4，并生成新的 group scale。"""
    if real.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(real.shape)}")

    out_dim, in_dim = real.shape
    if in_dim % dst_group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by dst_group_size={dst_group_size}")

    num_groups = in_dim // dst_group_size
    grouped = real.view(out_dim, num_groups, dst_group_size)

    # FP4_TABLE 的最大幅值是 6.0，所以新的 scale 用 max_abs / 6。
    max_abs = grouped.abs().amax(dim=-1)
    new_scale = torch.clamp(max_abs / 6.0, min=eps)

    normalized = grouped / new_scale.unsqueeze(-1)
    codebook = FP4_TABLE.view(1, 1, 1, 16)
    dist = (normalized.unsqueeze(-1) - codebook).abs()
    nibble = dist.argmin(dim=-1).to(torch.uint8)

    # 打包回 int8：偶数列放低 4 bit，奇数列放高 4 bit。
    nibble = nibble.view(out_dim, in_dim)
    if in_dim % 2 != 0:
        raise ValueError(f"in_dim={in_dim} must be even for nibble packing")
    low = nibble[:, 0::2]
    high = nibble[:, 1::2]
    packed_u8 = low | (high << 4)
    packed_i8 = packed_u8.view(torch.int8)
    return packed_i8, new_scale.float()


def regroup_fp4_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    src_group_size: int,
    dst_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    real = dequant_fp4_grouped(packed, scale, src_group_size=src_group_size)
    return quantize_real_to_fp4(real, dst_group_size=dst_group_size)


def convert_config(src_dir: Path, out_dir: Path, dst_group_size: int) -> None:
    config_path = src_dir / "config.json"
    if not config_path.exists():
        return

    config = load_json(config_path)
    quant_config = config.setdefault("quantization_config", {})
    quant_config["group_size"] = dst_group_size
    quant_config["quant_method"] = "fp8"
    quant_config.pop("weight_block_size", None)
    quant_config.pop("activation_scheme", None)
    quant_config.pop("fmt", None)
    quant_config.pop("scale_fmt", None)
    save_json(out_dir / "config.json", config)


def should_skip_copy(rel_path: Path) -> bool:
    name = rel_path.name
    if rel_path == Path("config.json"):
        return True
    if rel_path == Path("model.safetensors.index.json"):
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    return False


def copy_non_weight_artifacts(src_dir: Path, out_dir: Path) -> None:
    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if rel == Path("."):
            continue
        if should_skip_copy(rel):
            continue

        dst = out_dir / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def read_existing_shard_keys_and_size(path: Path) -> tuple[set[str], int]:
    with safe_open(path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        total_size = 0
        for key in keys:
            tensor = reader.get_tensor(key)
            total_size += tensor.numel() * tensor.element_size()
    return keys, total_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert packed FP4 tensors to group_size=128 FP4.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src-group-size", type=int, default=32)
    parser.add_argument("--dst-group-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑。若输出 shard 已存在且包含该 shard 所需全部 tensor，则直接跳过。",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="每处理多少个 shard 打印一次进度，默认每个 shard 都打印。",
    )
    parser.add_argument(
        "--skip-input-format-check",
        action="store_true",
        help="跳过 config.json 的输入格式检查。仅当你确认目录虽然写着 fp8，但实际 expert 权重仍是 packed FP4 时使用。",
    )
    args = parser.parse_args()

    src_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    if looks_like_fp8_config(src_dir) and not args.skip_input_format_check:
        raise ValueError(
            "input config.json looks like an FP8 checkpoint already. "
            "If your actual expert weights are still packed FP4 and only config.json is misleading, "
            "rerun with --skip-input-format-check."
        )

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to continue")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(index_path)
    weight_map = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file_name in weight_map.items():
        by_file.setdefault(file_name, []).append(name)

    emitted_map: dict[str, str] = {}
    ordered_files = sorted(by_file)
    total_files = len(ordered_files)
    print(f"starting conversion: {total_files} shard(s)", flush=True)

    for file_idx, file_name in enumerate(ordered_files, start=1):
        src_file = src_dir / file_name
        out_file = out_dir / file_name
        expected_names = set(by_file[file_name])

        if args.resume and out_file.exists():
            existing_keys, _ = read_existing_shard_keys_and_size(out_file)
            if expected_names.issubset(existing_keys):
                emitted_map.update({name: file_name for name in expected_names})
                if file_idx == 1 or file_idx % args.progress_every == 0 or file_idx == total_files:
                    print(
                        f"[{file_idx}/{total_files}] skipping existing completed shard: {file_name}",
                        flush=True,
                    )
                continue
            print(
                f"[{file_idx}/{total_files}] existing shard incomplete, regenerating: {file_name}",
                flush=True,
            )
        elif file_idx == 1 or file_idx % args.progress_every == 0 or file_idx == total_files:
            print(f"[{file_idx}/{total_files}] converting shard: {file_name}", flush=True)

        state_dict: dict[str, torch.Tensor] = {}
        with safe_open(src_file, framework="pt", device="cpu") as reader:
            key_set = set(reader.keys())
            consumed: set[str] = set()
            for name in by_file[file_name]:
                if name in consumed:
                    continue
                if name not in key_set:
                    continue
                tensor = reader.get_tensor(name)

                if is_fp4_expert_weight(name, tensor):
                    scale_name = name.removesuffix(".weight") + ".scale"
                    if scale_name not in key_set:
                        raise KeyError(f"missing scale tensor for {name}: {scale_name}")
                    new_weight, new_scale = regroup_fp4_tensor(
                        tensor,
                        reader.get_tensor(scale_name),
                        src_group_size=args.src_group_size,
                        dst_group_size=args.dst_group_size,
                    )
                    state_dict[name] = new_weight
                    state_dict[scale_name] = new_scale
                    emitted_map[name] = file_name
                    emitted_map[scale_name] = file_name
                    consumed.add(scale_name)
                    continue

                if name.endswith(".scale"):
                    # 非 expert 的 scale 先原样保留；如果目标后端也要求改布局，需要再单独处理。
                    state_dict[name] = tensor.float()
                else:
                    state_dict[name] = tensor
                emitted_map[name] = file_name

        save_file(state_dict, out_file, metadata={"format": "pt"})

    total_size = 0
    for file_name in sorted(set(emitted_map.values())):
        _, shard_size = read_existing_shard_keys_and_size(out_dir / file_name)
        total_size += shard_size

    save_json(
        out_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(emitted_map.items())),
        },
    )
    convert_config(src_dir, out_dir, args.dst_group_size)
    copy_non_weight_artifacts(src_dir, out_dir)

    print(f"converted checkpoint written to {out_dir}")


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    main()
