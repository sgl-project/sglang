# SPDX-License-Identifier: Apache-2.0
"""Rewrite a MiniMax-H3 transformer with native QKV rows, so it can stay mapped.

    python3 -m sglang.multimodal_gen.tools.convert_minimax_h3_native_qkv \
        <model dir> <out dir>

The released checkpoints interleave the Q, K and V rows of every attention head.
SGLang reorders them at load, which needs an anonymous host copy of the fused QKV
weights (about 11 GiB per DiT) that the page cache cannot drop. This tool applies the
same permutation offline and sets `checkpoint_qkv_layout: native` in
`transformer/config.json`; the loader then keeps the weights on the file mapping.

`<model dir>` holds `transformer/` next to the other components: the `FL2VA/` or
`Ref2VA/` partition of MiniMax-H3, or the root of MiniMax-H3-Image. Shards are
rewritten in `<out dir>/transformer`; everything else is linked to the source, so keep
the source in place. The output needs about as much disk as the transformer.

Trade-off: under layerwise offload the QKV weights are read from disk on every pass
instead of staying in host memory. Quantized, LoRA-merged or already native
transformers are rejected or unsupported; LoRA on the converted weights is untested.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

_QKV_SUFFIX = ".attn.qkv_proj.weight"
_COPY_CHUNK_BYTES = 1 << 24


def _read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header and the offset of the tensor data."""
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def _native_qkv_rows(raw: bytes, heads: int) -> bytes:
    """Per-head [q, k, v] row blocks -> [q_all, k_all, v_all], on raw bytes."""
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(heads, 3, -1)
    return np.ascontiguousarray(rows.transpose(1, 0, 2)).tobytes()


def _qkv_spans(
    header: dict, data_start: int, heads: int, head_dim: int, path: Path
) -> list[tuple[int, int, str]]:
    spans = []
    for name, info in header.items():
        if name == "__metadata__" or not name.endswith(_QKV_SUFFIX):
            continue
        rows = 3 * heads * head_dim
        if len(info["shape"]) != 2 or info["shape"][0] != rows:
            raise ValueError(
                f"{path.name}: {name} has shape {info['shape']}, expected {rows} "
                "grouped QKV rows"
            )
        begin, end = info["data_offsets"]
        if (end - begin) % rows:
            raise ValueError(f"{path.name}: {name} does not hold whole rows")
        spans.append((data_start + begin, data_start + end, name))
    return sorted(spans)


def _copy(src, dst, count: int) -> None:
    while count:
        chunk = src.read(min(_COPY_CHUNK_BYTES, count))
        if not chunk:
            raise ValueError("unexpected end of file")
        dst.write(chunk)
        count -= len(chunk)


def convert_shard(source: Path, target: Path, heads: int, head_dim: int) -> int:
    """Copy one shard with native QKV rows; returns how many tensors changed."""
    header, data_start = _read_header(source)
    spans = _qkv_spans(header, data_start, heads, head_dim, source)
    with source.open("rb") as src, target.open("wb") as dst:
        position = 0
        for begin, end, _ in spans:
            _copy(src, dst, begin - position)
            dst.write(_native_qkv_rows(src.read(end - begin), heads))
            position = end
        _copy(src, dst, source.stat().st_size - position)
    return len(spans)


def _convert_into(model_dir: Path, out_dir: Path) -> int:
    source = model_dir / "transformer"
    config = json.loads((source / "config.json").read_text())
    if config.get("checkpoint_qkv_layout") == "native":
        raise ValueError(f"{source} is already in the native QKV layout")
    if config.get("quantization_config"):
        raise ValueError("quantized transformers are not supported")
    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{source} holds no .safetensors shards")

    target = out_dir / "transformer"
    target.mkdir(parents=True)
    changed = sum(
        convert_shard(shard, target / shard.name, heads, head_dim) for shard in shards
    )
    expected = int(config.get("num_layers", 0)) + int(
        config.get("token_refiner_num_layers", 0)
    )
    if expected and changed != expected:
        raise ValueError(f"converted {changed} QKV tensors, expected {expected}")
    for entry in source.iterdir():
        if entry.name == "config.json":
            config["checkpoint_qkv_layout"] = "native"
            (target / entry.name).write_text(json.dumps(config, indent=2) + "\n")
        elif entry.suffix != ".safetensors":
            shutil.copyfile(entry.resolve(), target / entry.name)
    for entry in sorted(model_dir.iterdir()):
        if entry.name != "transformer":
            # Hub snapshots hold links into a shared blob folder: follow them once
            target_path = entry.resolve() if entry.is_symlink() else entry
            (out_dir / entry.name).symlink_to(target_path)
    return changed


def convert(model_dir: Path, out_dir: Path) -> int:
    """Write the converted model to out_dir; returns how many tensors changed."""
    model_dir = model_dir.resolve()
    if not (model_dir / "transformer" / "config.json").is_file():
        raise ValueError(f"{model_dir} has no transformer/config.json")
    if out_dir.exists():
        raise ValueError(f"{out_dir} already exists")
    # finish under a temporary name so a failure leaves no half-converted model
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        raise ValueError(f"{partial} exists; remove it first")
    try:
        changed = _convert_into(model_dir, partial)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    os.replace(partial, out_dir)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        changed = convert(args.model_dir, args.out_dir)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"rewrote {changed} QKV tensors into {args.out_dir / 'transformer'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
