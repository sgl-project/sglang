from __future__ import annotations

import argparse
from pathlib import Path

import torch


def metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    a = a.float()
    b = b.float()
    diff = a - b
    cosine = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1), b.reshape(1, -1)
    ).item()
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "cosine": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--sglang-dir", required=True)
    parser.add_argument("--prefix", default="transformer_blocks.0.video_to_audio_attn")
    parser.add_argument("--max-calls", type=int, default=4)
    args = parser.parse_args()

    keys = [
        "x",
        "context",
        "pe_cos",
        "pe_sin",
        "k_pe_cos",
        "k_pe_sin",
        "out",
    ]

    result: dict[str, object] = {
        "prefix": args.prefix,
        "official_dir": args.official_dir,
        "sglang_dir": args.sglang_dir,
        "calls": {},
    }
    for call_idx in range(args.max_calls):
        official_path = (
            Path(args.official_dir) / f"{args.prefix}.call{call_idx}.pt"
        )
        sglang_path = Path(args.sglang_dir) / f"{args.prefix}.call{call_idx}.pt"
        if not official_path.exists() or not sglang_path.exists():
            continue
        official = torch.load(official_path, map_location="cpu")
        sglang = torch.load(sglang_path, map_location="cpu")
        call_metrics = {}
        for key in keys:
            official_tensor = official.get(key)
            sglang_tensor = sglang.get(key)
            if isinstance(official_tensor, torch.Tensor) and isinstance(
                sglang_tensor, torch.Tensor
            ):
                call_metrics[key] = metric(official_tensor, sglang_tensor)
        result["calls"][str(call_idx)] = call_metrics

    print(result)


if __name__ == "__main__":
    main()
