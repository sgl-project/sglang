from __future__ import annotations

import argparse

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
    parser.add_argument("--official-step-dump", required=True)
    parser.add_argument("--sglang-coords-dump", required=True)
    args = parser.parse_args()

    official = torch.load(args.official_step_dump, map_location="cpu")
    sglang = torch.load(args.sglang_coords_dump, map_location="cpu")

    official_video = official["video_positions"]
    official_audio = official["audio_positions"]
    sglang_video = sglang["video_coords"]
    sglang_audio = sglang["audio_coords"]

    result = {
        "video_positions": metric(official_video, sglang_video),
        "audio_positions": metric(official_audio, sglang_audio),
        "video_time_positions": metric(
            official_video[:, 0:1, ...], sglang_video[:, 0:1, ...]
        ),
        "audio_time_positions": metric(
            official_audio[:, 0:1, ...], sglang_audio[:, 0:1, ...]
        ),
    }
    print(result)


if __name__ == "__main__":
    main()
