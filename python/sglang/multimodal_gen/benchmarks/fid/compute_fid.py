# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def compute_fid(
    ref_dir: Path,
    sample_dir: Path,
    device: str,
    num_workers: int,
    batch_size: int,
) -> float:
    try:
        from pytorch_fid import fid_score
    except ImportError as e:
        raise RuntimeError(
            "pytorch-fid is required. Install with: pip install pytorch-fid"
        ) from e

    return float(
        fid_score.calculate_fid_given_paths(
            [str(ref_dir), str(sample_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute FID between two folders of images using pytorch-fid."
    )
    parser.add_argument(
        "--ref-dir", type=str, required=True, help="Reference images folder"
    )
    parser.add_argument(
        "--sample-dir", type=str, required=True, help="Generated images folder"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Torch device string (e.g. cpu, cuda). Other devices may work if your PyTorch build supports them."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for Inception forward pass (pytorch-fid)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    ref_dir = Path(args.ref_dir)
    sample_dir = Path(args.sample_dir)

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {ref_dir}")
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    n_ref = count_images(ref_dir)
    n_samp = count_images(sample_dir)
    if n_ref == 0:
        raise RuntimeError(f"No images found in ref dir: {ref_dir}")
    if n_samp == 0:
        raise RuntimeError(f"No images found in sample dir: {sample_dir}")

    score = compute_fid(
        ref_dir=ref_dir,
        sample_dir=sample_dir,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    print(f"FID: {score:.4f}")
    print(f"ref: {ref_dir} ({n_ref} images)")
    print(f"samples: {sample_dir} ({n_samp} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
