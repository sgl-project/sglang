# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(8192)
    # Prefer tab for COCO-style TSVs.
    if "\t" in sample:
        return "\t"
    try:
        return csv.Sniffer().sniff(sample, delimiters=["\t", ","]).delimiter
    except csv.Error:
        return "\t"


def read_captions_tsv(captions_file: Path) -> list[dict]:
    delim = _sniff_delimiter(captions_file)
    with captions_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in captions file: {captions_file}")

    # Normalize common column names.
    for row in rows:
        # Some dumps use 'file_name', others 'filename'
        if "file_name" not in row and "filename" in row:
            row["file_name"] = row["filename"]
        if row.get("caption"):
            row["caption"] = row["caption"].replace("\n", " ").strip()

    required = {"caption", "file_name"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Captions file must include columns {sorted(required)}; missing: {sorted(missing)}. "
            f"Got columns: {sorted(rows[0].keys())}"
        )
    return rows


def read_coco_captions_json(caption_file: Path) -> list[dict]:
    with caption_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(
            "Expected COCO captions JSON with top-level lists 'images' and 'annotations'"
        )

    image_id_to_name: dict[int, str] = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = image.get("id")
        file_name = image.get("file_name")
        if isinstance(image_id, int) and isinstance(file_name, str) and file_name:
            image_id_to_name[image_id] = file_name

    rows: list[dict] = []
    seen_image_ids: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if not isinstance(image_id, int) or not isinstance(caption, str):
            continue
        if image_id not in image_id_to_name or image_id in seen_image_ids:
            continue
        cleaned_caption = caption.replace("\n", " ").strip()
        if not cleaned_caption:
            continue

        seen_image_ids.add(image_id)
        rows.append(
            {
                "image_id": image_id,
                "file_name": image_id_to_name[image_id],
                "caption": cleaned_caption,
            }
        )

    if not rows:
        raise ValueError(f"No caption rows found in COCO JSON: {caption_file}")
    return rows


def select_samples(
    rows: list[dict],
    coco_val_dir: Path,
    num_samples: int,
    seed: int,
    shuffle_rows: bool,
) -> list[dict]:
    candidates = []
    for row in rows:
        caption = (row.get("caption") or "").strip()
        file_name = (row.get("file_name") or "").strip()
        if not caption or not file_name:
            continue
        img_path = coco_val_dir / file_name
        if not img_path.exists():
            continue
        candidates.append(row)

    if not candidates:
        raise ValueError(
            f"No usable rows found. Checked for images under: {coco_val_dir}"
        )

    if shuffle_rows:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    selected = candidates[:num_samples]
    if len(selected) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but only found {len(selected)} with existing images under {coco_val_dir}."
        )
    return selected


def write_prompts_file(selected_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(row["caption"].strip() + "\n")


def prepare_ref_folder(
    selected_rows: list[dict],
    coco_val_dir: Path,
    ref_out_dir: Path,
    ref_size: int | None,
) -> None:
    ref_out_dir.mkdir(parents=True, exist_ok=True)

    if ref_size is None:
        # Symlink (or copy) originals.
        for row in selected_rows:
            src = coco_val_dir / row["file_name"].strip()
            dst = ref_out_dir / src.name
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        return

    # Resize to a fixed size like the xDiT benchmark (256x256).
    try:
        from PIL import Image
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "PIL/Pillow is required for --ref-size resizing. Install pillow or set --ref-size=0."
        ) from e

    target = (ref_size, ref_size)
    for row in selected_rows:
        src = coco_val_dir / row["file_name"].strip()
        dst = ref_out_dir / src.name
        if dst.exists():
            continue
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize(target, Image.Resampling.LANCZOS)
            im.save(dst, quality=95)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N COCO captions + reference images into a prompt.txt file and a reference image folder."
        )
    )
    parser.add_argument(
        "--caption-file",
        "--captions-file",
        dest="caption_file",
        type=str,
        required=True,
        help=(
            "COCO captions JSON, e.g. captions_val2014.json. TSV/CSV with "
            "caption and file_name columns is also supported."
        ),
    )
    parser.add_argument(
        "--coco-val-dir",
        type=str,
        required=True,
        help="Directory containing COCO val2014 images",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path.home() / "outputs" / "fid_flux_coco"),
        help="Base output directory",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of prompts/images to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sampling/shuffling"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle caption rows before selecting"
    )

    # Reference preprocessing
    parser.add_argument(
        "--ref-size",
        type=int,
        default=256,
        help=(
            "If >0, resize reference images to ref_size x ref_size (xDiT-style). "
            "If 0, use original sizes (symlink/copy)."
        ),
    )

    args = parser.parse_args()

    coco_val_dir = Path(args.coco_val_dir)
    out_dir = Path(args.out_dir)

    caption_file = Path(args.caption_file)

    if not caption_file.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
    if not coco_val_dir.exists():
        raise FileNotFoundError(f"COCO val dir not found: {coco_val_dir}")

    if args.ref_size < 0:
        raise ValueError("--ref-size must be >= 0")

    rows = (
        read_coco_captions_json(caption_file)
        if caption_file.suffix.lower() == ".json"
        else read_captions_tsv(caption_file)
    )
    selected = select_samples(
        rows=rows,
        coco_val_dir=coco_val_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        shuffle_rows=args.shuffle,
    )

    ref_tag = "ref_orig" if args.ref_size == 0 else f"ref{args.ref_size}"
    run_dir = out_dir / f"n{args.num_samples}_seed{args.seed}_{ref_tag}"
    prompts_path = run_dir / "prompt.txt"

    ref_size = None if args.ref_size == 0 else args.ref_size

    write_prompts_file(selected, prompts_path)
    prepare_ref_folder(selected, coco_val_dir, run_dir, ref_size=ref_size)

    n_ref = count_images(run_dir)
    if n_ref != args.num_samples:
        print(f"Warning: reference images count = {n_ref}, expected {args.num_samples}")

    print(f"Prepared prompts: {prompts_path}")
    print(f"Prepared refs:   {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
