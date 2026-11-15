#!/usr/bin/env python3
"""Evaluate a remote multimodal embedding service on MS COCO Captions.

This script evaluates image-text retrieval with the official COCO captions JSON:

  Text -> Image: each caption queries the image collection.
  Image -> Text: each image queries the caption collection; all captions of the
                 same image are considered relevant.

Embeddings are NOT computed locally. Inputs are sent to an OpenAI-compatible
embedding endpoint following scripts/req.py:

  POST http://127.0.0.1:30000/v1/embeddings
  {
    "model": "qwen3-vl-embedding-2b",
    "input": [{"text": "..."}, {"image": "..."}]
  }

The service URL defaults to the hardcoded local-forwarded remote endpoint
"http://127.0.0.1:30000".

Example:

  python scripts/evaluate_coco_captions_qwen_vl.py \
    --coco-root /data/coco \
    --model qwen3-vl-embedding-2b \
    --split val2017 \
    --batch-size-text 32 \
    --batch-size-image 4 \
    --output-json results/qwen3_vl_embedding_2b_coco_val2017.json

Expected local COCO layout when using --coco-root:

  /data/coco/val2017/*.jpg
  /data/coco/annotations/captions_val2017.json

Images are sent as absolute filesystem paths by default so the embedding server
can open the COCO files directly. If the service cannot read the same
filesystem, use --image-transport data-url instead.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SERVICE_URL = "http://127.0.0.1:30000"

COCO_URLS = {
    "val2017_images": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017_images": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations_trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


@dataclass(frozen=True)
class CocoEvalData:
    image_ids: list[int]
    image_paths: list[str]
    captions: list[str]
    caption_to_image_index: list[int]
    image_to_caption_indices: list[list[int]]


class RemoteEmbeddingClient:
    """Small client for the /v1/embeddings API shown in scripts/req.py."""

    def __init__(
        self,
        *,
        service_url: str = DEFAULT_SERVICE_URL,
        model: str = "qwen3-vl-embedding-2b",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        import requests

        self.service_url = service_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def embed(self, inputs: list[dict[str, str]]) -> list[list[float]]:
        payload = {"model": self.model, "input": inputs}
        url = self.service_url + "/v1/embeddings"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code} from {url}: {response.text[:1000]}"
                    )
                body = response.json()
                return parse_embedding_response(body, expected_count=len(inputs))
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(2 ** (attempt - 1), 8)
                print(
                    f"Embedding request failed on attempt {attempt}/{self.max_retries}: {exc}. "
                    f"Retrying in {sleep_s}s..."
                )
                time.sleep(sleep_s)
        raise RuntimeError(
            f"Embedding request failed after {self.max_retries} attempts: {last_error}"
        )


def parse_embedding_response(
    body: dict[str, Any], *, expected_count: int
) -> list[list[float]]:
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Embedding response has no list field 'data': {body}")

    rows = []
    for fallback_index, item in enumerate(data):
        if not isinstance(item, dict) or "embedding" not in item:
            raise ValueError(f"Embedding response item lacks 'embedding': {item}")
        index = item.get("index", fallback_index)
        rows.append((int(index), item["embedding"]))

    rows.sort(key=lambda x: x[0])
    embeddings = [embedding for _, embedding in rows]
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding response count mismatch: expected {expected_count}, got {len(embeddings)}"
        )
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MultiModal Embedding service on MS COCO Captions retrieval."
    )
    data = parser.add_argument_group("COCO data")
    data.add_argument(
        "--coco-root",
        type=Path,
        default=None,
        help="COCO root containing <split>/ and annotations/. Used to infer paths.",
    )
    data.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing COCO images, e.g. /data/coco/val2017.",
    )
    data.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="COCO captions JSON, e.g. /data/coco/annotations/captions_val2017.json.",
    )
    data.add_argument(
        "--split",
        default="val2017",
        choices=["val2017", "train2017"],
        help="COCO split to evaluate/download. Default: val2017.",
    )
    data.add_argument(
        "--download",
        action="store_true",
        help="Download official COCO split images and train/val annotations into --coco-root.",
    )
    data.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit the number of images for a quick run/debug. Default: all images in split.",
    )
    data.add_argument(
        "--captions-per-image",
        type=int,
        default=None,
        help="Optional cap on captions per image. Default: use all available captions.",
    )
    data.add_argument(
        "--seed", type=int, default=0, help="Seed used only when --shuffle is set."
    )
    data.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle images before applying --max-images.",
    )
    data.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip annotations whose image file is missing instead of failing.",
    )

    model = parser.add_argument_group("remote embedding service")
    model.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help=f"Embedding service base URL. Default: {DEFAULT_SERVICE_URL}.",
    )
    model.add_argument(
        "--model",
        default="qwen3-vl-embedding-2b",
        help="Model string sent in the /v1/embeddings payload.",
    )
    model.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds. Default: 300.",
    )
    model.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Embedding request retries. Default: 3.",
    )
    model.add_argument(
        "--batch-size-text",
        type=int,
        default=32,
        help="Batch size for caption/text embedding requests. Default: 32.",
    )
    model.add_argument(
        "--batch-size-image",
        type=int,
        default=4,
        help="Batch size for image embedding requests. Default: 4.",
    )
    model.add_argument(
        "--query-prompt",
        default=None,
        help="Optional prefix added to every caption before remote embedding.",
    )
    model.add_argument(
        "--image-prompt",
        default=None,
        help="Optional text included with every image object, e.g. 'Represent this image in embedding space.'.",
    )
    model.add_argument(
        "--image-transport",
        choices=["path", "data-url"],
        default="path",
        help="How to send local COCO images to the remote service. Default: path, using absolute paths.",
    )
    model.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings before dot product. Default: true.",
    )
    model.add_argument(
        "--cache-npz",
        type=Path,
        default=None,
        help="Optional .npz path for saving/loading computed embeddings.",
    )
    model.add_argument(
        "--reuse-cache",
        action="store_true",
        help="If --cache-npz exists, load embeddings from it instead of calling the service.",
    )

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Recall@K values. Default: 1 5 10.",
    )
    eval_group.add_argument(
        "--score-chunk-size",
        type=int,
        default=256,
        help="Number of queries per similarity chunk. Lower if memory is tight. Default: 256.",
    )
    eval_group.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write metrics and run metadata as JSON.",
    )
    eval_group.add_argument(
        "--save-ranks",
        type=Path,
        default=None,
        help="Optional .npz path to save per-query rank arrays.",
    )
    eval_group.add_argument(
        "--text-to-image-only",
        action="store_true",
        help="Only run caption -> image retrieval.",
    )
    eval_group.add_argument(
        "--image-to-text-only",
        action="store_true",
        help="Only run image -> caption retrieval.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--print-requirements",
        action="store_true",
        help="Print Python packages required for an actual evaluation run and exit.",
    )
    misc.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny metric/response-parser self-test that requires numpy only, then exit.",
    )
    return parser.parse_args()


def require_packages() -> tuple[Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency. Install dependencies in a virtual environment, for example:\n"
            "  uv venv .venv && source .venv/bin/activate\n"
            "  uv pip install numpy requests tqdm\n"
            f"Original import error: {exc}"
        ) from exc
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency fallback
        tqdm = lambda x, **_: x
    return np, tqdm


def print_requirements() -> None:
    print("Required for evaluation:")
    print("  numpy")
    print("  requests")
    print("Optional but useful:")
    print("  tqdm")


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Using existing download: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"Downloading {url} -> {dst}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        last_report = 0.0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if total and now - last_report > 2:
                print(
                    f"  {done / total * 100:5.1f}% ({done / 1e9:.2f}/{total / 1e9:.2f} GB)"
                )
                last_report = now
    tmp.replace(dst)


def unzip_if_needed(zip_path: Path, target_root: Path, expected_child: str) -> None:
    expected = target_root / expected_child
    if expected.exists():
        print(f"Using existing extracted data: {expected}")
        return
    print(f"Extracting {zip_path} -> {target_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_root)


def maybe_download_coco(coco_root: Path, split: str) -> None:
    if split not in {"val2017", "train2017"}:
        raise ValueError(f"Unsupported split for download: {split}")
    coco_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = coco_root / "downloads"

    image_key = f"{split}_images"
    image_zip = downloads_dir / f"{split}.zip"
    ann_zip = downloads_dir / "annotations_trainval2017.zip"

    download_file(COCO_URLS[image_key], image_zip)
    unzip_if_needed(image_zip, coco_root, split)

    download_file(COCO_URLS["annotations_trainval2017"], ann_zip)
    unzip_if_needed(ann_zip, coco_root, "annotations")


def infer_coco_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    images_dir = args.images_dir
    annotations = args.annotations
    if args.coco_root is not None:
        images_dir = images_dir or (args.coco_root / args.split)
        annotations = annotations or (
            args.coco_root / "annotations" / f"captions_{args.split}.json"
        )
    if images_dir is None or annotations is None:
        raise SystemExit(
            "Provide either --coco-root or both --images-dir and --annotations."
        )
    return images_dir, annotations


def load_coco_eval_data(
    images_dir: Path,
    annotations_path: Path,
    *,
    max_images: int | None,
    captions_per_image: int | None,
    shuffle: bool,
    seed: int,
    allow_missing_images: bool,
) -> CocoEvalData:
    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO annotations JSON not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images directory not found: {images_dir}")

    with annotations_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    images_by_id = {int(img["id"]): img for img in raw.get("images", [])}
    captions_by_image_id: dict[int, list[str]] = {}
    for ann in raw.get("annotations", []):
        image_id = int(ann["image_id"])
        caption = str(ann["caption"]).strip()
        if caption:
            captions_by_image_id.setdefault(image_id, []).append(caption)

    image_ids = [
        image_id for image_id in images_by_id if image_id in captions_by_image_id
    ]
    image_ids.sort()

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(image_ids)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        image_ids = image_ids[:max_images]

    selected_image_ids: list[int] = []
    image_paths: list[str] = []
    captions: list[str] = []
    caption_to_image_index: list[int] = []
    image_to_caption_indices: list[list[int]] = []
    missing: list[Path] = []

    for image_id in image_ids:
        img = images_by_id[image_id]
        path = images_dir / img["file_name"]
        if not path.exists():
            missing.append(path)
            if allow_missing_images:
                continue
            if len(missing) >= 5:
                break
            continue

        caps = captions_by_image_id[image_id]
        if captions_per_image is not None:
            if captions_per_image <= 0:
                raise ValueError("--captions-per-image must be positive")
            caps = caps[:captions_per_image]
        if not caps:
            continue

        image_index = len(selected_image_ids)
        cap_indices: list[int] = []
        for cap in caps:
            cap_indices.append(len(captions))
            captions.append(cap)
            caption_to_image_index.append(image_index)

        selected_image_ids.append(image_id)
        image_paths.append(str(path))
        image_to_caption_indices.append(cap_indices)

    if missing and not allow_missing_images:
        examples = "\n".join(f"  {p}" for p in missing[:5])
        raise FileNotFoundError(
            f"Missing COCO image files under {images_dir}. Examples:\n{examples}\n"
            "Use --allow-missing-images to skip them, or check --images-dir/--split."
        )
    if not selected_image_ids:
        raise RuntimeError("No evaluable COCO images/captions found.")

    return CocoEvalData(
        image_ids=selected_image_ids,
        image_paths=image_paths,
        captions=captions,
        caption_to_image_index=caption_to_image_index,
        image_to_caption_indices=image_to_caption_indices,
    )


def l2_normalize(x: Any, np: Any, eps: float = 1e-12) -> Any:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def image_to_data_url(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def make_text_inputs(
    texts: list[str], *, query_prompt: str | None
) -> list[dict[str, str]]:
    if query_prompt:
        return [{"text": f"{query_prompt}\n{text}"} for text in texts]
    return [{"text": text} for text in texts]


def make_image_inputs(
    image_paths: list[str],
    *,
    image_prompt: str | None,
    image_transport: str,
) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    for path in image_paths:
        image_value = (
            image_to_data_url(path)
            if image_transport == "data-url"
            else str(Path(path).resolve())
        )
        item = {"image": image_value}
        if image_prompt:
            item["text"] = image_prompt
        inputs.append(item)
    return inputs


def encode_remote_batches(
    client: RemoteEmbeddingClient,
    inputs: list[dict[str, str]],
    *,
    batch_size: int,
    normalize: bool,
    np: Any,
    tqdm: Any,
    desc: str,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    chunks: list[Any] = []
    for start in tqdm(range(0, len(inputs), batch_size), desc=desc):
        end = min(start + batch_size, len(inputs))
        embeddings = client.embed(inputs[start:end])
        arr = np.asarray(embeddings, dtype="float32")
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D embedding array, got shape {arr.shape}")
        chunks.append(arr)
    if not chunks:
        raise ValueError("No inputs to encode")
    out = np.concatenate(chunks, axis=0)
    if normalize:
        out = l2_normalize(out, np).astype("float32", copy=False)
    return out


def load_or_compute_embeddings(
    args: argparse.Namespace, data: CocoEvalData
) -> tuple[Any, Any, Any]:
    np, tqdm = require_packages()
    cache = args.cache_npz
    if args.reuse_cache and cache is not None and cache.exists():
        print(f"Loading embeddings from cache: {cache}")
        loaded = np.load(cache, allow_pickle=False)
        text_embeddings = loaded["text_embeddings"]
        image_embeddings = loaded["image_embeddings"]
        if args.normalize:
            text_embeddings = l2_normalize(text_embeddings, np).astype(
                "float32", copy=False
            )
            image_embeddings = l2_normalize(image_embeddings, np).astype(
                "float32", copy=False
            )
        return np, text_embeddings, image_embeddings

    print(f"Remote embedding service: {args.service_url}/v1/embeddings")
    print(f"Remote model: {args.model}")
    client = RemoteEmbeddingClient(
        service_url=args.service_url,
        model=args.model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    text_inputs = make_text_inputs(data.captions, query_prompt=args.query_prompt)
    print(
        f"Embedding {len(text_inputs)} captions remotely with batch size {args.batch_size_text}"
    )
    text_embeddings = encode_remote_batches(
        client,
        text_inputs,
        batch_size=args.batch_size_text,
        normalize=args.normalize,
        np=np,
        tqdm=tqdm,
        desc="Embedding captions",
    )

    image_inputs = make_image_inputs(
        data.image_paths,
        image_prompt=args.image_prompt,
        image_transport=args.image_transport,
    )
    print(
        f"Embedding {len(image_inputs)} images remotely with batch size {args.batch_size_image} "
        f"using image transport {args.image_transport!r}"
    )
    image_embeddings = encode_remote_batches(
        client,
        image_inputs,
        batch_size=args.batch_size_image,
        normalize=args.normalize,
        np=np,
        tqdm=tqdm,
        desc="Embedding images",
    )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings cache: {cache}")
        np.savez_compressed(
            cache,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            image_ids=np.array(data.image_ids, dtype=np.int64),
            caption_to_image_index=np.array(
                data.caption_to_image_index, dtype=np.int64
            ),
        )
    return np, text_embeddings, image_embeddings


def rank_metrics(ranks: Any, ks: list[int], np: Any) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def text_to_image_ranks(
    text_embeddings: Any,
    image_embeddings: Any,
    caption_to_image_index: list[int],
    *,
    chunk_size: int,
    np: Any,
    tqdm: Any,
) -> Any:
    targets = np.asarray(caption_to_image_index, dtype=np.int64)
    ranks = np.empty(len(targets), dtype=np.int64)
    for start in tqdm(range(0, len(targets), chunk_size), desc="Scoring text->image"):
        end = min(start + chunk_size, len(targets))
        scores = text_embeddings[start:end] @ image_embeddings.T
        target_cols = targets[start:end]
        target_scores = scores[np.arange(end - start), target_cols]
        ranks[start:end] = np.sum(scores > target_scores[:, None], axis=1) + 1
    return ranks


def image_to_text_ranks(
    image_embeddings: Any,
    text_embeddings: Any,
    image_to_caption_indices: list[list[int]],
    *,
    chunk_size: int,
    np: Any,
    tqdm: Any,
) -> Any:
    ranks = np.empty(len(image_to_caption_indices), dtype=np.int64)
    for start in tqdm(
        range(0, len(image_to_caption_indices), chunk_size), desc="Scoring image->text"
    ):
        end = min(start + chunk_size, len(image_to_caption_indices))
        scores = image_embeddings[start:end] @ text_embeddings.T
        for row, image_index in enumerate(range(start, end)):
            rel = image_to_caption_indices[image_index]
            best_rel_score = np.max(scores[row, rel])
            ranks[image_index] = int(np.sum(scores[row] > best_rel_score) + 1)
    return ranks


def run_self_test() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "--self-test requires numpy. Install it with: uv pip install numpy"
        ) from exc

    parsed = parse_embedding_response(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        },
        expected_count=2,
    )
    assert parsed == [[1.0, 0.0], [0.0, 1.0]], parsed

    # Three images, five captions. Caption 0/1 -> image 0, 2/3 -> image 1, 4 -> image 2.
    image_embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="float32"
    )
    text_embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]], dtype="float32"
    )
    caption_to_image_index = [0, 0, 1, 1, 2]
    image_to_caption_indices = [[0, 1], [2, 3], [4]]
    t2i = text_to_image_ranks(
        text_embeddings,
        image_embeddings,
        caption_to_image_index,
        chunk_size=2,
        np=np,
        tqdm=lambda x, **_: x,
    )
    i2t = image_to_text_ranks(
        image_embeddings,
        text_embeddings,
        image_to_caption_indices,
        chunk_size=2,
        np=np,
        tqdm=lambda x, **_: x,
    )
    assert t2i.tolist() == [1, 1, 1, 1, 1], t2i.tolist()
    assert i2t.tolist() == [1, 1, 1], i2t.tolist()
    print("Self-test passed.")


def main() -> int:
    args = parse_args()
    if args.print_requirements:
        print_requirements()
        return 0
    if args.self_test:
        run_self_test()
        return 0
    if args.text_to_image_only and args.image_to_text_only:
        raise SystemExit(
            "Choose at most one of --text-to-image-only and --image-to-text-only."
        )
    if args.score_chunk_size <= 0:
        raise SystemExit("--score-chunk-size must be positive.")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be positive.")

    if args.download:
        if args.coco_root is None:
            raise SystemExit("--download requires --coco-root.")
        maybe_download_coco(args.coco_root, args.split)

    images_dir, annotations = infer_coco_paths(args)
    print(f"Images dir:  {images_dir}")
    print(f"Annotations: {annotations}")
    data = load_coco_eval_data(
        images_dir,
        annotations,
        max_images=args.max_images,
        captions_per_image=args.captions_per_image,
        shuffle=args.shuffle,
        seed=args.seed,
        allow_missing_images=args.allow_missing_images,
    )
    print(f"Loaded {len(data.image_paths)} images and {len(data.captions)} captions")

    np, text_embeddings, image_embeddings = load_or_compute_embeddings(args, data)
    _, tqdm = require_packages()

    results: dict[str, Any] = {
        "service_url": args.service_url,
        "model": args.model,
        "split": args.split,
        "num_images": len(data.image_paths),
        "num_captions": len(data.captions),
        "embedding_dim": int(text_embeddings.shape[1]),
        "normalize": bool(args.normalize),
        "image_transport": args.image_transport,
        "k": sorted(set(args.k)),
    }
    ranks_to_save: dict[str, Any] = {}

    if not args.image_to_text_only:
        t2i_ranks = text_to_image_ranks(
            text_embeddings,
            image_embeddings,
            data.caption_to_image_index,
            chunk_size=args.score_chunk_size,
            np=np,
            tqdm=tqdm,
        )
        results["text_to_image"] = rank_metrics(t2i_ranks, args.k, np)
        ranks_to_save["text_to_image_ranks"] = t2i_ranks

    if not args.text_to_image_only:
        i2t_ranks = image_to_text_ranks(
            image_embeddings,
            text_embeddings,
            data.image_to_caption_indices,
            chunk_size=args.score_chunk_size,
            np=np,
            tqdm=tqdm,
        )
        results["image_to_text"] = rank_metrics(i2t_ranks, args.k, np)
        ranks_to_save["image_to_text_ranks"] = i2t_ranks

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote metrics: {args.output_json}")

    if args.save_ranks is not None:
        args.save_ranks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_ranks, **ranks_to_save)
        print(f"Wrote ranks: {args.save_ranks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
