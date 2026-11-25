#!/usr/bin/env python3
"""Evaluate a remote multimodal embedding service on MS COCO Captions.

This script evaluates image-text retrieval with the official COCO captions JSON:

  Text -> Image: each caption queries the image collection.
  Image -> Text: each image queries the caption collection; all captions of the
                 same image are considered relevant.

Tie-breaking policy:
  Zero-norm and non-finite embeddings are rejected before scoring. For valid
  embeddings, ties are broken conservatively (pessimistic / worst-case rank):
  non-relevant candidates with similarity >= the relevant target score are
  counted ahead of the target.

Embeddings are NOT computed locally. Inputs are sent to an OpenAI-compatible
embedding endpoint following scripts/req.py:

  POST http://127.0.0.1:30000/v1/embeddings
  {
    "model": "qwen3-vl-embedding-2b",
    "input": [{"text": "..."}, {"image": "..."}]
  }

The service URL defaults to "http://127.0.0.1:30000" and can be configured
with --host/--port (or overridden directly with --service-url).

Example:

  python benchmark/embedding/evaluate_embedding_accuracy_with_coco_captions.py \
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
import contextlib
import hashlib
import io
import json
import mimetypes
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_PORT = 30000
DEFAULT_SERVICE_URL = f"{DEFAULT_HOST}:{DEFAULT_PORT}"

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


def canonical_service_url(url: str) -> str:
    """Canonicalize an endpoint base URL (lowercase scheme/hostname, strip trailing '/')."""
    import urllib.parse

    raw = str(url).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        netloc = f"{userinfo}@{netloc}"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), netloc, path, parsed.query, parsed.fragment)
    )


def build_service_url(args: argparse.Namespace) -> str:
    """Resolve service URL from --service-url or --host/--port."""
    raw_service_url = getattr(args, "service_url", DEFAULT_SERVICE_URL)
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)
    if raw_service_url is not None and str(raw_service_url) != DEFAULT_SERVICE_URL:
        return canonical_service_url(str(raw_service_url))
    return canonical_service_url(f"{str(host).rstrip('/')}:{port}")


def parse_embedding_response(
    body: dict[str, Any], *, expected_count: int
) -> list[list[float]]:
    if expected_count < 0:
        raise ValueError(f"expected_count must be non-negative, got {expected_count}")
    if not isinstance(body, dict):
        raise ValueError(f"Embedding response body must be a JSON object: {body!r}")
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Embedding response has no list field 'data': {body}")
    if len(data) != expected_count:
        raise ValueError(
            f"Embedding response count mismatch: expected {expected_count}, got {len(data)}"
        )

    has_explicit_index = [
        ("index" in item) if isinstance(item, dict) else False for item in data
    ]
    if any(has_explicit_index) and not all(has_explicit_index):
        raise ValueError(
            "Embedding response mixes items with and without explicit 'index' fields"
        )

    ordered: list[list[float] | None] = [None] * expected_count
    seen_indices: set[int] = set()
    expected_dim: int | None = None
    for fallback_index, item in enumerate(data):
        if not isinstance(item, dict) or "embedding" not in item:
            raise ValueError(f"Embedding response item lacks 'embedding': {item}")
        raw_index = item.get("index", fallback_index)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"Embedding response item has invalid integer 'index': {raw_index!r}"
            )
        if raw_index < 0 or raw_index >= expected_count:
            raise ValueError(
                f"Embedding response index {raw_index} out of range [0, {expected_count})"
            )
        if raw_index in seen_indices:
            raise ValueError(
                f"Duplicate embedding response index {raw_index} in batch of {expected_count}"
            )
        embedding = item["embedding"]
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(
                f"Embedding response item has invalid 'embedding' vector: {item}"
            )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in embedding
        ):
            raise ValueError(
                f"Embedding response item contains non-numeric values: {item}"
            )
        if expected_dim is None:
            expected_dim = len(embedding)
        elif len(embedding) != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions in response: expected {expected_dim}, got {len(embedding)}"
            )
        seen_indices.add(raw_index)
        ordered[raw_index] = embedding

    if seen_indices != set(range(expected_count)):
        missing = sorted(set(range(expected_count)) - seen_indices)
        raise ValueError(
            f"Embedding response indices do not cover [0, {expected_count}); missing {missing}"
        )

    return [emb for emb in ordered if emb is not None]


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
        "--host",
        default=DEFAULT_HOST,
        help=f"Serving host (scheme + host). Used with --port when --service-url is not overridden. Default: {DEFAULT_HOST}.",
    )
    model.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Serving port. Used with --host when --service-url is not overridden. Default: {DEFAULT_PORT}.",
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


def validate_embeddings(x: Any, np: Any, *, name: str = "embeddings") -> Any:
    raw = np.asarray(x)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D {name} array, got shape {raw.shape}")
    if not np.all(np.isfinite(raw)):
        bad_rows = np.where(~np.all(np.isfinite(raw), axis=1))[0][:5].tolist()
        raise ValueError(
            f"{name} contains NaN or infinite values (example row indices: {bad_rows})"
        )
    arr = np.asarray(raw, dtype="float32")
    if not np.all(np.isfinite(arr)):
        bad_rows = np.where(~np.all(np.isfinite(arr), axis=1))[0][:5].tolist()
        raise ValueError(f"{name} overflows float32 (example row indices: {bad_rows})")
    norms = np.linalg.norm(arr, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        zero_rows = np.where((~np.isfinite(norms)) | (norms <= 0.0))[0][:5].tolist()
        raise ValueError(
            f"{name} contains zero-norm or invalid-norm vectors (example row indices: {zero_rows})"
        )
    return arr


def l2_normalize(x: Any, np: Any, *, name: str = "embeddings") -> Any:
    arr = validate_embeddings(x, np, name=name)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / denom).astype("float32", copy=False)


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


CACHE_SCHEMA_VERSION = 3


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_metadata(
    args: argparse.Namespace, data: CocoEvalData
) -> dict[str, Any]:
    service_url = build_service_url(args)
    resolved_image_paths = [str(Path(p).resolve()) for p in data.image_paths]
    image_digests: list[str] = []
    for p in resolved_image_paths:
        image_file = Path(p)
        if image_file.is_file():
            image_digests.append(_sha256_file(p))
        else:
            image_digests.append(f"path:{p}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "embeddings_are_raw": True,
        "service_url": service_url,
        "endpoint": f"{service_url}/v1/embeddings",
        "model": args.model,
        "split": getattr(args, "split", None),
        "query_prompt": args.query_prompt,
        "image_prompt": args.image_prompt,
        "image_transport": args.image_transport,
        "num_images": len(data.image_ids),
        "num_captions": len(data.captions),
        "captions_sha256": _sha256_json(data.captions),
        "image_paths_sha256": _sha256_json(resolved_image_paths),
        "image_contents_sha256": _sha256_json(image_digests),
        "image_to_caption_indices_sha256": _sha256_json(data.image_to_caption_indices),
    }


def save_embeddings_cache(
    cache: Path,
    *,
    raw_text_embeddings: Any,
    raw_image_embeddings: Any,
    args: argparse.Namespace,
    data: CocoEvalData,
    np: Any,
) -> None:
    raw_text = validate_embeddings(raw_text_embeddings, np, name="text_embeddings")
    raw_image = validate_embeddings(raw_image_embeddings, np, name="image_embeddings")
    metadata = build_cache_metadata(args, data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        text_embeddings=raw_text,
        image_embeddings=raw_image,
        image_ids=np.asarray(data.image_ids, dtype=np.int64),
        caption_to_image_index=np.asarray(data.caption_to_image_index, dtype=np.int64),
        cache_metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        ),
    )


def load_validated_embeddings_cache(
    cache: Path,
    *,
    args: argparse.Namespace,
    data: CocoEvalData,
    np: Any,
) -> tuple[Any, Any]:
    with np.load(cache, allow_pickle=False) as loaded:
        required_keys = {
            "text_embeddings",
            "image_embeddings",
            "image_ids",
            "caption_to_image_index",
            "cache_metadata_json",
        }
        missing_keys = sorted(required_keys - set(loaded.files))
        if missing_keys:
            raise ValueError(
                f"Cache {cache} is missing required validation fields {missing_keys}; "
                "re-run without --reuse-cache to regenerate it."
            )

        raw_meta = loaded["cache_metadata_json"]
        if raw_meta.ndim != 0:
            raise ValueError(f"Cache {cache} has malformed cache_metadata_json array.")
        try:
            cached_meta = json.loads(str(raw_meta.item()))
        except Exception as exc:
            raise ValueError(f"Cache {cache} has invalid JSON metadata: {exc}") from exc
        if not isinstance(cached_meta, dict):
            raise ValueError(f"Cache {cache} metadata must be a JSON object.")

        expected_meta = build_cache_metadata(args, data)
        for field, expected_val in expected_meta.items():
            actual_val = cached_meta.get(field)
            if actual_val != expected_val:
                raise ValueError(
                    f"Cache mismatch in {cache} for {field!r}: "
                    f"cached={actual_val!r}, current={expected_val!r}"
                )

        cached_image_ids = np.asarray(loaded["image_ids"], dtype=np.int64)
        expected_image_ids = np.asarray(data.image_ids, dtype=np.int64)
        if not np.array_equal(cached_image_ids, expected_image_ids):
            raise ValueError(
                f"Cache mismatch in {cache}: ordered image_ids do not match current dataset."
            )

        cached_c2i = np.asarray(loaded["caption_to_image_index"], dtype=np.int64)
        expected_c2i = np.asarray(data.caption_to_image_index, dtype=np.int64)
        if not np.array_equal(cached_c2i, expected_c2i):
            raise ValueError(
                f"Cache mismatch in {cache}: ordered caption_to_image_index does not match current dataset."
            )

        raw_text = validate_embeddings(
            loaded["text_embeddings"], np, name="cached text_embeddings"
        )
        raw_image = validate_embeddings(
            loaded["image_embeddings"], np, name="cached image_embeddings"
        )

    if raw_text.shape[0] != len(data.captions):
        raise ValueError(
            f"Cache mismatch in {cache}: expected {len(data.captions)} text embeddings, got {raw_text.shape[0]}"
        )
    if raw_image.shape[0] != len(data.image_ids):
        raise ValueError(
            f"Cache mismatch in {cache}: expected {len(data.image_ids)} image embeddings, got {raw_image.shape[0]}"
        )
    if raw_text.shape[1] != raw_image.shape[1]:
        raise ValueError(
            f"Cache mismatch in {cache}: text/image embedding dims differ "
            f"({raw_text.shape[1]} vs {raw_image.shape[1]})"
        )

    if args.normalize:
        return (
            l2_normalize(raw_text, np, name="cached text_embeddings"),
            l2_normalize(raw_image, np, name="cached image_embeddings"),
        )
    return raw_text, raw_image


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
        arr = validate_embeddings(embeddings, np, name=f"{desc}[{start}:{end}]")
        chunks.append(arr)
    if not chunks:
        raise ValueError("No inputs to encode")
    out = validate_embeddings(np.concatenate(chunks, axis=0), np, name=desc)
    if normalize:
        out = l2_normalize(out, np, name=desc)
    return out


def load_or_compute_embeddings(
    args: argparse.Namespace, data: CocoEvalData
) -> tuple[Any, Any, Any]:
    np, tqdm = require_packages()
    cache = args.cache_npz
    if args.reuse_cache and cache is not None and cache.exists():
        print(f"Loading embeddings from cache: {cache}")
        text_embeddings, image_embeddings = load_validated_embeddings_cache(
            cache, args=args, data=data, np=np
        )
        return np, text_embeddings, image_embeddings

    service_url = build_service_url(args)
    print(f"Remote embedding service: {service_url}/v1/embeddings")
    print(f"Remote model: {args.model}")
    client = RemoteEmbeddingClient(
        service_url=service_url,
        model=args.model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    text_inputs = make_text_inputs(data.captions, query_prompt=args.query_prompt)
    print(
        f"Embedding {len(text_inputs)} captions remotely with batch size {args.batch_size_text}"
    )
    raw_text_embeddings = encode_remote_batches(
        client,
        text_inputs,
        batch_size=args.batch_size_text,
        normalize=False,
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
    raw_image_embeddings = encode_remote_batches(
        client,
        image_inputs,
        batch_size=args.batch_size_image,
        normalize=False,
        np=np,
        tqdm=tqdm,
        desc="Embedding images",
    )

    if cache is not None:
        print(f"Saving raw embeddings cache: {cache}")
        save_embeddings_cache(
            cache,
            raw_text_embeddings=raw_text_embeddings,
            raw_image_embeddings=raw_image_embeddings,
            args=args,
            data=data,
            np=np,
        )

    if args.normalize:
        text_embeddings = l2_normalize(raw_text_embeddings, np, name="text_embeddings")
        image_embeddings = l2_normalize(
            raw_image_embeddings, np, name="image_embeddings"
        )
    else:
        text_embeddings = raw_text_embeddings
        image_embeddings = raw_image_embeddings
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
    """Compute 1-based caption -> image ranks.

    Validates that all vectors are finite and non-zero before scoring.
    Ties are broken conservatively (pessimistic rank): any non-target image
    whose score is >= the target image's score is counted ahead of the target.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    text_arr = validate_embeddings(text_embeddings, np, name="text_embeddings")
    image_arr = validate_embeddings(image_embeddings, np, name="image_embeddings")
    if text_arr.shape[1] != image_arr.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: text {text_arr.shape[1]} vs image {image_arr.shape[1]}"
        )
    if len(caption_to_image_index) != text_arr.shape[0]:
        raise ValueError(
            f"caption_to_image_index length ({len(caption_to_image_index)}) "
            f"does not match text_embeddings rows ({text_arr.shape[0]})"
        )

    targets = np.asarray(caption_to_image_index, dtype=np.int64)
    if np.any((targets < 0) | (targets >= image_arr.shape[0])):
        raise ValueError("caption_to_image_index contains out-of-range image indices")

    ranks = np.empty(len(targets), dtype=np.int64)
    all_cols = np.arange(image_arr.shape[0])
    for start in tqdm(range(0, len(targets), chunk_size), desc="Scoring text->image"):
        end = min(start + chunk_size, len(targets))
        scores = text_arr[start:end] @ image_arr.T
        if not np.all(np.isfinite(scores)):
            raise ValueError("Non-finite similarity scores encountered in text->image")
        target_cols = targets[start:end]
        row_idx = np.arange(end - start)
        target_scores = scores[row_idx, target_cols]
        non_target_mask = all_cols[None, :] != target_cols[:, None]
        ranks[start:end] = (
            np.sum((scores >= target_scores[:, None]) & non_target_mask, axis=1) + 1
        )
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
    """Compute 1-based image -> caption ranks.

    Validates that all vectors are finite and non-zero before scoring.
    For an image with multiple relevant captions, the best-scoring relevant
    caption is evaluated, and ties with non-relevant captions (score >= best
    relevant score) are broken conservatively (pessimistic rank).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    image_arr = validate_embeddings(image_embeddings, np, name="image_embeddings")
    text_arr = validate_embeddings(text_embeddings, np, name="text_embeddings")
    if image_arr.shape[1] != text_arr.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: image {image_arr.shape[1]} vs text {text_arr.shape[1]}"
        )
    if len(image_to_caption_indices) != image_arr.shape[0]:
        raise ValueError(
            f"image_to_caption_indices length ({len(image_to_caption_indices)}) "
            f"does not match image_embeddings rows ({image_arr.shape[0]})"
        )

    num_texts = text_arr.shape[0]
    ranks = np.empty(len(image_to_caption_indices), dtype=np.int64)
    for start in tqdm(
        range(0, len(image_to_caption_indices), chunk_size), desc="Scoring image->text"
    ):
        end = min(start + chunk_size, len(image_to_caption_indices))
        scores = image_arr[start:end] @ text_arr.T
        if not np.all(np.isfinite(scores)):
            raise ValueError("Non-finite similarity scores encountered in image->text")
        for row, image_index in enumerate(range(start, end)):
            rel = image_to_caption_indices[image_index]
            if not rel:
                raise ValueError(f"Image index {image_index} has no relevant captions")
            rel_arr = np.asarray(rel, dtype=np.int64)
            if np.any((rel_arr < 0) | (rel_arr >= num_texts)):
                raise ValueError(
                    f"Image index {image_index} has out-of-range caption indices"
                )
            best_rel_score = np.max(scores[row, rel_arr])
            non_rel_mask = np.ones(num_texts, dtype=bool)
            non_rel_mask[rel_arr] = False
            ranks[image_index] = int(
                np.sum((scores[row] >= best_rel_score) & non_rel_mask) + 1
            )
    return ranks


def _assert_raises_value_error(fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError from {fn.__name__}")


def run_self_test() -> None:
    import tempfile

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "--self-test requires numpy. Install it with: uv pip install numpy"
        ) from exc

    noop_tqdm = lambda x, **_: x

    # 1. Response parsing: valid reordering, fallback indices, and malformed index rejection.
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

    parsed_no_index = parse_embedding_response(
        {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]},
        expected_count=2,
    )
    assert parsed_no_index == [[1.0, 0.0], [0.0, 1.0]], parsed_no_index

    for bad_body in [
        # Duplicate index [0, 0]
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        },
        # Out-of-range / missing index [0, 2]
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 2, "embedding": [0.0, 1.0]},
            ]
        },
        # Negative index [-1, 0]
        {
            "data": [
                {"index": -1, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        },
        # Mixed explicit and implicit index (would collide on index 0)
        {
            "data": [
                {"embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        },
        # Boolean index
        {
            "data": [
                {"index": False, "embedding": [1.0, 0.0]},
                {"index": True, "embedding": [0.0, 1.0]},
            ]
        },
        # Ragged embedding dimensions
        {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [0.0, 1.0]},
            ]
        },
        # Empty embedding vector
        {
            "data": [
                {"index": 0, "embedding": []},
                {"index": 1, "embedding": [0.0, 1.0]},
            ]
        },
    ]:
        _assert_raises_value_error(parse_embedding_response, bad_body, expected_count=2)

    # 2. Ranking on clean embeddings.
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
        tqdm=noop_tqdm,
    )
    i2t = image_to_text_ranks(
        image_embeddings,
        text_embeddings,
        image_to_caption_indices,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    assert t2i.tolist() == [1, 1, 1, 1, 1], t2i.tolist()
    assert i2t.tolist() == [1, 1, 1], i2t.tolist()

    # 3. Zero and non-finite vectors must be rejected before normalization or scoring.
    zero_text = np.zeros((5, 2), dtype="float32")
    zero_image = np.zeros((3, 2), dtype="float32")
    _assert_raises_value_error(l2_normalize, zero_text, np)
    _assert_raises_value_error(
        text_to_image_ranks,
        zero_text,
        image_embeddings,
        caption_to_image_index,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    _assert_raises_value_error(
        image_to_text_ranks,
        zero_image,
        text_embeddings,
        image_to_caption_indices,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    nan_embeddings = image_embeddings.copy()
    nan_embeddings[0, 0] = np.nan
    _assert_raises_value_error(l2_normalize, nan_embeddings, np)
    _assert_raises_value_error(
        text_to_image_ranks,
        text_embeddings,
        nan_embeddings,
        caption_to_image_index,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    inf_embeddings = text_embeddings.copy()
    inf_embeddings[0, 0] = np.inf
    _assert_raises_value_error(
        image_to_text_ranks,
        image_embeddings,
        inf_embeddings,
        image_to_caption_indices,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )

    # 4. Pessimistic tie-breaking policy on valid non-zero embeddings:
    # Identical unit vectors produce identical similarity scores across all candidates,
    # so non-relevant candidates tie with the target and push rank to worst-case (not 1).
    identical_images = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype="float32")
    identical_texts = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype="float32"
    )
    tie_t2i = text_to_image_ranks(
        identical_texts,
        identical_images,
        caption_to_image_index,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    tie_i2t = image_to_text_ranks(
        identical_images,
        identical_texts,
        image_to_caption_indices,
        chunk_size=2,
        np=np,
        tqdm=noop_tqdm,
    )
    # 3 images -> 2 non-target images tie => rank 3 for every caption.
    assert tie_t2i.tolist() == [3, 3, 3, 3, 3], tie_t2i.tolist()
    # 5 captions: image 0 & 1 have 2 relevant (3 non-rel tie => rank 4); image 2 has 1 relevant (4 non-rel tie => rank 5).
    assert tie_i2t.tolist() == [4, 4, 5], tie_i2t.tolist()

    # 5. Cache validation: ordered inputs, model/prompt settings, and raw-embedding normalization.
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        tmp_path = Path(tmpdir)
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.write_bytes(b"fake-jpeg-a")
        img_b.write_bytes(b"fake-jpeg-b")

        base_data = CocoEvalData(
            image_ids=[10, 20],
            image_paths=[str(img_a), str(img_b)],
            captions=["caption a1", "caption a2", "caption b1"],
            caption_to_image_index=[0, 0, 1],
            image_to_caption_indices=[[0, 1], [2]],
        )
        base_args = argparse.Namespace(
            service_url=DEFAULT_SERVICE_URL,
            host="http://127.0.0.1",
            port=30000,
            model="model-a",
            split="val2017",
            query_prompt="q-prefix",
            image_prompt="i-prefix",
            image_transport="path",
            normalize=False,
            cache_npz=tmp_path / "cache.npz",
            reuse_cache=True,
        )
        raw_text = np.asarray([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]], dtype="float32")
        raw_image = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype="float32")
        cache_file = base_args.cache_npz
        save_embeddings_cache(
            cache_file,
            raw_text_embeddings=raw_text,
            raw_image_embeddings=raw_image,
            args=base_args,
            data=base_data,
            np=np,
        )

        # Unnormalized load via load_or_compute_embeddings preserves raw norms;
        # normalized load L2-normalizes from raw cache.
        _, loaded_raw_t, loaded_raw_i = load_or_compute_embeddings(base_args, base_data)
        assert np.allclose(loaded_raw_t, raw_text)
        assert np.allclose(loaded_raw_i, raw_image)

        norm_args = argparse.Namespace(**{**vars(base_args), "normalize": True})
        _, loaded_norm_t, loaded_norm_i = load_or_compute_embeddings(
            norm_args, base_data
        )
        assert np.allclose(np.linalg.norm(loaded_norm_t, axis=1), 1.0)
        assert np.allclose(np.linalg.norm(loaded_norm_i, axis=1), 1.0)

        # Reordered images with identical array shapes must fail validation.
        reordered_data = CocoEvalData(
            image_ids=[20, 10],
            image_paths=[str(img_b), str(img_a)],
            captions=["caption b1", "caption a1", "caption a2"],
            caption_to_image_index=[0, 1, 1],
            image_to_caption_indices=[[0], [1, 2]],
        )
        _assert_raises_value_error(
            load_or_compute_embeddings,
            base_args,
            reordered_data,
        )

        # Changed caption text with identical IDs and shapes must fail validation.
        altered_captions_data = CocoEvalData(
            image_ids=[10, 20],
            image_paths=[str(img_a), str(img_b)],
            captions=["caption a1", "DIFFERENT CAPTION", "caption b1"],
            caption_to_image_index=[0, 0, 1],
            image_to_caption_indices=[[0, 1], [2]],
        )
        _assert_raises_value_error(
            load_or_compute_embeddings,
            base_args,
            altered_captions_data,
        )

        # Equivalent endpoint representations (e.g. --service-url with trailing
        # slash/uppercase host vs matching --host/--port) must reuse the cache.
        equiv_url_args = argparse.Namespace(
            **{
                **vars(base_args),
                "service_url": "HTTP://127.0.0.1:30000/",
            }
        )
        _, equiv_t, equiv_i = load_or_compute_embeddings(equiv_url_args, base_data)
        assert np.allclose(equiv_t, raw_text)
        assert np.allclose(equiv_i, raw_image)

        # Cross-endpoint reuse (same model/inputs, different --service-url, --host,
        # or --port) as well as mismatched model, prompts, split, or transport
        # must fail validation instead of attributing cached vectors to another server.
        for override in [
            {"service_url": "http://127.0.0.1:30001"},
            {"service_url": "http://10.0.0.2:30000"},
            {"host": "http://localhost"},
            {"host": "http://10.0.0.2"},
            {"port": 30001},
            {"model": "model-b"},
            {"query_prompt": "different-q-prefix"},
            {"query_prompt": None},
            {"image_prompt": "different-i-prefix"},
            {"image_transport": "data-url"},
            {"split": "train2017"},
        ]:
            bad_args = argparse.Namespace(**{**vars(base_args), **override})
            _assert_raises_value_error(
                load_or_compute_embeddings,
                bad_args,
                base_data,
            )

        # Modified image file contents at the same path must fail validation.
        img_a.write_bytes(b"modified-jpeg-bytes")
        _assert_raises_value_error(
            load_or_compute_embeddings,
            base_args,
            base_data,
        )

        # Legacy/unvalidated cache missing metadata must fail validation.
        legacy_cache = tmp_path / "legacy.npz"
        np.savez_compressed(
            legacy_cache,
            text_embeddings=raw_text,
            image_embeddings=raw_image,
            image_ids=np.asarray(base_data.image_ids, dtype=np.int64),
            caption_to_image_index=np.asarray(
                base_data.caption_to_image_index, dtype=np.int64
            ),
        )
        legacy_args = argparse.Namespace(
            **{**vars(base_args), "cache_npz": legacy_cache}
        )
        _assert_raises_value_error(
            load_or_compute_embeddings,
            legacy_args,
            base_data,
        )

    print("Self-test passed.")


def main() -> int:
    args = parse_args()
    service_url = build_service_url(args)
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
    if args.port <= 0:
        raise SystemExit("--port must be positive.")

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
        "service_url": service_url,
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
