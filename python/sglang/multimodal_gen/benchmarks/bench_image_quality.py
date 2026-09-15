"""Score a directory of generated images with CLIP / CLIP-IQA / ImageReward.

Consumes the images and the manifest that produced them, so every score is tied
to the prompt it was generated from:

    python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \\
        --model-path black-forest-labs/FLUX.2-klein \\
        --request-manifest prompts.jsonl --save-output-dir out/

    python -m sglang.multimodal_gen.benchmarks.bench_image_quality \\
        --request-manifest prompts.jsonl --images out/ --out-prefix klein_4b

Absolute means are only comparable against a run that used the same prompt file
and the same metric variants. To compare two arms, generate both from a
byte-identical manifest and pass the first arm's per-image JSONL as --baseline:
the paired test is several times tighter than comparing the two means, because
it cancels the prompt-to-prompt spread that dominates each mean's error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.benchmarks.request_manifest import (
    ManifestRequest,
    load_request_manifest,
)
from sglang.multimodal_gen.test.quality_metrics import (
    CLIP_PROMPT_MODEL_NAME,
    IMAGE_REWARD_MODEL_NAME,
    QualityScores,
    check_quality,
    compute_clip_iqa,
    compute_clip_score,
    compute_image_rewards,
    load_quality_thresholds,
)

ALL_METRICS = ("clip_score", "clip_iqa", "image_reward")
DEFAULT_METRICS = ("clip_score", "clip_iqa")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# schedule_batch appends _<output_idx> to the stem when a request returns more
# than one image, so the parsed id is the manifest's id plus that suffix.
_OUTPUT_INDEX_SUFFIX = re.compile(r"_\d+$")


def _discover_images(images_dir: Path) -> list[tuple[int, str, Path]]:
    """Parse bench_offline_throughput's ``<index>-<request_id>.<ext>`` outputs."""
    found = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index_text, separator, request_id = path.stem.partition("-")
        if not separator or not index_text.isdigit():
            raise ValueError(
                f"{path.name} is not named <index>-<request_id>.<ext>; point --images "
                "at a --save-output-dir from bench_offline_throughput"
            )
        found.append((int(index_text), request_id, path))
    if not found:
        raise ValueError(f"No images with suffix {IMAGE_SUFFIXES} under {images_dir}")
    return found


def _pair_with_manifest(
    *,
    images: list[tuple[int, str, Path]],
    requests: list[ManifestRequest],
) -> list[tuple[int, ManifestRequest, Path]]:
    """Match each image to its manifest row on index *and* request id."""
    paired = []
    for index, request_id, path in images:
        if index >= len(requests):
            raise ValueError(
                f"{path.name} has index {index} but the manifest holds "
                f"{len(requests)} requests"
            )
        request = requests[index]
        # bench_offline_throughput sanitizes the id into the filename, so
        # compare against the same sanitization rather than the raw id.
        expected = _sanitize_request_id(request.request_id)
        if request_id != expected:
            if _OUTPUT_INDEX_SUFFIX.sub("", request_id) == expected:
                raise ValueError(
                    f"{path.name} is an extra output for manifest row {index}; "
                    "scoring pairs one image per row, so generate a single output "
                    "per request."
                )
            raise ValueError(
                f"PAIRING ERROR: {path.name} claims request id {request_id!r} but "
                f"manifest row {index} is {expected!r}. The images and the manifest "
                "are from different runs; scoring them would mislabel every prompt."
            )
        paired.append((index, request, path))
    return paired


def _sanitize_request_id(request_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._")
    return safe or "request"


def _score_one(
    *,
    image: np.ndarray,
    prompt: str,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """The in-process metrics only; image_reward is batched by _score_rewards."""
    scores = {}
    if "clip_score" in metrics:
        scores["clip_score"] = compute_clip_score(image=image, prompt=prompt)
    if "clip_iqa" in metrics:
        scores["clip_iqa"] = compute_clip_iqa(image=image)
    return scores


def _score_rewards(
    *,
    rows: list[dict[str, Any]],
    paths: list[Path],
) -> None:
    # One launch for the whole directory: the 1.7 GB checkpoint load dominates a
    # sweep, and the on-disk images go over by path rather than re-encoded.
    print(f"scoring image_reward for {len(rows)} images in one launch", flush=True)
    scores = compute_image_rewards(
        pairs=[(row["prompt"], path) for row, path in zip(rows, paths)]
    )
    for row, score in zip(rows, scores):
        row["image_reward"] = score


def _summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "sem": 0.0}
    std = statistics.stdev(values)
    return {"n": n, "mean": mean, "std": std, "sem": std / math.sqrt(n)}


def paired_delta(
    *,
    values: list[float],
    baseline_values: list[float],
) -> dict[str, float | None]:
    """Mean of the per-image differences, its standard error, and mean/sem."""
    differences = [value - base for value, base in zip(values, baseline_values)]
    summary = _summarize(differences)
    mean_delta, sem = summary["mean"], summary["sem"]
    if sem > 0:
        z = mean_delta / sem
    else:
        # A constant non-zero difference has no spread to divide by, and z=0 would
        # read as "no difference"; only an all-zero delta is genuinely z=0.
        z = 0.0 if mean_delta == 0 else None
    return {
        "n": summary["n"],
        "mean_delta": mean_delta,
        "sem": sem,
        "z": z,
    }


def _load_baseline_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row["index"])] = row
    return rows


def _paired_report(
    *,
    rows: list[dict[str, Any]],
    baseline_rows: dict[int, dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    report = {}
    for metric in metrics:
        values, baseline_values = [], []
        for row in rows:
            baseline_row = baseline_rows.get(row["index"])
            if baseline_row is None or baseline_row.get(metric) is None:
                continue
            # A paired test is only valid on the same prompt; an unequal pair
            # means the two runs used different manifests.
            if baseline_row["prompt"] != row["prompt"]:
                raise ValueError(
                    f"MANIFEST DRIFT at index {row['index']}: baseline prompt "
                    f"{baseline_row['prompt']!r} != {row['prompt']!r}"
                )
            values.append(row[metric])
            baseline_values.append(baseline_row[metric])
        if values:
            report[metric] = paired_delta(
                values=values, baseline_values=baseline_values
            )
    return report


def run_scoring(args: argparse.Namespace) -> dict[str, Any]:
    metrics = tuple(args.metrics)
    if args.case_id is not None:
        _check_case_metrics(case_id=args.case_id, metrics=metrics)
    manifest = load_request_manifest(args.request_manifest)
    paired = _pair_with_manifest(
        images=_discover_images(Path(args.images)),
        requests=manifest.requests,
    )
    if args.limit is not None:
        paired = paired[: args.limit]

    rows = []
    for position, (index, request, path) in enumerate(paired, start=1):
        image = np.asarray(Image.open(path).convert("RGB"))
        row = {
            "index": index,
            "request_id": request.request_id,
            "prompt": request.prompt,
            "image": path.name,
            **_score_one(image=image, prompt=request.prompt, metrics=metrics),
        }
        rows.append(row)
        if position % args.log_every == 0 or position == len(paired):
            print(f"scored {position}/{len(paired)}", flush=True)

    if "image_reward" in metrics:
        _score_rewards(rows=rows, paths=[path for _, _, path in paired])

    summary: dict[str, Any] = {
        "images": str(Path(args.images).resolve()),
        "request_manifest": manifest.path,
        "request_manifest_sha256": manifest.sha256,
        "num_images": len(rows),
        # A truncated run has the same manifest sha256 as a full one, so the
        # limit has to be recorded or the artifact overstates what it covers.
        "limit": args.limit,
        "metrics": {
            metric: _summarize([row[metric] for row in rows]) for metric in metrics
        },
        "metric_variants": {
            "clip_score": f"{CLIP_PROMPT_MODEL_NAME}, 100*cosine",
            "clip_iqa": "torchmetrics clip_iqa weights, data_range=255",
            "image_reward": IMAGE_REWARD_MODEL_NAME,
        },
    }

    if args.baseline is not None:
        summary["paired_vs_baseline"] = {
            "baseline": str(Path(args.baseline).resolve()),
            **_paired_report(
                rows=rows,
                baseline_rows=_load_baseline_rows(Path(args.baseline)),
                metrics=metrics,
            ),
        }

    if args.case_id is not None:
        summary["threshold_check"] = _threshold_report(rows=rows, case_id=args.case_id)

    _write_outputs(out_prefix=args.out_prefix, rows=rows, summary=summary)
    return summary


def _check_case_metrics(*, case_id: str, metrics: tuple[str, ...]) -> None:
    """Refuse a --case-id whose floors need a metric --metrics does not compute.

    check_quality counts an enforced-but-missing score as a failure, so every
    image would otherwise violate a metric that was never asked for.
    """
    thresholds = load_quality_thresholds(case_id)
    missing = [
        metric
        for metric, minimum in (
            ("clip_score", thresholds.min_clip_score),
            ("clip_iqa", thresholds.min_clip_iqa),
            ("image_reward", thresholds.min_image_reward),
        )
        if minimum is not None and metric not in metrics
    ]
    if missing:
        raise ValueError(
            f"--case-id {case_id} enforces {', '.join(missing)}, which is not in "
            f"--metrics {' '.join(metrics)}. Add the metric, or drop --case-id and "
            "compare the arms with --baseline."
        )


def _threshold_report(*, rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    thresholds = load_quality_thresholds(case_id)
    violations = []
    for row in rows:
        scores = QualityScores(
            clip_score=row.get("clip_score"),
            clip_iqa=row.get("clip_iqa"),
            image_reward=row.get("image_reward"),
        )
        for failure in check_quality(scores=scores, thresholds=thresholds):
            violations.append({"index": row["index"], "failure": failure})
    return {
        "case_id": case_id,
        "num_violations": len(violations),
        "violations": violations[:20],
    }


def _write_outputs(
    *,
    out_prefix: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    per_image_path = Path(f"{out_prefix}_per_image.jsonl")
    summary_path = Path(f"{out_prefix}_summary.json")
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with per_image_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {per_image_path} and {summary_path}")


def _positive_int(value: str) -> int:
    # --limit 0 would summarize an empty score list, and --log-every 0 divides by
    # zero on the first image; both only surface after the sweep has run.
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated images with CLIP, CLIP-IQA and ImageReward."
    )
    parser.add_argument(
        "--images",
        required=True,
        help="--save-output-dir from bench_offline_throughput",
    )
    parser.add_argument(
        "--request-manifest",
        required=True,
        help="the JSONL manifest that generated those images",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="writes <prefix>_per_image.jsonl and <prefix>_summary.json",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=list(DEFAULT_METRICS),
        help="image_reward additionally needs SGLANG_TEST_IMAGE_REWARD_PYTHON",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="per-image JSONL from another arm; adds a paired delta per metric",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="also check every image against that case's floors in "
        "test/server/quality_thresholds.json",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="score only the first N images; the means are then not comparable "
        "against a full run, but the per-image rows and paired deltas are",
    )
    parser.add_argument("--log-every", type=_positive_int, default=25)
    args = parser.parse_args()

    summary = run_scoring(args)
    for metric, stats in summary["metrics"].items():
        print(
            f"{metric}: mean={stats['mean']:.4f} sem={stats['sem']:.4f} n={stats['n']}"
        )
    for metric, stats in summary.get("paired_vs_baseline", {}).items():
        if isinstance(stats, dict) and "mean_delta" in stats:
            z = stats["z"]
            print(
                f"{metric} vs baseline: delta={stats['mean_delta']:+.4f} "
                f"z={'undefined' if z is None else format(z, '+.2f')} n={stats['n']}"
            )


if __name__ == "__main__":
    main()
