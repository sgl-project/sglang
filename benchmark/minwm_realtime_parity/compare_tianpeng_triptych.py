#!/usr/bin/env python3
"""Compare published, native minWM, and SGLang Tianpeng gap12 outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from common import sha256_file, write_json
from tianpeng_alignment import _run_ffmpeg_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-results", required=True)
    parser.add_argument("--sglang-results", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _raw_metrics(reference: Path, candidate: Path) -> dict:
    left = np.load(reference, mmap_mode="r")
    right = np.load(candidate, mmap_mode="r")
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(
            f"raw output mismatch: {left.shape}/{left.dtype} != "
            f"{right.shape}/{right.dtype}"
        )
    maximum = 0
    absolute_sum = 0
    squared_sum = 0
    different_values = 0
    exact_frames = 0
    for frame_index in range(left.shape[0]):
        delta = left[frame_index].astype(np.int16) - right[frame_index].astype(np.int16)
        absolute = np.abs(delta)
        maximum = max(maximum, int(absolute.max()))
        absolute_sum += int(absolute.sum(dtype=np.int64))
        squared_sum += int(np.square(delta, dtype=np.int64).sum(dtype=np.int64))
        frame_differences = int(np.count_nonzero(delta))
        different_values += frame_differences
        exact_frames += int(frame_differences == 0)
    values = int(left.size)
    mse = squared_sum / values
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "bitwise_equal": different_values == 0,
        "exact_frames": exact_frames,
        "different_values": different_values,
        "different_fraction": different_values / values,
        "max_abs_diff": maximum,
        "mean_abs_diff": absolute_sum / values,
        "mse": mse,
        "psnr": math.inf if mse == 0 else 10 * math.log10((255**2) / mse),
    }


def _write_player(path: Path, metrics: dict) -> None:
    summary = json.dumps(metrics["native_vs_sglang_raw"], indent=2, ensure_ascii=False)
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>MinWM Tianpeng gap12 三路对齐</title>
<style>
body{{font-family:system-ui;background:#101114;color:#eee;margin:24px}}
button{{font-size:16px;padding:10px 18px;margin:0 8px 18px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
video{{width:100%;background:#000}} h2{{font-size:17px}}
pre{{background:#181b20;padding:16px;overflow:auto}}
</style>
<h1>Tianpeng detailmix director gap12</h1>
<button onclick="playAll()">三路同步播放</button>
<button onclick="pauseAll()">暂停</button>
<button onclick="resetAll()">归零</button>
<div class="grid">
<section><h2>天鹏已发布 baseline</h2><video controls muted loop src="baseline.mp4"></video></section>
<section><h2>minWM 4220c8a 原生重放</h2><video controls muted loop src="native_minwm.mp4"></video></section>
<section><h2>SGLang MinWM</h2><video controls muted loop src="sglang.mp4"></video></section>
</div>
<h2>原生 minWM vs SGLang 原始 RGB</h2>
<pre>{summary}</pre>
<script>
const videos=[...document.querySelectorAll("video")];
function playAll(){{const t=Math.min(...videos.map(v=>v.currentTime));videos.forEach(v=>v.currentTime=t);void Promise.all(videos.map(v=>v.play()));}}
function pauseAll(){{videos.forEach(v=>v.pause());}}
function resetAll(){{pauseAll();videos.forEach(v=>v.currentTime=0);}}
</script>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    native = Path(args.native_results).resolve()
    sglang = Path(args.sglang_results).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline.mp4": native / "baseline.mp4",
        "native_minwm.mp4": native / "native_minwm.mp4",
        "sglang.mp4": sglang / "sglang.mp4",
    }
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / name)
    metrics = {
        "native_vs_sglang_raw": _raw_metrics(
            native / "native_minwm.npy", sglang / "sglang.npy"
        ),
        "native_vs_sglang_encoded": {
            "psnr": _run_ffmpeg_metric(
                "psnr", files["native_minwm.mp4"], files["sglang.mp4"]
            ),
            "ssim": _run_ffmpeg_metric(
                "ssim", files["native_minwm.mp4"], files["sglang.mp4"]
            ),
        },
        "published_vs_sglang_encoded": {
            "psnr": _run_ffmpeg_metric(
                "psnr", files["baseline.mp4"], files["sglang.mp4"]
            ),
            "ssim": _run_ffmpeg_metric(
                "ssim", files["baseline.mp4"], files["sglang.mp4"]
            ),
        },
        "videos": {name: {"sha256": sha256_file(output / name)} for name in files},
    }
    write_json(output / "comparison.json", metrics)
    _write_player(output / "index.html", metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
