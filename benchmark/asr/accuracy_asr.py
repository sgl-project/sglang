"""
ASR Accuracy Benchmark for SGLang (WER and RTFx)

This script evaluates transcription accuracy (WER) and runtime efficiency (RTFx)
for audio-capable models served by SGLang via the OpenAI-compatible Chat API.

Data flow:
HF ASR Dataset --(audio array/sr)--> WAV data URL --> /v1/chat/completions (non-stream)
 --> model output text --> normalization --> WER + RTFx metrics

Metrics:
- WER (%): Word Error Rate using `evaluate`'s "wer".
- RTFx: Sum of audio seconds processed / Sum of compute time seconds (higher is better).
- Latency stats: mean/median/95th percentile per request.

Usage example:
  python -m benchmark.asr.accuracy_asr \
    --port 30000 \
    --dataset openslr/librispeech_asr --subset clean --split test \
    --limit 100 --concurrency 4 \
    --max-tokens 128

Notes:
- Uses non-streaming chat.completions for simplicity and determinism.
- Sends audio as data:audio/wav;base64,... in messages[].content[].audio_url.url.
- Normalization defaults to simple English normalization. If your model/tokenizer
  provides a more appropriate normalization, switch with --normalize basic to
  disable English-specific rules or extend this script accordingly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, List, Optional, Tuple

import numpy as np
import openai
import requests


@dataclass
class Sample:
    audio: np.ndarray
    sr: int
    text: Optional[str]  # reference text (if available)


@dataclass
class BenchResult:
    latency_s: float
    tokens: int
    hyp: str
    ref: str
    audio_sec: float
    success: bool
    error: str = ""


def _duration_seconds(y: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return float(len(y)) / float(sr)


def _wav_data_url(
    y: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    mono: bool = True,
) -> str:
    """
    Convert a waveform to a base64 WAV data URL.
    - y: np.ndarray (1D or 2D)
    - sr: original sampling rate
    - target_sr: resample if needed (uses scipy.signal.resample if available)
    - mono: average channels if stereo/multi-channel
    """
    import soundfile as sf

    # Convert types
    if isinstance(y, list):
        y = np.asarray(y)

    if hasattr(y, "ndim") and y.ndim > 1 and mono:
        y = np.mean(y, axis=1)

    if sr != target_sr and sr > 0:
        try:
            from scipy.signal import resample as _resample
        except Exception:
            _resample = None
        if _resample is not None:
            num_samples = int(len(y) * float(target_sr) / float(sr))
            if num_samples > 0:
                y = np.asarray(_resample(y, num_samples))
                sr = target_sr

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def _normalize_english(s: str) -> str:
    # Simple English normalization: lowercase, keep alnum and spaces, collapse spaces
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_basic(s: str) -> str:
    # Minimal normalization, strip and collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def _fetch_default_model_id(base_url: str) -> Optional[str]:
    try:
        resp = requests.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception:
        return None
    return None


def _load_hf_asr_dataset(
    dataset_repo: str,
    split: str,
    subset: Optional[str],
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install datasets: pip install datasets") from e

    if subset:
        return load_dataset(dataset_repo, subset, split=split)
    else:
        return load_dataset(dataset_repo, split=split)


def _iter_asr_samples(
    dataset: Any,
    limit: int,
    skip_long: bool,
    max_duration_s: float,
) -> List[Sample]:
    results: List[Sample] = []
    for item in dataset:
        if limit is not None and len(results) >= limit:
            break
        audio = item.get("audio")
        if not audio:
            continue

        y = None
        sr = 0
        try:
            if isinstance(audio, dict) and "array" in audio:
                y = np.asarray(audio["array"])
                sr = int(audio.get("sampling_rate", 0))
            elif hasattr(audio, "array") and hasattr(audio, "sampling_rate"):
                y = np.asarray(getattr(audio, "array"))
                sr = int(getattr(audio, "sampling_rate"))
        except Exception:
            continue

        if y is None or sr <= 0:
            continue

        dur = _duration_seconds(y, sr)
        if dur <= 0.0:
            continue
        if skip_long and dur > max_duration_s:
            continue

        ref = None
        if "text" in item and isinstance(item["text"], str):
            ref = item["text"]
        elif "transcription" in item and isinstance(item["transcription"], str):
            ref = item["transcription"]

        results.append(Sample(audio=y, sr=sr, text=ref))
    return results


async def _transcribe_one(
    client: Any,
    model: str,
    prompt: str,
    audio_url: str,
    max_tokens: int,
    normalize_mode: str,
) -> Tuple[float, int, str]:
    """
    Non-streaming chat.completions call. Returns (latency_s, out_tokens, out_text).
    """
    st = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    et = time.perf_counter()
    latency = et - st
    text = resp.choices[0].message.content or ""
    out_tokens = int(getattr(resp, "usage", {}).get("completion_tokens", 0) or 0)

    if normalize_mode == "english":
        text = _normalize_english(text)
    elif normalize_mode == "basic":
        text = _normalize_basic(text)
    # else no normalization

    return latency, out_tokens, text


async def run_benchmark(
    port: int,
    base_url: Optional[str],
    model: Optional[str],
    dataset_repo: str,
    split: str,
    subset: Optional[str],
    limit: int,
    concurrency: int,
    max_tokens: int,
    target_sr: int,
    skip_long: bool,
    max_duration_s: float,
    normalize_mode: str,
) -> None:
    base = base_url or f"http://127.0.0.1:{port}"
    mdl = model
    if not mdl:
        mdl = await _fetch_default_model_id(base)
    if not mdl:
        mdl = "default"

    client = openai.AsyncOpenAI(api_key="sk", base_url=f"{base}/v1")

    # Load dataset
    ds = _load_hf_asr_dataset(dataset_repo, split, subset)
    samples = _iter_asr_samples(ds, limit, skip_long, max_duration_s)

    if not samples:
        print("No samples after filtering. Adjust --split/--limit/--max-duration-s.")
        return

    # Prepare references and audio URLs
    refs: List[str] = []
    audio_urls: List[str] = []
    audio_secs: List[float] = []

    for s in samples:
        data_url = _wav_data_url(s.audio, s.sr, target_sr=target_sr, mono=True)
        audio_urls.append(data_url)
        audio_secs.append(_duration_seconds(s.audio, s.sr))

        ref_text = s.text or ""
        if normalize_mode == "english":
            ref_text = _normalize_english(ref_text)
        elif normalize_mode == "basic":
            ref_text = _normalize_basic(ref_text)
        refs.append(ref_text)

    sem = asyncio.Semaphore(concurrency)
    prompt = "Please transcribe the audio into text."

    async def bound_task(aurl: str) -> BenchResult:
        async with sem:
            try:
                latency, tokens, hyp = await _transcribe_one(
                    client=client,
                    model=mdl,
                    prompt=prompt,
                    audio_url=aurl,
                    max_tokens=max_tokens,
                    normalize_mode=normalize_mode,
                )
                return BenchResult(
                    latency_s=latency,
                    tokens=tokens,
                    hyp=hyp,
                    ref="",  # fill after mapping indices
                    audio_sec=0.0,  # fill after mapping indices
                    success=True,
                )
            except Exception:
                return BenchResult(
                    latency_s=0.0,
                    tokens=0,
                    hyp="",
                    ref="",
                    audio_sec=0.0,
                    success=False,
                    error="".join(traceback.format_exception(*sys.exc_info())),
                )

    tasks = [bound_task(u) for u in audio_urls]

    # Run benchmark
    start = time.perf_counter()
    outs = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    # Attach refs and durations to each result in order
    for i, r in enumerate(outs):
        if i < len(refs):
            r.ref = refs[i]
        if i < len(audio_secs):
            r.audio_sec = audio_secs[i]

    # Compute WER
    try:
        from evaluate import load as eval_load

        wer_metric = eval_load("wer")
        predictions = [r.hyp for r in outs]
        references = [r.ref for r in outs]
        # Only compute WER on successful requests
        filtered_pairs = [
            (p, t) for (p, r), t in zip(zip(predictions, references), outs) if t.success
        ]
        if filtered_pairs:
            preds_flt, refs_flt = zip(*[(p, r) for (p, r) in filtered_pairs])
            wer_score = 100 * wer_metric.compute(
                references=list(refs_flt), predictions=list(preds_flt)
            )
        else:
            wer_score = float("nan")
    except Exception as e:
        print(
            f"WARNING: Could not compute WER (install with `pip install evaluate`): {e}"
        )
        wer_score = float("nan")

    # Compute RTFx and Latency stats
    latencies = [r.latency_s for r in outs if r.success]
    total_audio_sec = sum([r.audio_sec for r in outs if r.success])
    total_compute_sec = sum(latencies)
    rtfx = (total_audio_sec / total_compute_sec) if total_compute_sec > 0 else 0.0

    # Stats
    succ = sum(1 for r in outs if r.success)
    print("\n=== ASR Accuracy Benchmark ===")
    print(f"Total Requests: {len(outs)}")
    print(f"Successful Requests: {succ}")
    print(f"Total Test Time: {total_time:.4f} seconds")
    if latencies:
        print(f"Average Latency: {mean(latencies):.4f} seconds")
        print(f"Median Latency: {median(latencies):.4f} seconds")
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        print(f"95th Percentile Latency: {p95:.4f} seconds")
    else:
        print("No successful latencies to aggregate.")

    print(f"Total audio seconds (successful): {total_audio_sec:.2f}")
    print(f"Total compute seconds (successful): {total_compute_sec:.2f}")
    print(f"RTFx: {rtfx:.3f} (higher is better)")

    if not math.isnan(wer_score):
        print(f"WER: {wer_score:.3f}%")
    else:
        print("WER: N/A")

    # Short per-request lines (first 10)
    for i, r in enumerate(outs[: min(10, len(outs))]):
        status = "ok" if r.success else "fail"
        print(
            f"[{i}] {status} lat={r.latency_s:.3f}s tok={r.tokens} "
            f"audio={r.audio_sec:.2f}s hyp='{(r.hyp[:60] + '...') if len(r.hyp) > 60 else r.hyp}'"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument(
        "--base-url", type=str, default=None, help="Override http://host:port"
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model id; auto-detect if omitted",
    )

    ap.add_argument("--dataset", type=str, default="openslr/librispeech_asr")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--subset", type=str, default="clean")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)

    ap.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Resample audio to this sample rate",
    )
    ap.add_argument(
        "--skip-long", action="store_true", default=True, help="Skip long samples"
    )
    ap.add_argument(
        "--max-duration-s",
        type=float,
        default=30.0,
        help="Max allowed duration (seconds)",
    )

    ap.add_argument(
        "--normalize",
        type=str,
        default="english",
        choices=["english", "basic", "none"],
        help="Text normalization for hyp/ref before WER",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_url or f"http://{args.host}:{args.port}"
    asyncio.run(
        run_benchmark(
            port=args.port,
            base_url=args.base_url,
            model=args.model,
            dataset_repo=args.dataset,
            split=args.split,
            subset=args.subset,
            limit=args.limit,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            target_sr=args.target_sr,
            skip_long=args.skip_long,
            max_duration_s=args.max_duration_s,
            normalize_mode=args.normalize,
        )
    )


if __name__ == "__main__":
    main()
