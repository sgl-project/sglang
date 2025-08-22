"""
Async audio ASR benchmark against SGLang's OpenAI-compatible /v1/chat/completions.

Usage:
  1) Launch server (example, MiniCPM-o):
     python -m sglang.launch_server --model-path openbmb/MiniCPM-o-2_6 --port 30000 --trust-remote-code
  2) Run benchmark:
     python -m benchmark.asr.bench_sglang_asr --port 30000 --dataset openslr/librispeech_asr --split test --limit 8 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import openai
from tqdm import tqdm

from .asr_dataset import ASRDataset, ASRSample


@dataclass
class RequestMetrics:
    success: bool
    error: str = ""
    ttft: float = 0.0
    latency: float = 0.0
    output_tokens: int = 0
    generated_text: str = ""
    itl: List[float] = field(default_factory=list)  # inter-token latencies


def wav_data_url(y: np.ndarray, sr: int) -> str:
    """Encode waveform to WAV and return as data URL string.

    Shapes/dtypes:
      - y: 1D float np.ndarray on CPU; values scaled by soundfile
      - sr: int sample rate
    Returns:
      - str: 'data:audio/wav;base64,<...>'
    """
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


async def request_chat_completion(
    client: Any,
    audio_data_url: str,
    prompt: str,
    max_tokens: int,
) -> RequestMetrics:
    """Call OpenAI-compatible chat.completions via SDK (non-streaming)."""
    metrics = RequestMetrics(success=False)
    st = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_completion_tokens=max_tokens,
            max_tokens=max_tokens,
        )
        et = time.perf_counter()
        metrics.latency = et - st
        metrics.generated_text = resp.choices[0].message.content or ""
        metrics.output_tokens = int(
            getattr(resp, "usage", {}).get("completion_tokens", 0) or 0
        )
        metrics.success = True
        return metrics
    except Exception:
        exc = sys.exc_info()
        metrics.error = "".join(traceback.format_exception(*exc))
        return metrics


async def run_benchmark(
    port: int,
    dataset: str,
    split: str,
    subset: Optional[str],
    limit: int,
    concurrency: int,
    max_tokens: int,
    skip_long: bool,
) -> None:
    client = openai.AsyncOpenAI(api_key="sk", base_url=f"http://127.0.0.1:{port}/v1")
    ds = ASRDataset(path=dataset, split=split, subset=subset, skip_long=skip_long)
    samples: List[ASRSample] = ds.iter_samples(limit=limit)

    if len(samples) == 0:
        print(
            "No samples found after filtering. Consider disabling --skip-long or changing split."
        )
        return

    print(f"Collected {len(samples)} audio samples for benchmarking.")

    prompt = "Please transcribe the audio into text."
    sem = asyncio.Semaphore(concurrency)

    async def process_one(sample: ASRSample) -> RequestMetrics:
        async with sem:
            url = wav_data_url(sample.audio, sample.sr)
            return await request_chat_completion(
                client=client, audio_data_url=url, prompt=prompt, max_tokens=max_tokens
            )

    tasks = [process_one(s) for s in samples]
    ttfts, lats, succ, outs = [], [], 0, []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        m = await fut
        outs.append(m)
        if m.success:
            succ += 1
            ttfts.append(m.ttft)
            lats.append(m.latency)

    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0
    avg_lat = (sum(lats) / len(lats)) if lats else 0.0

    print("\n=== ASR Benchmark Summary ===")
    print(f"Total: {len(samples)}, Success: {succ}, Fail: {len(samples) - succ}")
    print(f"Average TTFT: {avg_ttft:.3f}s")
    print(f"Average Latency: {avg_lat:.3f}s")

    # Optional: print a short per-request line
    for i, m in enumerate(outs[: min(5, len(outs))]):
        status = "ok" if m.success else "fail"
        print(
            f"[{i}] {status} ttft={m.ttft:.3f}s lat={m.latency:.3f}s tokens={m.output_tokens}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dataset", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--skip-long", action="store_true", default=True)
    from sglang.test.test_utils import add_common_sglang_args_and_parse

    return add_common_sglang_args_and_parse(parser)


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_benchmark(
            port=args.port,
            dataset=args.dataset,
            split=args.split,
            subset=args.subset,
            limit=args.limit,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            skip_long=args.skip_long,
        )
    )


if __name__ == "__main__":
    main()
