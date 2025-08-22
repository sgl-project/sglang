"""
Async audio ASR benchmark against SGLang's OpenAI-compatible /v1/chat/completions.

Usage:
  1) Launch server (example, MiniCPM-o):
     python -m sglang.launch_server --model-path openbmb/MiniCPM-o-2_6 --port 30000 --trust-remote-code
  2) Run benchmark:
     python benchmark/asr/bench_sglang_asr.py --port 30000 --dataset openslr/librispeech_asr --split test --limit 8 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from tqdm import tqdm

try:
    # When running as a script: python benchmark/asr/bench_sglang_asr.py
    from asr_dataset import ASRDataset, ASRSample
except Exception:  # pragma: no cover
    # Fallback if executed differently
    from benchmark.asr.asr_dataset import ASRDataset, ASRSample


API_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


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


async def stream_chat_with_audio(
    base_url: str,
    audio_data_url: str,
    prompt: str,
    max_tokens: int,
) -> RequestMetrics:
    """POST to /v1/chat/completions with streaming and parse SSE chunks.

    This mirrors the VLLM-style async streaming loop, adapted for SGLang's
    chat completions API with multimodal audio messages.
    """
    api_url = f"{base_url}/v1/chat/completions"
    headers = {
        # SGLang accepts any bearer; use env if provided
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    # Keep text instruction clear and short
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "stream": True,
        # Include usage in stream if server supports it
        "stream_include_usage": True,
        "stream_continuous_usage_stats": True,
    }

    metrics = RequestMetrics(success=False)
    st = time.perf_counter()
    most_recent = st
    ttft_recorded = False

    async with aiohttp.ClientSession(trust_env=True, timeout=API_TIMEOUT) as session:
        try:
            async with session.post(api_url, headers=headers, data=json.dumps(payload)) as resp:
                if resp.status != 200:
                    metrics.error = resp.reason or f"HTTP {resp.status}"
                    return metrics

                async for raw_chunk in resp.content:
                    chunk = raw_chunk.strip()
                    if not chunk:
                        continue
                    # SGLang streams SSE lines prefixed with 'data: '
                    try:
                        line = chunk.decode("utf-8")
                    except Exception:
                        continue

                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    now = time.perf_counter()
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        # Skip malformed JSON fragments
                        continue

                    # Choices/delta content
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content_piece = delta.get("content")

                        if content_piece is not None:
                            if not ttft_recorded:
                                metrics.ttft = now - st
                                ttft_recorded = True
                            else:
                                metrics.itl.append(now - most_recent)
                            metrics.generated_text += content_piece
                            most_recent = now

                    # Usage token counts (if present)
                    if "usage" in data and isinstance(data["usage"], dict):
                        # SGLang may include completion_tokens here during streaming
                        metrics.output_tokens = int(data["usage"].get("completion_tokens") or 0)

        except Exception:
            exc = sys.exc_info()
            metrics.error = "".join(traceback.format_exception(*exc))
            return metrics

    metrics.latency = most_recent - st if most_recent > st else 0.0
    metrics.success = True
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
    base_url = f"http://127.0.0.1:{port}"
    ds = ASRDataset(path=dataset, split=split, subset=subset, skip_long=skip_long)
    samples: List[ASRSample] = ds.iter_samples(limit=limit)

    if len(samples) == 0:
        print("No samples found after filtering. Consider disabling --skip-long or changing split.")
        return

    print(f"Collected {len(samples)} audio samples for benchmarking.")

    prompt = "Please transcribe the audio into text."
    sem = asyncio.Semaphore(concurrency)

    async def process_one(sample: ASRSample) -> RequestMetrics:
        async with sem:
            url = wav_data_url(sample.audio, sample.sr)
            return await stream_chat_with_audio(
                base_url=base_url, audio_data_url=url, prompt=prompt, max_tokens=max_tokens
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

    if succ:
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_lat = sum(lats) / len(lats)
    else:
        avg_ttft = 0.0
        avg_lat = 0.0

    print("\n=== ASR Benchmark Summary ===")
    print(f"Total: {len(samples)}, Success: {succ}, Fail: {len(samples) - succ}")
    print(f"Average TTFT: {avg_ttft:.3f}s")
    print(f"Average Latency: {avg_lat:.3f}s")

    # Optional: print a short per-request line
    for i, m in enumerate(outs[: min(5, len(outs))]):
        status = "ok" if m.success else "fail"
        print(f"[{i}] {status} ttft={m.ttft:.3f}s lat={m.latency:.3f}s tokens={m.output_tokens}")


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
    return parser.parse_args()


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
