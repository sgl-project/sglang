"""MMStar multimodal benchmark for DFLASH/speculative decoding.

Connects to an existing SGLang server (e.g. with DFLASH or baseline), runs
MMStar VLM requests, and reports throughput, average acceptance length, etc.

Usage:
  # Start server (with or without DFLASH), e.g.:
  #   python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000
  #   (with DFLASH: add --speculative-algorithm DFLASH --speculative-draft-model-path ...)

  python benchmark/dflash/bench_dflash_mmstar.py --port 30000 --num-samples 100
  python benchmark/dflash/bench_dflash_mmstar.py --base-url http://127.0.0.1:30000 --concurrency 8 --output-md report.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import base64
import openai
import requests
from tqdm import tqdm

try:
    from sglang.utils import encode_image_base64
except ImportError:
    def encode_image_base64(image_path: str) -> str:
        """Fallback: read image file and return base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


@dataclass
class Sample:
    """One MMStar sample: image path + prompt pieces."""

    sample_id: str
    image_path: str
    prompt_prefix: str
    prompt_suffix: str


@dataclass
class RequestMetrics:
    """Per-request metrics from one completion."""

    completion_tokens: int
    prompt_tokens: int
    latency_s: float
    spec_accept_length: Optional[float] = None
    spec_verify_ct: Optional[int] = None
    response_text: Optional[str] = None


@dataclass
class BenchResult:
    """Aggregate benchmark result."""

    num_samples: int
    total_latency_s: float
    total_completion_tokens: int
    total_prompt_tokens: int
    throughput_toks_per_s: float
    avg_spec_accept_length: Optional[float] = None
    avg_spec_verify_ct: Optional[float] = None
    errors: int = 0
    per_request_metrics: list[RequestMetrics] = field(default_factory=list)


def _load_mmstar_samples(
    dataset_path: str = "MMStar/MMStar",
    split: str = "val",
    num_samples: Optional[int] = None,
    image_cache_dir: Optional[str] = None,
    image_pixels_limit: int = -1,
) -> list[Sample]:
    """Load MMStar dataset and prepare samples with local image paths."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    cache_dir = image_cache_dir or os.path.expanduser("~/.cache/mmstar/images")
    os.makedirs(cache_dir, exist_ok=True)

    # Local path or HF dataset id; do not use trust_remote_code (deprecated for datasets)
    ds = load_dataset(dataset_path, split=split)
    if num_samples is not None and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    samples: list[Sample] = []
    for i, row in enumerate(ds):
        # MMStar columns: question, image (PIL or path), options, answer; id may vary
        question = row.get("question") or row.get("Question") or row.get("query", "")
        image = row.get("image") or row.get("image_path") or row.get("img_path")
        if image is None:
            continue
        sample_id = row.get("id") or row.get("question_id") or str(i)
        if isinstance(sample_id, (int, float)):
            sample_id = str(int(sample_id))
        ext = "png"
        image_path = os.path.join(cache_dir, f"mmstar_{sample_id}.{ext}")
        if isinstance(image, str) and (os.path.isfile(image) or image.startswith("http")):
            image_path = image if os.path.isfile(image) else image_path
            if not os.path.isfile(image_path) and image.startswith("http"):
                try:
                    from PIL import Image
                    import urllib.request
                    urllib.request.urlretrieve(image, image_path)
                except Exception:
                    continue
        elif not os.path.exists(image_path):
            try:
                if hasattr(image, "save"):
                    image.save(image_path)
                else:
                    continue
            except Exception:
                continue
        if image_pixels_limit > 0:
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    w, h = img.size
                    if w * h > image_pixels_limit:
                        continue
            except Exception:
                pass
        # Simple prompt: question + "Answer:" for VLM
        prompt_prefix = (question if isinstance(question, str) else str(question)).strip()
        if not prompt_prefix.endswith("?"):
            prompt_prefix = prompt_prefix + "\nAnswer:"
        else:
            prompt_prefix = prompt_prefix + "\nAnswer:"
        prompt_suffix = ""
        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=os.path.abspath(image_path),
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
            )
        )
    return samples


def _flush_cache(base_url: str) -> None:
    """Flush server cache so that server_info spec metrics reflect only this run."""
    url = base_url.rstrip("/") + "/flush_cache"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception:
        pass


def _get_server_spec_metrics(base_url: str) -> tuple[Optional[float], Optional[float]]:
    """Query /server_info for avg_spec_accept_length (and related)."""
    url = base_url.rstrip("/") + "/server_info"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None
    internal_states = data.get("internal_states") or []
    if not internal_states:
        return None, None
    state = internal_states[0]
    avg_accept = state.get("avg_spec_accept_length")
    return (float(avg_accept) if avg_accept is not None else None, None)


def _image_url_for_sample(sample: Sample, use_base64: bool) -> str:
    """Build image URL for API: base64 (works with remote server) or file path (same machine only)."""
    if use_base64:
        # Encode image into request body so server does not need filesystem access.
        # Required when client and server run on different machines.
        b64 = encode_image_base64(sample.image_path)
        return f"data:image/png;base64,{b64}"
    # Local file path: only works when server can read the same path (client and server on same machine).
    path = os.path.abspath(sample.image_path).replace("\\", "/")
    return path if path.startswith("http") else f"file:///{path.lstrip('/')}"


def _build_extra_body(
    ignore_eos: bool,
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
    repetition_penalty: Optional[float],
) -> dict[str, Any]:
    """SGLang-specific params: pass via extra_body (same as MMMU bench_sglang.py + ignore_eos etc.)."""
    extra: dict[str, Any] = {"ignore_eos": ignore_eos}
    if enable_thinking is not None:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    if repetition_penalty is not None and repetition_penalty != 1.0:
        extra["repetition_penalty"] = repetition_penalty
    return extra


def _get_sampling_params(
    temperature: float,
    max_completion_tokens: int,
    stop: Optional[list[str]],
) -> dict[str, Any]:
    """Build top-level sampling params (same pattern as MMMU get_sampling_params + payload.update)."""
    params: dict[str, Any] = {
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if stop:
        params["stop"] = stop
    return params


async def _send_one(
    client: openai.AsyncOpenAI,
    sample: Sample,
    *,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    use_base64_image: bool = True,
    ignore_eos: bool = False,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    repetition_penalty: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> RequestMetrics:
    """Send one chat completion and return metrics (payload layout aligned with MMMU bench_sglang)."""
    image_url = _image_url_for_sample(sample, use_base64=use_base64_image)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.prompt_prefix},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": sample.prompt_suffix} if sample.prompt_suffix else None,
                ],
            }
        ],
        "extra_body": _build_extra_body(
            ignore_eos, enable_thinking, reasoning_effort, repetition_penalty
        ),
    }
    payload["messages"][0]["content"] = [x for x in payload["messages"][0]["content"] if x is not None]
    # Merge sampling params at top level (same as MMMU: payload.update(sampling_params))
    sampling_params = _get_sampling_params(temperature, max_completion_tokens, stop)
    payload.update(sampling_params)

    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(**payload)
        latency = time.perf_counter() - start
        usage = getattr(resp, "usage", None)
        completion_tokens = int(usage.completion_tokens) if usage else 0
        prompt_tokens = int(usage.prompt_tokens) if usage else 0
        content = None
        if resp.choices and len(resp.choices) > 0 and resp.choices[0].message is not None:
            content = getattr(resp.choices[0].message, "content", None) or ""
        return RequestMetrics(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            latency_s=latency,
            spec_accept_length=None,
            spec_verify_ct=None,
            response_text=content,
        )
    except Exception as e:
        print(f"[Request failed] {type(e).__name__}: {e}")
        return RequestMetrics(
            completion_tokens=0,
            prompt_tokens=0,
            latency_s=time.perf_counter() - start,
            spec_accept_length=None,
            spec_verify_ct=None,
            response_text=None,
        )


async def _run_bench(
    base_url: str,
    samples: list[Sample],
    *,
    concurrency: int,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    flush_cache_before: bool = False,
    use_base64_image: bool = True,
    ignore_eos: bool = False,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    repetition_penalty: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> BenchResult:
    """Run all samples with concurrency and collect metrics."""
    if flush_cache_before:
        _flush_cache(base_url)
    client = openai.AsyncOpenAI(
        api_key="sk-any",
        base_url=base_url.rstrip("/") + "/v1",
        timeout=60.0 * 60,
    )
    sem = asyncio.Semaphore(concurrency)
    per_request: list[RequestMetrics] = []
    errors = 0

    async def run_one(s: Sample) -> RequestMetrics:
        async with sem:
            m = await _send_one(
                client, s,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                use_base64_image=use_base64_image,
                ignore_eos=ignore_eos,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            if m.completion_tokens == 0 and m.prompt_tokens == 0:
                nonlocal errors
                errors += 1
            return m

    start = time.perf_counter()
    tasks = [run_one(s) for s in samples]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="MMStar"):
        m = await coro
        per_request.append(m)
    total_latency = time.perf_counter() - start

    total_completion = sum(m.completion_tokens for m in per_request)
    total_prompt = sum(m.prompt_tokens for m in per_request)
    throughput = total_completion / max(total_latency, 1e-6)

    # Server-side spec metrics (cumulative over this run)
    avg_spec_accept, _ = _get_server_spec_metrics(base_url)

    return BenchResult(
        num_samples=len(samples),
        total_latency_s=total_latency,
        total_completion_tokens=total_completion,
        total_prompt_tokens=total_prompt,
        throughput_toks_per_s=throughput,
        avg_spec_accept_length=avg_spec_accept,
        avg_spec_verify_ct=None,
        errors=errors,
        per_request_metrics=per_request,
    )


def _print_and_save_result(result: BenchResult, output_md: Optional[str], base_url: str) -> None:
    """Print summary and optionally write markdown report."""
    lines = [
        "=" * 60,
        "MMStar multimodal benchmark result",
        "=" * 60,
        f"  Base URL:          {base_url}",
        f"  Num samples:        {result.num_samples}",
        f"  Errors:             {result.errors}",
        f"  Total latency (s):  {result.total_latency_s:.2f}",
        f"  Total completion:   {result.total_completion_tokens} tokens",
        f"  Throughput:         {result.throughput_toks_per_s:.2f} tok/s",
        f"  Avg accept length:  {result.avg_spec_accept_length if result.avg_spec_accept_length is not None else 'N/A'}",
        "=" * 60,
    ]
    for line in lines:
        print(line)

    # When only one request, print the model output
    if result.num_samples == 1 and result.per_request_metrics:
        m0 = result.per_request_metrics[0]
        if m0.response_text is not None:
            print("")
            print("--- Model output (single request) ---")
            print(m0.response_text)
            print("---")

    if output_md:
        md = [
            "# MMStar multimodal benchmark",
            "",
            "## Settings",
            f"- Base URL: `{base_url}`",
            f"- Num samples: `{result.num_samples}`",
            "",
            "## Results",
            f"- **Throughput**: {result.throughput_toks_per_s:.2f} tok/s",
            f"- **Total completion tokens**: {result.total_completion_tokens}",
            f"- **Total latency (s)**: {result.total_latency_s:.2f}",
            f"- **Average spec accept length**: " + (f"{result.avg_spec_accept_length:.3f}" if result.avg_spec_accept_length is not None else "N/A"),
            f"- **Errors**: {result.errors}",
            "",
        ]
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"\nWrote report to: {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMStar multimodal benchmark for SGLang (DFLASH/speculative)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="SGLang server port (default 30000).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL (e.g. http://127.0.0.1:30000). If set, --port is ignored.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="MMStar/MMStar",
        help="HuggingFace dataset path for MMStar (default: MMStar/MMStar).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split. MMStar has 'val' (default); use e.g. 'test' if your dataset has it.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Max number of samples to run (default: 100).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent requests (default: 4).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Max new tokens per request (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0). Top-level in request, same as MMMU/llava_bench. With speculative decoding, outputs may still vary slightly.",
    )
    parser.add_argument(
        "--image-cache-dir",
        type=str,
        default=None,
        help="Directory to cache MMStar images (default: ~/.cache/mmstar/images).",
    )
    parser.add_argument(
        "--image-pixels-limit",
        type=int,
        default=-1,
        help="Skip images with pixels > this (default: -1, no limit).",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write markdown report to this file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name for API (default: default).",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Call /flush_cache before run so spec metrics reflect only this run.",
    )
    parser.add_argument(
        "--use-file-url",
        action="store_true",
        help="Send image as file:// URL instead of base64. Only use when client and server share the same filesystem (same machine). Default is base64 for remote server.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS and generate up to max_completion_tokens (fixed-length throughput). Passed via extra_body. Default: False.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        dest="enable_thinking",
        default=None,
        help="Enable thinking/reasoning mode (e.g. Qwen3 enable_thinking). Passed via extra_body.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Disable thinking mode. Default: do not set (use server/model default).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for reasoning models (low/medium/high). Passed via extra_body.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        metavar="R",
        help="Repetition penalty to reduce looping (default: 1.05). Set 1.0 to disable.",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default=None,
        metavar="STR",
        help="Comma-separated stop sequences, e.g. \"Actually, let's look at the image again\",\"\\n\\n\\n\\n\". Stops generation when any appears.",
    )
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}"

    print("Loading MMStar samples...")
    samples = _load_mmstar_samples(
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=args.num_samples,
        image_cache_dir=args.image_cache_dir,
        image_pixels_limit=args.image_pixels_limit,
    )
    if not samples:
        raise SystemExit("No samples loaded. Check --dataset-path and --split.")
    print(f"Running benchmark with {len(samples)} samples, concurrency={args.concurrency}")

    result = asyncio.run(
        _run_bench(
            base_url,
            samples,
            concurrency=args.concurrency,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            flush_cache_before=args.flush_cache,
            use_base64_image=not args.use_file_url,
            ignore_eos=args.ignore_eos,
            enable_thinking=getattr(args, "enable_thinking", None),
            reasoning_effort=args.reasoning_effort,
            repetition_penalty=args.repetition_penalty,
            stop=[s.strip() for s in args.stop.split(",") if s.strip()] if args.stop else None,
        )
    )

    _print_and_save_result(result, args.output_md, base_url)


if __name__ == "__main__":
    main()
