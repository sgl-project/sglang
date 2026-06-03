"""Microbenchmark for the multimodal-processor offload (SGLANG_ENABLE_MM_PROCESSOR_OFFLOAD).

It isolates the *exact* work the offload moves off the TokenizerManager asyncio
event loop -- the synchronous HF processor in `process_and_combine_mm_data`
(resize/patchify the images + tokenize the text) -- and drives it through asyncio
at a fixed concurrency, comparing the two modes:

  inline   : run the work directly in the coroutine (blocks the event loop)
             == SGLANG_ENABLE_MM_PROCESSOR_OFFLOAD off
  offload  : await loop.run_in_executor(threadpool, work)
             == SGLANG_ENABLE_MM_PROCESSOR_OFFLOAD on (SGLANG_MM_PROC_WORKERS workers)

CPU-only (no GPU / server / JIT) so it is cheap and portable. A 5ms "event-loop
ticker" runs during the load to measure how long the loop is starved -- a proxy
for the TTFT/streaming smoothness of *other* in-flight requests.

Metrics:
  * preprocessing throughput (req/s)            -- the headline win
  * event-loop max stall (ms)                   -- why TTFT improves under load
  * per-request preprocessing latency (ms)      -- see NOTE below

NOTE on per-request latency: the two modes are not apples-to-apples (Little's law
W = L / throughput). Inline blocks the loop, so only L~=1 request is ever truly
"in work" while the rest are frozen (its low latency hides the stall). Offload
runs all L=concurrency requests at once against the worker pool, so latency =
concurrency / throughput is naturally higher -- it reflects real concurrent
processing bounded by the GIL-limited fraction of the HF processor, plus pool
queue wait. In production the GPU forward paces the pipeline so preprocessing is
not saturated this deep; the transferable wins are throughput and loop responsiveness.

Example:
  python benchmark/bench_mm_processor_offload.py \
      --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
      --num-requests 96 --concurrency 32 --images-per-req 4 --workers 16
"""

import argparse
import asyncio
import concurrent.futures
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image, ImageDraw
from transformers import AutoProcessor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        required=True,
        help="HF model/processor path with an image processor (e.g. a Qwen-VL).",
    )
    p.add_argument("--num-requests", type=int, default=96)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--images-per-req", type=int, default=4)
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="offload thread-pool size (= SGLANG_MM_PROC_WORKERS)",
    )
    p.add_argument("--text-tokens", type=int, default=3800)
    p.add_argument("--image-width", type=int, default=1536)
    p.add_argument("--image-height", type=int, default=2170)
    p.add_argument(
        "--distinct-images",
        type=int,
        default=128,
        help="size of the synthetic image pool",
    )
    return p.parse_args()


def make_image(idx: int, w: int, h: int) -> Image.Image:
    bg = ((idx * 37) % 200 + 30, (idx * 71) % 200 + 30, (idx * 113) % 200 + 30)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    for k in range(12):
        y = 60 + k * (h // 13)
        d.rectangle(
            [80, y, w - 80, y + 90],
            fill=((idx + k * 17) % 255, (k * 23) % 255, (idx * 5) % 255),
        )
    d.text((90, 30), f"DOC {idx:05d}", fill=(255, 255, 255))
    return im


async def _ticker(stop, gaps, period=0.005):
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(period)
        now = time.perf_counter()
        gaps.append((now - last - period) * 1000.0)
        last = now


async def run_mode(mode, work, num_requests, concurrency, pool):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)
    lat = []

    async def req(i):
        async with sem:
            t = time.perf_counter()
            if mode == "inline":
                work(i)
            else:
                await loop.run_in_executor(pool, work, i)
            lat.append((time.perf_counter() - t) * 1000.0)

    stop = asyncio.Event()
    gaps = []
    tk = asyncio.create_task(_ticker(stop, gaps))
    t0 = time.perf_counter()
    await asyncio.gather(*(req(i) for i in range(num_requests)))
    wall = time.perf_counter() - t0
    stop.set()
    await tk
    lat.sort()
    pct = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
    print(f"\n===== {mode} (concurrency={concurrency}, n={num_requests}) =====")
    print(f"  throughput: {num_requests / wall:.2f} req/s   (wall {wall:.2f}s)")
    print(
        f"  per-req preprocessing ms: p50 {pct(lat, .5):.0f}  p90 {pct(lat, .9):.0f}  p99 {pct(lat, .99):.0f}"
    )
    print(
        f"  event-loop max stall ms: {max(gaps):.0f}  (p50 {statistics.median(gaps):.1f})"
    )
    return {"rps": num_requests / wall, "p99": pct(lat, 0.99), "loop_max_ms": max(gaps)}


async def main():
    args = parse_args()
    print(f"loading processor: {args.model_path}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    imgproc, tok = proc.image_processor, proc.tokenizer

    imgs = [
        make_image(i, args.image_width, args.image_height)
        for i in range(args.distinct_images)
    ]
    text = tok.decode(
        tok("doc page content " * 1000, add_special_tokens=False).input_ids[
            : args.text_tokens
        ]
    )
    ipr = args.images_per_req

    def work(req_idx):
        sel = [imgs[(req_idx * ipr + k) % len(imgs)] for k in range(ipr)]
        imgproc(images=sel, return_tensors="pt")
        tok(text, return_tensors="pt")

    work(0)  # warm
    t = time.perf_counter()
    for i in range(5):
        work(i)
    print(
        f"single preprocessing call: {(time.perf_counter() - t) / 5 * 1000:.0f} ms "
        f"({ipr} imgs @{args.image_width}x{args.image_height} + {args.text_tokens} tok)"
    )

    off = await run_mode("inline", work, args.num_requests, args.concurrency, None)
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="sgl-mmproc"
    )
    on = await run_mode("offload", work, args.num_requests, args.concurrency, pool)
    pool.shutdown()

    print("\n================ offload OFF -> ON ================")
    print(
        f"  throughput:           {off['rps']:.2f} -> {on['rps']:.2f} req/s  ({on['rps'] / off['rps']:.2f}x)"
    )
    print(
        f"  event-loop max stall: {off['loop_max_ms']:.0f} -> {on['loop_max_ms']:.0f} ms"
    )
    print(
        f"  per-req p99:          {off['p99']:.0f} -> {on['p99']:.0f} ms  (see NOTE in module docstring)"
    )


if __name__ == "__main__":
    asyncio.run(main())
