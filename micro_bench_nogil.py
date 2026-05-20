"""Micro-benchmark: measure CPU-bound Python work with/without GIL.

Tests whether free-threaded Python provides real speedup for concurrent
tokenization and image processing — the two main CPU workloads in SGLang.
"""

import os
import sys
import time
import threading
import concurrent.futures
from io import BytesIO

import numpy as np
from PIL import Image


def _generate_random_image(size=(1920, 1080)):
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue()


def _process_image(jpeg_bytes):
    """Simulate the image processing pipeline: decode → resize → normalize."""
    img = Image.open(BytesIO(jpeg_bytes))
    img = img.convert("RGB")
    img = img.resize((448, 448), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return arr


def _tokenize_text(tokenizer, text):
    """Tokenize a text string."""
    return tokenizer.encode(text)


def bench_image_processing(images, num_workers, label=""):
    """Process images concurrently and measure time."""
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(_process_image, images))
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {len(images)} images, {num_workers} workers: {elapsed*1000:.1f}ms "
          f"({elapsed*1000/len(images):.1f}ms/img)")
    return elapsed


def bench_image_serial(images, label=""):
    """Process images serially."""
    start = time.perf_counter()
    results = [_process_image(img) for img in images]
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {len(images)} images, serial:     {elapsed*1000:.1f}ms "
          f"({elapsed*1000/len(images):.1f}ms/img)")
    return elapsed


def bench_tokenization(tokenizer, texts, num_workers, label=""):
    """Tokenize texts concurrently."""
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(lambda t: _tokenize_text(tokenizer, t), texts))
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {len(texts)} texts, {num_workers} workers: {elapsed*1000:.1f}ms")
    return elapsed


def bench_tokenization_serial(tokenizer, texts, label=""):
    """Tokenize texts serially."""
    start = time.perf_counter()
    results = [_tokenize_text(tokenizer, t) for t in texts]
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {len(texts)} texts, serial:     {elapsed*1000:.1f}ms")
    return elapsed


def bench_mixed(tokenizer, images, texts, num_img_workers, num_tok_workers, label=""):
    """Run image processing and tokenization concurrently (simulates real serving)."""
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_img_workers) as img_pool, \
         concurrent.futures.ThreadPoolExecutor(max_workers=num_tok_workers) as tok_pool:
        img_futures = [img_pool.submit(_process_image, img) for img in images]
        tok_futures = [tok_pool.submit(_tokenize_text, tokenizer, t) for t in texts]
        concurrent.futures.wait(img_futures + tok_futures)
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {len(images)} imgs + {len(texts)} texts, "
          f"{num_img_workers}+{num_tok_workers} workers: {elapsed*1000:.1f}ms")
    return elapsed


def main():
    gil_status = "DISABLED" if not sys._is_gil_enabled() else "ENABLED"
    print(f"Python {sys.version}")
    print(f"GIL: {gil_status}")
    print()

    # Prepare test data
    print("Generating test images (1080p)...")
    num_images = 32
    images = [_generate_random_image() for _ in range(num_images)]
    print(f"Generated {num_images} images, ~{len(images[0])/1024:.0f}KB each")

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/disk3/models/Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
    )
    sample_text = "This is a test prompt for benchmarking tokenization performance. " * 20
    texts = [sample_text] * 64
    print(f"Prepared {len(texts)} texts, ~{len(sample_text)} chars each")
    print()

    # Warmup
    _process_image(images[0])
    _tokenize_text(tokenizer, texts[0])

    # === Image Processing Benchmark ===
    print("=" * 60)
    print("IMAGE PROCESSING BENCHMARK")
    print("=" * 60)

    serial_time = bench_image_serial(images, "serial")
    for workers in [2, 4, 8, 16]:
        t = bench_image_processing(images, workers, f"{workers}T")
        print(f"    → Speedup vs serial: {serial_time/t:.2f}x")
    print()

    # === Tokenization Benchmark ===
    print("=" * 60)
    print("TOKENIZATION BENCHMARK")
    print("=" * 60)

    serial_tok = bench_tokenization_serial(tokenizer, texts, "serial")
    for workers in [2, 4, 8]:
        t = bench_tokenization(tokenizer, texts, workers, f"{workers}T")
        print(f"    → Speedup vs serial: {serial_tok/t:.2f}x")
    print()

    # === Mixed Benchmark (simulates real serving) ===
    print("=" * 60)
    print("MIXED BENCHMARK (image + tokenization concurrent)")
    print("=" * 60)

    serial_total = serial_time + serial_tok
    print(f"  Serial total (img + tok): {serial_total*1000:.1f}ms")
    for img_w, tok_w in [(4, 2), (8, 4), (16, 8)]:
        t = bench_mixed(tokenizer, images, texts, img_w, tok_w, f"{img_w}+{tok_w}")
        print(f"    → Speedup vs serial: {serial_total/t:.2f}x")
    print()

    # === Pure Python CPU benchmark (no C extensions) ===
    print("=" * 60)
    print("PURE PYTHON CPU BENCHMARK (dict/list processing)")
    print("=" * 60)

    def python_cpu_work(n):
        """Simulate Python-heavy work like logprob processing."""
        result = {}
        for i in range(n):
            result[f"token_{i}"] = {
                "logprob": -float(i) / n,
                "rank": i,
                "decoded": f"tok{i}",
                "top_logprobs": [
                    {"token": f"t{j}", "logprob": -float(j)/10}
                    for j in range(5)
                ],
            }
        return result

    num_tasks = 32
    task_size = 500

    start = time.perf_counter()
    _ = [python_cpu_work(task_size) for _ in range(num_tasks)]
    serial_py = time.perf_counter() - start
    print(f"  [serial] {num_tasks} tasks: {serial_py*1000:.1f}ms")

    for workers in [2, 4, 8, 16]:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            _ = list(pool.map(lambda _: python_cpu_work(task_size), range(num_tasks)))
        t = time.perf_counter() - start
        print(f"  [{workers}T]     {num_tasks} tasks: {t*1000:.1f}ms → Speedup: {serial_py/t:.2f}x")


if __name__ == "__main__":
    main()
