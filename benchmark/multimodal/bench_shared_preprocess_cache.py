import argparse
import base64
import concurrent.futures
import io
import json
import statistics
import time
import urllib.request

from PIL import Image, ImageDraw

ROOT = "http://127.0.0.1:30000"


def call(base_url, payload):
    started = time.perf_counter()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    parts = []
    first = None
    with urllib.request.urlopen(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            for choice in data.get("choices", []):
                text = choice.get("delta", {}).get("content", "")
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    parts.append(text)
    text = "".join(parts)
    assert text, "empty model output"
    return dict(ttft=first, total=time.perf_counter() - started, text=text)


def benchmark_large_batch(base_url, model, label):
    images = []
    shapes = ((1920, 1440), (2000, 1501), (2000, 1501), (2000, 1501), (1920, 1440))
    colors = ("red", "blue", "green", "yellow", "purple")
    for index, (height, width) in enumerate(shapes):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (width // 5, height // 5, 4 * width // 5, 4 * height // 5),
            fill=colors[index],
        )
        draw.text((25, 25), str(index), fill="black")
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        images.append(stream.getvalue())
    images.append(images[0])
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(data).decode()
            },
        }
        for data in images
    ]
    content.append(
        {
            "type": "text",
            "text": "What color is the rectangle in the first image? Answer with one word.",
        }
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 12,
        "stream": True,
        "seed": 42,
    }
    cold = call(base_url, payload)
    warm = [call(base_url, payload) for _ in range(5)]
    assert len({result["text"] for result in [cold, *warm]}) == 1
    print(
        json.dumps(
            {
                "label": label,
                "images": 6,
                "unique_images": 5,
                "cold": cold,
                "warm": warm,
                "median_ttft": statistics.median(result["ttft"] for result in warm),
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=ROOT)
    ap.add_argument(
        "--large-batch",
        action="store_true",
        help="Also benchmark six large images with one duplicate",
    )
    args = ap.parse_args()
    base_url = args.base_url.rstrip("/")
    payloads = []
    for dimensions in ((448, 448), (1440, 1920)):
        image = Image.new("RGB", dimensions, "white")
        draw = ImageDraw.Draw(image)
        w, h = dimensions
        draw.rectangle((w // 5, h // 5, 4 * w // 5, 4 * h // 5), fill="red")
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {
                            "type": "text",
                            "text": "What color is the rectangle? Answer with one word.",
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 12,
            "stream": True,
            "seed": 42,
        }
        # Warm kernels and the GPU prefix first. Measure hot requests separately.
        cold = call(base_url, payload)
        warm = [call(base_url, payload) for _ in range(7)]
        assert len({x["text"] for x in warm}) == 1, "non-deterministic repeated output"
        payloads.append(payload)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "size": dimensions,
                    "cold": cold,
                    "warm": warm,
                    "median_ttft": statistics.median(x["ttft"] for x in warm),
                }
            ),
            flush=True,
        )
    if args.large_batch:
        benchmark_large_batch(base_url, args.model, args.label)
    # Verify multi-worker concurrency and duplicate images do not corrupt outputs.
    payload = payloads[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results = list(pool.map(lambda _: call(base_url, payload), range(8)))
    assert len({x["text"] for x in concurrent_results}) == 1
    print(
        json.dumps(
            {
                "label": args.label,
                "concurrent_requests": 8,
                "concurrency": 4,
                "results": concurrent_results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
