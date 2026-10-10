# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-noise native requests with a concurrent segmented HTTP batch."""

import argparse
import copy
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def send(url, payload, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/actions/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--segmented-url", required=True)
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="A valid action-generation JSON payload with fixed observation.noise",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 5, 4, 8])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--atol",
        type=float,
        required=True,
        help="Backend-appropriate action error tolerance",
    )
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    if not args.steps or any(n <= 0 for n in args.steps):
        parser.error("--steps must be positive")
    payload = json.loads(args.request.read_text())
    if "noise" not in payload.get("input", {}).get("observation", {}):
        parser.error(
            "Supply fixed input.observation.noise to compare identical trajectories"
        )
    requests = []
    for i, n in enumerate(args.steps):
        req = copy.deepcopy(payload)
        req["request_id"] = f"pi05-segment-check-{i}"
        req.setdefault("parameters", {})["num_inference_steps"] = n
        req.setdefault("runtime", {}).update(
            return_timing=True, cuda_graph=False, response_format="envelope"
        )
        requests.append(req)
    native = [send(args.native_url, req, args.timeout) for req in requests]
    barrier = threading.Barrier(len(requests))

    def concurrent_send(req):
        barrier.wait(timeout=args.timeout)
        return send(args.segmented_url, req, args.timeout)

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        segmented = list(pool.map(concurrent_send, requests))
    resumed = False
    for steps, ref, result in zip(args.steps, native, segmented, strict=True):
        expected = np.asarray(ref["data"][0]["action"]["values"])
        actual = np.asarray(result["data"][0]["action"]["values"])
        np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol)
        assert result["usage"]["denoise_steps"] == steps
        assert result["timings"]["actual_nfe"] == steps
        count = result["timings"]["segment_count"]
        resumed |= count > 1
        print(
            json.dumps(
                {
                    "steps": steps,
                    "segments": count,
                    "max_abs_error": float(np.max(np.abs(actual - expected))),
                }
            )
        )
    if not resumed:
        raise RuntimeError(
            "No continuation observed; increase batching delay/concurrency and rerun"
        )


if __name__ == "__main__":
    main()
