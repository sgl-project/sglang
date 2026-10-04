import json
import math
import statistics
import unittest

import torch
from sgl_kernel import rotary_embedding

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-a-test-1-gpu-small-amd")


class TestRopeCacheProbe(CustomTestCase):
    def test_cold_l2(self):
        torch.manual_seed(42494)
        eviction = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        angles = torch.randn(4096, 64, device="cuda")
        fp32_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        rows = []
        for dtype in (torch.float16, torch.bfloat16):
            typed_cache = fp32_cache.to(dtype)
            for neox in (True, False):
                for tokens in (1, 4, 32, 128, 512, 2048):
                    positions = torch.randint(4096, (tokens,), device="cuda")
                    q = torch.randn(tokens, 32 * 128, dtype=dtype, device="cuda")
                    k = torch.randn(tokens, 8 * 128, dtype=dtype, device="cuda")
                    buffers = [(q.clone(), k.clone()), (q.clone(), k.clone())]
                    caches = [typed_cache, fp32_cache]

                    def run(index):
                        query, key = buffers[index]
                        rotary_embedding(positions, query, key, 128, caches[index], neox)

                    run(0)
                    run(1)
                    torch.testing.assert_close(buffers[0][0], buffers[1][0], atol=0, rtol=0)
                    torch.testing.assert_close(buffers[0][1], buffers[1][1], atol=0, rtol=0)
                    # Use identical Q/K addresses for both timed paths.
                    buffers = [(q, k), (q, k)]
                    for _ in range(10):
                        run(0)
                        run(1)
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    samples = [[], []]
                    for sample in range(1500):
                        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                            # Read/write 512 MiB on this GPU before every timed invocation.
                            eviction.add_(1)
                            start.record()
                            run(index)
                            end.record()
                            end.synchronize()
                            samples[index].append(start.elapsed_time(end) * 1000)
                    medians = [statistics.median(s) for s in samples]
                    row = {
                        "dtype": str(dtype),
                        "neox": neox,
                        "tokens": tokens,
                        "typed_cache_us": medians[0],
                        "fp32_cache_us": medians[1],
                        "ratio": medians[1] / medians[0],
                        "bitwise_equal": True,
                        "round_ratios": [statistics.median(samples[1][i:i+500]) / statistics.median(samples[0][i:i+500]) for i in (0, 500, 1000)],
                    }
                    rows.append(row)
                    print("ROPE_CACHE_CASE " + json.dumps(row), flush=True)
        print(
            "ROPE_CACHE_SUMMARY "
            + json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cases": len(rows),
                    "samples_per_path": 1500,
                    "eviction_bytes": eviction.numel(),
                    "l2_scope": "512 MiB read/write before each invocation, outside event timing; one GPU",
                    "comparison": "FP32 cache vs query-dtype cache in the same candidate kernel, identical Q/K addresses; 3 consecutive rounds of 500 interleaved samples",
                    "geomean_ratio": math.exp(statistics.mean(math.log(r["ratio"]) for r in rows)),
                    "min_ratio": min(r["ratio"] for r in rows),
                    "max_ratio": max(r["ratio"] for r in rows),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
