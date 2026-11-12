# PPU nightly baseline skip policy

`nightly_baseline_skips.json` is the exact executable inventory. Every entry is a full JUnit `classname.method`; no prefix or wildcard matching is used. A skip not present in the inventory fails closed.

## Approved categories

| Test area | Reason for the baseline skip |
| --- | --- |
| RoPE `test_partial_rope` | `head_dim=64` is smaller than `rope_dim` 80, 96, or 128, so those parameter combinations are invalid. The 144 entries use the actual JUnit names, including the duplicate-value IDs `20` and `21`. |
| Timestep embedding performance test | Performance measurement is disabled by default; the correctness cases still execute. |
| Mamba stochastic rounding | Requires Blackwell SM100-family hardware and CUDA 12.8 or newer. |
| ModelOpt export | `nvidia-modelopt` is not installed in the PPU image. |
| Virtual experts sentinel base class | Pytest collects the abstract `_AlignBlockSizeSentinelBucketBase` class, but it is not an independent test class. |
| Nsight profiling | `nsys` is not available in the PPU image. |
| HiSparse page-size-one cases | These two paths are ROCm-specific. |
| Negative-penalty diversity | Upstream explicitly disables this assertion as flaky. |
| Rust OpenAI server cases | The embedded Rust server extension is not built for this PPU nightly lane. |
| Radix LPM | The existing test marks this class as skipped to reduce CI execution time. |
| Unified radix configurations | Skip only the configuration-specific cases declared by the existing test implementation, such as Full+Mamba, SWA-only, page-size, or sliding-window requirements. Every skipped class/method pair is enumerated from the real JUnit report. |
| NUMA topology | The two cases require 4-GPU Grace Blackwell or B200 hardware, not PPU ZW-810E. |

The inventory was derived from actual JUnit evidence, not inferred from decorators. Validation compares the observed skip set with the inventory in both directions and fails on any difference.

## Legacy deselection is separate

Two bare functions in `test_update_weights_from_tensor.py` and `test_update_weights_from_distributed.py` were never executed by the legacy unittest runner. They are handled by the CI-only exact-deselection policy, not by this skip inventory. PR evidence verification rejects any nonempty `deselect` field, so a required PR test cannot be hidden through deselection.
