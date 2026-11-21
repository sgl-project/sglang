## Summary

This PR documents the deployment and tuning results of Qwen3.8-27B on Ascend 910B4 NPUs.

## Changes

- Added BF16 and W8A8 deployment configurations.
- Documented the effective concurrency limitation caused by Mamba state cache and KV pool capacity.
- Added a 128k input / 1k output / 90% shared-prefix benchmark procedure.
- Added gsm8k accuracy results.
- Added Mean TTFT, Mean TPOT, P95 TPOT, and output throughput results.
- Added tuning recommendations for attention DP, chunked prefill, Mamba cache size, and decode graph settings.

## Results

| Metric | BF16 | W8A8 |
|---|---:|---:|
| gsm8k accuracy | 98.5% | 98.0% |
| Mean TTFT | 18.969 s | 15.788 s |
| Mean TPOT | 41.927 ms | 44.208 ms |
| P95 TPOT | 60.668 ms | 73.176 ms |
| Output throughput | 127.470 token/s | 131.798 token/s |
| Cache hit rate | 90.35% | 90.35% |

Both configurations satisfy the Mean TPOT <= 50 ms target. P95 TPOT remains above 50 ms, so the result is described as mean-latency compliant.

## Validation

- gsm8k: 200 requests
- Long-context workload: 128k input / 1k output
- Shared prefix: approximately 90%
- Client concurrency: 8
- Successful requests: 16/16
- `mint broken-links`: passed

Related issue: #<Issue 50461>
