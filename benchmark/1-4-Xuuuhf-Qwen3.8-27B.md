# Qwen3.8-27B MeetUp results on Ascend

Xuuuhf · 2026-11-14 · MeetUp task book items 1–4 (deployment, GSM8K and performance) and 5 (comparison).

Both models ran on four Ascend 910B3 NPUs (64 GB each) with SGLang `0.0.0.dev22121+g0fa9ff55f.d20261112`. GSM8K accuracy uses the first 200 test questions under the same EvalScope settings.

The native SGLang benchmark sent 20 requests at concurrency 20. Each request had 128,000 input tokens, reused 115,200 cached tokens (90%), and generated 1,000 tokens. Each row is the highest-output-throughput run observed for that model among completed runs with 90% cache reuse and mean TPOT ≤ 50 ms.

| Model | GSM8K accuracy | Mean TPOT (ms) | P99 TPOT (ms) | Mean TTFT (s) | Output tokens/s | Requests with TPOT ≤ 50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B BF16 | 199/200 (99.5%) | 36.86 | 66.06 | 39.44 | 250.12 | 16/20 |
| Qwen3.8-27B W8A8 | 198/200 (99.0%) | 38.30 | 64.50 | 33.29 | 267.18 | 14/20 |

Both runs used TP=4, attention DP=2, a 131,072-token context, a 16,384-token prefill chunk (8,192 per DP rank), and a limit of 20 running requests. NEXTN used three steps and four draft tokens. The W8A8 run used ModelSlim quantization. Both DP ranks received the shared prefix before timing.

Across these two runs, W8A8 delivered 6.82% more output tokens per second and answered one fewer GSM8K question correctly.
