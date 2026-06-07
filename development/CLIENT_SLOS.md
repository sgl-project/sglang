Here are the immediate client requirements:
- Model: zai-org/GLM-5.1 (FP8)
- Inference SLOs: 30 TPS (Total Latency - TTFT / total tokens) with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, minimum concurrency: 16, Cache hit: ~55% (benchmark found in development/benchmark.sh) 
- Support for key performant knobs, like TP, cuda graphs, radix cache

Deferred Client requirements ordered from most important to least:
1. 128k ISL, 1024 OSL.
2. nvfp4 and mxfp4 quantizated weight support.
3. other performant knobs like DP Attention, MTP (eagle), EP, chunked prefill, mixed chunked prefill, overlap scheduling, piecewise cuda graph. 

Downstream requirements after client deliverables:
1. Twilight (top-p selection instead of top-k)
2. Extensions as a general knob for the sglang engine
3. Integration into all other sglang features, like PD-Disagg