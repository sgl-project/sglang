Here are the immediate client requirements:
- Model: deepseek-ai/DeepSeek-V3.2 (FP8)
- Inference SLOs: 30 TPS per request with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, minimum concurrency: 16, Cache hit: ~55% (benchmark found in development/benchmark.sh) 
- Page size: 64 (technically not explicitly listed as a hard requirement, but significantly preferred and implementation should support different page sizes)
- Support for key performant knobs, like TP, cuda graphs, radix cache

Deferred Client requirements ordered from most important to least:
1. zai-org/GLM-5.1
2. 128k ISL, 1024 OSL.
3. nvfp4 and mxfp4 quantizated weight support.
4. other performant knobs like DP Attention, MTP (eagle), EP, chunked prefill, mixed chunked prefill, overlap scheduling, piecewise cuda graph. 

Downstream requirements after client deliverables:
1. Twilight (top-p selection instead of top-k)
2. Extensions as a general knob for the sglang engine
3. Integration into all other sglang features, like PD-Disagg