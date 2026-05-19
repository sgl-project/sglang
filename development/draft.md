I am in the middle of delivering a double sparsity implementation into SGLang.

Here are the immediate client requirements:
- Model: deepseek-ai/DeepSeek-V3.2 (FP8)
- Inference SLOs: 30 tokens/s with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, minimum concurrency: 16, Cache hit: ~55% (benchmark found in development/benchmark.sh) 
- Page size: 64 (technically not explicitly listed as a hard requirement, but significantly preferred and implementation should support different page sizes)

Deferred Client requirements ordered from most important to least:
1. zai-org/GLM-5.
2. 128k ISL, 1024 OSL.
3. nvfp4 and mxfp4 quantizated weight support.

Downstream requirements after client deliverables:
1. Twilight (top-p selection instead of top-k)
2. Extensions as a general knob for the sglang engine
3. Integration into all other sglang features, like PD-Disagg and HiSparse.

Double Sparsity Implementation Sources (listed from most recent to least recent)
1. Twilight: https://github.com/tsinghua-ideal/Twilight, with https://github.com/tsinghua-ideal/flash-topk-attention/tree/d8803b29961c44d77a747636ad4282bd7a9094af
2. Legacy SGL implementation (parent commit of commit that removed double sparsity): https://github.com/sgl-project/sglang/tree/29f56cb2304bf6699da78e4e5a738fb794babcfd/python/sglang/srt/layers/attention 
3. Original Author Implementation: https://github.com/andy-yang-1/DoubleSparse
4. Paper: https://arxiv.org/pdf/2408.07092

Hi-Sparse Implementation (Quite irrelevant, but an example of succesfull ship of performant sglang sparsity feature, design should ideally be inspired by this)
Guide: https://docs.sglang.io/docs/advanced_features/hisparse_guide
PRS: https://github.com/sgl-project/sglang/pull/20343, https://github.com/sgl-project/sglang/pull/23013, https://github.com/sgl-project/sglang/pull/21591


I am deciding between whether I should
1. Resume from the legacy sglang restoration: https://github.com/sgl-project/sglang/pull/22992
2. Resume from the current rewrite session: https://github.com/sgl-project/sglang/pull/25304/commits
3. Restart from scratch on a new branch of SGLang as this will be a huge downstream decision. I am leaning towards this as neither of the previous options were designed/created before the clients gave us these requirements

Help me first decide whether I should resume or restart from scratch on Sglang.