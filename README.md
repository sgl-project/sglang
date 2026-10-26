# Week 44

Base image: `lmsysorg/sglang:v0.5.22-rocm10-mi35x`

Branch `collector_26_10_qwen_3.8`, cut from SGLang main at `e1c4b273af` (26 Oct 2026). AITER is `ROCm/aiter` main at `9059673612`.

Pull requests already merged into SGLang main are in this collector because the branch starts from that main. They were not applied again.

## SGLang

| Pull request | Change | In this collector | In stock v0.5.22 | On SGLang main |
| --- | --- | --- | --- | --- |
| [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP | Yes, added on this branch | No | No, still open |
| [#40708](https://github.com/sgl-project/sglang/pull/40708) | 512-expert softmax MoE through AITER topk_gating | Yes, added on this branch | No | No, still open |
| [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce | Yes, added on this branch | No | No, still open |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quantize the Quark MXFP4 shared expert while loading | Yes, through the main base | No | Yes, merged 26 Oct 2026 |
| [#43516](https://github.com/sgl-project/sglang/pull/43516) | Qwen3.8 decode layer as two FlyDSL launches | Yes, added on this branch | No | No, still open |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | Quantize the BF16 GDN projections to MXFP4 online | Yes, added on this branch | No | No, still open |
| [`flydsl_reduce_int4` @ `8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | FlyDSL INT4 QuickReduce for eligible bf16 payloads | Yes, added on this branch | No | No |

## AITER

The collector reinstalls AITER main `9059673612`. The stock v0.5.22 image pins AITER `e7d2453` (25 Sep 2026). Every pull request below is on that AITER main, so it is in this collector through the checkout.

| Pull request | Change | In stock v0.5.22 | On AITER main |
| --- | --- | --- | --- |
| [#4383](https://github.com/ROCm/aiter/pull/4383) | Gluon MXFP4 and MXFP8 quant on gfx950 | No | Yes, merged 5 Oct 2026 |
| [#5334](https://github.com/ROCm/aiter/pull/5334) | E=512 softmax topk_gating prefill dispatch | Yes | Yes, merged 20 Sep 2026 |
| [#5835](https://github.com/ROCm/aiter/pull/5835) | Tuned bf16 GEMM configs for Qwen3.8 on gfx950 | No | Yes, merged 26 Sep 2026 |
| [#5998](https://github.com/ROCm/aiter/pull/5998) | Preshuffled AFP4WFP4 GEMM config, N=4608 K=8192 | No | Yes, merged 1 Oct 2026 |
| [#6145](https://github.com/ROCm/aiter/pull/6145) | Row-group FP4 paged MQA logits | No | Yes, merged 6 Oct 2026 |
| [#6398](https://github.com/ROCm/aiter/pull/6398) | Bind load_hip_runtime to the HIP library already loaded | No | Yes, merged 22 Oct 2026 |
