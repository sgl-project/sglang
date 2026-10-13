# CUDA TileLang raw FP8 KV validation

Use both `--dsa-prefill-backend tilelang` and `--dsa-decode-backend tilelang`
with `--kv-cache-dtype fp8_e4m3`. CUDA FP8 Tensor Cores (SM89+) are required.
The raw layout has no per-block scales; it is distinct from the scaled,
packed layout consumed by other CUDA DSA backends. Mixed backends, DCP,
attention CP and HiSparse are unsupported for this CUDA path.

CUDA routing is already supported by [PR #42957](https://github.com/sgl-project/sglang/pull/42957).
The earlier raw-layout approach was described in
[PR #36904](https://github.com/sgl-project/sglang/pull/36904).
The additional checks reject packed rows and ROCm's E4M3FNUZ format on CUDA.
The paged writer fuses the cast to raw FP8 with the write, including NoPE-only
rows where the RoPE input is absent.

## Component tests

```bash
python -m pytest -v \
  test/registered/unit/test_dsa_tilelang_fp8_validation.py \
  test/registered/kernels/ops/attention/test_tilelang_dsa_fp8_cuda.py \
  test/registered/mem_cache/test_tilelang_raw_fp8_pool.py
```

Kernel references use the same quantized inputs as the kernel, isolating
kernel correctness from quantization error. Coverage includes NoPE/RoPE,
8/16/32 heads, padded sparse indices, exact one-hot reads, all-masked rows,
an FP32 attention reference and a wrong-index negative control. Pool tests
check page-boundary writes, untouched rows and exported buffer byte lengths.
These component checks do not establish end-to-end PD transfer correctness.

## Sampled model accuracy

The following is a real evaluation performed on October 13, 2026 using
GLM-5.3-Flash-FP8, raw `fp8_e4m3` KV, TileLang prefill/decode and 8 H20-96GB
GPUs. The evaluated runtime was base commit
`7760d5264ac16b85392aefb1bf5c13ce38d9632e` with validation patch
`f9c6b44d964504c6ab9e0088c090aa8a10a5ff5e`. These model-level results have
not been rerun on the latest upstream base of this PR.

| Dataset / metric | Correct / scored | Score | Input TPS | Input TPS/GPU |
| --- | ---: | ---: | --- | --- |
| IFEval strict prompt | 35/40 | 87.50% | N/A | N/A |
| C-Eval validation | 50/52 | 96.15% | N/A | N/A |
| GSM8K | 48/50 | 96.00% | N/A | N/A |
| LiveCodeBench release_v6 pass@1 | 16/18 | 88.89% | N/A | N/A |

TPS fields are unavailable because this was an accuracy evaluation, not a
serving throughput measurement. Output TPS and total TPS are also unavailable.
All 160 selected samples remain in the denominators. IFEval loose prompt was
36/40 (90.00%), strict instruction 56/61 (91.80%), and loose instruction
57/61 (93.44%). LiveCodeBench was easy 6/6, medium 6/6 and hard 4/6.

Sampling used seed 42 without replacement: IFEval 40/541 proportionally
stratified by instruction count (0-shot), C-Eval 52/1346 with one sample per
subject (5-shot), GSM8K 50/1319 (4-shot), and LiveCodeBench 18/1055 with six
per difficulty (0-shot). The equal-difficulty LiveCodeBench sample is not a
population-weighted full-dataset score.

EvalScope 1.12.0 scored the model's native chat template and default thinking
mode. Generation used temperature 0, top_p 1, seed 42, one candidate,
non-streaming requests and concurrency 8. The output budget, including
reasoning, was 32768 tokens for IFEval/C-Eval/GSM8K and 65536 for
LiveCodeBench. The code sandbox had no network, host mounts or GPU, and used
two CPUs, 4 GiB memory and a six-second limit per code test.

Two samples exhausted the output budget without a final answer: C-Eval
`law` source index 6 and LiveCodeBench source index 965. Both count as
incorrect; neither was excluded or regenerated. The other failed
LiveCodeBench sample, source index 46, raised `IndexError` in generated
code. No API/transport failures occurred.

The initial sampled loader exposed an evaluator cache-ID mismatch. After
repair, dense evaluator IDs were mapped back to source IDs and complete
request hashes were audited against predictions and reviews. Two duplicate
historical requests remain in the request count (162 actual requests for
160 unique scored samples). Reused responses were selected by complete
request identity and first successful return, independently of correctness;
the invalid initial scoring output was not used.

There was no paired BF16 KV accuracy control. These sampled scores do not
measure an accuracy delta caused by FP8 KV or certify full-dataset accuracy.
Selected source IDs, sample hashes, exact scores and runtime metadata are in
[the machine-readable summary](glm53_tilelang_fp8_accuracy_sample.json).

## Evaluated serving configuration

PyTorch 2.14.1+cu130, CUDA 13.0, TileLang 0.1.12 and Transformers 5.19.0;
FP8 weights, BF16 activations and FP32 SSM state. This was a single-host
deployment, not PD disaggregation. The resolved chunk size was 8192;
speculative `NEXTN` resolved to `EAGLE` with no adaptive speculation.

```bash
sglang serve --model-path "$MODEL_PATH" \
  --tp-size 8 --ep-size 4 \
  --mem-fraction-static 0.72 --max-running-requests 32 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend deep_gemm \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --cuda-graph-max-bs-decode 32 --max-mamba-cache-size 384 \
  --enable-hierarchical-cache --hicache-host-memory-mode cache \
  --hicache-ratio 1 --hicache-write-policy write_back \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --host 0.0.0.0 --port 31000 --enable-metrics --enable-cache-report
```

`MODEL_PATH` points to the original GLM-5.3-Flash-FP8 weights. The command
records the evaluated runtime; it is not an assertion that every option is
unchanged on future releases. No throughput improvement is implied by these
accuracy or component tests.
