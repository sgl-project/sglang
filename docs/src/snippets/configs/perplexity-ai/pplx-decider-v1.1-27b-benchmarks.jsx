// PPLX-Decider-v1.1-27B per-cell benchmark numbers. See _deployment.jsx for the
// speed/accuracy schema.
//
// A decision is prefill-only: the answer is read from the logits at the end of
// the prompt, so TTFT is the whole request latency, there is no TPOT (and no
// derived interactivity), and throughput per GPU is prompt tokens/s per GPU
// (osl = 0). TTFT is the P50 of end-to-end /v1/systemone request time.
//
// GB300: one GPU, the cell's recipe (no extra flag, so the server resolves
// --disable-radix-cache, --chunked-prefill-size -1 and trtllm_mha itself),
// PR #42645, checkpoint revision 3b45dea. Same closed-loop client and request
// shapes as the v1 benchmarks file: a 256-token state with one 4-option choice
// question (382 tokens) and an 8,192-token state with the same question
// (8,224 tokens). Section 3 of the page has the full tables.
//
// Accuracy through /v1/systemone with the v1 page's conversion: Belebele
// eng_Latn test (900) and WinoGrande xl validation (1,267). The checkpoint's
// reference DecisionModel scores 97.67 / 92.50 on the same items.
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #42645",
    speed: [
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 52.6, tpot_ms: null, tokens_per_sec_per_gpu: 7203 },
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 931.4, tpot_ms: null, tokens_per_sec_per_gpu: 25770 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 333.7, tpot_ms: null, tokens_per_sec_per_gpu: 24620 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 5100.9, tpot_ms: null, tokens_per_sec_per_gpu: 25726 },
    ],
    accuracy: { belebele_pct: 97.67, winogrande_pct: 92.42 },
    notes: "Prefill-only model: TTFT is the full decision latency and throughput is prompt tokens/s/GPU. TPOT and interactivity do not apply.",
  },
]
