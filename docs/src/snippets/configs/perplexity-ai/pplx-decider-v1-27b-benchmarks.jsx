// PPLX-Decider-v1-27B per-cell benchmark numbers. See _deployment.jsx for the
// speed/accuracy schema.
//
// A decision is prefill-only: the answer is read from the logits at the end of
// the prompt, so TTFT is the whole request latency, there is no TPOT (and no
// derived interactivity), and throughput per GPU is prompt tokens/s per GPU
// (osl = 0). TTFT is the P50 of end-to-end /v1/systemone request time.
//
// GB300: one GPU, the cell's recipe (--disable-radix-cache), SGLang main @
// 70f0b7351e, checkpoint revision 5117a6c. Closed-loop client against
// /v1/systemone: every request carries a unique random state (only the system
// message is shared), 8 warmup requests, cache flushed before each level. isl
// is the mean prompt length per request: a 256-token state with one 4-option
// choice question (382), or an 8,192-token state with the same question
// (8,224). Section 3 of the page has the full tables and the four-question and image
// shapes.
//
// Accuracy through /v1/systemone with the page's own conversion: Belebele
// eng_Latn test (900) and WinoGrande xl validation (1,267). The checkpoint's
// reference DecisionModel scores 96.67 / 84.37 on the same items.
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "main @ 70f0b7351e",
    speed: [
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 51.4, tpot_ms: null, tokens_per_sec_per_gpu: 7413 },
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 881.9, tpot_ms: null, tokens_per_sec_per_gpu: 27039 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 324.7, tpot_ms: null, tokens_per_sec_per_gpu: 25297 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 4796.2, tpot_ms: null, tokens_per_sec_per_gpu: 27175 },
    ],
    accuracy: { belebele_pct: 96.67, winogrande_pct: 84.21 },
    notes: "Prefill-only model: TTFT is the full decision latency and throughput is prompt tokens/s/GPU. TPOT and interactivity do not apply.",
  },
]
