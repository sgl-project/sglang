// One entry per config cell `match` tuple. Left as bare-match stubs — the benchmark
// card shows "pending" until an entry carries a non-null speed metric.
//
// k3-track HAS measured 8k1k serving numbers (H200 commit 0dc8b030d; B300 commit
// d70f59487) — fill them per cell with the reproducible sglang_version once the K3
// build is pinned. Speed shape:
//   speed: [{ workload: {dataset, isl, osl, max_concurrency}, ttft_ms, tpot_ms,
//             tokens_per_sec_per_gpu }, ...]   // ttft/tpot = P50; tps = total(in+out)/GPU

export const benchmarks = [
  { match: { hw: "h200",  variant: "default", quant: "mxfp4", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "mxfp4", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "b300",  variant: "default", quant: "mxfp4", strategy: "low-latency",     nodes: "single"  } },
  { match: { hw: "b300",  variant: "default", quant: "mxfp4", strategy: "balanced",        nodes: "single"  } },
  { match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "multi-2" } },
];
