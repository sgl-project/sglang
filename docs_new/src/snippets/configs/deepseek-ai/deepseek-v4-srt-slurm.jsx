// Official NVIDIA SRT Slurm recipes curated for the DeepSeek-V4 cookbook.
// Keep this file as a single plain-data literal so Mintlify can hydrate it.

export const srtSlurmConfig = {
  repositoryUrl: "https://github.com/NVIDIA/srt-slurm-recipes",
  branch: "main",
  recipesPath: "../srt-slurm-recipes",

  model: { id: "pro", label: "DeepSeek-V4-Pro", subtitle: "1.6T" },
  hardware: { id: "gb300", label: "GB300", subtitle: "288GB · NVIDIA", gpusPerNode: 4 },

  workloads: [
    { id: "1k1k", label: "1K / 1K", subtitle: "input / output tokens" },
    { id: "8k1k", label: "8K / 1K", subtitle: "input / output tokens" },
  ],
  strategies: [
    { id: "low-latency", label: "Low-Latency", subtitle: "fastest response" },
    { id: "balanced", label: "Balanced", subtitle: "mid-curve" },
    { id: "high-throughput", label: "High-Throughput", subtitle: "maximum scale" },
  ],

  recipes: [
    {
      workload: "1k1k",
      strategy: "low-latency",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/1k1k/sglang/disagg/stp/disagg-1p1d-tp4-mxfp4.yaml",
      prefillNodes: 1,
      decodeNodes: 1,
      description: "TP4 prefill / TP4 decode",
    },
    {
      workload: "1k1k",
      strategy: "balanced",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/1k1k/sglang/disagg/stp/disagg-1p1d-dep4-mega-moe.yaml",
      prefillNodes: 1,
      decodeNodes: 1,
      description: "DEP4 with MegaMoE",
    },
    {
      workload: "1k1k",
      strategy: "high-throughput",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/1k1k/sglang/disagg/stp/disagg-1p2d-dep4-to-dep8-mega-moe.yaml",
      prefillNodes: 1,
      decodeNodes: 2,
      description: "DEP4 prefill / DEP8 decode with MegaMoE",
    },
    {
      workload: "8k1k",
      strategy: "low-latency",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/8k1k/sglang/disagg/disagg-low-latency-1p1d-tp4-tp4-mtp.yaml",
      prefillNodes: 1,
      decodeNodes: 1,
      description: "TP4 prefill / TP4 decode with MTP",
    },
    {
      workload: "8k1k",
      strategy: "balanced",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/8k1k/sglang/disagg/disagg-mid-curve-1p1d-dep4-dep8-mtp.yaml",
      prefillNodes: 1,
      decodeNodes: 2,
      description: "DEP4 prefill / DEP8 decode with MTP",
    },
    {
      workload: "8k1k",
      strategy: "high-throughput",
      path: "recipes/multi-node/DeepSeek-V4-Pro/GB300/8k1k/sglang/disagg/disagg-high-conc-8p1d-dep4-dep8-mtp.yaml",
      prefillNodes: 8,
      decodeNodes: 2,
      description: "high-concurrency DEP4 / DEP8 with MTP",
    },
  ],
};
