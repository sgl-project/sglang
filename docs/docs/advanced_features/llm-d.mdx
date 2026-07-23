---
title: "llm-d"
metatags:
    description: "Run SGLang at fleet scale with llm-d: prefix-aware routing, distributed KV-cache management, prefill/decode disaggregation, SLO-aware autoscaling on Kubernetes."
---

[llm-d](https://llm-d.ai/) is a Kubernetes-native distributed inference framework for serving large language models at scale across a fleet of inference servers. SGLang is a supported inference engine in llm-d: llm-d coordinates a fleet of SGLang instances across a cluster so that performance holds up under real production traffic, achieving the fastest "time to state-of-the-art (SOTA) performance" for key OSS models across most hardware accelerators.

llm-d is a [CNCF Sandbox project](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/) founded by Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA.

## What llm-d adds to an SGLang deployment

A single SGLang server is fast, and RadixAttention already maximizes cache reuse within each replica. But at scale the picture changes: across many replicas, cache locality breaks under round-robin load balancing as related requests scatter and radix-cache hit rates collapse, long prompts inflate time-to-first-token, and accelerators sit underused. llm-d adds the cluster-level layer that the engine does not aim to provide on its own:

- **[Prefix-aware routing](https://llm-d.ai/docs/guides/precise-prefix-cache-aware).** Instead of round-robin, the llm-d Router scores each replica on prefix-cache locality and current load, routing each request to the replica most likely to already hold its prefix — raising RadixAttention hit rates on multi-turn and shared-prefix workloads while avoiding saturated servers.
- **[Distributed KV-cache management](https://llm-d.ai/docs/guides#advanced-kv-cache-management).** A global index tracks which token blocks live on which replica, and tiered offloading spills cache to CPU memory or local SSD, extending the working set beyond accelerator HBM.
- **[Prefill/decode disaggregation](https://llm-d.ai/docs/guides/pd-disaggregation).** Prompt processing and token generation run on separate workers, with KV-cache moved over high-speed interconnects, lowering TTFT and steadying per-token latency on long prompts.
- **[SLO-aware autoscaling](https://llm-d.ai/docs/guides/workload-autoscaling) and [flow control](https://llm-d.ai/docs/guides/flow-control).** Scale SGLang pools on real inference signals (queue depth, true demand) rather than raw GPU utilization, with multi-tenant fairness and priority dispatch.
- **One control plane for mixed fleets.** llm-d schedules across engines, so platform teams can serve SGLang and vLLM pools behind the same gateway, policies, and observability instead of running parallel stacks.

These capabilities are composable. Most teams start by adding prefix-aware routing over an existing SGLang pool, then layer in the rest as specific bottlenecks appear.

## Kubernetes-native gateway

llm-d builds on the Gateway API Inference Extension, so SGLang pools are managed through standard Kubernetes resources (Gateway, HTTPRoute, InferencePool) and work with supported gateway providers such as Istio, GKE Inference Gateway, and agentgateway, rather than a bespoke routing tier.

## Performance

llm-d publishes reproducible benchmarks from production-scale deployments on [Prism](https://prism.llm-d.ai/). One representative result: prefix-aware routing delivered **3x higher output throughput and 2x faster TTFT** than round-robin load balancing (Llama 3.1 70B). The mechanism carries over directly to SGLang, where RadixAttention makes the cluster-level cache hit rate a function of routing.

## Get started

- **Deploy the optimized baseline with the [Quickstart](https://llm-d.ai/docs/getting-started/quickstart), selecting SGLang as the inference server.** It stands up an intelligent router (the llm-d Router) over an SGLang pool on Kubernetes in a tested configuration.
- Browse the [well-lit path guides](https://llm-d.ai/docs/guides), each a tested recipe for one of the capabilities above, and add the optimization that fits your workload.
- Read the [Introduction](https://llm-d.ai/docs/getting-started) and [Architecture overview](https://llm-d.ai/docs/architecture) to see how the scheduler, gateway, and model servers wrap your SGLang deployment.

Questions and contributions are welcome on [GitHub](https://github.com/llm-d/llm-d) and [Slack](https://llm-d.ai/slack).

## Current scope

SGLang is supported across the well-lit paths — including intelligent inference scheduling, precise prefix-cache routing (SGLang publishes KV-cache events that the llm-d Router subscribes to), tiered KV-cache management, prefill/decode disaggregation, flow control, and autoscaling. The one current exception is Multi-Node Wide Expert Parallelism, which is vLLM-specific today. See the llm-d documentation for the latest per-engine support status.
