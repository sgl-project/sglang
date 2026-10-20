# Programmable KV Cache 社区工作索引

核查日期：2026-10-20。日期列按 America/Phoenix（UTC−7）换算；状态来自 GitHub API 快照。已合入依据 `merged_at`，Closed 且无合入时间的 PR 标记为“关闭未合入”。

本索引覆盖本次报告引用及相邻实现，共 52 项；并非仓库中所有 KV Cache 工作。

[返回汇报正文](/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/docs/programmable-kvc/community-progress-report-2026-10-20.zh-CN.md)

## sgl-project/sglang

| 编号与标题 | 状态 | 合入日期 | 最近更新 |
| --- | --- | --- | --- |
| [#21846 \[Roadmap\]: SGLang Distributed KVCache System For Agentic Workload](https://github.com/sgl-project/sglang/issues/21846) | Open | — | 2026-09-27 |
| [#27574 \[Agentic Inference\] Programmatic KV Cache for Agentic Workloads](https://github.com/sgl-project/sglang/issues/27574) | Closed | — | 2026-10-12 |
| [#43098 Agentic Cache Scheduling System Roadmap (Q4)](https://github.com/sgl-project/sglang/issues/43098) | Open | — | 2026-10-19 |
| [#36224 \[RFC\] First-Class, Versioned KV Hint Envelope for SGLang](https://github.com/sgl-project/sglang/issues/36224) | Open | — | 2026-09-09 |
| [#36208 \[RFC\] Per-Token-Range KV-Cache Retention Priority](https://github.com/sgl-project/sglang/issues/36208) | Open | — | 2026-08-24 |
| [#24656 \[RFC\] Agent-Aware KV Cache Phase 1 for Agentic Workloads](https://github.com/sgl-project/sglang/issues/24656) | Open | — | 2026-09-21 |
| [#45128 \[RFC\] linker: Prefetch SSD-resident KV while the request is queued](https://github.com/sgl-project/sglang/issues/45128) | Open | — | 2026-10-19 |
| [#44721 \[RFC\] scheduler: Overlap prefill scheduling and KV loading with model execution](https://github.com/sgl-project/sglang/issues/44721) | Open | — | 2026-10-19 |
| [#32903 \[RFC\] KVCR as a HiCacheStorage backend for peer-to-peer KV reuse](https://github.com/sgl-project/sglang/issues/32903) | Open | — | 2026-08-24 |
| [#39991 \[RFC\] Align KV cache events with vLLM's schema so shared consumers have a standard format](https://github.com/sgl-project/sglang/issues/39991) | Open | — | 2026-10-05 |
| [#33394 \[RFC\]\[sgl-router\] Recoverable KV Placement State with Snapshots and Event Replay](https://github.com/sgl-project/sglang/issues/33394) | Closed | — | 2026-10-10 |
| [#44221 \[RFC\]\[Simulator\] Support Closed-Loop Agentic Workloads](https://github.com/sgl-project/sglang/issues/44221) | Open | — | 2026-10-19 |
| [#44635 \[RFC\] Context-parallel support for the unified L3 key scheme](https://github.com/sgl-project/sglang/issues/44635) | Open | — | 2026-10-19 |
| [#20535 \[HiCache\] Add L2 prefetch-buffer-only memory mode](https://github.com/sgl-project/sglang/pull/20535) | 关闭未合入 | — | 2026-10-12 |
| [#38891 feat: add kv hint envelope to request transport](https://github.com/sgl-project/sglang/pull/38891) | 已合入 | 2026-09-23 | 2026-09-23 |
| [#29436 feat: first-class session identity in SGLang](https://github.com/sgl-project/sglang/pull/29436) | 已合入 | 2026-06-28 | 2026-06-28 |
| [#29173 feat: Session-reference-aware Unified Radix Cache for agentic multi-turn workloads](https://github.com/sgl-project/sglang/pull/29173) | 已合入 | 2026-08-01 | 2026-08-01 |
| [#27970 \[mem_cache\] soft retention: priority decay via retention_seconds](https://github.com/sgl-project/sglang/pull/27970) | Open | — | 2026-08-06 |
| [#29199 Add experimental agent-aware KV cache hints](https://github.com/sgl-project/sglang/pull/29199) | 关闭未合入 | — | 2026-08-08 |
| [#30796 feat(hicache): retain hinted prompt KV in Mooncake L3](https://github.com/sgl-project/sglang/pull/30796) | 关闭未合入 | — | 2026-08-11 |
| [#32710 \[Radix Cache\] Add Rust TreeCore backend with shared parity tests](https://github.com/sgl-project/sglang/pull/32710) | 已合入 | 2026-08-31 | 2026-08-31 |
| [#37306 \[Rust TreeCore\] Support external cache linker](https://github.com/sgl-project/sglang/pull/37306) | 已合入 | 2026-09-10 | 2026-09-10 |
| [#37205 \[Unified Cache\]\[4/N\]: Add Mooncake backend for external linker](https://github.com/sgl-project/sglang/pull/37205) | 已合入 | 2026-08-31 | 2026-08-31 |
| [#37381 \[Unified Cache\]\[5/N\]: Integrate external linker mode end to end](https://github.com/sgl-project/sglang/pull/37381) | 已合入 | 2026-09-03 | 2026-09-03 |
| [#41343 \[Feat\] unified_cache: Support hybrid SWA + Mamba models in the Mooncake direct linker](https://github.com/sgl-project/sglang/pull/41343) | 已合入 | 2026-10-17 | 2026-10-17 |
| [#39727 \[HiCache\] Add the page_unified host KV layout and the unified L3 key](https://github.com/sgl-project/sglang/pull/39727) | 已合入 | 2026-10-15 | 2026-10-15 |
| [#39815 \[HiCache\] Add page unified layout for linker](https://github.com/sgl-project/sglang/pull/39815) | Open | — | 2026-10-15 |
| [#44723 \[Feat\] scheduler: Overlap prefill scheduling and KV loading with model execution](https://github.com/sgl-project/sglang/pull/44723) | Open / Draft | — | 2026-10-19 |
| [#36409 \[HiCache\] Add KVCR as an L3 storage backend for hint-driven cross-instance KV reuse](https://github.com/sgl-project/sglang/pull/36409) | Open | — | 2026-09-17 |
| [#37482 feat(agent sessions): attribute stored KV cache blocks to sessions](https://github.com/sgl-project/sglang/pull/37482) | 已合入 | 2026-09-14 | 2026-09-14 |
| [#33370 \[Feature\] Add process-local in-memory KV indexer and Router integration](https://github.com/sgl-project/sglang/pull/33370) | 已合入 | 2026-08-19 | 2026-08-19 |
| [#40601 \[KV Cache\] Align additive KV event metadata with shared consumers](https://github.com/sgl-project/sglang/pull/40601) | Open / Draft | — | 2026-10-04 |
| [#42455 \[KV Events\] Publish LoRA adapter names and namespaced block hashes](https://github.com/sgl-project/sglang/pull/42455) | 已合入 | 2026-10-11 | 2026-10-11 |
| [#42817 \[KV events\] Publish opaque LoRA block IDs and adapter names](https://github.com/sgl-project/sglang/pull/42817) | 已合入 | 2026-10-14 | 2026-10-14 |
| [#43386 \[kvevents\] fix missing topics in replayed kv events](https://github.com/sgl-project/sglang/pull/43386) | 已合入 | 2026-10-11 | 2026-10-11 |
| [#34407 \[sgl-router\] Worker Snapshot Support for Recoverable KV Placement State in sgl-router](https://github.com/sgl-project/sglang/pull/34407) | Open | — | 2026-09-13 |
| [#32729 \[Router\] Add KV event gap replay for sgl-router](https://github.com/sgl-project/sglang/pull/32729) | Open | — | 2026-07-28 |
| [#44611 \[Feat\] mem_cache: tag Mooncake linker offloads with the writing session](https://github.com/sgl-project/sglang/pull/44611) | Open | — | 2026-10-19 |
| [#40896 \[Unified Cache\] Abort requests whose external linker load fails instead of crashing the scheduler](https://github.com/sgl-project/sglang/pull/40896) | 已合入 | 2026-10-12 | 2026-10-12 |
| [#44526 \[Feat\] linker: Support runtime attach and detach of the unified cache external linker](https://github.com/sgl-project/sglang/pull/44526) | Open | — | 2026-10-19 |
| [#43831 \[Perf\] unified_cache: Skip write-through of pages the external linker just loaded](https://github.com/sgl-project/sglang/pull/43831) | Open | — | 2026-10-16 |
| [#44623 \[Fix\] cache: Support DeepSeek-V4.1 external linker pools](https://github.com/sgl-project/sglang/pull/44623) | 已合入 | 2026-10-19 | 2026-10-19 |
| [#44977 \[Feat\] unified_cache: Support FULL + MAMBA MLA models (Kimi-K3) in the direct external linker](https://github.com/sgl-project/sglang/pull/44977) | Open | — | 2026-10-19 |
| [#39726 \[HiCache\] Add the page-unified KV load-back JIT kernel ](https://github.com/sgl-project/sglang/pull/39726) | 已合入 | 2026-09-27 | 2026-09-27 |
| [#39728 \[Feat\] HiCache: Page unified e2e integration](https://github.com/sgl-project/sglang/pull/39728) | Open | — | 2026-10-19 |

## vllm-project/vllm

| 编号与标题 | 状态 | 合入日期 | 最近更新 |
| --- | --- | --- | --- |
| [#57103 \[RFC\]: Programmable KV Cache: Composable Policies for Agentic Serving](https://github.com/vllm-project/vllm/issues/57103) | Open | — | 2026-10-18 |
| [#59708 \[RFC\]: Modular and Extensible APIs & Execution Surface for Orchestrator-Driven KV Hint Policies](https://github.com/vllm-project/vllm/issues/59708) | Open | — | 2026-10-05 |
| [#62382 \[RFC\]\[KV Cache\] Implementation of programmable KV cache (#57103)](https://github.com/vllm-project/vllm/pull/62382) | Open / Draft | — | 2026-10-18 |
| [#59714 \[Feature\]\[Programmatic KV\] G1-only Retain via PRIORITY with KvHintEnvelope integration](https://github.com/vllm-project/vllm/pull/59714) | Open / Draft | — | 2026-10-06 |
| [#53423 \[Feature\] Add first-class KV hints request envelope for programmatic KV management](https://github.com/vllm-project/vllm/pull/53423) | 已合入 | 2026-09-21 | 2026-09-21 |
| [#58198 \[KV Offload\] Request-free CPU->GPU prefetch primitive](https://github.com/vllm-project/vllm/pull/58198) | Open / Draft | — | 2026-10-18 |
| [#38514 \[Core\] Priority-based KV-cache retention via kv.retain hints (#37003)](https://github.com/vllm-project/vllm/pull/38514) | Open | — | 2026-10-19 |

数据与正文保存在 [github-status.json](/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/docs/programmable-kvc/report-evidence-2026-10-20/github-status.json)。本地 SGLang 代码基线为 `f02458fff127d918cb573914e5e6d5dd8b6fb8d1`；公开条目更新可能领先本地代码，实施前需再次核对目标分支。
