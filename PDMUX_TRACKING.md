# PDMux model integration tracker

This is a working record for upstreaming PDMux support for GLM-5.3-Flash and
DeepSeek-V4.1-Flash. The accompanying draft PR tracks the work; this document
does not change runtime behavior or establish a new support claim.

## Current work

| Model | Current work | Next update |
| --- | --- | --- |
| GLM-5.3-Flash | Five stacked branches cover model forward, Mamba/HiCache state, PDMux scheduling, DP/EP alignment, and capped-prefill/full-device-decode overlap. | Link each focused PR as it opens; record GPU correctness and performance results. |
| DeepSeek-V4.1-Flash | [Integration PR #41562](https://github.com/sgl-project/sglang/pull/41562) currently contains standard prefill, DP attention, and DSpark together. | Keep the integration PR as the reference while focused PRs are prepared later; link them here when available. |

The GLM branch stack, in dependency order:

1. [`feat/glm53-pdmux-model`](https://github.com/Li-brua/sglang/tree/feat/glm53-pdmux-model) — layerwise model forward.
2. [`fix/glm53-pdmux-mamba-hicache`](https://github.com/Li-brua/sglang/tree/fix/glm53-pdmux-mamba-hicache) — Mamba and HiCache state handling.
3. [`feat/glm53-pdmux-core`](https://github.com/Li-brua/sglang/tree/feat/glm53-pdmux-core) — prefill/decode scheduling.
4. [`feat/glm53-pdmux-parallel`](https://github.com/Li-brua/sglang/tree/feat/glm53-pdmux-parallel) — DP/EP rank alignment and communication groups.
5. [`feat/glm53-pdmux-overlap`](https://github.com/Li-brua/sglang/tree/feat/glm53-pdmux-overlap) — overlapped SM reservations. This is the complete GLM integration branch.

## Progress

- [ ] Open and link the focused GLM PRs in stack order.
- [ ] Run the GLM multi-GPU correctness and sustained-load matrix with custom all-reduce enabled.
- [ ] Add GLM latency, throughput, and memory comparisons for the intended overlap configuration.
- [ ] Record the focused DeepSeek-V4.1-Flash PRs when the current integration PR is split.
- [ ] Run the DeepSeek-V4.1-Flash TP8 correctness and performance matrix for standard prefill, DP attention, and DSpark.
- [ ] Track speculative decoding support separately for GLM; the current GLM stack rejects MTP/EAGLE with PDMux.

When both model paths need the same scheduler or communication change, the
focused PRs should identify one owning change and link to it. The tracking PR
will be updated as PRs merge and validation results become available.
