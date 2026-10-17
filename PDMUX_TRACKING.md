# GLM-5.3-Flash PDMux integration tracker

[Tracking PR #41861](https://github.com/sgl-project/sglang/pull/41861) records
the four GLM-5.3-Flash implementation PRs, their dependencies and validation.
Related model integrations are linked below for reference.

## GLM-5.3-Flash: four implementation PRs

| Order | PR | Branch | Scope | Status |
| --- | --- | --- | --- | --- |
| 1 | [#42411](https://github.com/sgl-project/sglang/pull/42411) | `feat/glm53-pdmux` | GLM model forward, split-prefill scheduling, bounded layer submission, and capped-prefill/full-device-decode overlap. | Ready for review |
| 2 | [#42412](https://github.com/sgl-project/sglang/pull/42412) | `fix/glm53-mamba-hicache` | Mamba admission/checkpoint reservations, HiCache load-back rollback and aligned host registration. | Draft |
| 3 | [#42413](https://github.com/sgl-project/sglang/pull/42413) | `feat/glm53-pdmux-dp` | Rank-consistent DP/EP communication, raw versus padded token metadata and idle-rank handling. | Draft |
| 4 | [#42414](https://github.com/sgl-project/sglang/pull/42414) | `feat/glm53-pdmux-mtp` | Checkpoint single-layer MTP through EAGLE/NEXTN, final-slice draft handoff and decode-lane verification. | Draft |

Merge order: **#42411 → #42412 → #42413 → #42414**.
All four PRs target `sgl-project/sglang:main`. Their heads are stacked, so
later PRs initially include the earlier commits. After each predecessor merges,
rebase the remaining heads onto upstream `main`, remove the integrated
predecessor commits, and update the links and validation at their new heads.
Each later PR links a focused comparison against its immediate predecessor.
The documentation tracking PR is separate from these runtime dependencies.

The original experiment is restored as
[`glm53-flash-pdmux-old`](https://github.com/Li-brua/sglang/tree/glm53-flash-pdmux-old)
at `991adaa3bd`, retaining its original upstream base `cee293628e` and history.
[`feat/glm53-flash-pdmux`](https://github.com/Li-brua/sglang/tree/feat/glm53-flash-pdmux)
now matches the four-commit integration stack at `bd5264d872`. All child heads
share the first commit `5651d486f5`, using the busiest DP rank's scheduler token
count (`max`) and retaining `max_split_forward_layers`. No extra revert or fixup
commit is needed. The stack was synchronized with upstream `main` at `5b5d721239`;
GLM split forward retains upstream's batch-owned residual stream and capture
ownership across slices. Its multiplex directory matches the original experiment,
while model, DP metadata and attention runtime APIs still include upstream changes.

## Validation and performance

Each cumulative head passed its focused CPU suite on macOS, Python 3.12 and
real CPU PyTorch. CUDA streams, kernels and collectives are mocked where required.

| Cumulative head | Passed | Skipped |
| --- | --- | --- |
| GLM PDMux | 39 | 0 |
| + Mamba/HiCache | 134 | 2 |
| + DP/EP | 163 | 2 |
| + MTP | 177 | 2 |

Changed Python files passed AST parsing, isort 7.0.0, ruff 0.15.1 lint/format and
`git diff --check`. The MTP suite includes 14 tests using normal runtime imports.
The large unified radix-cache matrix remains unvalidated on this host because
native HiCache hashing requires little-endian Linux and CUDA fixtures need a GPU.

The benchmark author reports approximately **40,000–50,000 additional logical
total tokens/s** over the roughly 252,000 tokens/s baseline with configuration D
(`max_split_forward_layers: 1`) on the complete non-speculative experiment.
The reported workload uses concurrency 20, TP/DP/EP 8, TileLang DSA, DeepGEMM,
HiCache, about 140k average input tokens and 98.8% cache hits. Logical throughput
includes cached input tokens. This result has not been reproduced after rebasing
to upstream main; it is not an isolated result for a single child PR and does not
validate MTP throughput.

The author's latest best configuration uses a layer cap of **2**, token budget
65536, `manual_divisions: [[112, 0, 1]]`, and `overlap_decode_full_sm: true`, with
TP/DP/EP 8, TileLang DSA, DeepGEMM, HiCache size 40 and no speculative decoding.
The author reports that the reorganized stack has not reproduced the original
best performance. After restoring the `max` policy, the full CPU suite passed
177 tests with 2 skipped; GPU performance has not been revalidated. With a cap
of 2, switching from `max` to `sum` changes the slice count only when total
prefill tokens exceed 32768 and the busiest rank does not exceed that threshold.

Remaining GPU checks:

- [ ] Output equivalence and accuracy against the same model/configuration without PDMux.
- [ ] TP8/DP8/EP8 stress with uneven ranks, chunked prefill, HiCache pressure and custom all-reduce.
- [ ] Controlled non-speculative throughput/latency comparisons and a launch/stream timeline for configuration D.
- [ ] MTP accuracy, acceptance behavior and target graph replay; compare identical MTP configurations with PDMux on/off.

MTP launch and validation guidance:
[`GLM53_FLASH_PDMUX_MTP.md`](https://github.com/Li-brua/sglang/blob/feat/glm53-pdmux-mtp/GLM53_FLASH_PDMUX_MTP.md).
The first three heads gate speculative decoding; the fourth accepts the
checkpoint's single-layer EAGLE/NEXTN path and keeps PDMux drafts eager.

## Related integration

[DeepSeek-V4.1-Flash PDMux integration #41562](https://github.com/sgl-project/sglang/pull/41562)
continues as a single independent PR covering standard prefill, DP attention
and DSpark. Its implementation and validation are tracked in that PR.

## Merge progress

- [x] Open and link the four GLM PRs.
- [ ] Merge #42411 and narrow the remaining diffs.
- [ ] Merge #42412 and narrow the remaining diffs.
- [ ] Merge #42413 and narrow the MTP diff.
- [ ] Merge #42414 after MTP GPU validation.

Shared scheduler or communication changes should have one owning implementation
PR, with later model integrations linking that dependency.
