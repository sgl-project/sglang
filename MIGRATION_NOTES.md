# Cache-mode PP admission on v0.5.22

## Pinned sources

- Base: official `v0.5.22`, `baa9befa381511cd4d56f83980951e5a45fffc6d`.
- Development branch: `codex/pp-admission-v0522-20261016`.
- Feature source: the final reviewed `f21a8b84bb` delta against its original
  `74308cf1d6` base, not the intermediate implementation history.
- The local `main` was also fast-forwarded to `7098ec387c`, but is deliberately
  NOT the development base. Pair this branch with the v0.5.22 image/runtime.

The release's dependency files, Dockerfile, Rust sources and model files are
unchanged. In particular, no unrelated DeepSeek/OwedOutput fix or FlashKV SDK
compatibility shim is included. Image preparation and publication are separate.

## Execution path

`SGLANG_PP_L3_CACHE_ASYNC_ADMISSION=1` selects static PD-prefill, PP > 1,
HiCache + L3, cache mode, FULL/KV-derived DSA. Non-target configurations keep
native scheduling. Existing unsupported-profile validation remains at startup.

1. Each stage progresses its native prefetch and stage-local TP/CP cache ACKs.
2. Native `PPConsensusStore`/bootstrap polling moves requests into waiting_queue.
3. The Gate collects readiness, prepares a full-prefix MIN and keeps the
   necessary prefix lease/capacity credit until native selection takes over.
4. PP0 chooses the candidate order and continuation end; peers replay them.
5. Native PrefillAdder performs budget accounting, allocation and H2D.

The migration does not restore the old bootstrap RID ring. `PrefillControl`
now carries native transfer/release RIDs plus the optional `AdmissionUpdate`.
Admission rides the existing forward transfer fold and return release stream:
no extra PP message, channel or collective is introduced. C posts these messages
before model waits; flag0 retains native send placement and bare RID-list payloads.
Bootstrap processing and the regular PP / PD-decode loops remain unchanged.

The Gate, configuration predicate and prefix-lease units retain the reviewed
`f21a8b84bb` behavior (AST-identical; import whitespace normalized). No removed
attempt-500, final layout rejection, restrict guards,
chaos, direct-vote transport or extreme-buffer implementation was reintroduced.
Native v0.5.22 prefix lengths, H2D token-count returns, adder factory, multimodal
retirement and buffer staging metrics are preserved rather than backported away.

## Validation and remaining limits

- Focused suite: **75 tests + 466 subtests passed**, including four-process Gloo
  TP2PP2/TP1PP4, real Gate votes with transfer RIDs, SPF/FCFS continuation replay,
  and the production event-loop simulation at PP2/4/8/16.
- Native RID intersection, empty transfer/release payloads, startup-slot gating,
  and both flag0/flag1 message order/send placement are covered.
- All five focused files are registered with CPU CI and pass their actual
  `python3 file.py -f` entry points. Repository-selected Ruff rules, changed-source
  compilation and whitespace checks pass; unrelated upstream lint is not fixed.
- Native bootstrap, ordinary PP, PD-decode and decode release-relay bodies were
  checked AST-identical to the release. Dependencies, Rust, native buffer
  pipeline and the DeepSeek model file were checked against the release.
- Full-import native idle/attach/HiCache-event and PPConsensusStore suites were
  attempted but cannot collect on this control host: missing IPython and
  parameterized, followed by an incomplete-package import error; the local
  Torch/CUDA driver mismatch warning remains. These are NOT counted as passes.
- CPU/Gloo results are not GPU, NCCL/RDMA, HTTP/PD or performance acceptance.
  This migration did not contact or modify the restarting DSW.

Existing policy tradeoffs remain: C budgets timeout from the submitted span and
uses the ingress prefetch attempt; B retains native hash-based timeout/retries.
Matching policy names alone does not isolate consensus overhead. GPU comparison,
actual storage SDK compatibility and end-to-end model output checks remain due.
