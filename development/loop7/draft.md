# Loop 7 Draft — DS Long-Context Recall R&D (Tier-2 / AC-10), the deferred high-priority work

> Written 2026-05-31, after **Loop 6 closed at its Minimum Acceptable Scope** (`.humanize/rlcr/2026-05-30_06-27-19`).
> Loop 6 landed the full **Tier-1 engineering spine** (TokenLabelTable footprint → mem-lift → admission →
> TTFT) and closed AC-5 **directional** (conc-16 P99 TTFT 13.13 s < 22 at the full-context Option-B point).
> Tier-2 (DS long-context **recall** R&D) was **explicitly deferred to its own loop** per the plan's Lower
> Bound and the owner's Round-24 close. **This is that loop, and it is the high-priority carryover.**
> Feed this through `gen-plan` once the scope below is confirmed.

---

## Objective

Close the **DS long-context recall gap** on DeepSeek-V3.2 FP8. Today DS recalls **4K 75% / 16K 5% / 64K 0%**
(NIAH) vs DSA **100% at every length using the same 2048 budget and the same decode kernel**. The whole point
of this loop is to make DS *competitive on recall*, which Loop 6 deliberately did not touch (it fixed
admission/TTFT, not selection quality).

This is the **AC-10 / Tier-2** work carried out of Loop 6. It is **gated-open**: the strategic gate
`runs/20260530_dsv32_loop6/ds_on_v32_decision.md` (Loop-6 AC-1) already resolved to *pursue* Tier-2 recall
R&D **strictly after** the Tier-1 spine landed. The Tier-1 spine has landed → the gate is open.

---

## Why this loop exists (the recall root cause — established in Loop 6)

DS decode is **sound**: dense DS (seq ≤ 2048) recalls **100%**, and DS MMLU == DSA MMLU (89.00%). So the
recall collapse at length is **not** a decode bug. It is the product of **two compounding limits**:

1. **The selection budget is kernel-locked at `index_topk = 2048`.** The shared `flashmla_kv` **decode**
   kernel asserts `indices.shape[-1] == self.dsa_index_topk` (`python/sglang/srt/layers/attention/dsa_backend.py`,
   `_forward_flashmla_kv`) during CUDA-graph capture; `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does **not** bypass it.
   So **DS cannot spend more than 2048 tokens of budget on a hard/long prompt without a new decode kernel.**
2. **DS's offline channel-mask selector is inferior to V3.2's *trained* DSA indexer at that same 2048 budget.**
   V3.2's trained indexer reliably places the needle inside its 2048; DS's offline channel-importance
   projection does not at 16K/64K. This is a **selection-quality** gap, not a budget-size gap *per se* — but
   the two interact: a better selector helps within 2048, and a wider budget helps a fixed selector.

(Full as-built state and evidence: [`development/past_implementations/study/08-current-system-architecture.md`](../past_implementations/study/08-current-system-architecture.md),
the recall characterization in `runs/20260528_dsv32_mvp/ac12_analysis.md`, and the strategic gate
`runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.)

---

## Scope — IN (the two R&D directions, from the strategic gate doc)

### Tier-2.A — PRIMARY: adjustable-`top_k` sparse decode kernel
A `flashmla_kv`-style **decode** kernel (mirroring the native NSA/DSA sparse-matmul decode) that exposes an
**adjustable `top_k`** by relaxing the `indices.shape[-1] == dsa_index_topk` hard cap — as a **new, opt-in DS
decode path**, NOT by weakening the assert on the default DSA path. This lets DS spend a **larger** selection
budget (e.g. 4096 / 8192) on long prompts and is the only lever that can lift recall when the bottleneck is
the 2048 cap itself. Heavy: it is a CUDA-graph-safe sparse-attention decode kernel with a fixed-shape ABI of
its own.

### Tier-2.B — SECONDARY: learned / query-aware DS selector
A **query-aware or learned** DS selector that places the needle inside the existing 2048 budget better than
the offline channel-mask projection — **no kernel change required** (stays within the locked ABI), so it is
much cheaper to try first as a recall-uplift probe. Candidate: a lightweight learned scorer, or pulling the
**top-p / nucleus selection (Twilight, roadmap Loop 11)** forward — top-p can spend more of the 2048 budget
on hard prompts adaptively.

### Tier-2.C — secondary engineering scope: 128k servability
Extends Loop-6's 64K servability (`runs/20260530_dsv32_loop6/ac8_servability/ac8_64k_servability.md`) to the
**128k** context the deferred client requirement (roadmap §6 Loop 7) needs — KV-budget / admission to *serve*
128k. Servability is separate from recall; both are needed for the 128k deliverable.

---

## Scope — OUT

- **Re-litigating the Tier-1 spine.** The lifted **DS int8 / `mem_fraction_static`=0.7 / radix-on / TP=8**
  operating point and the directional AC-5 result stand as the Loop-7 baseline; do not regress them.
- **The strict all-concurrency client SLO** (`P99 TTFT < 22s` AND `≥ 30 TPS/req` at *every* conc). That is a
  **separate downstream** concern: DS per-request decode TPS is ≤ DSA structurally, and conc-64 ≥ 30 is
  unattainable even for DSA (29.4) — so it is an operating-point / DSA-side question, not recall R&D. Only in
  scope here if the owner explicitly merges the two.
- **GLM-5.1 / nvfp4 / multi-node / knob-compat** — their own roadmap loops (§6/§9/§10).

---

## Acceptance criteria (draft — `gen-plan` will formalize positive/negative tests)

1. **Recall uplift, measured.** NIAH 4K/16K/64K recall **delta vs the Loop-5/Loop-6 DS baseline 75 / 5 / 0**,
   on real hardware, DS-vs-DSA on the same node. `gen-plan` sets the binding uplift gate (e.g. 16K materially
   > 5%); a recorded+characterized result is the floor (DEC-3-style), a strict recall target is the stretch.
2. **(If Tier-2.A) the new decode kernel is CUDA-graph-safe and opt-in**: bit-exact selection contract
   (the R23 deterministic tie-break carries over), zero-alloc under graph replay, the **default DSA path's
   `dsa_index_topk` assert is untouched**, fp16/DSA default unchanged.
3. **(If Tier-2.B) the new selector is flag-gated** with the offline channel-mask as default; selection
   equivalence is **NIAH non-regression**, not bitwise (selector granularity changed).
4. **(If Tier-2.C) 128k `/generate` serves** (no HTTP 400) at the lifted op-point, or the new ceiling is
   documented.
5. **No Tier-1 regression**: the Loop-6 admission/TTFT spine and the directional AC-5 conc-16 result still
   hold at the chosen op-point.

---

## Hardware / operating point

Same as Loop 6: single node, **8×H200 (TP=8)**, V3.2 FP8, page_size 64, fp8 KV, `flashmla_kv` prefill+decode,
overlap-schedule + piecewise-cuda-graph disabled, radix-on via the config-bound fixture, **DS int8 compact
table at `mem_fraction_static`=0.7**. Reuse the Loop-5/Loop-6 serve/bench scripts (`development/serve_double_sparsity.sh`,
`benchmark.sh`) and the NIAH harness — no new serve/bench scaffolding.

---

## Inputs / handoffs to read first

- **Strategic gate (the open gate):** `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`
- **As-built system state:** `development/past_implementations/study/08-current-system-architecture.md`
- **Recall characterization:** `runs/20260528_dsv32_mvp/ac12_analysis.md` (+ `ac12_results/`)
- **The kernel cap to relax (Tier-2.A):** `indices.shape[-1] == dsa_index_topk` in
  `python/sglang/srt/layers/attention/dsa_backend.py`
- **The selector to improve (Tier-2.B):** the DS selection path in
  `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` +
  `token_label_write.py` (channel-mask projection)
- **Loop-6 process record:** `.humanize/rlcr/2026-05-30_06-27-19/` (goal-tracker, round summaries)

---

## Pending decisions (resolve in `gen-plan` discussion mode)

- **Which direction leads?** Tier-2.B (cheap, no-kernel, try first) vs Tier-2.A (the only lever if 2048 itself
  is the wall). Recommend B-as-probe then A-if-needed, but A is the higher-ceiling, higher-cost path.
- **Recall gate hardness:** strict recall target vs DEC-3-style recorded directional uplift.
- **Does 128k servability (Tier-2.C) belong here or its own loop?** It is engineering, not recall R&D.
- **Theory-over-pragmatism (standing owner preference):** prefer the theoretically correct
  adjustable-budget/learned-selector design over a cheap hack even at higher engineering cost.
