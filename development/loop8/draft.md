# Loop 8 Draft — GLM-5.1 DS bring-up (this is roadmap **Loop 10**, pulled forward)

> Written 2026-06-07, after **Loop 7 closed** (`.humanize/rlcr/2026-06-01_09-27-07/`, decision
> `development/loop7/m12_final_decision.md`). The client re-prioritized to **GLM-5.1 (deferred client #1)**,
> so roadmap **Loop 10** is pulled ahead of Loops 8/9 and executed in the next on-disk loop dir
> (`development/loop8/` = roadmap Loop 10). Feed this through `gen-plan` once scope is confirmed.

---

## Objective

Bring up the **opt-in Double-Sparsity (DS) path on GLM-5.1-FP8**, single-node TP=8, and re-run the accuracy +
SLO gates — **without** building a GLM-specific standalone DS path and **without** regressing the model's
native default. This is a model bring-up, not recall R&D (that was Loop 7).

## Model + weights (already staged)

- **Weights are already in `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/`** (HF cache layout; 142
  FP8 safetensor shards under `snapshots/<hash>/`). No download needed.
- Architecture: **`GlmMoeDsaForCausalLM` / `model_type: glm_moe_dsa`** — MLA attention + a **native trained
  DSA indexer** (`self_attn.indexer` / `indexers_proj`) + 256-expert MoE; FP8 e4m3 block-quant (128×128);
  78 layers, hidden 6144, kv_lora 512 / q_lora 2048 / qk_rope 64 / qk_nope 192 / v_head 256; ~198k max ctx.

## Key finding — §4.0 question is ANSWERED for GLM-5.1

GLM-5.1 **ships a native trained DSA indexer**, and **`is_deepseek_dsa()` returns True for it** (confirmed:
config exposes `index_topk: 2048`, `index_head_dim: 128`, `index_n_heads: 32`;
`python/sglang/srt/configs/model_config.py:111`). So GLM-5.1 already routes through the **same
`dsa_backend.py`** as DeepSeek-V3.2 (and at the same 2048 budget). ⇒ **Same posture as V3.2: DSA-native is the default; DS is the
opt-in fallback**, valuable only where the trained indexer underperforms (e.g. long-context recall — the
Loop-7 regime). DS is *not* the primary path here; it is the reversible opt-in knob, default-off.

## Scope — IN

1. **Compatibility — wire DS into the preexisting backend (the load-bearing requirement).** GLM-5.1 uses the
   existing DSA backend (`dsa_backend.py`); the DS solution must **reuse our current wiring pattern** — bind
   DS into that preexisting backend (the bind site + `TokenLabelTable` + the selection/label-write hooks),
   **not** a separate GLM DS backend. Concretely: generalize the DS model-forward hooks that are today
   DeepSeek-specific (`deepseek_v2.py`: `_select_topk_indices` / `forward_absorb_prepare` /
   `_write_token_labels` / the channel-mask bind) onto the **GLM model forward**, so DS attaches to GLM the
   same way it attaches to V3.2.
2. **Calibrate a GLM-5.1 channel mask** (the offline importance projection) for the GLM MLA shapes, and bring
   up the DS serving path (TP=8, FP8, page 64).
3. **Re-run the gates on GLM-5.1**: accuracy (MMLU within tolerance of the DSA-native default) + the client
   SLOs (≥30 TPS/req, P99 TTFT < 22 s at the client workload) + DS-vs-DSA-native non-regression.

## Scope — OUT

- **Re-litigating V3.2** or the Loop-7 recall R&D (the learned-selector follow-on is its own loop, roadmap 11).
- **nvfp4/mxfp4, multi-node TP, the knob-compat matrix** — their own roadmap loops (8/9 deferred behind this).
- Closing any GLM long-context recall gap — first **bring it up and characterize**; recall R&D is downstream.

## Acceptance criteria (draft — `gen-plan` formalizes)

1. GLM-5.1 **serves** with DS opt-in on TP=8 FP8; the DSA-native default path is **byte-identical when DS is
   off** (no regression to the shipped model).
2. DS attaches via the **existing `dsa_backend.py` wiring** (no standalone GLM DS backend); the DeepSeek-only
   model-forward hooks are generalized, not duplicated.
3. GLM-5.1 channel mask calibrated + loaded; DS decode is coherent (dense-DS / within-budget sanity).
4. Accuracy + SLO gates recorded DS-vs-DSA-native on the same node; result characterized (uplift or parity or
   documented gap).

## Hardware / inputs to read first

- **Hardware:** single node 8×H200 (TP=8), FP8 e4m3, page 64, fp8 KV.
- **The preexisting backend DS wires into:** `python/sglang/srt/layers/attention/dsa_backend.py`
  (`is_deepseek_dsa` → `use_dsa`); the DSA recognition at `configs/model_config.py:102-114`.
- **The DS wiring to generalize:** `deepseek_v2.py` (DS hooks) + `double_sparsity/` (selection_kernel,
  token_label_write, channel-mask calibration).
- **GLM model forward (where the DS hooks land):** `python/sglang/srt/models/glm4_moe.py` registers
  `GlmMoeDsaForCausalLM` and already shares `models/deepseek_common/deepseek_weight_loader.py` — so the
  DeepSeek-common infra is partly reusable and the generalized DS hooks have a natural home here.
- **Prior art:** the Loop-7 decision (`development/loop7/m12_final_decision.md`) + the DS↔V3.2 bring-up record.

## Pending decisions (resolve in `gen-plan` discussion)

- **How much of the DeepSeek DS model-forward hook can be shared vs GLM-specialized.** `glm4_moe.py` handles
  `GlmMoeDsaForCausalLM` and already shares `deepseek_common`, so part is reusable — but the MLA dims differ
  (v_head 256, qk_nope 192) and the indexer is wider (`index_head_dim` 128 × 32 heads). Where is the clean
  shared-vs-specialized boundary, and does the DS channel-mask calibration assume DeepSeek-only shapes?
- **Is DS even worth landing on GLM-5.1 given its native indexer?** Default expectation (per §4.0): land it as
  the reversible opt-in fallback + characterize; do not chase parity-beating recall in this bring-up loop.
