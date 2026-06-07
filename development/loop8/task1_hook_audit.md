# Loop 8 / task1 — Per-hook audit: DeepSeek-only shape assumptions in the inherited DS path vs GLM-5.1

**Routing:** `analyze`. Attempted via `/humanize:ask-codex` first; Codex `exec` could not read the
checkout (`bwrap: No permissions to create new namespace` — its sandbox cannot create a mount
namespace, so `rg`/`sed`/`git` all fail and it correctly refused to fabricate file:line evidence).
Audit performed directly against the working tree at commit `d018026f9`; an independent Codex
cross-check was then run with the relevant excerpts inlined (see `task1_codex_crosscheck.md`).

**Shape deltas under audit (GLM-5.1 vs DeepSeek-V3.2):**

| field | V3.2 | GLM-5.1 |
|-------|------|---------|
| qk_nope_head_dim | 128 | **192** |
| v_head_dim | 128 | **256** |
| qk_rope_head_dim | 64 | 64 |
| kv_lora_rank | 512 | 512 |
| q_lora_rank | 1536 | **2048** |
| num_hidden_layers | 61 | **78** |
| DSA index_head_dim × index_n_heads | 128 × 64* | 128 × 32 |
| index_topk | 2048 | 2048 |
| calibration label_dim | 16 | **GLM-native (e.g. 24/32, DEC-3)** |

The label store is `[num_heads_local, label_dim]` channels selected from the projected **K_noPE**
(nope_dim-wide). The selection indices live in **nope-space** (`[0, qk_nope_head_dim)`).

---

## HOOK 1 — `dsa_backend.py :: DSAttnBackend._write_token_labels` (1557–1623)

**VERDICT: GLM-SAFE (already parametric; one latent default to harden at bind — see task2).**

EVIDENCE:
- `head_width = kv_proj_out.shape[-1] // H_local` (1611) — derived from the **projection output**,
  not `layer.v_head_dim`. This was the BL-20260528-dsv32-ds-decode-degeneration fix; it is correct
  for GLM (per-head `kv_b_proj` width = `qk_nope + v_head` = 192+256 = 448 → `[T,H,448]`).
- `nope_dim = self._ds_qk_nope_head_dim` (1602) then `kv_proj_out.view(T,H_local,head_width)[..., :nope_dim]`
  (1612) — reshape-before-slice (BL-20260527-reshape-before-slice-mla), picks K-noPE columns not V.
  For GLM, slicing `[:192]` of the 448-wide per-head block is correct.
- `kv_lora_rank = k.shape[-1]` (1595) — read from the tensor, not hardcoded 512.
- **Latent trap (not in this hook):** `_ds_qk_nope_head_dim` is read at backend `__init__`
  (`dsa_backend.py:519`) as `int(getattr(server_args, "_ds_qk_nope_head_dim", 128))` — **default 128**.
  It is set to the model's real `self.qk_nope_head_dim` only by `_bind_double_sparsity_runtime_data`
  (`deepseek_v2.py:2004`). If, on GLM, the backend reads the default before bind publishes 192, the
  hook would slice `[:128]` of a 192-wide nope block → **wrong labels, no crash**. This is the single
  real GLM risk and is closed by the **bind-time shape verification** (task2), which asserts the
  published nope_dim equals the model's `qk_nope_head_dim` and that the mask's `head_dim == qk_nope`.
- Docstring (1566–1568) says "128-d K_noPE" / "512-d latent" — **documentation-only**; the code is
  parametric. Optional docstring de-DeepSeek-ing in task2/task3.

## HOOK 2 — `deepseek_v2.py :: DeepseekV2AttentionMLA._select_topk_indices` (2102–~2300) + graph-safe path

**VERDICT: GLM-SAFE.**

EVIDENCE:
- `queries_for_ds = q_nope if q_nope is not None else q_lora` (2197) — the selector consumes
  projected Q-noPE; its head_dim is `qk_nope_head_dim` (192 GLM), read from the tensor.
- Metadata resolution goes through `ForwardContext` (`get_attn_backend()`, TBO-unwrapped) per
  BL-20260527-ds-metadata-via-forward-context (2146–2177, 2217–2234) — model-agnostic.
- No `==128` / nope-width / power-of-two head_dim assumption anywhere in the branch; dispatch into
  `retrieve_topk_graph_safe` passes through `channel_mask`/`token_label_table` (both label_dim-shaped).

## HOOK 3 — `double_sparsity/token_label_write.py` (`token_label_write`, `invalidate_token_label_slots`)

**VERDICT: GLM-SAFE.**

EVIDENCE:
- `H, label_dim = channel_selection_layer.shape` (77) — both read from the selection tensor.
- `nope_dim` is implicit in `k_nope.shape[-1]`; gather index `sel_idx` is `[T,H,label_dim]` expanded
  from the selection — no hardcoded width. int8/fp16 paths both shape-agnostic.
- Docstring (lines 1–8) says "128-d projected nope K" — **documentation-only**.

## HOOK 4 — `double_sparsity/selection_kernel.py` (`retrieve_topk_graph_safe` / `_compute_token_scores_kernel` / `_logical_score_kernel` / `project_query_onto_channels`)

**VERDICT: GLM-SAFE. No Triton block-size needs a nope_dim/head_dim-aware launch.**

EVIDENCE:
- Scoring is in **label_dim space**: `bs, num_heads, label_dim = q_proj.shape` (320);
  `label_dim_pow2 = _next_pow2(max(label_dim,1))` (336); the kernel masks `d_offs < label_dim`
  (130, 252). label_dim 24/32 → pow2 32 is fine. **No `==128`, no head_dim assumption** in the kernel.
- `project_query_onto_channels` (364–391): gathers `channel_selection` from `queries[bs,H,head_dim]`
  → `[bs,H,label_dim]`; validates `channel_selection.shape[0] == num_heads` (383). head_dim is read
  from `queries.shape` (382). The only implicit requirement is that selection index values are
  `< head_dim` (= qk_nope 192) — enforced offline by calibration and at load by `channel_mask`
  (`cs_max < head_dim`, channel_mask.py:206) and to be re-asserted at bind (task2).
- `TOKEN_BLOCK = _next_pow2(min(token_block, max_tokens))` (335) — sequence-driven, not head-driven.

## HOOK 5 — `deepseek_v2.py :: forward_absorb_prepare` (1719+) + bind (`_bind_double_sparsity_runtime_data` 1863–2033, `finalize_double_sparsity_bind` 1848–1861)

**VERDICT: GLM-SAFE (functionally); bind is where task2's verification + hard-error must land.**

EVIDENCE:
- `forward_absorb_prepare` attaches `kv_b_proj` onto `attn_mqa` under `if self.use_double_sparsity ...`
  (1735–1736) — DS-only, leaves non-DS byte-identical. The hook width is derived downstream from the
  projection output (HOOK 1), so attn_mqa's absorbed `v_head_dim=512` does not affect labels.
- Bind reads everything from config/mask: `label_dim = int(local_mask.label_dim)` (1968),
  table sized from `kv_pool.size + page_size` (1967), `num_layers_local = config.num_hidden_layers`
  (1955 → 78 GLM), `num_heads_local = self.num_local_heads`. Publishes `_ds_qk_nope_head_dim =
  self.qk_nope_head_dim` (2004) and `_ds_channel_selection` on the selector device (1999–2003).
- **GAP:** bind does NOT cross-check `mask.head_dim` against `self.qk_nope_head_dim`, nor
  `mask.label_dim`/page/dtype against the runtime, nor `channel_selection.max() < qk_nope`. The startup
  validator explicitly **defers** the head_dim check (validator.py:303–306: "best-effort … attention
  layer not yet constructed"), and `channel_mask.validate_against_runtime` (channel_mask.py:391) is
  **defined/exported but never called**. → A V3.2 mask (head_dim=128) on GLM passes startup and binds,
  then `project_query_onto_channels`/`token_label_write` silently gather the first 128 of GLM's 192
  nope channels (indices 0..127 are in-range) — **wrong selection, no crash**. AC-3's negative test
  ("loading a head_dim=128 mask against GLM must FAIL") and AC-2's hard-error policy both require this
  to be caught. **This is task2's deliverable**, and the bind site is where head_dim is authoritative.

## HOOK 6 — `double_sparsity/calibrate.py` (`_extract_mla_nope_prefix`, config shape derivation) + `channel_mask.py` (`load_channel_mask`, `validate_against_runtime`)

**VERDICT: GLM-SAFE (config-driven). `validate_against_runtime` exists but is unwired (task2 wires it).**

EVIDENCE (calibrate.py):
- `_extract_mla_nope_prefix` (115–134): `flat.reshape(-1, num_heads, nope_dim + suffix_dim)[..., :nope_dim]`
  — reshape-before-slice, BL-20260527-reshape-before-slice-mla. Parametric in `nope_dim`/`suffix_dim`.
- Reads `qk_nope_head_dim`, `v_head_dim`, `head_dim` from config (471–480); `k_head_dim = qk_nope_head_dim`
  (480 → 192 GLM); `importance = zeros((num_layers, num_heads, k_head_dim))` (505). Reads
  `qk_rope_head_dim` **directly** from config (484–486) with the BL-20260527-mla-config-rope-dim-derivation
  guard (488–494) — does NOT derive it from `hidden_size//num_heads`.
- Fallback `H = num_heads_hint or 128`, `D = head_dim_hint or 128` (438–439) lives in the non-MLA hint
  path; for GLM the config has the MLA fields so the real path (471+) runs. (Justify; verify in task4.)

EVIDENCE (channel_mask.py):
- `load_channel_mask` validates `channel_selection` 3-D `[L,H,label_dim]` (160), `label_dim` metadata ==
  `channel_selection.shape[-1]` (193), and **channel indices in `[0, head_dim)`** (206–209). For GLM the
  recorded `head_dim` metadata must be 192 (the nope width) — produced by calibration (task5).
- `validate_against_runtime` (391) checks dtype/head_dim/page_size/label_dim vs runtime, taking
  `model_head_dim`. **Never called** (only re-exported in `__init__.py`). task2 wires it at the bind
  site with `model_head_dim = self.qk_nope_head_dim`.

---

## SUMMARY

- **GLM-safe by inheritance / already parametric (5 of 6 hook areas):** `_write_token_labels`,
  `_select_topk_indices`, `token_label_write.py`, `selection_kernel.py` (all kernels), `calibrate.py`
  extraction. The V3.2 bring-up (Loops 0–7) already removed the head-dim hardcodes; widths are read
  from projection outputs / config / tensor shapes, and the Triton kernels are **label_dim-driven**.
- **No `head_dim=192` kernel/reshape specialization is needed (task3 → none).** No Triton block-size is
  head_dim/nope_dim-shaped; all are `label_dim`- or `seq_len`-driven via `_next_pow2`. AC-2.1's
  "narrowest specialization" is therefore **zero kernel changes**, which is the correct outcome under
  DEC-1 (no abstraction without an audit-proven break — and there is no break).
- **The one real gap is the missing bind-time shape verification (task2):** nothing asserts the
  calibrated mask's `head_dim`/`label_dim`/page/dtype match GLM's authoritative MLA dims, and
  `validate_against_runtime` is dead code. A wrong-shape (e.g. V3.2 head_dim=128) mask on GLM would
  bind and silently mis-select. task2 adds a hard-error-naming-the-field check at the bind site
  (DS explicitly requested) covering the full GLM shape set + `channel_selection.max() < qk_nope` +
  the `_ds_qk_nope_head_dim`-default consistency, leaving the DSA path untouched when DS is off.
- **Per-hook deliverable (AC-2):** hooks 2,3,4 and the calibrate extraction = *documented GLM-safe
  source evidence*; hook 1 = GLM-safe + the bind-time default-hardening; hook 5/6 bind = the
  *config-driven patch* (wire `validate_against_runtime` + the GLM shape-set verification).
