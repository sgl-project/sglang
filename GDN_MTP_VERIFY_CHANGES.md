# GDN MTP verify — sglang changes (hybrid WY verify + Triton recovery)

Scope: the GDN MTP speculative-decode **verify + recovery** path in sglang, for
`gdn-mtp-cache-mode=none`. Validated on Qwen3.5-397B-A17B-NVFP4, GB300 (sm_103), TP4, conc 256,
draft-len 3 (T=4). Pairs with the FlashInfer-side WY-kernel changes
(`flashinfer-pr3720/repo/GDN_WY_VERIFY_CHANGES.md`).

## Files changed (5, net vs upstream)
```
python/sglang/srt/environ.py                                           +24   env flags
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py       +57   recovery routing + persist-read
python/sglang/srt/layers/attention/linear/gdn_backend.py              +161   A_log/dt_bias cast fix + Option A
python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py    +74   WY verify routing + feed bf16 A_log
python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py       +12   out= param (Option A)
```
Two commits: `b32880b7ec` (hybrid base — routing + recovery gate) and `f6aaeaa623`
(A_log/dt_bias cast fix + Option A stash-elim).

---

## 1. WY verify routing — `gdn_flashinfer.py` (commit b32880, essential)
Route the `none`-mode verify to the FlashInfer WY output-only kernel; keep `full` mode on the
FI state kernel.

```python
_use_wy = (
    get_global_server_args().gdn_mtp_cache_mode == "none"
    and envs.SGLANG_GDN_WY_VERIFY.get()
)
if _use_wy:
    from flashinfer.gdn_kernels import gated_delta_rule_mtp_wy_output_only as gated_delta_rule_mtp_bf16_state
else:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp as gated_delta_rule_mtp_bf16_state
```

## 2. Hybrid recovery gate — `hybrid_linear_attn_backend.py` + `environ.py` (commit b32880, essential)
Choose the accepted-state recovery kernel. `SGLANG_GDN_FI_RECOVERY=0` → Triton recover kernel
(the shipped hybrid); `=1` → FlashInfer state kernel.

```python
use_fi_recovery = (
    envs.SGLANG_GDN_FI_RECOVERY.get()
    and decode_kernel.__class__.__name__ == "FlashInferGDNKernel"
    and getattr(decode_kernel, "use_state_pool", False)
)
if use_fi_recovery:
    ...  # FlashInfer state kernel (gdn_decode_bf16_state)
else:
    fused_sigmoid_gating_delta_rule_recover_final_state(...)   # Triton recovery
```
Env flags added: `SGLANG_GDN_WY_VERIFY` (default True), `SGLANG_GDN_FI_RECOVERY` (default True).
Shipped hybrid = WY verify + Triton recovery (`SGLANG_GDN_FI_RECOVERY=0`).

## 3. A_log/dt_bias bf16 pre-cast — `gdn_backend.py` + `gdn_flashinfer.py` (commit f6aaeaa, relevant / default-on / lossless)
The WY kernel reads A_log/dt_bias as bf16; the per-step fp32→bf16 cast was being baked into the
captured decode graph (1 per GDN layer per step). Pre-cast the weights to bf16 **once during
eager prefill warmup**, store on the layer, feed bf16 to the WY verify (gated to none/WY so the
full-mode fp32 state kernel keeps full precision). Bit-identical to the kernel's own cast →
no accept-length / accuracy change.

```python
# gdn_backend.py forward_extend (top) — capture-guarded, populates in eager prefill warmup:
if getattr(layer, "_gdn_A_log_bf16", None) is None and not torch.cuda.is_current_stream_capturing():
    layer._gdn_A_log_bf16   = layer.A_log.detach().to(torch.bfloat16).contiguous()
    layer._gdn_dt_bias_bf16 = layer.dt_bias.detach().to(torch.bfloat16).contiguous()

# at the verify call — WY/none only; full mode keeps the fp32 weights:
_use_wy_verify = (get_global_server_args().gdn_mtp_cache_mode == "none"
                  and envs.SGLANG_GDN_WY_VERIFY.get()
                  and getattr(layer, "_gdn_A_log_bf16", None) is not None)
_verify_A_log   = layer._gdn_A_log_bf16   if _use_wy_verify else layer.A_log
_verify_dt_bias = layer._gdn_dt_bias_bf16 if _use_wy_verify else layer.dt_bias
```
Validated: `bfloat16_copy` cast 450→0 in the decode trace; gsm8k 0.555, AL 3.39 (unchanged).

## 4. Option A stash-elim — 4 files (commit f6aaeaa, **env-gated `SGLANG_GDN_STASH_ELIM`, default off, ~0 e2e**)
In `cache_mode=none` the verify conv output (k/v) is copied into a persistent "stash" so the
deferred recovery can read it after the CUDA graph recycles activations. Option A removes those
k/v copies: the verify conv writes its output directly into a persistent per-layer buffer, and
recovery reads k/v as strided views of it.

```python
# causal_conv1d_triton.py — causal_conv1d_update gains out=:
def causal_conv1d_update(..., out: Optional[torch.Tensor] = None):
    out = torch.empty_like(x) if out is None else out   # write into a caller buffer
```
```python
# gdn_backend.py — verify conv writes into a persistent token-major [max_bs, draft, conv_dim]:
pbuf = self._conv_out_persist.get(layer.layer_id) or torch.empty(...)
_conv_out_arg = pbuf[:batch_size].transpose(1, 2)       # matches empty_like strides
mixed_qkv_processed = causal_conv1d_update(..., out=_conv_out_arg)
# ...and the k/v recovery-stash copies are skipped (stash_entry["conv_dims"] stored instead).
```
```python
# hybrid_linear_attn_backend.py — recovery reads k/v from the persist buffer (no copy):
def _persist_kv(layer_id, stash, b_rows):
    persist = persist_per_layer[layer_id]
    q_dim, k_dim, v_dim, Hk, Dk, Hv, Dv = stash["conv_dims"]
    mixed = persist[:b_rows].reshape(b_rows * cache_steps, q_dim + k_dim + v_dim)
    k = mixed[:, q_dim:q_dim + k_dim].view(1, b_rows * cache_steps, Hk, Dk)   # token stride = conv_dim
    v = mixed[:, q_dim + k_dim:q_dim + k_dim + v_dim].view(1, b_rows * cache_steps, Hv, Dv)
    return k, v
```
Bit-exact correctness; gsm8k 0.565, AL 3.40, no hang. **But e2e flat** — the stash copies were
off the step critical path (they overlap the per-layer MoE/GEMM), so this only *cleans the
verify region* (~8.5µs of copies), it does not move throughput. Hence default-off.

---

## Env flags (set on the serving process)
| flag | default | effect |
|---|---|---|
| `SGLANG_GDN_WY_VERIFY` | True | route none-mode verify to the WY output-only kernel |
| `SGLANG_GDN_FI_RECOVERY` | True | recovery kernel: True=FlashInfer state, **0=Triton (hybrid)** |
| `SGLANG_GDN_STASH_OVERLAP` | False | fork the recovery stash copies to a side stream |
| `SGLANG_GDN_STASH_ELIM` | False | Option A: conv writes persist buffer, recovery reads it (no k/v stash copy) |

(The FlashInfer-side WY flags — `SGLANG_GDN_WY_NATIVE_T`, `SGLANG_GDN_WY_STRIDED_QKV`,
`SGLANG_GDN_WY_NATIVE_AB` — are read directly in the FlashInfer wrapper, not registered here.)

## Essential vs optional
| change | files | role | keep |
|---|---|---|---|
| WY verify routing | gdn_flashinfer | required to use the WY kernel | ✅ |
| hybrid recovery gate | hybrid_backend, environ | required for the Triton-recovery hybrid | ✅ |
| A_log/dt_bias cast fix | gdn_backend, gdn_flashinfer | lossless, removes a per-step cast | ✅ |
| Option A stash-elim | environ, causal_conv1d, gdn_backend, hybrid_backend | env-gated, **~0 e2e** (region cleanup only) | ⚠️ optional |

Option A is the sglang analog of the dropped FlashInfer state-kernel tweak — it can be removed
if the PR should carry only the throughput-relevant changes (the +5.4% comes entirely from the
FlashInfer strided-qkv + native-a/b).

## Validation
- gsm8k (Jane #26520 protocol: 1319 ex, 8-shot, max_tokens 16384, T=0.6): **0.9794** (none+FI
  hybrid) = 0.9794 (none+Triton) ≈ 0.9779 (full+Triton) — lossless. (Earlier 0.55 numbers were a
  truncated-CoT eval config, not a regression.)
- Bit-exact gates on the FlashInfer side (`max|Δ|=0`); decode-only nsys + perfetto traces.

## Decode step latency at concurrency 256 (GB300, controlled)
Cleanest decode metric: the GPU period from one `step[TARGET_VERIFY]` to the next at sustained
batch=256 — **accept-length-independent** (the verify forward processes bs×draft_tokens
regardless of accepts) and **prefill-excluded**. Measured from the once-per-step
`VerifyTreeGreedy` kernel's consecutive GPU timestamps; all configs run back-to-back on the
SAME node (fixed seed, `num_prompts=256=max_concurrency`, in256/out1024) so node/seed variance
cancels (±0.5 ms IQR).

| config (cache-mode + backend) | verify / recovery | step latency (ms) | vs hybrid |
|---|---|--:|--:|
| **`none` + FlashInfer (this PR's hybrid)** | **WY verify + Triton recovery** | **45.8–46.9** | — |
| `full` + Triton (baseline) | Triton verify (state cached) | ~47.5 | +1–4% |
| `none` + Triton | Triton verify + Triton recovery | ~50.2 | +7–10% |
| `none` + FlashInfer, FI-state verify + Triton recovery | FlashInfer state (non-WY) + Triton | 49.0 | +4.6% |
| `none` + FlashInfer, FI-state verify + FI recovery (≈ #26520) | FlashInfer state + FlashInfer | 58.7 | +25.2% |

**The hybrid (WY verify + Triton recovery) has the lowest per-step decode latency — faster than
every other config, including full+Triton.** Isolating the two contributions (same-node
FlashInfer-decode variants, recovery or verify held fixed):
- **WY output-only verify: −4.6%/step** vs the FlashInfer state kernel (recovery=Triton fixed; 46.9 vs 49.0 ms).
- **Triton recovery: −16.4%/step** vs FlashInfer recovery (verify=FI-state fixed; 49.0 vs 58.7 ms).

This is why the original `none`+FlashInfer (FI-state verify + FI recovery, 58.7 ms) was *slower*
than Triton: the hybrid fixes it via Triton recovery (−16%, dominant) + WY verify (−4.6%).
