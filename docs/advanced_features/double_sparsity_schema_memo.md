# Channel-Mask Schema Compatibility Memo (GLM-5.1 / 128K ISL / FP4)

This memo records the forward-compatibility argument for the Double Sparsity channel-mask file schema against the three deferred-but-tracked targets from DEC-6:

1. **GLM-5.1** (deferred-but-hard, per user CMT-1 / DEC-6 split)
2. **128K input sequence length**
3. **FP4 (nvfp4 / mxfp4) quantized weights**

Goal: zero schema rewrites once any of these targets ship. The file format and validator are designed so a calibrated artifact produced today for DeepSeek-V3.2 (FP8) loads cleanly on a future GLM-5.1 / 128K / FP4 build without bumping `schema_version`.

## Schema fields (post-CMT-6 reworked)

| Field | Type | Purpose | Cross-target stance |
|-------|------|---------|---------------------|
| `channel_selection` | int32 `[L, H, label_dim]` tensor | Per-(layer, head) channel indices | L and H scale per model. GLM-5.1's expected layer/head counts are recorded in its own calibration run; the schema does not encode an absolute upper bound. |
| `channel_weights` | float32 `[L, H, label_dim]` tensor | Normalised importance per selected channel | dtype is float32 regardless of model weights / KV dtype — projection accumulates in fp32, kernels dequant as needed. |
| `schema_version` | str | Loader compatibility gate | Frozen at `"1"` for the V3.2-FP8 / GLM-5.1 / 128K / FP4 cohort. Bumped only on incompatible changes. |
| `dtype` | str | `kv_cache_dtype` calibrated for | Currently `"fp8_e4m3"` or `"bfloat16"`. FP4 enablement adds `"nvfp4"` / `"mxfp4"` here — both work with the same schema; only the kernel branch in `page_signature_write` learns the new dequant. |
| `head_dim` | str of int | Model head_dim sanity check | GLM-5.1 typically uses head_dim=128 (same as V3.2). 128K ISL does not change head_dim. FP4 does not change head_dim. |
| `page_size` | str of int | Runtime page granularity | Decoupled from sequence length: 128K ISL just means more pages of the same `page_size`. Schema admits any positive page size; validator restricts to `{32, 64, 128}` for AC-3 compliance. |
| `label_dim` | str of int | Compressed projection width | Currently 16. The selector buffer matches; bumping `label_dim` is a configuration change, not a schema change. |
| `created_at` | ISO-8601 | Audit | No cross-target concern. |
| `content_sha256` | hex str | Content identity | Tensor content is the same regardless of the model class; the hash mechanism applies unchanged. |
| `extra_metadata.*` | str | Free namespace | Where future per-model fields land (e.g. `glm5_indexer_variant`, `fp4_block_size`) without bumping `schema_version`. |

## Per-target analysis

### GLM-5.1

GLM-5.1 is expected to ship the same `nsa.Indexer` interface as DeepSeek-V3.2 (this is the user's explicit guidance per DEC-6 / DEC-10). Under that assumption:

- The capability-check validator (`is_deepseek_nsa(hf_config)` proxy → eventually an interface-presence check) generalises without changing the schema.
- GLM-5.1's L / H may differ from V3.2; the schema is shape-agnostic.
- GLM-5.1 may expose a different KV-cache layout (per-tile FP8 scales may live at a different offset). The `page_signature_write` Triton kernel reads the offset from a model-config constant, not from the schema, so the file is portable.

Schema delta needed: **none**. Calibrate GLM-5.1 with the same `calibrate.py` once weights ship; the resulting file passes the existing loader.

### 128K input sequence length

128K ISL is purely a runtime concern. The schema has no length-dependent fields:

- `max_pages` is sized at server allocation time (1M context / page_size=64 → 15,625 pages at 128K ISL → 2,000 pages — well within the 480 MB / rank operating point).
- The selector ABI (`retrieve_topk -> (selected_indices, valid_lengths)`) handles arbitrary sequence lengths because `selected_indices[bs, max_top_k]` is sized by `max_top_k`, not by seq_len.
- TPOT / TTFT SLO scaling to 128K is the responsibility of the kernel optimisation pass, not the schema.

Schema delta needed: **none**.

### FP4 (nvfp4 / mxfp4) weights

FP4 weight quantisation affects the K-projection layer's weight dtype but NOT the KV cache dtype. The channel mask file records the KV cache dtype (`dtype` field), so:

- An FP4-weights / FP8-KV deployment uses an `fp8_e4m3` channel mask (no change from V3.2).
- An FP4-weights / BF16-KV deployment uses a `bfloat16` channel mask.
- A future FP4-KV deployment (currently not a target) would add `nvfp4` / `mxfp4` to the allowed `dtype` set; the schema admits the string without bumping `schema_version`. The `page_signature_write` kernel learns the new dequant rule, but the file format does not change.

Schema delta needed: **none for nvfp4 / mxfp4 weights**. A future FP4-KV target would not change the schema either, only the validator's allowed-dtype set.

## Validator outcome

The validator (per AC-4) needs the following additions to support the deferred targets:

1. **GLM-5.1**: when GLM-5.1's HF config exposes the indexer interface, extend `is_deepseek_nsa(hf_config)` (or replace with a capability check via `hasattr(layer, 'indexer')`) so the validator stops refusing GLM-5.1 models.
2. **128K**: no validator change.
3. **FP4 weights**: no validator change.
4. **FP4 KV (future)**: extend `_BACKEND_BY_DTYPE` to include the FP4 KV path and the corresponding flashmla backend.

None of these require the file format to change. Tasks 11/12 (kernels) will need their own per-target work; the schema absorbs those changes without rewrite.

## Conclusion

**No schema fields need to be added or modified to admit GLM-5.1, 128K ISL, or FP4 weights.** The schema's existing fields and `extra_metadata` free namespace cover all three targets. The validator's allowed-dtype set is the only place that will grow, and that growth is additive, not schema-breaking. Loader and saver code remain at `schema_version="1"`.

The forward-compatibility argument is the same one that justifies dropping `model_revision_sha` and `tp_world_size` (CMT-6): bookkeeping fields proliferate without catching the bugs they pretend to catch, while content + shape checks cover the actual failure modes.
