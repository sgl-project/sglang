# DS decode degeneration — DSA-baseline control (Round 2)

Same model (/cluster-storage/models/deepseek-ai/DeepSeek-V3.2), same `dsa`
attention backend, same fp8_e4m3 KV, same flashmla_kv prefill/decode backends,
TP=8, page=64, Option B knobs. Only difference: Double Sparsity.

Prompt: "The capital of France is", temp=0, max_new_tokens=24.

- **DS-on** (serve_double_sparsity.sh): `" Paris. DDDDDDDDDDDD..."`  → DEGENERATE after prefill.
- **DSA baseline** (serve_native_nsa.sh): `" Paris. 法国的首都是巴黎。\nThe capital of Italy is Rome. 意大利的首都是罗马。\nThe capital of"` → COHERENT.

## Conclusion
The V3.2 DSA serving stack (model load, FP8 KV, flashmla_kv read, regular CUDA
graph, dense prefill) is CORRECT. The degeneration is **Double-Sparsity-specific**:
it appears only once DS selection drives decode attention (prefill used dense
MHA_ONE_SHOT and was correct in both). Root cause is in the DS decode path:
`_select_topk_indices` → `logical_to_physical` → the physical page table fed to
flashmla_kv decode. Next-round target.

Artifacts: ds_generate_probe.json (DS), dsa_baseline_generate_probe.json (baseline).
