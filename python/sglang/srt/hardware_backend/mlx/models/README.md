# sglang MLX models

Native MLX model implementations for the sglang MLX backend.  Each
module exposes a `Model` class that the sglang `MlxModelRunner` can
construct and load weights for, independent of `mlx_lm.models.*`.

## Layout

| File | Status | Notes |
|---|---|---|
| `qwen3_5_moe.py` | Phase 2 — text-only | Qwen3.6-35B-A3B; hybrid 3:1 attention, 256 experts / 8 active, attn_output_gate, shared expert |

## Adding a new model

1. Implement `TextModelArgs` (or equivalent) with the HF config fields
   the architecture needs.  Add a `from_hf_config` classmethod for HF
   config parsing.
2. Implement the architecture in sglang:
   * For `nn.Linear`-style layers, write your own.
   * For experts, alias `mlx_lm.models.switch_layers.SwitchGLU` (or
     `QuantizedSwitchGLU`) so 4-bit `QuantizedSwitchLinear` weights
     load via `Model.load_weights` and the
     `SGLANG_MLX_FUSE_SWIGLU` patcher recognises the module.
   * For the linear-attention primitive, import `gated_delta_update`
     from `mlx_lm.models.gated_delta`.
   * For masks, import `create_attention_mask` / `create_ssm_mask`
     from `mlx_lm.models.base`.
3. Add a `Model.sanitize(weights)` that handles:
   * Dropping unused subtrees (`mtp.*`, `model.visual.*`,
     `vision_tower.*`).
   * Stripping the `language_model.` prefix (HF composite-model layout
     vs the sglang flat layout).
   * HF → MLX weight-shape remapping (`conv1d.weight` transpose,
     RMSNorm `(1 - gamma) → gamma` shift when the file is a fresh HF
     export).
4. Add a `load(path)` helper that reads `config.json`, merges all
   `model*.safetensors` shards, calls `nn.quantize` with a
   per-layer predicate (so `QuantizedLinear` modules are created only
   for paths whose `.scales` exist in the weights), then
   `Model.load_weights`.
5. Add a `_is_<arch>(model_path)` function in `model_runner.py` that
   reads `config.json` cheaply (one JSON read) and returns True for
   the architectures your model handles.
6. Hook the loader into `MlxModelRunner._load_model` with the
   `_is_<arch>(self.model_path)` check.

## Tests

`test/registered/unit/hardware_backend/mlx/test_<arch>.py` should
cover:

* Default `TextModelArgs` shape matches the target config.
* `from_hf_config` parses the relevant HF fields.
* `Model` constructs with a small config (so random init doesn't OOM
  on the real-model config) and exposes the attention contract attrs
  (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`rope`/`scale`) for the
  full-attention layers.
* `sanitize` drops / strips / remaps weights as expected.
* `_is_<arch>` matches multimodal, text-only, and text_config-wrapped
  HF configs, and rejects unrelated architectures and malformed
  configs.

Use a small config (2-4 layers, ≤8 experts) for the construction /
forward tests so the random init memory is well under test limits.
