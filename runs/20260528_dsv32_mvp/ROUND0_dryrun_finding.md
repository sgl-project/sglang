# Round 0 — Calibration one-block dry-run finding

Date: 2026-05-28
Host: 8× H200 (143 GB free each), single node.
Command log: `calibrate_dryrun_20260528-103632.log`

## What was run
The new `calibrate.py --dry-run-blocks 1` path against the real cluster weights:

```
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 8 \
    --label-dim 16 --page-size 64 --num-samples 1 --block-size 512 --seed 42 \
    --dataset runs/20260528_dsv32_mvp/dryrun_prompts.txt \
    --dry-run-blocks 1 -v
```

## Result: BLOCKED at config load (before dtype/device matters)

`AutoConfig.from_pretrained(..., trust_remote_code=True)` raised:

```
ValueError: The checkpoint you are trying to load has model type `deepseek_v32`
but Transformers does not recognize this architecture.
```

This is **not** a dtype/device/upcast problem — it fails at config resolution, before any weights load.

## Root cause
- The plan's calibration approach assumes the model can be loaded with HF
  `AutoModelForCausalLM.from_pretrained(..., device_map="auto")`.
- The DeepSeek-V3.2 checkpoint has **no `auto_map`** and ships **no remote
  modeling/configuration `.py`** (only `config.json`, safetensors shards,
  tokenizer files), so `trust_remote_code=True` has nothing to load.
- transformers 5.8.1 `CONFIG_MAPPING` has `deepseek_v2`, `deepseek_v3`,
  `deepseek_v4` — but **not** `deepseek_v32`.
- SGLang serves V3.2 via its **own** model class
  (`python/sglang/srt/models/deepseek_v2.py::DeepseekV32ForCausalLM`) and treats
  `deepseek_v32` as an *unregistered* HF model type
  (`python/sglang/srt/utils/hf_transformers_patches.py`). It never registers a
  HF AutoModel modeling class for it.

Conclusion: stock HF `AutoModelForCausalLM` cannot load DeepSeek-V3.2. The plan's
`device_map="auto"` premise is correct in spirit (sharded native-FP8 load) but
the HF entrypoint is unavailable for this architecture as written.

## Validated path forward (config-only probe, PASSED)
Remapping the config `model_type` to `deepseek_v3` (and `architectures` to
`DeepseekV3ForCausalLM`) builds a valid `DeepseekV3Config` that preserves every
field calibration needs:

```
AutoConfig built as DeepseekV3Config | model_type=deepseek_v3
  num_hidden_layers = 61
  qk_nope_head_dim = 128
  qk_rope_head_dim = 64
  v_head_dim = 128
  kv_lora_rank = 512
  has quantization_config: True   # FP8 block-quant preserved
```

DeepSeek-V3.2 differs from V3 only by the DSA sparse-attention **indexer**, which
is irrelevant to channel-importance calibration — calibration only needs the MLA
`kv_b_proj` / `q_b_proj` projections, which are identical in both. So loading the
V3.2 FP8 weights under the transformers `deepseek_v3` modeling is a viable
calibration load path.

## Next-round plan (the AC-4 load redesign)
1. In `calibrate.py`, when the on-disk `model_type` is `deepseek_v32` (an
   unregistered architecture), load the config, remap `model_type=deepseek_v3` /
   `architectures=[DeepseekV3ForCausalLM]`, and pass the remapped config to
   `AutoModelForCausalLM.from_pretrained(..., config=remapped, device_map="auto",
   torch_dtype="auto")`.
2. Expect V3.2-only indexer weights to be reported as unexpected keys; confirm
   the MLA projections load and the Method-1 Q/K hooks fire on all 61 layers
   (the existing "hooks did not fire" guard validates this).
3. Re-run `--dry-run-blocks 1`; the dtype/device report should then show FP8
   weights present (no bf16 upcast) sharded across the 8 GPUs.
4. Alternative if the v3-remap forward is wrong: drive the forward through
   SGLang's own model loader instead of HF AutoModel.

## Status
- Code changes that ARE landed and correct for once the load works:
  `device_map="auto"` + `torch_dtype="auto"`, input-embedding device routing,
  and the `--dry-run-blocks` mode.
- The mask `/models/dsv32-fp8-channel-mask.safetensors` is still absent (root
  blocker unchanged). The load redesign above unblocks it next round.
- Pile-val (`mit-han-lab/pile-val-backup`) is NOT cached locally; the full
  calibration (task4) will need it downloaded or a local `--dataset` substitute.
