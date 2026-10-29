# Qwen3.8 Flash Next frozen base

Every switch keeps its original name and code default.

`gdn_inproj_conv.py` and `gdn_inproj_conv_core.py` replace the external-core
PYTHONPATH layer. Only their import path changed. `shared_down_tgv.py` and
`moe_c1_pin.py` retain the snapshot plugins under the explicit plugin names
`q38_shared_down_tgv,q38_moe_c1_pin`. The plugin loader recognizes these names
from a source checkout without requiring copied dist-info or an installation.
They are not automatically registered when SGLANG_PLUGINS is unset.

Set Q38_MOE_C1_PIN_FILE to this directory's c1-pins.json and
SGLANG_FLASHINFER_AUTOTUNE_CACHE=1, as the snapshot launcher helper did. This
preserves the three-key reproducibility policy, not an optimal-tactic claim.
The original nvidia_megakernel plugin remains an independent package at
YAMY1234/sgl-nvidia-mega-fork@5474b2cdbc61; no new companion fork is necessary.
Do not enable its Qwen3.5 engine for FN.

The frozen serving image supplies the Rust mem_cache extension. Run
`bash python/sglang/srt/q38/prepare_runtime.sh` inside that image before serving
if the checkout lacks it. The extension is an ignored runtime dependency, never a
git artifact.

SGLANG_Q38C_HC_CLUSTER_M4ONLY defaults to 0: both ordinary and shared HC
retain the frozen cluster2 path at every admitted row count. With 1, actual
M4 uses cluster2 while M8/M12 and all other admitted non-M4 rows use the native
160-thread single-CTA-per-part implementation. The gate is resolved once into
the JIT specialization; it does not add per-forward environment reads.
