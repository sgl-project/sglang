# Mixture-of-LoRA Route Decode Patch

This branch vendors the previously PYTHONPATH-based MoL route-decode patch into
this SGLang checkout.

## What changed

- `python/apply_mol_patch/` contains the runtime route-decode patch package.
- `python/sglang/__init__.py` applies the patch at import time.
- `python/sglang/srt/managers/io_struct.py` defines the MoL router request and
  response dataclasses used by the patch.

Set `SGLANG_DISABLE_MOL_PATCH=1` to import plain SGLang without installing the
MoL runtime hooks.

## Launch shape

Use this checkout directly on `PYTHONPATH`; do not prepend the old external
overlay path:

```bash
export PYTHONPATH=/path/to/sglang-v0.5.13-pr/python
python3 -m sglang.launch_server \
  --model-path <model> \
  --served-model-name zai-org/glm-5.1 \
  --tp 8 \
  --enable-lora \
  --lora-paths l0_chat=<L0> l1_specialist=<L1> \
  --max-loras-per-batch 2 \
  --max-lora-rank 16 \
  --lora-use-virtual-experts \
  --disable-overlap-schedule \
  --disable-cuda-graph
```

`--disable-overlap-schedule` is required for the route-decode KV trim/requeue
path. CUDA graph is commonly disabled in our current GLM-5.1 multi-LoRA tests
because clean direct multi-LoRA serving showed CUDA-graph-related corruption.
