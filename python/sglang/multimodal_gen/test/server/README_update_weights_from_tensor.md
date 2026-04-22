# Diffusion `update_weights_from_tensor` README

This document describes the tensor-based in-place weight update flow for diffusion models in `sglang.multimodal_gen`.

## Endpoint

- `POST /update_weights_from_tensor`

## Request Schema

- `serialized_named_tensors: List[Union[str, bytes]]` (required)
- `load_format: Optional[str]` (`None`/`direct` or `"flattened_bucket"`)
- `target_modules: Optional[List[str]]` (for example: `["transformer"]`, `["transformer", "vae"]`)
- `weight_version: Optional[str]`


## TP Payload Rules

- `len(serialized_named_tensors)` must be either:
  - `1`, or
  - `tp_size`
- If length is `tp_size`, each TP rank consumes the payload at its own index.
- If length is `1`, all TP ranks consume index `0`.

## Module Payload Rules

- Single-module update (`target_modules` has one module):
  - Payload can be passed directly.
- Multi-module update:
  - Payload must be a dict keyed by module name:

```python
{
  "transformer": <module_payload>,
  "vae": <module_payload>,
}
```

## Supported Module Payload Formats

- `load_format=None` (or `direct`-style payload):
  - `[(param_name, tensor), ...]`
- `load_format="flattened_bucket"`:
  - `{"flattened_tensor": tensor, "metadata": [...]}`

## Safety Semantics

- Only selected modules are updated (`target_modules` aware).
- Update is all-or-nothing across requested modules:
  - on failure, already-updated modules are rolled back to previous disk weights.
- TP synchronization barrier is used in scheduler path to avoid mixed-rank model state.
