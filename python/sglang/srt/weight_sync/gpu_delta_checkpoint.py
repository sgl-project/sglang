"""Observe canonical checkpoint metadata without changing weight values.

Installed at the common iterator-based loader boundary, not in model classes.
This records source metadata only; GPU delta layout admission still decides
which model, quantization, aliases, and physical buffers it can safely update.
"""

from __future__ import annotations

import inspect
from functools import wraps

_DTYPE_NAMES = {
    "torch.bfloat16": "BF16",
    "torch.float16": "F16",
    "torch.float32": "F32",
    "torch.float8_e4m3fn": "F8_E4M3",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int32": "I32",
    "torch.int64": "I64",
}


def install_canonical_weight_observer(model, *, is_draft: bool = False):
    """Wrap one model's loader once, including its later direct reload calls.

    Draft construction is excluded by the existing draft build scope. Explicit
    ``is_nextn`` calls on a target retain the model loader's existing behavior.
    The observer never reads tensor values, copies tensors, or synchronizes a
    device. Unsupported dtypes and duplicate names are checked at delta
    admission, not during ordinary loading of the ``(name, tensor)`` iterator.
    """
    if is_draft:
        # Later reloads need not run inside the construction-only draft scope.
        model._gpu_delta_metadata_is_draft = True
    original = model.load_weights
    if getattr(model, "_gpu_delta_metadata_is_draft", False) or getattr(
        original, "_gpu_delta_metadata_observer", False
    ):
        return
    signature = inspect.signature(original)
    nextn = signature.parameters.get("is_nextn")

    @wraps(original)
    def load_weights(weights, *args, **kwargs):
        if nextn is not None:
            arguments = signature.bind_partial(weights, *args, **kwargs).arguments
            if arguments.get("is_nextn", nextn.default):
                return original(weights, *args, **kwargs)

        model._gpu_delta_load_generation = (
            getattr(model, "_gpu_delta_load_generation", 0) + 1
        )
        inventory = {}
        model._gpu_delta_canonical_inventory = inventory
        model._gpu_delta_duplicate_source_names = False
        model._gpu_delta_metadata_complete = False
        exhausted = False

        def observe():
            nonlocal exhausted
            for name, tensor in weights:
                dtype = str(tensor.dtype)
                if name in inventory:
                    model._gpu_delta_duplicate_source_names = True
                inventory[name] = {
                    "shape": list(tensor.shape),
                    "dtype": _DTYPE_NAMES.get(dtype, dtype),
                }
                yield name, tensor
            exhausted = True

        result = original(observe(), *args, **kwargs)
        model._gpu_delta_metadata_complete = exhausted
        return result

    load_weights._gpu_delta_metadata_observer = True
    model.load_weights = load_weights
