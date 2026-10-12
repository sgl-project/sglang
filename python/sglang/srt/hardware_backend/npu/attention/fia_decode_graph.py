"""Capture adapter for isolated FIA custom op; requires the inspected TorchNPU registry.

Import after npu_fia_decode_custom, before compiling/capturing. No native handler
is replaced. During capture only, lower this custom call to native FIA.out so
existing task-update bindings remain visible. Hardware validation is mandatory.
"""

import torch
import torch_npu
from torch_npu.npu._npugraph_handlers.ifa_handler import _IFAv1DefaultHandler
from torch_npu.npu._npugraph_handlers.npugraph_handler import register_npu_graph_handler
from sglang.srt.hardware_backend.npu.attention.fia_decode import fia_decode_bsh  # noqa: F401 - register custom schema before accessing torch.ops


_PARAMETER_NAMES = (
    "query",
    "key_cache",
    "value_cache",
    "block_table",
    "block_size",
    "num_heads",
    "num_key_value_heads",
    "scale",
    "actual_seq_lengths_kv",
    "atten_mask",
    "sparse_mode",
    "output_hidden_size",
)


def _native_arguments(args, kwargs):
    if len(args) > len(_PARAMETER_NAMES):
        raise ValueError("Unexpected FIA custom-op positional argument count")
    values = dict(zip(_PARAMETER_NAMES, args))
    for name, value in (kwargs or {}).items():
        if name not in _PARAMETER_NAMES or name in values:
            raise ValueError(f"Unexpected or duplicate FIA argument: {name}")
        values[name] = value
    missing = set(_PARAMETER_NAMES) - values.keys()
    if missing:
        raise ValueError(f"Missing FIA arguments: {sorted(missing)}")
    native_args = (values["query"], values["key_cache"], values["value_cache"])
    native_kwargs = {
        name: values[name]
        for name in (
            "block_table",
            "block_size",
            "num_heads",
            "num_key_value_heads",
            "scale",
            "actual_seq_lengths_kv",
            "atten_mask",
            "sparse_mode",
        )
    }
    native_kwargs["input_layout"] = "BSH"
    return native_args, native_kwargs, values["output_hidden_size"]


# Registry keys are func.__name__, not dispatcher namespace-qualified schema names.
_NATIVE_CUSTOM_NAME = torch.ops.sglang_npu.fia_decode_bsh.default.__name__


@register_npu_graph_handler(["fia_decode_bsh", _NATIVE_CUSTOM_NAME])
class FIADecodeBSHCaptureHandler(_IFAv1DefaultHandler):
    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        native_args, native_kwargs, width = _native_arguments(args, kwargs)
        # Reuse the actual installed IFA handler's workspace + output allocation.
        # Its returned func is native FIA.out; the registry records that native
        # name, so replay resolves the original IFA handler, not this adapter.
        actual_func, actual_args, actual_kwargs = _IFAv1DefaultHandler.prepare_capture(
            torch_npu.npu_fused_infer_attention_score,
            native_args,
            native_kwargs,
        )
        output = actual_kwargs["out"][0]
        if tuple(output.shape) != (native_args[0].shape[0], 1, width):
            raise ValueError(
                f"Native FIA output shape disagrees with custom fake: {output.shape}"
            )
        if (
            output.dtype != native_args[0].dtype
            or output.device != native_args[0].device
        ):
            raise ValueError(
                "Native FIA output dtype/device disagrees with custom fake"
            )
        return actual_func, actual_args, actual_kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        # Custom schema returns one Tensor, whereas native FIA returns TensorList.
        return kwargs["out"][0]
