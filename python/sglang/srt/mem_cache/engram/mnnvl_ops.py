"""Row lookup boundary for the Mooncake MNNVL Engram backend.

The custom op only moves immutable byte rows into the model's gather. Hashing,
quantization interpretation, scales, masking and collectives stay in the
model.
"""

from __future__ import annotations

import torch

_adapters = {}
_operation = None


def install_ops():
    global _operation
    if _operation is not None:
        return

    @torch.library.custom_op("sglang::engram_mnnvl_lookup", mutates_args=("out",))
    def operation(adapter_id: int, ids: torch.Tensor, out: torch.Tensor) -> None:
        dispatch(_adapters[adapter_id], ids, out)

    @operation.register_fake
    def fake(adapter_id: int, ids: torch.Tensor, out: torch.Tensor) -> None:
        return None

    _operation = operation


def dispatch(adapter, ids, out):
    if not ids.is_cuda or not out.is_cuda or ids.device != out.device:
        raise ValueError(
            "Engram MNNVL supports only CUDA IDs/output on the same device"
        )
    if adapter.state.closed:
        raise RuntimeError("Engram MNNVL backend is closed")
    module = adapter.module
    weight = adapter.fetch("weight", ids)
    if adapter.kind == "sglang_deepseek":
        from sglang.srt.layers.engram import FP8_BLOCK_SIZE, engram_gather

        scale = adapter.fetch("scale", ids)
        positions = torch.arange(ids.numel(), dtype=torch.int64, device=ids.device)
        engram_gather(
            weight.data_ptr(),
            scale.data_ptr(),
            positions,
            out.reshape(-1, module.dim),
            module.dim,
            FP8_BLOCK_SIZE,
            mxfp4=module.mxfp4,
        )
    elif adapter.kind == "sglang_qwen":
        values = weight.view(adapter.planes["weight"]["dtype"])
        out.copy_(values.to(torch.bfloat16))
    else:
        raise RuntimeError("Unknown Engram MNNVL adapter")


def owned_rows(module, ids):
    out = module.new_rows(ids)
    _operation(module._engram_mnnvl_id, ids, out)
    return out


def gather(module, input_ids, out=None):
    shape = (*input_ids.shape, module.embedding_dim)
    if out is None:
        out = module.allocate_output(shape, input_ids.device)
    if (
        tuple(out.shape) != shape
        or out.dtype != torch.bfloat16
        or out.device != input_ids.device
    ):
        raise ValueError("SGLang PLE output layout mismatch")
    _operation(module._engram_mnnvl_id, input_ids, out)
    return out


def bind(adapter):
    adapter.op_id = id(adapter)
    _adapters[adapter.op_id] = adapter
    module = adapter.module
    module._engram_mnnvl_id = adapter.op_id
    if adapter.kind == "sglang_deepseek":
        module._engram_row_source = lambda ids: owned_rows(module, ids)
    elif adapter.kind == "sglang_qwen":
        module._engram_gather_source = lambda ids, out=None: gather(
            module, ids, out
        )
    else:
        raise RuntimeError("Unknown Engram MNNVL adapter")


def forget(adapters):
    for adapter in adapters:
        _adapters.pop(adapter.op_id, None)
        module = adapter.module
        if adapter.kind == "sglang_deepseek":
            module._engram_row_source = None
        elif adapter.kind == "sglang_qwen":
            module._engram_gather_source = None
