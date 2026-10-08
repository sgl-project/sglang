"""Lossless M4 input projection with the original two-part BF16 reduction."""

import logging
import weakref

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import _module, _pack_bf16
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@cache_once
def _inproj_module():
    return load_jit(
        "sm120_lossless_bf16_inproj",
        cuda_files=["gemm/sm120_lossless_bf16_inproj.cuh"],
        cuda_wrappers=[("run", "sglang::LosslessInprojTc4Split<18560,4096,2>::run")],
        extra_cuda_cflags=["-O3"],
    )


def prepare_lossless_bf16_inproj(layer):
    weight = layer.weight
    if not (
        envs.SGLANG_SM120_LOSSLESS_BF16_INPROJ.get()
        and weight.shape == (18560, 4096)
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and not weight.requires_grad
        and torch.cuda.get_device_capability(weight.device) == (12, 0)
    ):
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare lossless input projection before graph capture")
    payload, metadata = _pack_bf16(weight)
    for name, size, dtype in (
        ("_lossless_inproj_payload", weight.numel() // 2, torch.uint32),
        ("_lossless_inproj_metadata", weight.numel() // 256, torch.uint64),
    ):
        old = getattr(layer, name, None)
        if old is None or old.shape != (size,) or old.device != weight.device:
            layer.register_buffer(
                name,
                torch.empty(size, dtype=dtype, device=weight.device),
                persistent=False,
            )
    layer._lossless_inproj_payload[: payload.numel()].copy_(payload)
    layer._lossless_inproj_metadata.copy_(metadata)
    restored = torch.empty_like(weight)
    _module(*weight.shape).restore(
        layer._lossless_inproj_payload, layer._lossless_inproj_metadata, restored
    )
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Lossless input projection failed the full BF16 bit audit")
    layer._lossless_inproj_source = weakref.ref(weight)
    logger.info("Prepared lossless BF16 input projection: every weight bit verified")


def try_lossless_bf16_inproj(x, layer, bias=None):
    source = getattr(layer, "_lossless_inproj_source", None)
    if not (
        envs.SGLANG_SM120_LOSSLESS_BF16_INPROJ.get()
        and source is not None
        and source() is layer.weight
        and x.shape == (4, 4096)
        and x.dtype == torch.bfloat16
        and x.device == layer.weight.device
        and x.is_contiguous()
        and bias is None
        and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    ):
        return None
    out = torch.empty((4, 18560), dtype=x.dtype, device=x.device)
    partial = torch.empty((2, 4, 18560), dtype=torch.float32, device=x.device)
    _inproj_module().run(
        x, layer._lossless_inproj_payload, layer._lossless_inproj_metadata, out, partial
    )
    return out
