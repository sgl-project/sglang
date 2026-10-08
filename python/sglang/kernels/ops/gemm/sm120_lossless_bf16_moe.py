"""Lossless BF16 storage for SM120 small-M, non-gated ReLU-squared MoE.

Every routed expert is evaluated with the original BF16 bits and tensor-core
accumulation order. Original weights remain available for larger batches.
"""

import logging
import weakref

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import _pack_bf16

logger = logging.getLogger(__name__)


@cache_once
def _module():
    return load_jit(
        "sm120_lossless_bf16_moe",
        cuda_files=["gemm/sm120_lossless_bf16_moe.cuh"],
        cuda_wrappers=[
            ("run", "sglang::LosslessBf16Moe::run"),
            ("restore", "sglang::LosslessBf16Moe::restore"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def prepare_lossless_bf16_moe(layer):
    from sglang.srt.environ import envs

    if not envs.SGLANG_SM120_LOSSLESS_BF16_MOE.get():
        return
    w1, w2 = layer.w13_weight, layer.w2_weight
    config = layer.moe_runner_config
    if (
        w1.shape != (512, 2688, 1024)
        or w2.shape != (512, 1024, 2688)
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or not w1.is_contiguous()
        or not w2.is_contiguous()
        or w1.device.type != "cuda"
        or w1.device != w2.device
        or torch.cuda.get_device_capability(w1.device) != (12, 0)
        or config.is_gated
        or config.activation != "relu2"
        or layer.moe_ep_size != 1
        or layer.moe_tp_size != 1
        or getattr(layer, "w13_weight_bias", None) is not None
        or getattr(layer, "w2_weight_bias", None) is not None
    ):
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Lossless expert preparation must precede graph capture")
    buffers = []
    used_bytes = 0
    for name, weight, padded_k in (("w13", w1, 1024), ("w2", w2, 2816)):
        experts, n, k = weight.shape
        specs = (
            (f"_lossless_{name}_payload", experts * n * padded_k // 2, torch.uint32),
            (f"_lossless_{name}_metadata", experts * n * padded_k // 256, torch.uint64),
        )
        for buffer_name, size, dtype in specs:
            old = getattr(layer, buffer_name, None)
            if old is None or old.shape != (size,) or old.device != weight.device:
                layer.register_buffer(
                    buffer_name,
                    torch.empty(size, dtype=dtype, device=weight.device),
                    persistent=False,
                )
        payload, metadata = (getattr(layer, entry[0]) for entry in specs)
        restored = torch.empty((n, padded_k), dtype=weight.dtype, device=weight.device)
        offset = 0
        for expert in range(experts):
            original = weight[expert].detach().cpu()
            if padded_k != k:
                original = torch.nn.functional.pad(original, (0, padded_k - k))
            data, meta = _pack_bf16(original)
            # The low 32 bits contain uint32-word offsets; the high bits encode
            # the original exponent base and whether the block uses raw bits.
            meta.view(torch.int64).add_(offset)
            payload[offset : offset + data.numel()].copy_(data)
            begin = expert * n * padded_k // 256
            expert_meta = metadata[begin : begin + meta.numel()]
            expert_meta.copy_(meta)
            _module().restore(payload, expert_meta, restored)
            if not torch.equal(
                restored.cpu().view(torch.int16), original.view(torch.int16)
            ):
                raise RuntimeError(
                    f"Lossless BF16 bit audit failed: {name} expert {expert}"
                )
            offset += data.numel()
        used_bytes += offset * 4 + metadata.numel() * 8
        buffers.extend((payload, metadata))
    layer._lossless_bf16_moe_sources = (weakref.ref(w1), weakref.ref(w2))
    layer._lossless_bf16_moe_ptrs = (w1.data_ptr(), w2.data_ptr())
    layer._lossless_bf16_moe_buffers = tuple(buffers)
    logger.info(
        "Prepared lossless BF16 experts: stream bytes=%d; all bits verified", used_bytes
    )


def lossless_bf16_moe_buffers(layer):
    sources = getattr(layer, "_lossless_bf16_moe_sources", None)
    if sources is None:
        return None
    if sources[0]() is not layer.w13_weight or sources[1]() is not layer.w2_weight:
        return None
    if layer._lossless_bf16_moe_ptrs != (
        layer.w13_weight.data_ptr(),
        layer.w2_weight.data_ptr(),
    ):
        return None
    return layer._lossless_bf16_moe_buffers


def lossless_bf16_moe(x, buffers, ids, weights, *, out=None):
    rows = x.shape[0]
    intermediate = torch.empty((rows * 22, 2816), dtype=x.dtype, device=x.device)
    partial = torch.empty((rows * 22, 1024), dtype=x.dtype, device=x.device)
    if out is None:
        out = torch.empty_like(x)
    _module().run(x, *buffers, ids, weights, intermediate, partial, out)
    return out
