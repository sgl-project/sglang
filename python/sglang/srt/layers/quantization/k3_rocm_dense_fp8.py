"""Static per-tensor FP8 (amax / 448 weight scale, unit activation scale) for
Kimi-K3 BF16 dense linears on ROCm; decode GEMMs there are weight-bandwidth bound.

``SGLANG_K3_TARGET_DENSE_FP8``: ``off`` | ``front`` (merged MoE front) |
``wide`` (front + KDA q/k/v/g).
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max


def scope() -> str:
    value = envs.SGLANG_K3_TARGET_DENSE_FP8.get().lower()
    if value not in ("off", "front", "wide"):
        raise ValueError(
            f"SGLANG_K3_TARGET_DENSE_FP8={value!r}; expected off|front|wide"
        )
    return value


def front_enabled() -> bool:
    return scope() in ("front", "wide")


def qkvg_enabled() -> bool:
    return scope() == "wide"


@torch.no_grad()
def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16/fp16 -> ([N, K] E4M3, scalar fp32 scale) with scale = amax / 448."""
    w = weight.float()
    amax = w.abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX).reshape(())
    q = (w / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()
    return q, scale.to(torch.float32)


# One-slot producer -> consumer handoff for a pre-quantized activation. A kernel
# that already holds the normalized row (attn_res aggregation) can emit its E4M3
# copy; the next StaticFP8Weight call consumes it only if it receives that very
# tensor (or a same-sized view of it; see _same_data), so a stale or reused buffer
# can never be picked up. Anything else falls back to quantizing in the GEMM wrapper.
_prequant_slot: tuple[torch.Tensor, torch.Tensor] | None = None


def offer_prequantized(x: torch.Tensor, xq: torch.Tensor) -> None:
    global _prequant_slot
    _prequant_slot = (x, xq)


def _same_data(src: torch.Tensor, x: torch.Tensor) -> bool:
    # The slot holds a strong reference to src, so a tensor with the same
    # address, size and dtype must alias its data.
    return x is src or (
        x.dtype == src.dtype
        and x.numel() == src.numel()
        and x.data_ptr() == src.data_ptr()
        and x.is_contiguous()
        and src.is_contiguous()
    )


def _take_prequantized(x: torch.Tensor) -> torch.Tensor | None:
    global _prequant_slot
    slot, _prequant_slot = _prequant_slot, None
    if slot is not None and _same_data(slot[0], x):
        return slot[1]
    return None


class StaticFP8Weight:
    """E4M3 [N, K] weight for aiter.gemm_a8w8_bpreshuffle; ``__call__`` is x @ W^T."""

    _MAX_TOKENS = 1 << 15

    def __init__(self, weight: torch.Tensor):
        from aiter.ops.shuffle import shuffle_weight

        q, scale = quantize_weight(weight)
        self.n, self.k = q.shape
        # Only the pre-shuffled copy is kept; module-weight views that alias it are
        # placeholders (their layers are routed here and never read them directly).
        self.weight = shuffle_weight(q, layout=(16, 16))
        del q
        self.w_scale = scale.reshape(1, 1).expand(1, self.n).contiguous()
        self.unit = torch.ones(1, dtype=torch.float32, device=weight.device)
        self.x_scale = torch.ones(
            self._MAX_TOKENS, 1, dtype=torch.float32, device=weight.device
        )

    def __call__(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        import aiter

        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        dtype = out_dtype or x.dtype
        if dtype not in (torch.bfloat16, torch.float16) or m > self._MAX_TOKENS:
            raise ValueError(
                f"K3 ROCm FP8 linear: unsupported out dtype {dtype} or M={m}"
            )
        if m == 0:
            y = x2.new_empty((0, self.n), dtype=dtype)
        else:
            xq = _take_prequantized(x)
            if xq is None:
                xq = torch.empty(x2.shape, dtype=_FP8, device=x2.device)
                aiter.static_per_tensor_quant(xq, x2.contiguous(), self.unit)
            else:
                xq = xq.reshape(x2.shape)
            y = aiter.gemm_a8w8_bpreshuffle(
                xq, self.weight, self.x_scale[:m], self.w_scale, dtype=dtype
            )
        return y.reshape(*x.shape[:-1], self.n)


class StaticFP8LinearMethod:
    """Drop-in ``quant_method`` for an already-loaded unquantized linear layer."""

    def __init__(self, fp8: StaticFP8Weight):
        self.fp8 = fp8

    def apply(
        self, layer, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.fp8(x)
        return y if bias is None else y + bias

    def process_weights_after_loading(self, layer) -> None:  # already converted
        return None


@torch.no_grad()
def convert_linear(layer: torch.nn.Module) -> StaticFP8Weight:
    """Convert ``layer.weight`` in place and route the layer through FP8."""
    fp8 = StaticFP8Weight(layer.weight.data)
    layer.weight.data = fp8.weight  # drop the bf16 tensor; nothing else reads it
    layer.quant_method = StaticFP8LinearMethod(fp8)
    return fp8
