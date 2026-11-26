"""Static per-tensor FP8 (amax / 448 weight scale, unit activation scale) for
Kimi-K3 BF16 dense linears on ROCm; decode GEMMs there are weight-bandwidth bound.

``SGLANG_K3_TARGET_DENSE_FP8``: ``off`` | ``qkvg`` (KDA q/k/v/g) | ``front``
(merged MoE front) | ``wide`` (front + KDA q/k/v/g). The front scopes replace the
fp32 front (exact router logits) with a bf16-output FP8 GEMM, so they are opt-in.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max


def scope() -> str:
    value = envs.SGLANG_K3_TARGET_DENSE_FP8.get().lower()
    if value not in ("off", "qkvg", "front", "wide"):
        raise ValueError(
            f"SGLANG_K3_TARGET_DENSE_FP8={value!r}; expected off|qkvg|front|wide"
        )
    return value


def front_enabled() -> bool:
    return scope() in ("front", "wide")


def qkvg_enabled() -> bool:
    return scope() in ("qkvg", "wide")


@torch.no_grad()
def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16/fp16 -> ([N, K] E4M3, scalar fp32 scale) with scale = amax / 448."""
    w = weight.float()
    amax = w.abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX).reshape(())
    q = (w / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()
    return q, scale.to(torch.float32)


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
        dtype = out_dtype or x.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"K3 ROCm FP8 linear: unsupported out dtype {dtype}")
        y = x2.new_empty((x2.shape[0], self.n), dtype=dtype)
        # x_scale is sized for one chunk; larger prefill chunks run in pieces.
        for start in range(0, x2.shape[0], self._MAX_TOKENS):
            xc = x2[start : start + self._MAX_TOKENS].contiguous()
            xq = torch.empty(xc.shape, dtype=_FP8, device=xc.device)
            aiter.static_per_tensor_quant(xq, xc, self.unit)
            y[start : start + xc.shape[0]] = aiter.gemm_a8w8_bpreshuffle(
                xq, self.weight, self.x_scale[: xc.shape[0]], self.w_scale, dtype=dtype
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
