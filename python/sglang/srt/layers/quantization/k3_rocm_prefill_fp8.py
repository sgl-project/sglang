"""Prefill-only FP8 copies of Kimi-K3's dense BF16 linears on ROCm.

``SGLANG_K3_TARGET_DENSE_FP8`` (k3_rocm_dense_fp8) replaces weights with static
FP8 for every batch size. This module keeps the BF16 weights, which decode and
the AMD monokernel read, and adds a per-channel E4M3 copy of each selected
layer next to them. A call with at least the group's row threshold quantizes
the activation per token and runs ``aiter.gemm_a8w8_bpreshuffle``; every other
call takes the stock BF16 path.

``SGLANG_K3_PREFILL_DENSE_FP8``: comma list of groups (empty = off; nothing is
built). ``SGLANG_K3_PREFILL_DENSE_FP8_MIN_TOKENS``: row threshold for every
group, raised per group by ``GROUP_MIN_TOKENS``.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

# Rows per GEMM call; rows are independent. Above 16384 rows, AITER's tuned
# bpreshuffle shapes have given run-to-run different results on gfx950.
MAX_ROWS_PER_CALL = 1 << 14

# Per-rank (N, K) of each group at TP8; a layer of another shape is left BF16.
GROUP_SHAPES = {
    "in_proj": (6144, 7168),
    "o_proj": (7168, 1536),
    "sdown": (7168, 768),
    "up": (7168, 3584),
    "qkv_a": (2112, 7168),
    "g_proj": (1536, 7168),
    "q_b": (2304, 1536),
    "kv_b": (3072, 512),
}

# Below these row counts the FP8 call (activation quant + GEMM) measured slower
# than the BF16 GEMM on MI355X.
GROUP_MIN_TOKENS = {
    "o_proj": 1536,
    "sdown": 4096,
    "g_proj": 4096,
    "q_b": 8192,
    "kv_b": 14336,
}

# (group, attention attribute) of the MLA projections; KDA layers have none.
_MLA_GROUPS = (
    ("qkv_a", "fused_qkv_a_proj_with_mqa"),
    ("g_proj", "g_proj"),
    ("q_b", "q_b_proj"),
    ("kv_b", "kv_b_proj"),
)


def parse_groups(value: str) -> tuple[str, ...]:
    """Groups named by ``value`` in order, without duplicates."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    for name in names:
        if name == "front":
            raise ValueError(
                "SGLANG_K3_PREFILL_DENSE_FP8: the MoE front is served by "
                "SGLANG_K3_TARGET_DENSE_FP8=front"
            )
        if name not in GROUP_SHAPES:
            raise ValueError(
                f"SGLANG_K3_PREFILL_DENSE_FP8: {name!r} is not one of "
                f"{list(GROUP_SHAPES)}"
            )
    return tuple(dict.fromkeys(names))


def group_min_tokens(group: str, base_min_tokens: int) -> int:
    return max(base_min_tokens, GROUP_MIN_TOKENS.get(group, 0))


def row_chunks(m: int, rows: Optional[int] = None) -> list[tuple[int, int]]:
    """(start, end) of each GEMM call for an M-row batch."""
    rows = rows or MAX_ROWS_PER_CALL
    return [(start, min(start + rows, m)) for start in range(0, m, rows)]


@torch.no_grad()
def quantize_per_channel(
    weight: torch.Tensor, fp8_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> ([N, K] FP8, [N, 1] fp32 scale) with scale = row amax / FP8 max."""
    fp8_max = torch.finfo(fp8_dtype).max
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / fp8_max).clamp(min=1e-12)
    q = (w / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale


class PrefillFP8Weight:
    """The FP8 copy of one BF16 linear weight and the checks that admit a call."""

    def __init__(self, group: str, weight: torch.Tensor, min_tokens: int):
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        self.group = group
        self.n, self.k = weight.shape
        self.min_tokens = min_tokens
        # A copy is only valid while the layer still holds the tensor it came from.
        self.source_ptr = weight.data_ptr()
        q, scale = quantize_per_channel(weight, aiter.dtypes.fp8)
        self.weight = shuffle_weight(q, layout=(16, 16))
        self.w_scale = scale.t().contiguous()  # [1, N]

    @property
    def nbytes(self) -> int:
        return (
            self.weight.numel() * self.weight.element_size()
            + self.w_scale.numel() * self.w_scale.element_size()
        )

    def admits(self, layer: torch.nn.Module, x: torch.Tensor, bias) -> bool:
        return (
            bias is None
            and not torch.compiler.is_compiling()
            and x.ndim == 2
            and x.shape[0] >= self.min_tokens
            and x.shape[1] == self.k
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and layer.weight.data_ptr() == self.source_ptr
        )

    def run(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        import aiter

        xq, x_scale = aiter.per_token_quant_hip(
            x.contiguous(), quant_dtype=aiter.dtypes.fp8
        )
        chunks = row_chunks(x.shape[0])
        if out is None and len(chunks) == 1:
            return self._gemm(xq, x_scale)
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        for start, end in chunks:
            out[start:end].copy_(self._gemm(xq[start:end], x_scale[start:end]))
        return out

    def _gemm(self, xq: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        import aiter

        return aiter.gemm_a8w8_bpreshuffle(
            xq, self.weight, x_scale, self.w_scale, dtype=torch.bfloat16
        )


class PrefillFP8LinearMethod(UnquantizedLinearMethod):
    """UnquantizedLinearMethod that runs admitted calls on the FP8 copy."""

    def __init__(self, fp8: PrefillFP8Weight):
        super().__init__()
        self.fp8 = fp8

    def apply(self, layer, x, bias=None):
        if self.fp8.admits(layer, x, bias):
            return self.fp8.run(x)
        return super().apply(layer, x, bias)

    def apply_raw_into(self, layer, x: torch.Tensor, output: torch.Tensor) -> bool:
        """FP8 GEMM into ``output`` for a call site that reads the raw weight
        (the shared down-projection). False: the caller runs BF16."""
        if (
            self.fp8.admits(layer, x, None)
            and output.dtype == torch.bfloat16
            and output.shape == (x.shape[0], self.fp8.n)
        ):
            self.fp8.run(x, output)
            return True
        return False


def iter_targets(model, groups: tuple[str, ...]) -> Iterator[tuple]:
    """(group, name, module, moe) of each selected layer; ``moe`` is set for
    the shared down-projection, whose call site lives in ``KimiK3MoE``."""
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:  # PPMissingLayer
            continue
        if "in_proj" in groups and getattr(attn, "fused_qkvg_proj", None) is not None:
            yield (
                "in_proj",
                f"layers.{i}.self_attn.fused_qkvg_proj",
                attn.fused_qkvg_proj,
                None,
            )
        if "o_proj" in groups and getattr(attn, "o_proj", None) is not None:
            yield "o_proj", f"layers.{i}.self_attn.o_proj", attn.o_proj, None
        for group, attr in _MLA_GROUPS:
            if group in groups and getattr(attn, attr, None) is not None:
                yield group, f"layers.{i}.self_attn.{attr}", getattr(attn, attr), None
        moe = getattr(layer, "mlp", None)
        up = getattr(moe, "routed_expert_up_proj", None)
        if "up" in groups and up is not None:
            yield "up", f"layers.{i}.mlp.routed_expert_up_proj", up, None
        shared = getattr(moe, "shared_experts", None)
        if "sdown" in groups and getattr(shared, "down_proj", None) is not None:
            yield (
                "sdown",
                f"layers.{i}.mlp.shared_experts.down_proj",
                shared.down_proj,
                moe,
            )


def is_plain_bf16_linear(module: torch.nn.Module, group: str) -> bool:
    """Only an unquantized, contiguous BF16 weight of the group's shape; a layer
    that SGLANG_K3_TARGET_DENSE_FP8 already converted is FP8 and fails this."""
    weight = getattr(module, "weight", None)
    data = None if weight is None else weight.data
    return (
        type(getattr(module, "quant_method", None)) is UnquantizedLinearMethod
        and type(data) is torch.Tensor
        and data.dtype == torch.bfloat16
        and tuple(data.shape) == GROUP_SHAPES[group]
        and data.is_contiguous()
    )


def build_prefill_fp8_copies(model) -> None:
    """Build the copies for the selected groups and route their layers. Safe to
    call again: post_load_weights runs from load_weights and loader hooks."""
    groups = parse_groups(envs.SGLANG_K3_PREFILL_DENSE_FP8.get())
    if not groups:
        return
    from sglang.srt.utils import is_hip

    if not is_hip():
        raise RuntimeError("SGLANG_K3_PREFILL_DENSE_FP8 needs ROCm (aiter)")
    base_min_tokens = envs.SGLANG_K3_PREFILL_DENSE_FP8_MIN_TOKENS.get()
    built, skipped, nbytes = 0, [], 0
    for group, name, module, moe in iter_targets(model, groups):
        if isinstance(module.quant_method, PrefillFP8LinearMethod):
            continue
        if not is_plain_bf16_linear(module, group):
            skipped.append(name)
            continue
        fp8 = PrefillFP8Weight(
            group, module.weight.data, group_min_tokens(group, base_min_tokens)
        )
        module.quant_method = PrefillFP8LinearMethod(fp8)
        if moe is not None:
            moe._prefill_fp8_sdown = module.quant_method
        built += 1
        nbytes += fp8.nbytes
    logger.info(
        "Kimi-K3 prefill FP8 copies: groups=%s layers=%d, %.2f GiB",
        ",".join(groups),
        built,
        nbytes / (1 << 30),
    )
    if skipped:
        logger.info("Kimi-K3 prefill FP8 copies: left BF16: %s", skipped)
