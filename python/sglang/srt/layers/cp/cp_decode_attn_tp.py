"""CP Decode Attention TP context.

When CP (Context Parallel) mode sets tp_size=1 (repeat weights), decode can
partition attention weights across CP ranks matching normal TP behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.runtime_context import get_device, get_exec, get_lora, get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# HF architectures whose attention linears are replicated with tp_size=1 under
# context parallelism, so slicing them to the local CP partition during decode is
# equivalent to a normal TP-layout GEMM. Any model not on this list must not use
# CP decode attention TP. Single arch-string source of truth for the whitelist.
CP_DECODE_ATTN_TP_SUPPORTED_ARCHS: Tuple[str, ...] = (
    # DeepSeek-V4
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
    "DeepseekV4ForCausalLMDSpark",
    # GLM-5.x (inherits DeepseekV2 attention; DSA path)
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
)

_global_cp_decode_attn_tp_ctx: CpDecodeAttnTpContext | None = None


def can_defer_cp_decode_attn_tp() -> bool:
    """Whether a DSA layer may leave its decode sum to its stage boundary.

    This is a construction-time choice, shared by the weight dispatch and the
    boundary declaration. Only the full-TP CP topology is supported for now;
    the other configurations retain the inline reduction.
    """
    parallel = get_parallel()
    return (
        parallel.enable_cp_decode_attn_tp
        and parallel.enable_prefill_cp
        and parallel.attn_cp_size == parallel.tp_size
        and parallel.attn_cp_size > 1
        and parallel.attn_tp_size == 1
        and parallel.attn_dp_size == 1
        and not parallel.enable_layernorm_sp
        and get_device().device == "cuda"
        and not get_lora().enable_lora
        and not get_exec().overlap.enable_two_batch_overlap
        and not get_exec().comm.enable_quant_communications
    )


def get_cp_decode_attn_tp_ctx() -> CpDecodeAttnTpContext:
    """Return the global CpDecodeAttnTpContext singleton."""
    global _global_cp_decode_attn_tp_ctx
    if _global_cp_decode_attn_tp_ctx is None:
        _global_cp_decode_attn_tp_ctx = CpDecodeAttnTpContext()
    return _global_cp_decode_attn_tp_ctx


def configure_cp_decode_attn_tp(linear, *, defer_all_reduce: bool = False) -> None:
    """Validate the output projection before serving and guard its derived cache.

    Clearing the Python shard cache on a weight update is insufficient: captured
    CUDA graphs still reference the old copies. Until updates refresh those
    buffers in place, use the existing model-owned update rejection protocol.
    This is set at construction, not when a rank first populates its cache.
    """
    if defer_all_reduce and linear.reduce_results:
        raise ValueError(
            "CP decode attention TP deferral requires reduce_results=False"
        )
    if get_parallel().enable_cp_decode_attn_tp and get_parallel().attn_cp_size > 1:
        linear._derived_weight_cache_error = (
            "Online weight updates are not supported with CP decode attention TP: "
            "cached weight/scale shards and captured CUDA graphs would retain old "
            "weights. Restart without --enable-cp-decode-attn-tp to update weights."
        )


class CpDecodeAttnTpContext:
    """Slices replicated attention weights across CP ranks during decode."""

    def __init__(self):
        enable_attn_tp = get_parallel().enable_cp_decode_attn_tp

        if enable_attn_tp and get_parallel().attn_cp_size > 1:
            self.decode_tp_rank = get_parallel().attn_cp_rank
            self.decode_tp_size = get_parallel().attn_cp_size
            logger.info("Enable CP decode attention TP")
        else:
            self.decode_tp_rank = None
            self.decode_tp_size = None

        self.use_decode_attn_tp = False
        self._slice_cache: Dict = {}

    @property
    def is_enabled(self) -> bool:
        return self.decode_tp_size is not None and self.decode_tp_size > 1

    def set_decode_attn_tp(self, forward_batch: ForwardBatch):
        if not self.is_enabled:
            self.use_decode_attn_tp = False
            return
        # Skip during prefill context parallel (needs all heads); apply on every
        # other forward, which includes decode.
        self.use_decode_attn_tp = not is_cp_active(
            forward_batch
        ) and not dsa_use_prefill_cp(forward_batch)

    def _slice(
        self, tensor: torch.Tensor, dim: int, *, is_scale: bool = False
    ) -> torch.Tensor:
        if dim not in (0, 1) or tensor.ndim <= dim:
            raise ValueError(
                f"CP decode attn TP: cannot slice {tensor.shape=} at {dim=}"
            )
        if tensor.shape[dim] % self.decode_tp_size:
            raise ValueError(
                f"CP decode attn TP: shape[{dim}]={tensor.shape[dim]} "
                f"not divisible by decode_tp_size={self.decode_tp_size}"
            )
        chunk = tensor.shape[dim] // self.decode_tp_size
        sliced = tensor.narrow(dim, self.decode_tp_rank * chunk, chunk)
        if is_scale and tensor.dtype == torch.int32 and tensor.ndim == 2:
            # DeepGEMM packs four UE8M0 scales per int32 in MN-major order.
            # An N slice retains the full matrix's column stride, while making
            # a K slice contiguous changes it to row-major. Re-layout the shard
            # without unpacking/requantizing; _activate caches this allocation.
            from deep_gemm.utils import get_tma_aligned_size

            aligned_mn = get_tma_aligned_size(sliced.shape[0], sliced.element_size())
            packed = torch.empty_strided(
                (aligned_mn, sliced.shape[1]),
                (1, aligned_mn),
                dtype=sliced.dtype,
                device=sliced.device,
            )[: sliced.shape[0]]
            packed.copy_(sliced)
            return packed
        return sliced if dim == 0 else sliced.contiguous()

    def _slice_aiter_bpreshuffled_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slice an AITER B-preshuffled weight along logical K."""
        from aiter.ops.shuffle import shuffle_weight

        from sglang.srt.layers.quantization.fp8_utils import (
            unshuffle_aiter_fp8_weight,
        )

        logical = unshuffle_aiter_fp8_weight(tensor)
        return shuffle_weight(self._slice(logical, 1), layout=(16, 16))

    # ==================== Unified activate/restore ====================

    def _activate(self, obj, attr_name: str, dim: int):
        """Replace obj.attr_name with its TP-sliced version. No-op if attr is None."""
        tensor = getattr(obj, attr_name, None)
        if tensor is None:
            return
        is_param = isinstance(tensor, torch.nn.Parameter)
        raw = tensor.data if is_param else tensor
        if not isinstance(raw, torch.Tensor) or raw.dim() <= dim:
            raise ValueError(
                f"CP decode attn TP: {type(obj).__name__}.{attr_name} "
                f"is not sliceable along dimension {dim}"
            )

        cache_key = (id(obj), attr_name)
        cache = self._slice_cache.get(cache_key)
        if cache is None:
            if (
                attr_name == "weight"
                and dim == 1
                and getattr(obj, "aiter_bpreshuffled", False)
            ):
                sliced = self._slice_aiter_bpreshuffled_weight(raw)
            else:
                sliced = self._slice(
                    raw, dim, is_scale=attr_name in ("weight_scale_inv", "weight_scale")
                )
            # Cache shards to avoid recomputation and keep CUDA graph pointers stable.
            cache = (raw, sliced, is_param)
            self._slice_cache[cache_key] = cache

        if cache[2]:
            tensor.data = cache[1]
        else:
            setattr(obj, attr_name, cache[1])

    def _restore(self, obj, attr_name: str):
        cache = self._slice_cache.get((id(obj), attr_name))
        if cache is None:
            return
        orig, _, is_param = cache
        if is_param:
            getattr(obj, attr_name).data = orig
        else:
            setattr(obj, attr_name, orig)

    # ==================== Linear helpers ====================

    def _get_linear_attrs(self, linear_instance) -> List[Tuple]:
        """Return (obj, attr_name, dim) list for a linear layer."""
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        if isinstance(linear_instance, RowParallelLinear):
            dim = 1
        elif isinstance(linear_instance, ColumnParallelLinear):
            dim = 0
        else:
            return []

        attrs = [(linear_instance, "weight", dim)]
        for scale_name in ("weight_scale_inv", "weight_scale"):
            scale = getattr(linear_instance, scale_name, None)
            if scale is not None:
                if dim == 1 and scale.dtype == torch.int32 and scale.ndim == 2:
                    # DeepGEMM UE8M0 packs four 128-element K-block scales per
                    # int32. Checking the padded scale shape alone misses a
                    # logical K boundary inside one of these packed groups.
                    if linear_instance.weight.shape[1] % (self.decode_tp_size * 512):
                        raise ValueError(
                            "CP decode attn TP: each packed UE8M0 K shard must "
                            "contain a multiple of 512 weight columns"
                        )
                attrs.append((linear_instance, scale_name, dim))
        return attrs

    # ==================== Context manager ====================

    @contextmanager
    def maybe_use_decode_attn_tp(
        self,
        forward_batch: ForwardBatch,
        modules: list,
        tensor_attrs: List[Tuple] = None,
        radix_attn: Optional[RadixAttention] = None,
        defer_all_reduce: bool = False,
    ):
        """Activate decode attention TP for the duration of the block.

        Args:
            modules: Linear layers (ColumnParallel/RowParallel) to slice.
            tensor_attrs: (obj, attr_name, dim) tuples for absorbed weights.
            radix_attn: RadixAttention instance whose tp_q_head_num should be
                overridden to match the sliced head count during decode TP.
            defer_all_reduce: The caller's boundary declares and completes the
                CP-rank sum. Otherwise keep the existing inline row reduction.
        """
        self.set_decode_attn_tp(forward_batch)
        if not self.use_decode_attn_tp:
            yield
            return

        all_attrs = []  # (obj, attr_name) pairs to restore
        size_overrides = []  # (linear, size_attr, orig_size)
        row_parallel_decode_flags = []  # (RowParallelLinear, orig_flag) to restore
        orig_tp_q_head_num = None
        try:
            for linear in modules:
                for obj, attr_name, dim in self._get_linear_attrs(linear):
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))
                from sglang.srt.layers.linear import RowParallelLinear

                size_attr = (
                    "input_size_per_partition"
                    if isinstance(linear, RowParallelLinear)
                    else "output_size_per_partition"
                )
                orig_size = getattr(linear, size_attr)
                setattr(linear, size_attr, orig_size // self.decode_tp_size)
                size_overrides.append((linear, size_attr, orig_size))

                # Set the decode attn TP flag on RowParallelLinear instances
                if isinstance(linear, RowParallelLinear):
                    row_parallel_decode_flags.append(
                        (linear, linear.use_decode_attn_tp)
                    )
                    if defer_all_reduce and linear.reduce_results:
                        raise ValueError(
                            "CP decode attention TP deferral requires reduce_results=False"
                        )
                    # This flag requests the *inline* row sum; deferred sums
                    # are owned by the boundary even though weights are sliced.
                    linear.use_decode_attn_tp = not defer_all_reduce

            if tensor_attrs:
                for obj, attr_name, dim in tensor_attrs:
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))

            if radix_attn is not None:
                orig_tp_q_head_num = radix_attn.tp_q_head_num
                radix_attn.tp_q_head_num = orig_tp_q_head_num // self.decode_tp_size
            yield
        finally:
            if radix_attn is not None and orig_tp_q_head_num is not None:
                radix_attn.tp_q_head_num = orig_tp_q_head_num
            for linear, orig_flag in reversed(row_parallel_decode_flags):
                linear.use_decode_attn_tp = orig_flag
            for linear, size_attr, orig_size in reversed(size_overrides):
                setattr(linear, size_attr, orig_size)
            for obj, attr_name in reversed(all_attrs):
                self._restore(obj, attr_name)
