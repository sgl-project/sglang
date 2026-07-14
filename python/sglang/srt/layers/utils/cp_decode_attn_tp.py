# SPDX-License-Identifier: Apache-2.0
"""CP Decode Attention TP context.

When CP (Context Parallel) mode sets tp_size=1 (repeat weights), decode can
partition attention weights across CP ranks matching normal TP behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_global_cp_decode_attn_tp_ctx: CpDecodeAttnTpContext | None = None


def get_cp_decode_attn_tp_ctx() -> CpDecodeAttnTpContext:
    """Return the global CpDecodeAttnTpContext singleton."""
    global _global_cp_decode_attn_tp_ctx
    if _global_cp_decode_attn_tp_ctx is None:
        _global_cp_decode_attn_tp_ctx = CpDecodeAttnTpContext()
    return _global_cp_decode_attn_tp_ctx


class CpDecodeAttnTpContext:
    """Slices replicated attention weights across CP ranks during decode."""

    def __init__(self):
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        disable_attn_tp = envs.SGLANG_CP_DISABLE_DECODE_SLICE.get()

        if dsa_enable_prefill_cp and not disable_attn_tp:
            self.decode_tp_rank = get_parallel().attn_cp_rank
            self.decode_tp_size = get_parallel().attn_cp_size
            logger.info("Enable CP decode attention TP")
        else:
            self.decode_tp_rank = None
            self.decode_tp_size = None
            logger.info("Disable CP decode attention TP")

        self.use_decode_attn_tp = False
        self._slice_cache: Dict = {}

    @property
    def is_enabled(self) -> bool:
        return self.decode_tp_size is not None and self.decode_tp_size > 1

    def set_decode_attn_tp(self, forward_batch: ForwardBatch):
        if not self.is_enabled:
            self.use_decode_attn_tp = False
            return
        self.use_decode_attn_tp = not dsa_use_prefill_cp(forward_batch)

    def _slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim in (0, 1)
        chunk = tensor.shape[dim] // self.decode_tp_size
        sliced = tensor.narrow(dim, self.decode_tp_rank * chunk, chunk)
        return sliced if dim == 0 else sliced.contiguous()

    # ==================== Unified activate/restore ====================

    def _activate(self, obj, attr_name: str, dim: int):
        """Replace obj.attr_name with its TP-sliced version. No-op if attr is None."""
        tensor = getattr(obj, attr_name, None)
        if tensor is None:
            return
        is_param = isinstance(tensor, torch.nn.Parameter)
        raw = tensor.data if is_param else tensor
        assert isinstance(raw, torch.Tensor) and raw.dim() > dim, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name} is not sliceable "
            f"(type={type(tensor).__name__}, dim={raw.dim()}, required_dim>{dim})"
        )
        assert raw.shape[dim] % self.decode_tp_size == 0, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name}.shape[{dim}]={raw.shape[dim]} "
            f"not divisible by decode_tp_size={self.decode_tp_size}"
        )

        cache_key = (id(obj), attr_name)
        cache = self._slice_cache.get(cache_key)
        if cache is None:
            cache = (raw, self._slice(raw, dim), is_param)
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
            if getattr(linear_instance, scale_name, None) is not None:
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
    ):
        """Activate decode attention TP for the duration of the block.

        Args:
            modules: Linear layers (ColumnParallel/RowParallel) to slice.
            tensor_attrs: (obj, attr_name, dim) tuples for absorbed weights.
            radix_attn: RadixAttention instance whose tp_q_head_num should be
                overridden to match the sliced head count during decode TP.
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
                    linear.use_decode_attn_tp = True

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
