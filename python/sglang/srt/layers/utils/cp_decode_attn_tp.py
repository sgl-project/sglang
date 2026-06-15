# SPDX-License-Identifier: Apache-2.0
"""CP Decode Attention TP context.

When CP (Context Parallel) mode sets tp_size=1 (repeat weights), decode can
partition attention weights across CP ranks matching normal TP behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class CpDecodeAttnTpContext:
    """Context for managing CP-mode decode attention TP partitioning.

    This class encapsulates the logic for determining whether decode attention TP
    should be enabled, and provides utilities for slicing weights and computing
    local dimensions.
    """

    def __init__(self):
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        enable_attn_tp = envs.SGLANG_CP_DECODE_ATTN_TP.get()

        if dsa_enable_prefill_cp and enable_attn_tp:
            self.decode_tp_rank = get_attention_cp_rank()
            self.decode_tp_size = get_attention_cp_size()
            logger.info("Enable CP decode attention TP")
        else:
            self.decode_tp_rank = None
            self.decode_tp_size = None
            logger.info("Disable CP decode attention TP")

        # State for current forward: whether to use attention TP
        self.use_decode_attn_tp = False

        # Per-linear cache for sliced state.
        # Keyed by id(linear_instance).
        self._slice_cache: Dict[int, dict] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if CP decode attention TP is enabled (has valid rank and size)."""
        return self.decode_tp_size is not None and self.decode_tp_size > 1

    def set_decode_attn_tp(self, forward_batch: "ForwardBatch"):
        """Set whether to use attention TP for current forward.

        Attention TP is used when:
        1. CP mode is enabled (tp_size=1, weights repeated)
        2. We're not in CP prefill mode (which needs all heads)
        """
        if not self.is_enabled:
            self.use_decode_attn_tp = False
            return

        self.use_decode_attn_tp = not dsa_use_prefill_cp(forward_batch)

    @contextmanager
    def maybe_use_decode_attn_tp(self, forward_batch: ForwardBatch, modules):
        self.set_decode_attn_tp(forward_batch)
        if not self.use_decode_attn_tp:
            yield
            return

        activated = []
        try:
            for module in modules:
                self.activate(module)
                activated.append(module)
            yield
        finally:
            for module in reversed(activated):
                self.restore(module)

    # ============== Unified initialization for Linear layers ==============

    def init_slices(
        self,
        linear_instance,
        dim: int,  # 0 for column, 1 for row
    ) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """Initialize sliced weight and scales.

        Args:
            linear_instance: The Linear instance (to get weight and scales).
            dim: 0=column (slice dim 0), 1=row (slice dim 1)

        Returns:
            Tuple of (sliced_weight, sliced_size, scales)
        """
        assert dim in (0, 1), f"dim must be 0 or 1, but got {dim}"
        weight = linear_instance.weight.data
        if not self.is_enabled:
            if dim == 0:
                return weight, weight.shape[0], {}
            else:
                return weight, weight.shape[1], {}

        rank, size = self.decode_tp_rank, self.decode_tp_size

        assert (
            weight.shape[dim] % size == 0
        ), f"weight shape {weight.shape} {dim=} is not divisible by {size}"
        if dim == 0:
            # Column parallel: slice along dim 0
            chunk = weight.shape[0] // size
            start, end = rank * chunk, (rank + 1) * chunk
            sliced_weight = weight[start:end]
            sliced_size = sliced_weight.shape[0]
        else:
            # Row parallel: slice along dim 1
            chunk = weight.shape[1] // size
            start, end = rank * chunk, (rank + 1) * chunk
            sliced_weight = weight[:, start:end].contiguous()
            sliced_size = sliced_weight.shape[1]

        # Slice scales (independently compute chunk size from the scale's own shape,
        # may have different shapes, e.g. block quantization)
        scales = {}
        for attr in ("weight_scale_inv", "weight_scale"):
            s = getattr(linear_instance, attr, None)
            if s is not None:
                s = s.data
                assert (
                    s.shape[dim] % size == 0
                ), f"scale {attr} shape {s.shape} {dim=} is not divisible by {size}"
                if dim == 0:
                    sc = s.shape[0] // size
                    scales[attr] = s[rank * sc : (rank + 1) * sc]
                else:
                    sc = s.shape[1] // size
                    scales[attr] = s[:, rank * sc : (rank + 1) * sc].contiguous()

        return sliced_weight, sliced_size, scales

    def activate(
        self,
        linear_instance,
    ):
        """Activate attention TP for a linear layer.

        Lazily initializes sliced weight/scales on first call for each linear,
        then replaces weight.data and scale .data with sliced versions.
        The caller must call restore() after the GEMM to revert to originals.

        Args:
            linear_instance: The Linear layer instance (ColumnParallelLinear or
                RowParallelLinear). Dim is inferred from the type.
        """
        if not self.use_decode_attn_tp:
            return
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        if isinstance(linear_instance, RowParallelLinear):
            dim = 1
            assert linear_instance.cp_decode_tp_ctx is not None
        elif isinstance(linear_instance, ColumnParallelLinear):
            dim = 0
        else:
            return

        cache_key = id(linear_instance)
        cache = self._slice_cache.get(cache_key)

        # First time init: compute slices and save originals
        if cache is None:
            size_attr = (
                "output_size_per_partition" if dim == 0 else "input_size_per_partition"
            )
            sliced_weight, sliced_size, scales = self.init_slices(linear_instance, dim)
            cache = {
                "orig_weight": linear_instance.weight.data,
                "orig_size": getattr(linear_instance, size_attr),
                "orig_scales": {
                    attr: getattr(linear_instance, attr).data for attr in scales.keys()
                },
                "sliced_weight": sliced_weight,
                "sliced_size": sliced_size,
                "sliced_scales": scales,
                "size_attr": size_attr,
            }
            self._slice_cache[cache_key] = cache

        # Activate: replace weight.data and size with sliced versions
        linear_instance.weight.data = cache["sliced_weight"]
        setattr(linear_instance, cache["size_attr"], cache["sliced_size"])
        for attr, sliced in cache["sliced_scales"].items():
            getattr(linear_instance, attr).data = sliced

    def restore(self, linear_instance):
        """Restore original weights after attention TP mode.

        Must be called after activate() once the GEMM is done.
        No-op if attention TP is not active or was never activated for this linear.
        """
        if not self.use_decode_attn_tp:
            return
        cache = self._slice_cache.get(id(linear_instance))
        if cache is None:
            return
        linear_instance.weight.data = cache["orig_weight"]
        setattr(linear_instance, cache["size_attr"], cache["orig_size"])
        for attr, orig in cache["orig_scales"].items():
            getattr(linear_instance, attr).data = orig
