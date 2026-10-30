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
from sglang.srt.runtime_context import get_parallel

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


def get_cp_decode_attn_tp_ctx() -> CpDecodeAttnTpContext:
    """Return the global CpDecodeAttnTpContext singleton."""
    global _global_cp_decode_attn_tp_ctx
    if _global_cp_decode_attn_tp_ctx is None:
        _global_cp_decode_attn_tp_ctx = CpDecodeAttnTpContext()
    return _global_cp_decode_attn_tp_ctx


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

    def _slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim in (0, 1)
        chunk = tensor.shape[dim] // self.decode_tp_size
        sliced = tensor.narrow(dim, self.decode_tp_rank * chunk, chunk)
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
            if (
                attr_name == "weight"
                and dim == 1
                and getattr(obj, "aiter_bpreshuffled", False)
            ):
                sliced = self._slice_aiter_bpreshuffled_weight(raw)
            else:
                sliced = self._slice(raw, dim)
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

    def prepare_humming_decode_shard(self, layer, weight_schema, input_schema):
        """Build a rank-local Humming layer from canonical schema tensors.

        Called after conversion/requantization, BEFORE packing/padding. Schema
        input/output dimensions describe channel/group/block scales and packed
        K weights (or packed N zero points); a shard boundary must land on a
        whole storage unit. Never slice an inference-layout Humming tensor.
        """
        if not self.is_enabled:
            return
        from humming.layer import HummingMethod

        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        if isinstance(layer, RowParallelLinear):
            dim_name, logical_dim = "input_dim", 1
        elif isinstance(layer, ColumnParallelLinear):
            dim_name, logical_dim = "output_dim", 0
        else:
            raise ValueError("CP Humming shard requires a parallel linear")
        if len(layer.output_partition_sizes) != 1:
            raise ValueError("CP Humming shard does not support merged linears")
        n, k = layer.output_partition_sizes_sum, layer.input_size_per_partition
        shape = [n, k]
        extent = shape[logical_dim]
        if extent % self.decode_tp_size:
            raise ValueError("CP Humming logical dimension is not divisible by CP size")
        shape[logical_dim] //= self.decode_tp_size
        attrs = weight_schema.get_tensors_attrs(
            shape_n=n,
            shape_k=k,
            param_dtype=layer.param_dtype,
            has_bias=layer.with_bias,
        )
        local_attrs = weight_schema.get_tensors_attrs(
            shape_n=shape[0],
            shape_k=shape[1],
            param_dtype=layer.param_dtype,
            has_bias=layer.with_bias,
        )
        shard = torch.nn.Module()
        for name, spec in attrs.items():
            tensor = getattr(layer, name).detach()
            if tuple(tensor.shape) != tuple(spec["shape"]):
                raise ValueError(f"CP Humming canonical tensor shape mismatch: {name}")
            axis = spec.get("extra_attrs", {}).get(dim_name)
            if axis is not None:
                units = tensor.shape[axis]
                # This check covers packed words and scale blocks, including
                # non-aligned boundaries that floor-divided schema shapes hide.
                if not units or extent % units or units % self.decode_tp_size:
                    raise ValueError(
                        f"CP Humming shard boundary splits a packed word/scale block: {name}"
                    )
                count = units // self.decode_tp_size
                tensor = tensor.narrow(axis, self.decode_tp_rank * count, count)
            if tuple(tensor.shape) != tuple(local_attrs[name]["shape"]):
                raise ValueError(f"CP Humming shard schema mismatch: {name}")
            # Transform may process tensors in place; do not alias full weights.
            shard.register_parameter(
                name,
                torch.nn.Parameter(tensor.clone().contiguous(), requires_grad=False),
            )
        shard.register_buffer("locks", torch.zeros_like(layer.locks))
        HummingMethod.prepare_layer_meta(
            layer=shard,
            shape_n=shape[0],
            shape_k=shape[1],
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            has_bias=layer.with_bias,
            torch_dtype=layer.param_dtype,
        )
        HummingMethod.transform_humming_layer(shard)
        # Keep a plain dict, not a child module: model weight-loader traversals
        # must not process this already-transformed layout a second time.
        names = (
            "weight",
            "weight_scale",
            "weight_scale_2",
            "zero_point",
            "bias",
            "locks",
        )
        layer._cp_decode_humming_state = {
            name: getattr(shard, name, None) for name in names
        }
        layer._cp_decode_humming_state["humming_metas"] = shard.humming_metas
        shard_bytes = sum(
            value.numel() * value.element_size()
            for value in layer._cp_decode_humming_state.values()
            if isinstance(value, torch.Tensor)
        )
        logger.debug(
            "Prepared CP decode Humming shard %s rank=%d/%d N=%d K=%d: %.2f MiB",
            getattr(layer, "prefix", type(layer).__name__),
            self.decode_tp_rank,
            self.decode_tp_size,
            shape[0],
            shape[1],
            shard_bytes / (1024 * 1024),
        )

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
        humming_states = []
        orig_tp_q_head_num = None
        try:
            for linear in modules:
                if hasattr(linear, "humming_metas"):
                    state = getattr(linear, "_cp_decode_humming_state", None)
                    if state is None:
                        raise RuntimeError(
                            "CP decode Humming shard was not prepared before weight transform"
                        )
                    original = {name: getattr(linear, name, None) for name in state}
                    humming_states.append((linear, original))
                    for name, value in state.items():
                        setattr(linear, name, value)
                else:
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
            for linear, original in reversed(humming_states):
                for name, value in original.items():
                    setattr(linear, name, value)
